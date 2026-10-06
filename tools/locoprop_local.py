"""Numerical and capture primitives embedded into the generated training scripts.

Keep this module independent of the training driver: generators copy its source
so remote /tmp scripts do not need an import path back to this file.
"""

import math

import torch
from torch import Tensor


_LOCO_CAPTURE = {}


def loco_begin_capture(step, microbatches, active, layers, accumulate=True, rank=0, micro_cap=0):
    _LOCO_CAPTURE.clear()
    _LOCO_CAPTURE.update(step=step, microbatches=microbatches, micro=0,
                         active=active, layers=set(layers), accumulate=accumulate,
                         rank=rank, micro_cap=micro_cap, counts={})


def loco_set_microbatch(index):
    _LOCO_CAPTURE["micro"] = index


def loco_sample_buffer(rows, dim, hidden, world=1):
    if rows <= 0:
        raise ValueError("LocoProp sample_tokens must be positive (capture is bounded)")
    return torch.zeros(max(1, (rows + world - 1) // world), dim + 2 * hidden)


@torch.library.custom_op("nanogpt_loco::capture", mutates_args=("samples",),
                         tags=(torch.Tag.cudagraph_unsafe,))
def loco_capture(x: Tensor, pre: Tensor, dpre: Tensor, samples: Tensor, layer: int) -> None:
    # Opaque to Dynamo/AOT. Only the compact buffer is mutable; the full
    # activation/gradient inputs are never cloned by functionalization. Python
    # schedule/RNG state must be read each invocation, so opt out of CUDA graphs.
    state = _LOCO_CAPTURE
    if not state.get("active", False) or layer not in state["layers"]:
        return
    micro, micros = state["micro"], state["microbatches"]
    if not state["accumulate"] and micro != micros - 1:
        return
    x = x.reshape(-1, x.shape[-1])
    pre = pre.reshape(-1, pre.shape[-1])
    dpre = dpre.reshape_as(pre)
    capacity = samples.shape[0]
    if state["accumulate"]:
        budget = min(capacity, micros * x.shape[0])
        if state["micro_cap"] > 0:
            budget = min(budget, micros * state["micro_cap"])
        start, end = micro * budget // micros, (micro + 1) * budget // micros
    else:
        start, end = 0, min(capacity, x.shape[0])
    count = end - start
    if count == 0:
        return
    # Stratified jitter covers token positions without consuming training RNG.
    generator = torch.Generator(device=x.device)
    generator.manual_seed((state["step"] * 1000003 + layer * 1009 +
                           micro * 9176 + state["rank"] * 65537) % (2**63 - 1))
    lo = torch.arange(count, device=x.device) * x.shape[0] // count
    hi = (torch.arange(count, device=x.device) + 1) * x.shape[0] // count
    indices = lo + (torch.rand(count, device=x.device, generator=generator) * (hi - lo)).long()
    dim, hidden = x.shape[1], pre.shape[1]
    out = samples[start:end]
    out[:, :dim].copy_(x.index_select(0, indices))
    out[:, dim:dim + hidden].copy_(pre.index_select(0, indices))
    out[:, dim + hidden:].copy_(dpre.index_select(0, indices))
    state["counts"][layer] = end


@loco_capture.register_fake
def _loco_capture_fake(x, pre, dpre, samples, layer):
    return None


def loco_samples(samples, layer, dim):
    count = _LOCO_CAPTURE.get("counts", {}).get(layer, 0)
    hidden = (samples.shape[1] - dim) // 2
    rows = samples[:count]
    return rows[:, :dim], rows[:, dim:dim + hidden], rows[:, dim + hidden:]


class LocoMLPFunction(torch.autograd.Function):
    """Same BF16 linear/ReLU-squared math; bounded capture during backward.

    Recompute post-activations in backward instead of retaining both pre/post.
    Capture stays an opaque buffer write under torch.compile, without hooks.
    """

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, samples, layer):
        w1_cast, w2_cast = w1.to(x.dtype), w2.to(x.dtype)
        pre = torch.nn.functional.linear(x, w1_cast, b1.to(x.dtype))
        post = pre.relu().square()
        y = torch.nn.functional.linear(post, w2_cast, b2.to(x.dtype))
        ctx.save_for_backward(x, w1_cast, w2_cast, pre, samples)
        ctx.layer = layer
        ctx.dtypes = (w1.dtype, b1.dtype, w2.dtype, b2.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, pre, samples = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_pre = pre.reshape(-1, pre.shape[-1])
        flat_dy = dy.reshape(-1, dy.shape[-1])
        post = flat_pre.relu().square()
        dpost = flat_dy @ w2
        dpre = dpost * (flat_pre.relu() * 2)
        loco_capture(flat_x, flat_pre, dpre, samples, ctx.layer)
        dw2, db2 = flat_dy.mT @ post, flat_dy.sum(0)
        dw1, db1 = dpre.mT @ flat_x, dpre.sum(0)
        dx = (dpre @ w1).reshape_as(x)
        a, b, c, d = ctx.dtypes
        return dx, dw1.to(a), db1.to(b), dw2.to(c), db2.to(d), None, None


class LocoReLUSquareFunction(torch.autograd.Function):
    """Capture while preserving Linear module hooks on covariance refresh steps."""

    @staticmethod
    def forward(ctx, x, pre, samples, layer):
        ctx.save_for_backward(x, pre, samples)
        ctx.layer = layer
        return pre.relu().square()

    @staticmethod
    def backward(ctx, dpost):
        x, pre, samples = ctx.saved_tensors
        dpre = dpost * (2 * pre.relu())
        loco_capture(x, pre, dpre, samples, ctx.layer)
        return None, dpre, None, None


def _loco_matching_value(change, pre0, dpre, gamma, regularizer, target_space="post"):
    """Matching value from a projected affine displacement and its penalty."""
    z = pre0 + change
    if target_space == "pre":
        remainder = 0.5 * change.square()
    elif target_space == "post":
        positive = pre0.clamp_min(0)
        same_region = positive * change.square() + change.pow(3) / 3
        crossed = positive.square() * (2 * positive / 3 - z)
        remainder = torch.where(pre0 > 0, torch.where(z > 0, same_region, crossed),
                                z.clamp_min(0).pow(3) / 3)
    else:
        raise ValueError("target_space must be 'pre' or 'post'")
    return ((remainder + gamma * dpre * change).sum(dtype=torch.float64) / change.shape[0]
            + regularizer)


def loco_matching_objective(delta, x, pre0, dpre, gamma, prox, target_space="post", bias_delta=None,
                            linear_grad=None):
    """Actual regularized matching objective, relative to its value at delta=0.

    Anchoring at the captured preactivation eliminates BF16/FP32 replay error.
    The piecewise Bregman remainder avoids subtracting large cubic potentials.
    Feature dimensions are summed, rows averaged, matching err.T @ x / N.
    """
    change = x @ delta.mT
    if bias_delta is not None:
        change = change + bias_delta.float()
    # The unchanged bias penalty would cancel between post-step candidates.
    regularizer = 0.5 * prox * delta.square().sum(dtype=torch.float64)
    if linear_grad is not None:
        # The caller supplies the full mean-token weight gradient. Samples
        # estimate only the Bregman curvature; do not resample the linear term.
        regularizer = regularizer + gamma * (linear_grad * delta).sum(dtype=torch.float64)
    return _loco_matching_value(change, pre0, dpre, gamma if linear_grad is None else 0.,
                                regularizer, target_space)


@torch.no_grad()
def loco_solve(x, pre0, dpre, raw_grad, *, steps=4, inner_lr=0.1, gamma=1.0,
               prox=0.1, target_space="post", min_cos=0.0, require_decrease=True,
               max_backtracks=20, armijo=1e-4, output_dtype=torch.float32,
               local_opt="sgd", rms_avg=None, rms_mom=None, beta1=0.999,
               beta2=0.9, rms_eps=1e-5, reset_rms=False, lr_decay=False,
               rms_style="torch", scale_momentum=False, linear_grad=None):
    if steps < 0 or max_backtracks < 0:
        raise ValueError("steps and max_backtracks must be nonnegative")
    if not all(math.isfinite(v) for v in (inner_lr, gamma, prox, armijo, rms_eps)):
        raise ValueError("LocoProp solver settings must be finite")
    if inner_lr <= 0 or gamma < 0 or prox < 0 or not 0 < armijo < 1:
        raise ValueError("invalid LocoProp learning rate, target, proximal term or Armijo coefficient")
    if local_opt not in {"sgd", "rmsprop"}:
        raise ValueError("local_opt must be 'sgd' or 'rmsprop'")
    if rms_style not in {"torch", "tensorflow"}:
        raise ValueError("rms_style must be 'torch' or 'tensorflow'")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and rms_eps > 0):
        raise ValueError("invalid RMSProp settings")
    if x.ndim != 2 or pre0.ndim != 2 or pre0.shape != dpre.shape or x.shape[0] != pre0.shape[0]:
        raise ValueError("incompatible LocoProp sample shapes")
    x, pre0, dpre = x.float(), pre0.float(), dpre.float()
    delta = torch.zeros(pre0.shape[1], x.shape[1], device=x.device)
    if linear_grad is not None:
        if linear_grad.shape != delta.shape:
            raise ValueError("linear_grad must have the weight shape")
        linear_grad = linear_grad.float()
    info = dict(loss0=0.0, lossK=0.0, cos_desc=0.0, accepted=False,
                backtracks=0, local_steps=0, inner_lr=0.0, reason="zero_gradient",
                scaled_momentum=bool(scale_momentum))
    if x.shape[0] == 0 or raw_grad is None:
        info["reason"] = "missing_samples_or_gradient"
        return delta.to(output_dtype), info, None
    inputs = (x, pre0, dpre, raw_grad) + (() if linear_grad is None else (linear_grad,))
    if not all(bool(torch.isfinite(t).all()) for t in inputs):
        info["reason"] = "nonfinite_input"
        return delta.to(output_dtype), info, None
    avg = mom = None
    tensorflow_rms = local_opt == "rmsprop" and rms_style == "tensorflow"
    if local_opt == "rmsprop":
        if rms_avg is None or reset_rms:
            avg = torch.ones_like(delta) if tensorflow_rms else torch.zeros_like(delta)
        else:
            avg = rms_avg.float().clone()
        mom = torch.zeros_like(delta) if rms_mom is None or reset_rms else rms_mom.float().clone()
    value = loco_matching_objective(delta, x, pre0, dpre, gamma, prox, target_space,
                                    linear_grad=linear_grad)
    initial_post = pre0.relu().square()
    trial_lr = inner_lr
    tensorflow_backtrack_factor = 1.0
    for index in range(steps):
        change = x @ delta.mT
        residual = (change if target_space == "pre" else
                    (pre0 + change).relu().square() - initial_post)
        if linear_grad is None:
            grad = (residual + gamma * dpre).mT @ x / x.shape[0] + prox * delta
        else:
            grad = residual.mT @ x / x.shape[0] + gamma * linear_grad + prox * delta
        if not bool(torch.isfinite(grad).all()):
            info["reason"] = "nonfinite_gradient"
            break
        if float(grad.norm()) == 0:
            break
        next_avg, next_mom, direction = avg, mom, grad
        if local_opt == "rmsprop":
            next_avg = beta2 * avg + (1 - beta2) * grad.square()
            if tensorflow_rms:
                normalized_grad = grad / (next_avg + rms_eps).sqrt()
            else:
                next_mom = beta1 * mom + grad / (next_avg.sqrt() + rms_eps)
                direction = next_mom
        if not tensorflow_rms:
            slope = (grad * direction).sum(dtype=torch.float64)
            if not bool(torch.isfinite(slope)) or float(slope) <= 0:
                direction, slope = grad, grad.square().sum(dtype=torch.float64)
                next_mom = grad.clone()
        decay_fraction = max(1 - index / max(steps, 1), 0.25) if lr_decay else 1.0
        # TF's momentum slot stores the LR-weighted displacement. Its local
        # decay is relative to the reference LR, with backtracking independent
        # of the preceding iteration's decay. Keep the legacy torch path intact.
        lr = (inner_lr * tensorflow_backtrack_factor if tensorflow_rms else trial_lr) * decay_fraction
        found = False
        momentum_fraction = 1.0
        if tensorflow_rms and scale_momentum:
            # A line search must shrink the entire proposed displacement.
            # Shrinking only the new gradient leaves an arbitrarily large
            # carried momentum term that cannot approach the current iterate.
            proposed_mom = beta1 * mom + lr * normalized_grad
            proposed_slope = (grad * proposed_mom).sum(dtype=torch.float64)
            if not bool(torch.isfinite(proposed_slope)) or float(proposed_slope) <= 0:
                proposed_mom = lr * grad
        for _ in range(max_backtracks + 1):
            if tensorflow_rms:
                # Recompute from the original momentum for each trial LR;
                # rejected trials must not accumulate momentum or slot decay.
                next_mom = (momentum_fraction * proposed_mom if scale_momentum else
                            beta1 * mom + lr * normalized_grad)
                displacement_slope = (grad * next_mom).sum(dtype=torch.float64)
                if not bool(torch.isfinite(displacement_slope)) or float(displacement_slope) <= 0:
                    next_mom = lr * grad
                    displacement_slope = (grad * next_mom).sum(dtype=torch.float64)
                candidate = delta - next_mom
                decrease_bound = value - armijo * displacement_slope
            else:
                candidate = delta - lr * direction
                decrease_bound = value - armijo * lr * slope
            candidate_value = loco_matching_objective(candidate, x, pre0, dpre, gamma, prox, target_space,
                                                       linear_grad=linear_grad)
            if bool(torch.isfinite(candidate_value)) and bool(candidate_value <= decrease_bound):
                delta, value = candidate, candidate_value
                avg, mom = next_avg, next_mom
                found = True
                info["local_steps"] += 1
                info["inner_lr"] = lr * momentum_fraction
                if tensorflow_rms:
                    tensorflow_backtrack_factor = lr * momentum_fraction / (inner_lr * decay_fraction)
                else:
                    trial_lr = lr
                break
            if tensorflow_rms and scale_momentum:
                momentum_fraction *= 0.5
            else:
                lr *= 0.5
            info["backtracks"] += 1
        if not found:
            info["reason"] = "backtracking_exhausted"
            break
    corr = delta.to(output_dtype)
    # Evaluate the actual stored displacement after the last update/rounding.
    value = loco_matching_objective(corr.float(), x, pre0, dpre, gamma, prox, target_space,
                                    linear_grad=linear_grad)
    grad_norm, corr_norm = raw_grad.float().norm(), corr.float().norm()
    denom = (grad_norm * corr_norm).clamp_min(1e-30)
    cosine = -(corr.float() * raw_grad.float()).sum() / denom
    finite = bool(torch.isfinite(corr).all() & torch.isfinite(value) & torch.isfinite(cosine)
                  & torch.isfinite(grad_norm) & torch.isfinite(corr_norm))
    accepted = finite and float(corr_norm) > 0 and float(grad_norm) > 0
    accepted = accepted and float(cosine) >= min_cos
    accepted = accepted and (not require_decrease or float(value) < 0)
    info.update(lossK=float(value), cos_desc=float(cosine), accepted=accepted)
    if accepted:
        info["reason"] = "accepted"
    elif float(corr_norm) > 0:
        info["reason"] = "acceptance_gate"
    return corr, info, (avg, mom) if accepted and local_opt == "rmsprop" else None


@torch.no_grad()
def loco_correction_scale(corr, base_norm, *, alpha=1.0, cap=0.2,
                          norm_target=0.0, norm_to_base=False):
    if not all(math.isfinite(v) and v >= 0 for v in (alpha, cap, norm_target)):
        raise ValueError("LocoProp correction scales must be finite and nonnegative")
    norm = corr.float().norm()
    zero = norm.new_zeros(())
    if not bool(torch.isfinite(norm) & torch.isfinite(base_norm)) or float(norm) == 0 or float(base_norm) <= 0:
        return zero
    # Never amplify a solved displacement merely to fill a norm target.
    scale = norm.new_tensor(alpha)
    limit = norm_target if norm_target > 0 else (1.0 if norm_to_base else 0.0)
    if limit > 0:
        scale = torch.minimum(scale, limit * base_norm / norm)
    if cap > 0:
        scale = torch.minimum(scale, cap * base_norm / norm)
    return scale


@torch.no_grad()
def loco_post_step_scale(corr, scale, base_delta, x, pre0, dpre, *, gamma=1.0,
                         prox=0.1, target_space="post", max_backtracks=20,
                         bias_delta=None, current_weight=None, reference_weight=None,
                         return_info=False, linear_grad=None):
    """Gate at the outer step's affine state, including its native bias update.

    Project the base and correction once for scalar backtracking. An apparently
    successful trial is then checked using the actual stored-dtype addition and
    a fresh projection, so caching cannot admit a rounded no-op or false decrease.
    Optional current/reference weights model parameter rounding exactly; callers
    requesting diagnostics receive the verified candidate for an exact copy.
    """
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be nonnegative")
    if (current_weight is None) != (reference_weight is None):
        raise ValueError("current_weight and reference_weight must be supplied together")
    info = dict(attempts=0, backtracks=0, final_checks=0, reason="disabled", candidate=None)
    def result(accepted_scale):
        return (accepted_scale, info) if return_info else accepted_scale
    zero = scale.new_zeros(())
    if float(scale) <= 0:
        return result(zero)
    x, pre0, dpre = x.float(), pre0.float(), dpre.float()
    stored_corr = corr
    base_delta, corr = base_delta.float(), corr.float()
    base_change, corr_change = x @ base_delta.mT, x @ corr.mT
    if bias_delta is not None:
        base_change = base_change + bias_delta.float()
    base_sq = base_delta.square().sum(dtype=torch.float64)
    cross = (base_delta * corr).sum(dtype=torch.float64)
    corr_sq = corr.square().sum(dtype=torch.float64)
    linear_base = linear_corr = 0.
    sample_gamma = gamma
    if linear_grad is not None:
        linear_grad = linear_grad.float()
        linear_base = gamma * (linear_grad * base_delta).sum(dtype=torch.float64)
        linear_corr = gamma * (linear_grad * corr).sum(dtype=torch.float64)
        sample_gamma = 0.
    baseline = _loco_matching_value(base_change, pre0, dpre, sample_gamma,
                                    0.5 * prox * base_sq + linear_base, target_space)
    if not bool(torch.isfinite(baseline)):
        info["reason"] = "nonfinite_baseline"
        return result(zero)
    # This matching objective is convex along any affine weight direction.
    # A nonnegative directional derivative rules out every positive scale;
    # repeatedly halving that direction cannot repair it.
    residual = (base_change if target_space == "pre" else
                (pre0 + base_change).relu().square() - pre0.relu().square())
    slope = ((residual + sample_gamma * dpre) * corr_change).sum(dtype=torch.float64) / x.shape[0] + prox * cross + linear_corr
    if not bool(torch.isfinite(slope)) or float(slope) >= 0:
        info["reason"] = "non_descent_direction"
        return result(zero)
    for index in range(max_backtracks + 1):
        info["attempts"] += 1
        regularizer = 0.5 * prox * (base_sq + 2 * scale.double() * cross + scale.double().square() * corr_sq)
        regularizer = regularizer + linear_base + scale * linear_corr
        value = _loco_matching_value(base_change + scale * corr_change, pre0, dpre,
                                     sample_gamma, regularizer, target_space)
        if bool(torch.isfinite(value)) and bool(value < baseline):
            if current_weight is not None:
                candidate = current_weight.detach().clone()
                candidate.add_(stored_corr, alpha=float(scale))
                unchanged = bool(torch.equal(candidate, current_weight))
                combined = candidate.float() - reference_weight.float()
            else:
                candidate = None
                combined = base_delta + scale * corr
                unchanged = bool(torch.equal(combined, base_delta))
            if unchanged:
                # All smaller positive scales are also below storage precision.
                info["reason"] = "stored_noop"
                return result(zero)
            info["final_checks"] += 1
            exact_value = loco_matching_objective(combined, x, pre0, dpre, gamma, prox,
                                                   target_space, bias_delta=bias_delta, linear_grad=linear_grad)
            if bool(torch.isfinite(exact_value)) and bool(exact_value < baseline):
                info.update(reason="accepted", candidate=candidate)
                return result(scale)
        if index < max_backtracks:
            info["backtracks"] += 1
        scale = scale * 0.5
    info["reason"] = "no_decrease"
    return result(zero)
