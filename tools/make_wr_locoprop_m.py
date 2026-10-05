#!/usr/bin/env python3
"""Generate a WR train_gpt.py variant with sampled MLP c_fc LocoProp-M.

The generated script keeps the baseline WR optimizer path intact, then adds a
state-decoupled local matching-loss displacement to owned MLP c_fc matrices.
It is intentionally narrow: c_fc only, sampled tokens, no extra full-model
forward/backward passes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SHARED_LOCAL_SOURCE = Path(__file__).with_name("locoprop_local.py").read_text()


WR_LOCOM_CONFIG = r'''
def _wr_locom_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def _wr_locom_parse_layer_set(spec: str, total_layers: int = 11) -> set[int]:
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return set(range(total_layers))
    if spec in {"", "none"}:
        return set()
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers

WR_LOCOM_ENABLED = _wr_locom_env_flag("WR_LOCOM_ENABLED", "1")
WR_LOCOM_LAYERS_SPEC = os.environ.get("WR_LOCOM_LAYERS", "all")
WR_LOCOM_LAYER_SET = _wr_locom_parse_layer_set(WR_LOCOM_LAYERS_SPEC)
WR_LOCOM_SAMPLE_TOKENS = int(os.environ.get("WR_LOCOM_SAMPLE_TOKENS", "2048"))
WR_LOCOM_LOCAL_STEPS = int(os.environ.get("WR_LOCOM_STEPS", "4"))
WR_LOCOM_INNER_LR = float(os.environ.get("WR_LOCOM_INNER_LR", "0.1"))
WR_LOCOM_TARGET_GAMMA = float(os.environ.get("WR_LOCOM_TARGET_GAMMA", "1.0"))
WR_LOCOM_PROX = float(os.environ.get("WR_LOCOM_PROX", "0.1"))
WR_LOCOM_ALPHA = float(os.environ.get("WR_LOCOM_ALPHA", "1.0"))
WR_LOCOM_NORM_TO_BASE = _wr_locom_env_flag("WR_LOCOM_NORM_TO_BASE", "0")
WR_LOCOM_NORM_CAP = float(os.environ.get("WR_LOCOM_NORM_CAP", "0.20"))
WR_LOCOM_REQUIRE_LOSS_DECREASE = _wr_locom_env_flag("WR_LOCOM_REQUIRE_LOSS_DECREASE", "1")
WR_LOCOM_MIN_COS_DESC = float(os.environ.get("WR_LOCOM_MIN_COS_DESC", "0.0"))
WR_LOCOM_START_STEP = int(os.environ.get("WR_LOCOM_START_STEP", "0"))
WR_LOCOM_END_STEP = int(os.environ.get("WR_LOCOM_END_STEP", "1000000000"))
WR_LOCOM_INTERVAL = int(os.environ.get("WR_LOCOM_INTERVAL", "1"))
WR_LOCOM_LOG_STEPS = {
    int(x)
    for x in os.environ.get("WR_LOCOM_LOG_STEPS", "0,1,2,10,50,125,250,500").split(",")
    if x.strip()
}

WR_LOCOM_MAX_BACKTRACKS = int(os.environ.get("WR_LOCOM_MAX_BACKTRACKS", "20"))
WR_LOCOM_ACCUM_SAMPLES = _wr_locom_env_flag("WR_LOCOM_ACCUM_SAMPLES", "1")
WR_LOCOM_CAPTURE_BUFFERS = {}

WR_LOCOM_CURRENT_STEP = -1
WR_LOCOM_OWNED_LAYER_SET: set[int] = set()
WR_LOCOM_LAYER_CORR: dict[int, tuple] = {}
WR_LOCOM_APPLY_STATS: list[str] = []

def _wr_locom_active(step: int) -> bool:
    return (
        WR_LOCOM_ENABLED
        and WR_LOCOM_LOCAL_STEPS > 0
        and step >= WR_LOCOM_START_STEP
        and step < WR_LOCOM_END_STEP
        and step % max(WR_LOCOM_INTERVAL, 1) == 0
    )

def _wr_locom_layer_active(layer_idx: int) -> bool:
    return layer_idx in WR_LOCOM_LAYER_SET

@torch.no_grad()
def _wr_locom_begin_step(step: int):
    global WR_LOCOM_CURRENT_STEP
    WR_LOCOM_CURRENT_STEP = step
    loco_begin_capture(step, grad_accum_steps, _wr_locom_active(step),
                       WR_LOCOM_LAYER_SET, WR_LOCOM_ACCUM_SAMPLES, dist.get_rank())
    WR_LOCOM_LAYER_CORR.clear()
    WR_LOCOM_APPLY_STATS.clear()

def initialize_wr_loco_capture(model):
    if not WR_LOCOM_ENABLED:
        return
    dim, hidden = model.mlp_bank.shape[-1], model.mlp_bank.shape[-2]
    for layer in sorted(WR_LOCOM_LAYER_SET):
        if layer < 0 or layer >= 11:
            raise ValueError("WR LocoProp layer index must be in 0..10")
        buffer = loco_sample_buffer(WR_LOCOM_SAMPLE_TOKENS, dim, hidden, world_size)
        buffer = buffer.to(device=model.mlp_bank.device)
        model.register_buffer(f"_wr_loco_samples_{layer}", buffer, persistent=False)
        WR_LOCOM_CAPTURE_BUFFERS[layer] = buffer

@torch.no_grad()
def _wr_locom_gather_sample(t: Tensor) -> Tensor:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t.float()
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0).float()

@torch.no_grad()
def attach_wr_locoprop_m_optimizer(loco_model: nn.Module, optimizer) -> None:
    global WR_LOCOM_OWNED_LAYER_SET
    if not WR_LOCOM_ENABLED:
        return
    mlp_param = optimizer._param_by_label["mlp_bank"]
    mlp_cfg = optimizer.param_cfgs[mlp_param]
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    start_idx = local_rank * mlp_cfg.chunk_size
    end_idx = start_idx + mlp_cfg.chunk_size
    owned = set()
    num_mlp_real = 22
    for global_idx in range(start_idx, end_idx):
        if global_idx >= num_mlp_real:
            continue
        if global_idx % 2 == 0:
            layer_idx = global_idx // 2
            if layer_idx in WR_LOCOM_LAYER_SET:
                owned.add(layer_idx)
    WR_LOCOM_OWNED_LAYER_SET = owned
    print0(
        f"wr_locom_owner rank={local_rank} world={world_size} "
        f"owned_layers={sorted(owned)} global_sample_tokens={WR_LOCOM_SAMPLE_TOKENS}",
        console=True,
    )

@torch.no_grad()
def prepare_wr_locoprop_m(loco_model: nn.Module, step: int) -> None:
    if not _wr_locom_active(step):
        return
    stats = []
    for layer_idx in sorted(WR_LOCOM_LAYER_SET):
        sx, sp, sg = loco_samples(WR_LOCOM_CAPTURE_BUFFERS[layer_idx], layer_idx,
                                  loco_model.mlp_bank.shape[-1])
        if sx.shape[0] == 0:
            continue
        # Collectives must occur in identical layer order on every rank.
        x, pre0, dpre = _wr_locom_gather_sample(sx), _wr_locom_gather_sample(sp), _wr_locom_gather_sample(sg)
        if layer_idx not in WR_LOCOM_OWNED_LAYER_SET:
            continue
        # The bank's full gradient is reduce-scattered inside optimizer.step.
        # Solve against the shared sample gradient here, then check alignment
        # with that reduced full gradient immediately before applying.
        raw_grad = dpre.mT @ x / x.shape[0]
        corr, diag, _ = loco_solve(
            x, pre0, dpre, raw_grad, steps=WR_LOCOM_LOCAL_STEPS,
            inner_lr=WR_LOCOM_INNER_LR, gamma=WR_LOCOM_TARGET_GAMMA,
            prox=WR_LOCOM_PROX, min_cos=WR_LOCOM_MIN_COS_DESC,
            require_decrease=WR_LOCOM_REQUIRE_LOSS_DECREASE,
            max_backtracks=WR_LOCOM_MAX_BACKTRACKS, output_dtype=loco_model.mlp_bank.dtype,
        )
        if diag["accepted"]:
            WR_LOCOM_LAYER_CORR[layer_idx] = (corr, loco_model.mlp_bank[layer_idx, 0].detach().float().clone(), x, pre0, dpre)
        if step in WR_LOCOM_LOG_STEPS and len(stats) < 6:
            stats.append(
                f"l{layer_idx}:loss0={diag['loss0']:.3e},lossK={diag['lossK']:.3e}"
                f",corr_norm={float(corr.float().norm()):.3e}"
                f",grad_norm={float(raw_grad.norm()):.3e}"
                f",cos_desc={diag['cos_desc']:.3f},accepted={int(diag['accepted'])},tokens={x.size(0)}"
                f",backtracks={diag['backtracks']},local_steps={diag['local_steps']}"
                f",inner_lr={diag['inner_lr']:.3e},reason={diag['reason']}"
            )

    if step in WR_LOCOM_LOG_STEPS and stats:
        print0("wr_locom_prepare step=" + str(step) + " " + " | ".join(stats), console=True)

@torch.no_grad()
def _wr_locom_apply_mlp_chunk_corrections(optimizer, p_slice: Tensor, p_cfg, p_state: dict, v_chunk: Tensor, local_rank: int, raw_grad_chunk: Tensor) -> None:
    if not (WR_LOCOM_ENABLED and p_cfg.label == "mlp_bank" and _wr_locom_active(WR_LOCOM_CURRENT_STEP)):
        return
    start_idx = local_rank * p_cfg.chunk_size
    for mat_idx in range(p_cfg.chunk_size):
        global_idx = start_idx + mat_idx
        if global_idx >= 22 or global_idx % 2 == 1:
            continue
        layer_idx = global_idx // 2
        context = WR_LOCOM_LAYER_CORR.pop(layer_idx, None)
        if context is None:
            continue
        corr, reference, x, pre0, dpre = context
        raw_grad = raw_grad_chunk[mat_idx]
        denominator = (raw_grad.norm() * corr.float().norm()).clamp_min(1e-30)
        cosine = -(corr.float() * raw_grad).sum() / denominator
        if not bool(torch.isfinite(cosine) & torch.isfinite(denominator)) or float(cosine) < WR_LOCOM_MIN_COS_DESC or float(raw_grad.norm()) == 0:
            if WR_LOCOM_CURRENT_STEP in WR_LOCOM_LOG_STEPS:
                WR_LOCOM_APPLY_STATS.append(f"l{layer_idx}:accepted_apply=0,cos_desc={float(cosine):.3f},reason=full_gradient_gate")
            continue

        lr_mul = p_cfg.lr_mul
        if p_cfg.per_matrix_lr_mul is not None:
            lr_mul *= p_cfg.per_matrix_lr_mul[mat_idx]
        base_step_norm = v_chunk[mat_idx].float().norm().mul(p_cfg.lr * lr_mul)
        corr_norm = corr.float().norm().clamp_min(1e-12)
        scale = loco_correction_scale(corr, base_step_norm, alpha=WR_LOCOM_ALPHA,
                                     cap=WR_LOCOM_NORM_CAP, norm_to_base=WR_LOCOM_NORM_TO_BASE)
        scale = loco_post_step_scale(corr, scale, p_slice[mat_idx].float() - reference, x, pre0, dpre,
                                    gamma=WR_LOCOM_TARGET_GAMMA, prox=WR_LOCOM_PROX,
                                    max_backtracks=WR_LOCOM_MAX_BACKTRACKS)
        if float(scale) == 0:
            if WR_LOCOM_CURRENT_STEP in WR_LOCOM_LOG_STEPS:
                WR_LOCOM_APPLY_STATS.append(f"l{layer_idx}:accepted_apply=0,scale=0,reason=post_step_gate")
            continue
        optimizer._loco_full_add_lr_t.fill_(float(scale))
        neg_corr = corr.neg()
        NorMuonAndAdam._cautious_wd_and_update_inplace(
            p_slice[mat_idx].view(torch.uint16),
            p_state["mantissa"][mat_idx],
            neg_corr,
            optimizer._zero_t,
            optimizer._loco_full_add_lr_t,
        )
        if WR_LOCOM_CURRENT_STEP in WR_LOCOM_LOG_STEPS and len(WR_LOCOM_APPLY_STATS) < 8:
            WR_LOCOM_APPLY_STATS.append(
                f"l{layer_idx}:base_step={float(base_step_norm):.3e}"
                f",corr_norm={float(corr_norm):.3e},scale={float(scale):.3e}"
                f",accepted_apply=1,cos_desc={float(cosine):.3f}"
            )

@torch.no_grad()
def flush_wr_locoprop_m_apply_stats(step: int) -> None:
    if step in WR_LOCOM_LOG_STEPS and WR_LOCOM_APPLY_STATS:
        print0("wr_locom_apply step=" + str(step) + " " + " | ".join(WR_LOCOM_APPLY_STATS[:8]), console=True)
    WR_LOCOM_APPLY_STATS.clear()

class FusedLinearReLUSquareLocoMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, samples, layer_idx: int):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)
        y = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post, samples)
        ctx.layer_idx = int(layer_idx)
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post, samples = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        dW2 = post.T @ flat_grad
        dpre = linear_relu_square(flat_grad, W2, aux=pre)
        loco_capture(flat_x, pre, dpre, samples, ctx.layer_idx)
        dW1 = dpre.T @ flat_x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2, None, None

ReLUSqrdMLPLocoM = FusedLinearReLUSquareLocoMFunction.apply

def _wr_locom_mlp(x: Tensor, W1: Tensor, W2: Tensor, layer_idx: int, training: bool) -> Tensor:
    # Static branch; active windows are read only by the opaque capture op.
    if WR_LOCOM_ENABLED and _wr_locom_layer_active(layer_idx):
        return ReLUSqrdMLPLocoM(x, W1, W2, WR_LOCOM_CAPTURE_BUFFERS[layer_idx], layer_idx)
    return ReLUSqrdMLP(x, W1, W2)
'''


def replace_exact(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found:\n{old[:300]}")
    return text.replace(old, new, 1)


def generate(source: Path, output: Path, train_steps: int) -> None:
    text = source.read_text()
    text = replace_exact(
        text,
        "from triton_kernels import XXT, XTX, ba_plus_cAA, FusedLinearReLUSquareFunction, FusedLinearReLUSquareWithDiagFunction, FusedSoftcappedCrossEntropy, transpose_add, transpose_copy\n",
        "from triton_kernels import XXT, XTX, ba_plus_cAA, FusedLinearReLUSquareFunction, FusedLinearReLUSquareWithDiagFunction, FusedSoftcappedCrossEntropy, linear_relu_square, transpose_add, transpose_copy\n",
    )
    text = replace_exact(
        text,
        "ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply\nReLUSqrdMLPWithDiag = FusedLinearReLUSquareWithDiagFunction.apply\n",
        "ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply\nReLUSqrdMLPWithDiag = FusedLinearReLUSquareWithDiagFunction.apply\n" + SHARED_LOCAL_SOURCE + WR_LOCOM_CONFIG,
    )
    text = replace_exact(
        text,
        "args = Hyperparameters()\n",
        "args = Hyperparameters()\n"
        f"args.num_scheduled_iterations = {train_steps}  # generated by tools/make_wr_locoprop_m.py\n"
        "args.num_extension_iterations = 0\n"
        "args.val_loss_every = int(os.environ.get(\"SCREEN_VAL_EVERY\", str(args.val_loss_every)))\n",
    )
    text = text.replace(
        "mlp_out = ReLUSqrdMLP(mlp_in, c_fc, c_proj)",
        "mlp_out = _wr_locom_mlp(mlp_in, c_fc, c_proj, i, self.training)",
    )
    text = replace_exact(
        text,
        "        self.optimizer = NorMuonAndAdam(\n"
        "            model.named_parameters(),\n"
        "            param_table=self.param_table,\n"
        "            scatter_order=list(self.param_table),  # Dict order defines scatter priority\n"
        "            work_order=self.work_order,\n"
        "            adam_defaults=adam_defaults,\n"
        "            normuon_defaults=normuon_defaults,\n"
        "            loco_diag_model=getattr(model, \"_orig_mod\", model) if LOCO_FEATURE_ACTIVE else None,\n"
        "        )\n"
        "        if LOCO_FULL_ACTIVE:\n",
        "        self.optimizer = NorMuonAndAdam(\n"
        "            model.named_parameters(),\n"
        "            param_table=self.param_table,\n"
        "            scatter_order=list(self.param_table),  # Dict order defines scatter priority\n"
        "            work_order=self.work_order,\n"
        "            adam_defaults=adam_defaults,\n"
        "            normuon_defaults=normuon_defaults,\n"
        "            loco_diag_model=getattr(model, \"_orig_mod\", model) if (LOCO_FEATURE_ACTIVE or WR_LOCOM_ENABLED) else None,\n"
        "        )\n"
        "        if WR_LOCOM_ENABLED:\n"
        "            attach_wr_locoprop_m_optimizer(getattr(model, \"_orig_mod\", model), self.optimizer)\n"
        "        if LOCO_FULL_ACTIVE:\n",
    )
    text = replace_exact(
        text,
        "        grad_chunk = grad_chunk.float()  # FP32 for momentum\n",
        "        grad_chunk = grad_chunk.float()  # FP32 for momentum\n"
        "        loco_raw_grad_chunk = grad_chunk.clone() if (WR_LOCOM_ENABLED and p_cfg.label == \"mlp_bank\" and _wr_locom_active(WR_LOCOM_CURRENT_STEP)) else None\n",
    )
    text = replace_exact(
        text,
        "        return p_slice\n\n    def _loco_diag_col_update_fn(self):\n",
        "        _wr_locom_apply_mlp_chunk_corrections(self, p_slice, p_cfg, p_state, v_chunk, rank, loco_raw_grad_chunk)\n"
        "        return p_slice\n\n    def _loco_diag_col_update_fn(self):\n",
    )
    text = replace_exact(
        text,
        "    def advance_schedule(self, step: int):\n        if LOCO_FULL_ACTIVE:\n",
        "    def advance_schedule(self, step: int):\n        _wr_locom_begin_step(step)\n        if LOCO_FULL_ACTIVE:\n",
    )
    text = replace_exact(
        text,
        "        self._prepare_loco_diag_buffers(step)\n        self._prepare_loco_full_buffers(step)\n        self.optimizer.step(do_adam=do_adam)\n        if LOCO_FEATURE_ACTIVE:\n",
        "        self._prepare_loco_diag_buffers(step)\n        self._prepare_loco_full_buffers(step)\n        prepare_wr_locoprop_m(getattr(self.model, \"_orig_mod\", self.model), step)\n        self.optimizer.step(do_adam=do_adam)\n        flush_wr_locoprop_m_apply_stats(step)\n        if LOCO_FEATURE_ACTIVE:\n",
    )
    text = replace_exact(
        text,
        "model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)\ntraining_manager = TrainingManager(model)\n",
        "initialize_wr_loco_capture(model)\n"
        "if WR_LOCOM_ENABLED and not _wr_locom_env_flag(\"WR_LOCOM_COMPILE\", \"1\"):\n"
        "    print0(\"WR LocoProp-M: eager model requested\", console=True)\n"
        "else:\n"
        "    model = torch.compile(model, dynamic=False, fullgraph=True)\n"
        "training_manager = TrainingManager(model)\n",
    )
    text = replace_exact(
        text,
        "if NEWTONV_PAIRED_CASES:\n    print0(f\"Using NEWTONV_PAIRED_CASES={','.join(NEWTONV_PAIRED_CASES)}\", console=True)\n",
        "if NEWTONV_PAIRED_CASES:\n"
        "    print0(f\"Using NEWTONV_PAIRED_CASES={','.join(NEWTONV_PAIRED_CASES)}\", console=True)\n"
        "print0(\n"
        "    f\"WR LocoProp-M generated run: total_steps={args.num_scheduled_iterations + args.num_extension_iterations} \"\n"
        "    f\"enabled={WR_LOCOM_ENABLED} layers={WR_LOCOM_LAYERS_SPEC} steps={WR_LOCOM_LOCAL_STEPS} \"\n"
        "    f\"sample_tokens={WR_LOCOM_SAMPLE_TOKENS} inner_lr={WR_LOCOM_INNER_LR} \"\n"
        "    f\"gamma={WR_LOCOM_TARGET_GAMMA} prox={WR_LOCOM_PROX} alpha={WR_LOCOM_ALPHA} \"\n"
        "    f\"norm_to_base={WR_LOCOM_NORM_TO_BASE} norm_cap={WR_LOCOM_NORM_CAP} \"\n"
        "    f\"require_loss_decrease={WR_LOCOM_REQUIRE_LOSS_DECREASE} min_cos_desc={WR_LOCOM_MIN_COS_DESC}\",\n"
        "    console=True,\n"
        ")\n",
    )
    text = text.replace(
        "for idx in range(grad_accum_steps):\n",
        "for idx in range(grad_accum_steps):\n" + "__LOCO_MICRO__\n",
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == "__LOCO_MICRO__":
            previous = lines[index - 1]
            indent = previous[:len(previous) - len(previous.lstrip())] + "    "
            lines[index] = indent + "loco_set_microbatch(idx)"
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="train_gpt.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    generate(Path(args.source), Path(args.output), args.steps)


if __name__ == "__main__":
    main()
