import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("locoprop_local", ROOT / "tools/locoprop_local.py")
loco = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = loco
spec.loader.exec_module(loco)


def test_matching_objective_gradient_including_prox():
    x = torch.tensor([[1., 2.], [-1., .5]])
    pre = torch.tensor([[.2, -.3], [-.1, .4]])
    dpre = torch.tensor([[.01, .02], [-.01, .03]])
    delta = torch.tensor([[.2, -.1], [-.3, .4]], requires_grad=True)
    value = loco.loco_matching_objective(delta, x, pre, dpre, 1.2, .3)
    actual, = torch.autograd.grad(value, delta)
    residual = (pre + x @ delta.T).relu().square() - pre.relu().square() + 1.2 * dpre
    expected = residual.T @ x / len(x) + .3 * delta
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("target_space", ["post", "pre"])
def test_zero_gradient_has_exactly_zero_correction_with_bf16_capture(target_space):
    torch.manual_seed(1)
    x, pre = torch.randn(16, 3).bfloat16(), torch.randn(16, 8).bfloat16()
    corr, info, state = loco.loco_solve(x, pre, torch.zeros_like(pre), torch.zeros(8, 3),
                                      target_space=target_space)
    assert torch.count_nonzero(corr) == 0
    assert info["lossK"] == 0 and not info["accepted"]
    assert state is None


def test_backtracking_stabilizes_high_curvature_and_checks_final_iterate():
    x, pre, dpre = torch.tensor([[10.]]), torch.tensor([[2.]]), torch.tensor([[.1]])
    corr, info, _ = loco.loco_solve(x, pre, dpre, dpre.T @ x, inner_lr=10., steps=4)
    assert info["accepted"] and info["backtracks"] > 0
    assert info["cos_desc"] > 0 and info["lossK"] < 0
    actual = loco.loco_matching_objective(corr, x, pre, dpre, 1., .1)
    assert float(actual) == info["lossK"]


def test_matching_decrease_can_increase_squared_error():
    x, pre, dpre = torch.eye(2), torch.tensor([[1.], [3.]]), torch.tensor([[1.], [-1.]])
    corr, info, _ = loco.loco_solve(x, pre, dpre, dpre.T @ x, steps=1, inner_lr=.9)
    assert info["accepted"] and info["lossK"] < 0
    target = pre.square() - dpre
    initial = (pre.square() - target).square().mean()
    final = ((pre + x @ corr.T).relu().square() - target).square().mean()
    assert final > initial


def test_uphill_and_nonfinite_corrections_are_rejected():
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 1), torch.full((2, 1), .1)
    _, uphill, _ = loco.loco_solve(x, pre, dpre, -dpre.T @ x)
    assert uphill["cos_desc"] < 0 and not uphill["accepted"]
    _, invalid, _ = loco.loco_solve(x, pre, dpre * float("nan"), dpre.T @ x)
    assert invalid["reason"] == "nonfinite_input" and not invalid["accepted"]


def test_rmsprop_state_commits_only_on_accepted_correction():
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 1), torch.full((2, 1), .1)
    avg, mom = torch.ones(1, 1), torch.ones(1, 1)
    _, info, state = loco.loco_solve(x, pre, dpre, -dpre.T @ x, local_opt="rmsprop",
                                   rms_avg=avg, rms_mom=mom)
    assert not info["accepted"] and state is None
    assert float(avg) == float(mom) == 1.


def test_cap_includes_alpha_and_never_amplifies_small_corrections():
    corr = torch.tensor([[2.]])
    scale = loco.loco_correction_scale(corr, torch.tensor(1.), alpha=5., cap=.2)
    assert float(corr.norm() * scale) == pytest.approx(.2)
    small = torch.tensor([[.01]])
    assert float(loco.loco_correction_scale(small, torch.tensor(1.), norm_target=.2)) == 1.
    assert float(loco.loco_correction_scale(corr * float("inf"), torch.tensor(1.))) == 0


def test_post_step_gate_rejects_correction_after_outer_step_passed_local_minimum():
    x, pre, dpre = torch.ones(1, 1), torch.ones(1, 1), torch.full((1, 1), .1)
    corr, info, _ = loco.loco_solve(x, pre, dpre, dpre.T @ x, steps=1)
    assert info["accepted"]
    accepted = loco.loco_post_step_scale(corr, torch.tensor(1.), torch.zeros_like(corr), x, pre, dpre)
    rejected = loco.loco_post_step_scale(corr, torch.tensor(1.), torch.full_like(corr, -.1), x, pre, dpre)
    assert float(accepted) > 0 and float(rejected) == 0


def test_accumulated_capture_is_compact_paired_and_does_not_change_rng():
    samples = loco.loco_sample_buffer(6, 1, 1)
    loco.loco_begin_capture(2, 3, True, {0})
    before = torch.get_rng_state().clone()
    for micro in range(3):
        loco.loco_set_microbatch(micro)
        x = (torch.arange(20.) + micro * 100).view(-1, 1)
        loco.loco_capture(x, x + 1, x + 2, samples, 0)
    sx, pre, dpre = loco.loco_samples(samples, 0, 1)
    assert len(sx) == 6 and samples.untyped_storage().nbytes() == 6 * 3 * 4
    torch.testing.assert_close(pre, sx + 1)
    torch.testing.assert_close(dpre, sx + 2)
    assert set((sx.flatten() // 100).tolist()) == {0., 1., 2.}
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_custom_backward_matches_native_mlp(dtype):
    torch.manual_seed(3)
    native = [torch.randn(2, 4, 3).to(dtype), torch.randn(7, 3), torch.randn(7),
              torch.randn(3, 7), torch.randn(3)]
    native = [v.requires_grad_() for v in native]
    custom = [v.detach().clone().requires_grad_() for v in native]
    x, w1, b1, w2, b2 = native
    pre = torch.nn.functional.linear(x, w1.to(dtype), b1.to(dtype))
    y = torch.nn.functional.linear(pre.relu().square(), w2.to(dtype), b2.to(dtype))
    buffer = loco.loco_sample_buffer(4, 3, 7)
    loco.loco_begin_capture(0, 1, True, {0})
    out = loco.LocoMLPFunction.apply(*custom, buffer, 0)
    torch.testing.assert_close(out, y, rtol=0, atol=0)
    dy = torch.randn_like(out)
    y.backward(dy)
    out.backward(dy)
    for a, b in zip(native, custom):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert loco.loco_samples(buffer, 0, 3)[0].shape == (4, 3)


@pytest.mark.parametrize("backend_name", ["aot_eager", "inductor"])
def test_capture_under_fullgraph_compile_has_no_step_recompilation(backend_name):
    assert torch.Tag.cudagraph_unsafe in torch.ops.nanogpt_loco.capture.default.tags
    compilations = []
    def backend(gm, inputs):
        compilations.append(gm)
        return torch._dynamo.backends.registry.lookup_backend(backend_name)(gm, inputs)
    def mlp(x, w1, b1, w2, b2, samples):
        return loco.LocoMLPFunction.apply(x, w1, b1, w2, b2, samples, 0)
    compiled = torch.compile(mlp, fullgraph=True, backend=backend)
    params = [torch.randn(2, 3), torch.randn(4, 3), torch.randn(4), torch.randn(3, 4), torch.randn(3)]
    params = [v.requires_grad_() for v in params]
    samples = loco.loco_sample_buffer(2, 3, 4)
    for step, active in enumerate((True, False, True)):
        loco.loco_begin_capture(step, 1, active, {0})
        compiled(*params, samples).sum().backward()
        assert len(loco.loco_samples(samples, 0, 3)[0]) == (2 if active else 0)
        for p in params:
            p.grad = None
    assert len(compilations) == 1
