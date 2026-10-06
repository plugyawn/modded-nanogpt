"""Independent double-precision references for opt-in legacy TF RMS arithmetic."""

import pytest
import torch
from torch import nn

from test_locoprop_local import loco
from test_locoprop_generators import runtime


def tensorflow_reference(x, pre, dpre, *, steps, inner_lr, beta1, beta2, eps,
                         prox=.1, gamma=1., decay=False, avg=None, mom=None):
    """Legacy TF RMS recurrence, with the explicitly added Armijo safeguards."""
    x, pre, dpre = x.double(), pre.double(), dpre.double()
    delta = torch.zeros(pre.shape[1], x.shape[1], dtype=torch.float64)
    avg = torch.ones_like(delta) if avg is None else avg.double().clone()
    mom = torch.zeros_like(delta) if mom is None else mom.double().clone()
    def objective(change):
        shift = x @ change.T
        z = pre + shift
        return ((z.relu().pow(3) / 3 - pre.relu().pow(3) / 3
                 - pre.relu().square() * shift + gamma * dpre * shift).sum() / len(x)
                + prox / 2 * change.square().sum())
    value, factor, backtracks, last_lr = objective(delta), 1., 0, 0.
    for j in range(steps):
        grad = (((pre + x @ delta.T).relu().square() - pre.relu().square()
                 + gamma * dpre).T @ x / len(x) + prox * delta)
        next_avg = beta2 * avg + (1 - beta2) * grad.square()
        normalized = grad / (next_avg + eps).sqrt()
        fraction = max(1 - j / steps, .25) if decay else 1.
        lr = inner_lr * fraction * factor
        for _ in range(21):
            next_mom = beta1 * mom + lr * normalized
            slope = (grad * next_mom).sum()
            if not torch.isfinite(slope) or slope <= 0:
                next_mom = lr * grad
                slope = (grad * next_mom).sum()
            candidate = delta - next_mom
            next_value = objective(candidate)
            if torch.isfinite(next_value) and next_value <= value - 1e-4 * slope:
                delta, value, avg, mom = candidate, next_value, next_avg, next_mom
                factor, last_lr = lr / (inner_lr * fraction), lr
                break
            lr *= .5
            backtracks += 1
        else:
            break
    return delta, avg, mom, backtracks, last_lr


@pytest.mark.parametrize("steps,decay", [(1, False), (2, True), (4, True)])
def test_tensorflow_rms_matches_double_precision_reference_with_local_decay(steps, decay):
    x = torch.tensor([[1., .1], [.2, .8], [.4, .3]])
    pre = torch.tensor([[1., .7], [.9, .6], [.8, 1.1]])
    dpre = torch.tensor([[.1, -.02], [.04, .03], [.05, -.01]])
    params = dict(steps=steps, inner_lr=.04, beta1=.4, beta2=.7, eps=.03, decay=decay)
    expected, avg, mom, backtracks, last_lr = tensorflow_reference(x, pre, dpre, **params)
    corr, info, state = loco.loco_solve(x, pre, dpre, dpre.T @ x, steps=steps,
        inner_lr=.04, beta1=.4, beta2=.7, rms_eps=.03, lr_decay=decay,
        local_opt="rmsprop", rms_style="tensorflow")
    assert info["accepted"] and info["local_steps"] == steps
    assert info["backtracks"] == backtracks == 0
    assert info["inner_lr"] == pytest.approx(last_lr)
    torch.testing.assert_close(corr.double(), expected, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(state[0].double(), avg, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(state[1].double(), mom, rtol=2e-6, atol=1e-8)


def test_tensorflow_rms_backtracking_recomputes_momentum_and_preserves_decay_fraction():
    x, pre, dpre = torch.tensor([[10.]]), torch.tensor([[2.]]), torch.tensor([[.1]])
    initial_avg, initial_mom = torch.ones(1, 1), torch.full((1, 1), .0001)
    params = dict(steps=3, inner_lr=10., beta1=.4, beta2=.7, eps=.03,
                  decay=True, avg=initial_avg, mom=initial_mom)
    expected, avg, mom, backtracks, last_lr = tensorflow_reference(x, pre, dpre, **params)
    corr, info, state = loco.loco_solve(x, pre, dpre, dpre.T @ x, steps=3,
        inner_lr=10., beta1=.4, beta2=.7, rms_eps=.03, lr_decay=True,
        local_opt="rmsprop", rms_style="tensorflow", rms_avg=initial_avg, rms_mom=initial_mom)
    assert info["accepted"] and info["local_steps"] == 3
    assert info["backtracks"] == backtracks > 0
    assert info["inner_lr"] == pytest.approx(last_lr)
    torch.testing.assert_close(corr.double(), expected, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(state[0].double(), avg, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(state[1].double(), mom, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(initial_avg, torch.ones_like(initial_avg), rtol=0, atol=0)
    torch.testing.assert_close(initial_mom, torch.full_like(initial_mom, .0001), rtol=0, atol=0)


@pytest.mark.parametrize("reset", [False, True])
def test_tensorflow_rms_state_continuation_and_reset(reset):
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 1), torch.full((2, 1), .1)
    initial_avg, initial_mom = torch.full((1, 1), .4), torch.full((1, 1), .002)
    expected, avg, mom, _, _ = tensorflow_reference(x, pre, dpre, steps=2,
        inner_lr=.03, beta1=.4, beta2=.7, eps=.03,
        avg=None if reset else initial_avg, mom=None if reset else initial_mom)
    corr, info, state = loco.loco_solve(x, pre, dpre, dpre.T @ x, steps=2,
        inner_lr=.03, beta1=.4, beta2=.7, rms_eps=.03, reset_rms=reset,
        local_opt="rmsprop", rms_style="tensorflow", rms_avg=initial_avg, rms_mom=initial_mom)
    assert info["accepted"]
    for actual, wanted in zip((corr, *state), (expected, avg, mom)):
        torch.testing.assert_close(actual.double(), wanted, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(initial_avg, torch.full_like(initial_avg, .4), rtol=0, atol=0)
    torch.testing.assert_close(initial_mom, torch.full_like(initial_mom, .002), rtol=0, atol=0)


def test_tensorflow_rms_zero_gradient_retains_zero_correction_guard():
    x, pre = torch.ones(2, 1), torch.ones(2, 1)
    avg, mom = torch.full((1, 1), .4), torch.full((1, 1), .002)
    corr, info, state = loco.loco_solve(x, pre, torch.zeros_like(pre), torch.zeros(1, 1),
        local_opt="rmsprop", rms_style="tensorflow", rms_avg=avg, rms_mom=mom)
    assert torch.count_nonzero(corr) == 0 and not info["accepted"] and state is None
    assert avg.item() == pytest.approx(.4) and mom.item() == pytest.approx(.002)


@pytest.mark.parametrize("alpha", [0., 1.])
def test_record_tensorflow_rms_initialization_and_application_state_commit(monkeypatch, alpha):
    for key, value in {"WR_LOCOM_LOCAL_OPT": "rmsprop", "WR_LOCOM_RMS_STYLE": "tensorflow",
                       "WR_LOCOM_SAMPLE_TOKENS": "2", "WR_LOCOM_STEPS": "2",
                       "WR_LOCOM_RMS_BETA1": ".4", "WR_LOCOM_RMS_BETA2": ".7",
                       "WR_LOCOM_RMS_EPS": ".03", "WR_LOCOM_INNER_LR": ".03"}.items():
        monkeypatch.setenv(key, value)
    rt = runtime("make_wr_record_locoprop_m")
    monkeypatch.setattr(rt.dist, "get_rank", lambda: 0)
    rt.WR_LOCOM_ALPHA = alpha
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(1)
    model.blocks = nn.ModuleList([block])
    layer = block.mlp
    layer.fc.weight.data.fill_(1.)
    layer.fc.bias.data.zero_()
    torch.testing.assert_close(layer._loco_rms_avg, torch.ones_like(layer.fc.weight), rtol=0, atol=0)
    assert torch.count_nonzero(layer._loco_rms_mom) == 0
    rt._wr_locom_begin_step(0)
    x, pre = torch.ones(2, 1), torch.ones(2, 4)
    dpre = torch.full_like(pre, .1)
    loco.loco_capture(x, pre, dpre, layer._loco_samples, 0)
    layer.fc.weight.grad = dpre.T @ x
    rt.prepare_wr_locom_m(model, 0)
    staged = rt.WR_LOCOM_CORR[0][-1]
    assert staged is not None
    torch.testing.assert_close(layer._loco_rms_avg, torch.ones_like(layer.fc.weight), rtol=0, atol=0)
    layer.fc.weight.data.sub_(.001)
    before = layer.fc.weight.detach().clone()
    rt.apply_wr_locom_m(model, None, 0)
    if alpha:
        assert not torch.equal(layer.fc.weight, before)
        torch.testing.assert_close(layer._loco_rms_avg, staged[0], rtol=0, atol=0)
        torch.testing.assert_close(layer._loco_rms_mom, staged[1], rtol=0, atol=0)
    else:
        torch.testing.assert_close(layer.fc.weight, before, rtol=0, atol=0)
        torch.testing.assert_close(layer._loco_rms_avg, torch.ones_like(layer.fc.weight), rtol=0, atol=0)
        assert torch.count_nonzero(layer._loco_rms_mom) == 0


def test_rms_style_validation_and_legacy_default(monkeypatch):
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 1), torch.full((2, 1), .1)
    with pytest.raises(ValueError, match="rms_style"):
        loco.loco_solve(x, pre, dpre, dpre.T @ x, rms_style="invalid")
    monkeypatch.delenv("WR_LOCOM_RMS_STYLE", raising=False)
    assert runtime("make_wr_record_locoprop_m").WR_LOCOM_RMS_STYLE == "torch"
    monkeypatch.setenv("WR_LOCOM_RMS_STYLE", "invalid")
    with pytest.raises(ValueError, match="WR_LOCOM_RMS_STYLE"):
        runtime("make_wr_record_locoprop_m")
