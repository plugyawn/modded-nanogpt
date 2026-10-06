"""Regressions for expensive rejected directions and stale solver centers."""
import pytest
import torch
from torch import nn

from test_locoprop_local import loco
from test_locoprop_generators import runtime


def test_line_search_can_shrink_a_large_carried_tensorflow_momentum():
    x, pre, dpre = torch.tensor([[10.]]), torch.tensor([[2.]]), torch.tensor([[.1]])
    avg, mom = torch.ones(1, 1), torch.full((1, 1), .5)
    arguments = dict(steps=1, inner_lr=.001, beta1=.999, local_opt="rmsprop",
                     rms_style="tensorflow", rms_avg=avg, rms_mom=mom)
    old, old_info, _ = loco.loco_solve(x, pre, dpre, dpre.T @ x, **arguments)
    corr, info, state = loco.loco_solve(x, pre, dpre, dpre.T @ x,
                                     scale_momentum=True, **arguments)
    assert not old_info["accepted"] and old_info["backtracks"] == 21
    assert torch.count_nonzero(old) == 0
    assert info["accepted"] and info["local_steps"] == 1
    assert loco.loco_matching_objective(corr, x, pre, dpre, 1., .1) < 0
    torch.testing.assert_close(state[1], -corr)
    torch.testing.assert_close(mom, torch.full_like(mom, .5), rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0., 1.])
def test_post_base_solver_and_gate_share_current_affine_center(monkeypatch, alpha):
    for key, value in {"WR_LOCOM_CENTER": "post_base", "WR_LOCOM_LOCAL_OPT": "rmsprop",
                       "WR_LOCOM_RMS_STYLE": "tensorflow", "WR_LOCOM_SCALE_MOMENTUM": "1",
                       "WR_LOCOM_SAMPLE_TOKENS": "2", "WR_LOCOM_STEPS": "3"}.items():
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
    rt._wr_locom_begin_step(0, model=model)
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 4), torch.full((2, 4), .1)
    loco.loco_capture(x, pre, dpre, layer._loco_samples, 0)
    layer.fc.weight.grad = dpre.T @ x
    rt.prepare_wr_locom_m(model, 0)
    assert rt.WR_LOCOM_CORR[0][0] is None
    # Native displacement has passed the original local minimizer; its bias
    # also changed. The old-center gate would reject further negative motion.
    layer.fc.weight.data.sub_(.5)
    layer.fc.bias.data.add_(.2)
    reference, avg = layer.fc.weight.detach().clone(), layer._loco_rms_avg.clone()
    rt.apply_wr_locom_m(model, None, 0)
    displacement = layer.fc.weight.detach() - reference
    if alpha:
        assert torch.count_nonzero(displacement) > 0
        centered_pre = torch.full_like(pre, .7)
        assert loco.loco_matching_objective(displacement, x, centered_pre, dpre, 1., .1) < 0
        assert "gate_reason=accepted" in rt.WR_LOCOM_APPLY_STATS[0]
    else:
        assert torch.count_nonzero(displacement) == 0
        torch.testing.assert_close(layer._loco_rms_avg, avg, rtol=0, atol=0)


def test_inactive_schedule_uses_native_graph_and_reuses_two_variants(monkeypatch):
    monkeypatch.setenv("WR_LOCOM_SAMPLE_TOKENS", "2")
    monkeypatch.setenv("WR_LOCOM_INTERVAL", "2")
    rt = runtime("make_wr_record_locoprop_m")
    monkeypatch.setattr(rt.dist, "get_rank", lambda: 0)
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(2)
    model.blocks = nn.ModuleList([block])
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(block.mlp, backend=backend, fullgraph=True)
    for step in range(6):
        rt._wr_locom_begin_step(step, model=model)
        compiled(torch.ones(2, 2)).sum().backward()
        samples = loco.loco_samples(block.mlp._loco_samples, 0, 2)[0]
        assert len(samples) == (2 if step % 2 == 0 else 0)
        block.mlp.zero_grad(set_to_none=True)
    assert len(graphs) == 2
    assert "autograd_function_apply" not in str(graphs[1].graph)


def test_full_gradient_preserves_first_order_direction_despite_opposing_token_sample():
    x, pre = torch.tensor([[1., 0.], [0., 1.]]), torch.ones(2, 1)
    full = torch.tensor([[.2, -.1]])
    for dpre in (torch.full_like(pre, 20.), torch.full_like(pre, -20.)):
        corr, info, _ = loco.loco_solve(x, pre, dpre, full * 524288,
            steps=1, inner_lr=.01, gamma=2., linear_grad=full)
        torch.testing.assert_close(corr, -.02 * full)
        assert info['accepted'] and info['backtracks'] == 0
        zero = torch.zeros_like(corr, requires_grad=True)
        objective = loco.loco_matching_objective(zero, x, pre, dpre, 2., .1,
                                                 linear_grad=full)
        torch.testing.assert_close(torch.autograd.grad(objective, zero)[0], 2 * full)


def test_full_gradient_stored_gate_uses_same_objective_with_native_bias_shift():
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 1), torch.full((2, 1), -10.)
    full = torch.ones(1, 1)
    corr, base, bias = torch.tensor([[-.1]]), torch.tensor([[.05]]), torch.tensor([.1])
    current, reference = torch.tensor([[1.05]]), torch.ones(1, 1)
    scale, info = loco.loco_post_step_scale(corr, torch.tensor(1.), base, x, pre, dpre,
        linear_grad=full, bias_delta=bias, current_weight=current,
        reference_weight=reference, return_info=True)
    assert scale == 1 and info['reason'] == 'accepted'
    combined = info['candidate'] - reference
    value = lambda delta: loco.loco_matching_objective(delta, x, pre, dpre, 1., .1,
                                                       bias_delta=bias, linear_grad=full)
    assert value(combined) < value(base)
    # The old sample's linear term points in the opposite direction.
    old = loco.loco_post_step_scale(corr, torch.tensor(1.), base, x, pre, dpre,
                                    bias_delta=bias)
    assert old == 0


@pytest.mark.parametrize('center', ['pre_base', 'post_base'])
def test_full_gradient_normalizes_global_sum_once_and_survives_native_grad_mutation(monkeypatch, center):
    monkeypatch.setenv('WR_LOCOM_LINEAR_TERM', 'full')
    monkeypatch.setenv('WR_LOCOM_CENTER', center)
    monkeypatch.setenv('WR_LOCOM_SAMPLE_TOKENS', '2')
    rt = runtime('make_wr_record_locoprop_m')
    monkeypatch.setattr(rt.dist, 'get_rank', lambda: 0)
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(1)
    model.blocks = nn.ModuleList([block])
    layer = block.mlp
    layer.fc.weight.data.fill_(1.)
    layer.fc.bias.data.zero_()
    full = torch.full_like(layer.fc.weight, .2)
    for global_tokens in (2, 16, 524288):
        rt.batch_size = global_tokens
        rt._wr_locom_begin_step(0, model=model)
        loco.loco_capture(torch.ones(2, 1), torch.ones(2, 4),
                         torch.full((2, 4), -3.), layer._loco_samples, 0)
        layer.fc.weight.grad = full * global_tokens
        rt.prepare_wr_locom_m(model, 0)
        layer.fc.weight.grad.zero_()
        torch.testing.assert_close(rt.WR_LOCOM_LINEAR_GRADS[0], full)
        assert 0 in rt.WR_LOCOM_CORR
