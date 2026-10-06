import ast
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import json
import subprocess
import sys
import types

import pytest
import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.multiprocessing as mp

from test_locoprop_local import loco


ROOT = Path(__file__).resolve().parents[1]


def generator(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Linear(nn.Linear):
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


def runtime(name):
    gen = generator(name)
    module = types.ModuleType("test_runtime_" + name)
    module.__dict__.update(vars(loco))
    module.__name__ = "test_runtime_" + name
    sys.modules[module.__name__] = module
    module.__dict__.update(nn=nn, Tensor=Tensor, dist=dist, os=os, Path=Path,
                           Linear=Linear, print0=lambda *a, **k: None)
    if name == "make_track3_locoprop_m":
        source = gen.LOCOM_CONFIG + gen.LOCOM_MLP + gen.LOCOM_HELPERS
    elif name == "make_wr_record_locoprop_m":
        source = gen.LOCOM_BLOCK + gen.MLP_BLOCK
    else:
        source = gen.WR_LOCOM_CONFIG
    exec(compile(source, module.__name__, "exec"), vars(module))
    return module


@pytest.mark.parametrize("name", ["make_track3_locoprop_m", "make_wr_record_locoprop_m", "make_wr_locoprop_m"])
def test_generator_embeds_shared_solver_and_compiles(name, tmp_path):
    gen = generator(name)
    if name == "make_track3_locoprop_m":
        sources = [ROOT / "records/track_3_optimization/train_gpt_simple.py",
                   ROOT / "records/track_3_optimization/results/20260505_newton_muon/train_gpt_simple_newton_muon.py",
                   ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt"]
    elif name == "make_wr_record_locoprop_m":
        sources = [ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt"]
    else:
        sources = [ROOT / "train_gpt.py"]
    for index, source in enumerate(sources):
        output = tmp_path / f"{name}_{index}.py"
        if name == "make_wr_record_locoprop_m":
            gen.generate(source, output, 125, 3105)
        else:
            gen.generate(source, output, 125)
        content = output.read_text()
        compile(content, str(output), "exec")
        assert "nanogpt_loco::capture" in content and "loco_solve(" in content
        assert "register_hook(" not in content
        assert "loco_set_microbatch(" in content


@pytest.mark.parametrize("name", ["make_track3_locoprop_m", "make_wr_record_locoprop_m"])
def test_generated_mlp_and_solver_run_with_compilation(name, monkeypatch):
    monkeypatch.setenv("TRACK3_LOCOM_SAMPLE_TOKENS", "8")
    monkeypatch.setenv("WR_LOCOM_SAMPLE_TOKENS", "8")
    rt = runtime(name)
    block = nn.Module()
    block.mlp = rt.MLP(3, 0) if name == "make_track3_locoprop_m" else rt.MLP(3)
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    mlp = torch.compile(block.mlp, backend="aot_eager", fullgraph=True)
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    if name == "make_track3_locoprop_m":
        rt.attach_locoprop_m_optimizer(model, opt)
        rt.set_locoprop_m_current_step(0)
        loco.loco_begin_capture(0, 2, True, {0})
    else:
        # Runtime setter normally runs inside an initialized distributed driver.
        monkeypatch.setattr(rt.dist, "get_rank", lambda: 0)
        rt._wr_locom_begin_step(0, 2)
    for micro in range(2):
        loco.loco_set_microbatch(micro)
        mlp(torch.randn(2, 4, 3).bfloat16()).float().square().sum().backward()
    sx, _, _ = loco.loco_samples(block.mlp._loco_samples, 0, 3)
    assert len(sx) == 8
    if name == "make_track3_locoprop_m":
        rt.prepare_locoprop_m(model, 0)
    else:
        rt.prepare_wr_locom_m(model, 0)
    before = block.mlp.fc.weight.detach().clone()
    update = block.mlp.fc.weight.grad.float().clone()
    opt.step()
    if name == "make_track3_locoprop_m":
        rt._locom_apply_owned_param_(block.mlp.fc.weight, update, .001)
    else:
        rt.apply_wr_locom_m(model, opt, 0)
    assert torch.isfinite(block.mlp.fc.weight).all()
    correction = block.mlp.fc.weight.detach() - before + .001 * update
    assert correction.norm() <= .2 * .001 * update.norm() + 1e-6


def test_record_rms_state_is_staged_and_only_committed_on_stored_update(monkeypatch):
    monkeypatch.setenv("WR_LOCOM_LOCAL_OPT", "rmsprop")
    monkeypatch.setenv("WR_LOCOM_SAMPLE_TOKENS", "2")
    rt = runtime("make_wr_record_locoprop_m")
    monkeypatch.setattr(rt.dist, "get_rank", lambda: 0)
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(1)
    model.blocks = nn.ModuleList([block])
    layer = block.mlp
    layer.fc.weight.data.fill_(1.)
    layer.fc.bias.data.zero_()
    for step, alpha in [(0, 1.), (1, 0.)]:
        rt.WR_LOCOM_ALPHA = alpha
        rt._wr_locom_begin_step(step)
        x = torch.ones(2, 1)
        pre = torch.nn.functional.linear(x, layer.fc.weight, layer.fc.bias)
        dpre = torch.full_like(pre, .1)
        loco.loco_capture(x, pre, dpre, layer._loco_samples, 0)
        layer.fc.weight.grad = dpre.T @ x
        old_avg, old_mom = layer._loco_rms_avg.clone(), layer._loco_rms_mom.clone()
        rt.prepare_wr_locom_m(model, step)
        assert rt.WR_LOCOM_CORR[0][-1] is not None
        torch.testing.assert_close(layer._loco_rms_avg, old_avg, rtol=0, atol=0)
        torch.testing.assert_close(layer._loco_rms_mom, old_mom, rtol=0, atol=0)
        layer.fc.weight.data.sub_(.001)
        before = layer.fc.weight.detach().clone()
        rt.apply_wr_locom_m(model, None, step)
        if alpha > 0:
            assert not torch.equal(layer.fc.weight, before)
            assert not torch.equal(layer._loco_rms_avg, old_avg)
            assert not torch.equal(layer._loco_rms_mom, old_mom)
            # The small FP32 update need not immediately change a BF16 forward.
            assert "stored_corr=" in rt.WR_LOCOM_APPLY_STATS[0]
            assert "gate_attempts=" in rt.WR_LOCOM_APPLY_STATS[0]
        else:
            torch.testing.assert_close(layer.fc.weight, before, rtol=0, atol=0)
            torch.testing.assert_close(layer._loco_rms_avg, old_avg, rtol=0, atol=0)
            torch.testing.assert_close(layer._loco_rms_mom, old_mom, rtol=0, atol=0)


def test_record_rms_solver_honors_explicit_momentum_reset_and_decay(monkeypatch):
    settings = {"WR_LOCOM_LOCAL_OPT": "rmsprop", "WR_LOCOM_RMS_BETA1": ".4",
                "WR_LOCOM_RMS_BETA2": ".7", "WR_LOCOM_RMS_EPS": ".03",
                "WR_LOCOM_RESET_RMS": "1", "WR_LOCOM_LR_DECAY": "1",
                "WR_LOCOM_SAMPLE_TOKENS": "2", "WR_LOCOM_STEPS": "3"}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    rt = runtime("make_wr_record_locoprop_m")
    monkeypatch.setattr(rt.dist, "get_rank", lambda: 0)
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(1)
    model.blocks = nn.ModuleList([block])
    layer = block.mlp
    layer._loco_rms_avg.fill_(.2)
    layer._loco_rms_mom.fill_(.3)
    rt._wr_locom_begin_step(0)
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, 4), torch.full((2, 4), .1)
    loco.loco_capture(x, pre, dpre, layer._loco_samples, 0)
    layer.fc.weight.grad = dpre.T @ x
    expected, info, state = loco.loco_solve(x, pre, dpre, layer.fc.weight.grad, steps=3,
                                           local_opt="rmsprop", rms_avg=layer._loco_rms_avg,
                                           rms_mom=layer._loco_rms_mom, beta1=.4, beta2=.7,
                                           rms_eps=.03, reset_rms=True, lr_decay=True)
    assert info["accepted"]
    rt.prepare_wr_locom_m(model, 0)
    context = rt.WR_LOCOM_CORR[0]
    torch.testing.assert_close(context[0], expected, rtol=0, atol=0)
    for actual, wanted in zip(context[-1], state):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    torch.testing.assert_close(layer._loco_rms_avg, torch.full_like(layer._loco_rms_avg, .2))
    torch.testing.assert_close(layer._loco_rms_mom, torch.full_like(layer._loco_rms_mom, .3))


@pytest.mark.parametrize("reason", ["stored_noop", "non_descent_direction"])
def test_record_rejected_application_preserves_rms_state(monkeypatch, reason):
    monkeypatch.setenv("WR_LOCOM_LOCAL_OPT", "rmsprop")
    rt = runtime("make_wr_record_locoprop_m")
    model, block = nn.Module(), nn.Module()
    block.mlp = rt.MLP(1)
    model.blocks = nn.ModuleList([block])
    layer = block.mlp
    layer.fc.weight.data.fill_(1.)
    layer.fc.bias.data.zero_()
    shape = layer.fc.weight.shape
    x, pre, dpre = torch.ones(2, 1), torch.ones(2, shape[0]), torch.full((2, shape[0]), .1)
    corr = torch.full(shape, -1e-10 if reason == "stored_noop" else .01)
    reference = torch.full(shape, .99)
    staged = (torch.full(shape, .3), torch.full(shape, .4))
    rt.WR_LOCOM_CORR[0] = (corr, reference, x, pre, dpre, torch.zeros(shape[0]), staged)
    before = layer.fc.weight.detach().clone()
    rt.apply_wr_locom_m(model, None, 0)
    torch.testing.assert_close(layer.fc.weight, before, rtol=0, atol=0)
    assert torch.count_nonzero(layer._loco_rms_avg) == torch.count_nonzero(layer._loco_rms_mom) == 0
    assert "accepted_apply=0" in rt.WR_LOCOM_APPLY_STATS[0]
    assert f"gate_reason={reason}" in rt.WR_LOCOM_APPLY_STATS[0]


def test_record_local_optimizer_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("WR_LOCOM_LOCAL_OPT", raising=False)
    rt = runtime("make_wr_record_locoprop_m")
    assert rt.WR_LOCOM_LOCAL_OPT == "sgd"
    mlp = rt.MLP(1)
    assert mlp._loco_rms_avg is None and mlp._loco_rms_mom is None
    monkeypatch.setenv("WR_LOCOM_LOCAL_OPT", "invalid")
    with pytest.raises(ValueError, match="WR_LOCOM_LOCAL_OPT"):
        runtime("make_wr_record_locoprop_m")


@pytest.mark.parametrize("enabled,layers", [("0", "all"), ("1", "none")])
def test_record_disabled_mlp_has_no_capture_storage_and_native_gradients(monkeypatch, enabled, layers):
    import copy
    monkeypatch.setenv("WR_LOCOM_ENABLED", enabled)
    monkeypatch.setenv("WR_LOCOM_LAYERS", layers)
    rt = runtime("make_wr_record_locoprop_m")
    class ForbiddenCapture:
        @staticmethod
        def apply(*args):
            raise AssertionError("native baseline must not call custom capture backward")
    rt.LocoMLPFunction = ForbiddenCapture
    mlp = rt.MLP(3)
    native = copy.deepcopy(mlp)
    assert mlp._loco_samples is None and mlp._loco_rms_avg is None and mlp._loco_rms_mom is None
    x = torch.randn(2, 4, 3).bfloat16().requires_grad_()
    nx = x.detach().clone().requires_grad_()
    out = mlp(x)
    expected = native.proj(native.fc(nx).relu().square())
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    dy = torch.randn_like(out)
    out.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, nx.grad, rtol=0, atol=0)
    for actual, reference in zip(mlp.parameters(), native.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)


def test_newton_muon_covariance_hooks_and_capture_survive_compiled_model(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACK3_LOCOM_SAMPLE_TOKENS", "8")
    rt = runtime("make_track3_locoprop_m")
    source = ROOT / "records/track_3_optimization/results/20260505_newton_muon/train_gpt_simple_newton_muon.py"
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in {"accum_xtx_", "precond_stat_hook"}]
    rt.torch = torch
    exec(compile(ast.Module(body=functions, type_ignores=[]), "newton_muon", "exec"), vars(rt))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = rt.MLP(3, 0)
        def forward(self, x):
            return self.mlp(x)
    model = Model()
    model.compile(fullgraph=True, backend="aot_eager")
    x = torch.randn(2, 4, 3).bfloat16()
    loco.loco_begin_capture(0, 1, True, {0})
    model(x).float().sum().backward()
    mlp = model.mlp
    mlp.fc.weight._stats_ref = dict(accum=mlp.fc_xtx, count=mlp.fc_count)
    mlp.proj.weight._stats_ref = dict(accum=mlp.proj_xtx, count=mlp.proj_count)
    hooks = [module.register_forward_hook(rt.precond_stat_hook) for module in (mlp.fc, mlp.proj)]
    try:
        loco.loco_begin_capture(64, 1, True, {0})
        # The generated driver bypasses the compiled root on refresh steps.
        model.forward(x).float().sum().backward()
        flat = x.flatten(0, -2).float()
        torch.testing.assert_close(mlp.fc_xtx, flat.T @ flat / len(flat))
        assert mlp.fc_count == 1 and mlp.proj_count == 1 and mlp.proj_xtx.norm() > 0
        assert len(loco.loco_samples(mlp._loco_samples, 0, 3)[0]) == 8
    finally:
        for hook in hooks:
            hook.remove()
    output = tmp_path / "newton_muon.py"
    generator("make_track3_locoprop_m").generate(source, output, 125)
    assert "(model.forward if precond_hooks else compiled_model)" in output.read_text()


def test_newton_muon_checkpoint_preserves_covariance_inverse_aliases(monkeypatch, tmp_path):
    rt = runtime("make_track3_locoprop_m")
    source = ROOT / "records/track_3_optimization/results/20260505_newton_muon/train_gpt_simple_newton_muon.py"
    classes = [node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "Muon"]
    exec(compile(ast.Module(body=classes, type_ignores=[]), "newton_muon", "exec"), vars(rt))
    model = nn.Module()
    block = nn.Module()
    block.mlp = rt.MLP(3, 0)
    model.blocks = nn.ModuleList([block])
    mlp = block.mlp
    mlp.fc.weight._stats_ref = dict(accum=mlp.fc_xtx, count=mlp.fc_count)
    mlp.proj.weight._stats_ref = dict(accum=mlp.proj_xtx, count=mlp.proj_count)
    opt = rt.Muon([mlp.fc.weight, mlp.proj.weight])
    opt.attach_preconditioner()
    opt._precond_cov.mul_(2)
    opt._precond_K.mul_(.5)
    cov, inverse = opt._precond_cov.clone(), opt._precond_K.clone()
    opt._precond_has_inv = True
    opt.global_step = 100
    rt.TRACK3_CHECKPOINT_STEPS = {100}
    rt.TRACK3_CHECKPOINT_DIR = str(tmp_path)
    rt.maybe_save_track3_checkpoint(model, [opt], 100, 3000, 0, None)
    path, = tmp_path.glob("*.pt")
    opt._precond_cov.zero_()
    opt._precond_K.zero_()
    opt._precond_has_inv = False
    opt.global_step = 0
    rt.TRACK3_RESUME_CHECKPOINT = str(path)
    assert rt.maybe_load_track3_checkpoint(model, [opt]) == 100
    assert opt._precond_has_inv and opt.global_step == 100
    torch.testing.assert_close(opt._precond_cov, cov)
    torch.testing.assert_close(opt._precond_K, inverse)
    for p in (mlp.fc.weight, mlp.proj.weight):
        state = opt.state[p]["precond"]
        assert state["inv"].untyped_storage().data_ptr() == opt._precond_K.untyped_storage().data_ptr()
        assert state["cov"].untyped_storage().data_ptr() == opt._precond_cov.untyped_storage().data_ptr()
        assert state["accum"] is p._stats_ref["accum"]


def test_sharded_gathers_all_layers_before_owner_filter(monkeypatch):
    rt = runtime("make_wr_locoprop_m")
    rt.WR_LOCOM_LAYER_SET = {0, 1}
    rt.WR_LOCOM_OWNED_LAYER_SET = {1}
    model = nn.Module()
    model.mlp_bank = nn.Parameter(torch.randn(2, 2, 4, 3))
    model.mlp_bank.grad = torch.ones_like(model.mlp_bank)
    loco.loco_begin_capture(0, 1, True, {0, 1})
    for layer in range(2):
        buffer = loco.loco_sample_buffer(2, 3, 4)
        rt.WR_LOCOM_CAPTURE_BUFFERS[layer] = buffer
        loco.loco_capture(torch.ones(2, 3), torch.ones(2, 4), torch.ones(2, 4), buffer, layer)
    calls = []
    monkeypatch.setattr(rt, "_wr_locom_gather_sample", lambda t: calls.append(t.shape) or t)
    rt.prepare_wr_locoprop_m(model, 0)
    assert len(calls) == 6
    assert set(rt.WR_LOCOM_LAYER_CORR) <= {1}


def test_sharded_apply_checks_reduced_full_gradient_before_mutation():
    rt = runtime("make_wr_locoprop_m")
    rt.WR_LOCOM_CURRENT_STEP = 0
    p = torch.zeros(1, 1, 1)
    corr = torch.full((1, 1), -.01)
    rt.WR_LOCOM_LAYER_CORR[0] = (corr, p[0].clone(), torch.ones(1, 1),
                               torch.ones(1, 1), torch.full((1, 1), .1))
    cfg = types.SimpleNamespace(label="mlp_bank", chunk_size=1)
    # A shared sample may point downhill while the reduced full gradient points
    # the other way. Reject before invoking the CUDA update or touching state.
    rt._wr_locom_apply_mlp_chunk_corrections(None, p, cfg, {}, torch.ones_like(p),
                                          0, -torch.ones_like(p))
    assert torch.count_nonzero(p) == 0 and not rt.WR_LOCOM_LAYER_CORR
    assert "reason=full_gradient_gate" in rt.WR_LOCOM_APPLY_STATS[0]


def _distributed_owner_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        rt = runtime("make_wr_locoprop_m")
        rt.WR_LOCOM_LAYER_SET = {0, 1}
        rt.WR_LOCOM_OWNED_LAYER_SET = {rank}
        model = nn.Module()
        model.mlp_bank = nn.Parameter(torch.ones(2, 2, 4, 3))
        model.mlp_bank.grad = torch.ones_like(model.mlp_bank)
        loco.loco_begin_capture(0, 1, True, {0, 1}, rank=rank)
        for layer in range(2):
            buffer = loco.loco_sample_buffer(2, 3, 4)
            rt.WR_LOCOM_CAPTURE_BUFFERS[layer] = buffer
            loco.loco_capture(torch.ones(2, 3), torch.ones(2, 4),
                              torch.full((2, 4), .1), buffer, layer)
        rt.prepare_wr_locoprop_m(model, 0)
        assert set(rt.WR_LOCOM_LAYER_CORR) == {rank}
    finally:
        dist.destroy_process_group()


def test_sharded_owner_collectives_complete_on_two_ranks(tmp_path):
    mp.spawn(_distributed_owner_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _distributed_record_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        rt = runtime("make_wr_record_locoprop_m")
        rt.WR_LOCOM_LAYER_SET = {0, 1}
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for block in model.blocks:
            block.mlp = rt.MLP(1)
            block.mlp.fc.weight.data.fill_(1)
            block.mlp.fc.weight.grad = torch.full_like(block.mlp.fc.weight, .1)
        loco.loco_begin_capture(0, 1, True, {0, 1}, rank=rank)
        for layer, block in enumerate(model.blocks):
            loco.loco_capture(torch.ones(2, 1), torch.ones(2, 4),
                              torch.full((2, 4), .1), block.mlp._loco_samples, layer)
        rt.prepare_wr_locom_m(model, 0)
        assert set(rt.WR_LOCOM_CORR) == {0, 1}
        opt = torch.optim.SGD(model.parameters(), lr=.001)
        # Muon has gathered the baseline update, but its state exists only on
        # each parameter's owner. Both ranks must still apply the correction.
        opt.state[model.blocks[rank].mlp.fc.weight]["owner_only"] = True
        for block in model.blocks:
            block.mlp.fc.weight.data.sub_(.001)
        rt.apply_wr_locom_m(model, opt, 0)
        for block in model.blocks:
            p = block.mlp.fc.weight
            assert bool((p < .999).all())
            gathered = [torch.empty_like(p), torch.empty_like(p)]
            dist.all_gather(gathered, p)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_record_corrections_are_identical_across_optimizer_owners(tmp_path):
    mp.spawn(_distributed_record_worker, args=(str(tmp_path / "record_rendezvous"),), nprocs=2, join=True)


def test_sharded_fused_mlp_plumbing_preserves_weight_layout_under_compile():
    rt = runtime("make_wr_locoprop_m")
    def linear_relu_square(x, weight, aux=None):
        z = x @ weight.T
        return (z, z.relu().square()) if aux is None else z * (2 * aux.relu())
    rt.linear_relu_square = linear_relu_square
    native = [torch.randn(2, 4, 3), torch.randn(7, 3), torch.randn(7, 3)]
    native = [v.requires_grad_() for v in native]
    custom = [v.detach().clone().requires_grad_() for v in native]
    samples = loco.loco_sample_buffer(4, 3, 7)
    loco.loco_begin_capture(0, 1, True, {0})
    compiled = torch.compile(rt.ReLUSqrdMLPLocoM, fullgraph=True, backend="aot_eager")
    out = compiled(*custom, samples, 0)
    x, w1, w2 = native
    reference = (x @ w1.T).relu().square() @ w2
    torch.testing.assert_close(out, reference)
    dy = torch.randn_like(out)
    out.backward(dy)
    reference.backward(dy)
    for a, b in zip(native, custom):
        torch.testing.assert_close(a.grad, b.grad)
    assert len(loco.loco_samples(samples, 0, 3)[0]) == 4


@pytest.mark.parametrize("source_kind", ["simple", "record"])
def test_schedule_switch_is_continuous_and_pr287_uses_blending(source_kind, monkeypatch, tmp_path):
    monkeypatch.setenv("TRACK3_LR_SWITCH_STEP", "1600")
    monkeypatch.setenv("TRACK3_LR_AFTER_SWITCH", "pr287")
    monkeypatch.setenv("TRACK3_LR_AFTER_SWITCH_STEPS", "3105")
    rt = runtime("make_track3_locoprop_m")
    gen = generator("make_track3_locoprop_m")
    source = (ROOT / "records/track_3_optimization/train_gpt_simple.py" if source_kind == "simple" else
              ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt")
    output = tmp_path / "schedule.py"
    gen.generate(source, output, 3000)
    if source_kind == "record":
        content = output.read_text()
        assert 'os.environ.get("TRACK3_LR_SCHEDULE", "pr287")' in content
        assert 'os.environ.get("TRACK3_LR_SCHEDULE_STEPS", "3105")' in content
        assert 'os.environ.get("TRACK3_LR_POWER", "1.2")' in content
        # Use the generated source's defaults when evaluating its functions.
        rt.TRACK3_LR_SCHEDULE = "pr287"
        rt.TRACK3_LR_SCHEDULE_STEPS = 3105
        rt.TRACK3_LR_POWER = 1.2
        rt.TRACK3_LR_AFTER_SWITCH_POWER = 1.2
    names = {"_track3_active_lr_schedule", "_track3_pr287_lr", "_track3_schedule_lr",
             "_track3_blend_lr", "set_hparams"}
    functions = [node for node in ast.walk(ast.parse(output.read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "schedule", "exec"), vars(rt))
    group = dict(initial_lr=1., lr=1., power_c=.001)
    rt.optimizers = [types.SimpleNamespace(param_groups=[group])]
    rt.train_steps = 3000
    rt.set_hparams(1600)
    old_kind = "pr287" if source_kind == "record" else "linear"
    old_at_switch = rt._track3_schedule_lr(1600, group, old_kind, rt.TRACK3_LR_SCHEDULE_STEPS, rt.TRACK3_LR_POWER)
    assert group["lr"] == pytest.approx(old_at_switch)
    rt.set_hparams(1700)
    new_at_end = rt._track3_schedule_lr(1700, group, "pr287", 3105, rt.TRACK3_LR_AFTER_SWITCH_POWER)
    assert group["lr"] == pytest.approx(new_at_end)


@pytest.mark.parametrize("native_seed", [None, 28])
def test_rmsprop_checkpoint_restores_local_state(native_seed, monkeypatch, tmp_path):
    monkeypatch.setenv("TRACK3_LOCOM_LOCAL_OPT", "rmsprop")
    rt = runtime("make_track3_locoprop_m")
    model = nn.Module()
    block = nn.Module()
    block.mlp = rt.MLP(3, 0)
    model.blocks = nn.ModuleList([block])
    block.mlp._loco_rms_avg.fill_(.3)
    block.mlp._loco_rms_mom.fill_(.4)
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    rt.TRACK3_CHECKPOINT_STEPS = {2}
    rt.TRACK3_CHECKPOINT_DIR = str(tmp_path)
    if native_seed is not None:
        rt.TRACK3_RESET_TRIAL_SEED = False
        rt.SEED = native_seed
    rt.maybe_save_track3_checkpoint(model, [opt], 2, 3000, 0, None)
    path, = tmp_path.glob("*.pt")
    if native_seed is not None:
        assert torch.load(path, weights_only=True)["seed"] == native_seed
        assert "seed28" in path.name
    block.mlp._loco_rms_avg.zero_()
    block.mlp._loco_rms_mom.zero_()
    rt.TRACK3_RESUME_CHECKPOINT = str(path)
    assert rt.maybe_load_track3_checkpoint(model, [opt]) == 2
    torch.testing.assert_close(block.mlp._loco_rms_avg, torch.full_like(block.mlp._loco_rms_avg, .3))
    torch.testing.assert_close(block.mlp._loco_rms_mom, torch.full_like(block.mlp._loco_rms_mom, .4))


def _distributed_checkpoint_worker(rank, rendezvous, checkpoint_dir):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        rt = runtime("make_track3_locoprop_m")
        rt.LOCO_M_LOCAL_OPT = "rmsprop"
        rt.LOCO_M_OWNED_LAYER_SET = {rank}
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for layer, block in enumerate(model.blocks):
            block.mlp = rt.MLP(1, layer)
        owned = model.blocks[rank].mlp
        owned._loco_rms_avg.fill_(rank + .3)
        owned._loco_rms_mom.fill_(rank + .4)
        opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        opt.step_count = 10 + rank
        opt.state[owned.fc.weight]["momentum_buffer"] = torch.full_like(owned.fc.weight, rank + .5)
        rt.TRACK3_CHECKPOINT_STEPS = {2}
        rt.TRACK3_CHECKPOINT_DIR = checkpoint_dir
        torch.manual_seed(100 + rank)
        rng = torch.get_rng_state().clone()
        rt.maybe_save_track3_checkpoint(model, [opt], 2, 3000, 0, None)
        path, = Path(checkpoint_dir).glob("*.pt")
        rt.TRACK3_RESUME_CHECKPOINT = str(path)
        owned._loco_rms_avg.zero_()
        owned._loco_rms_mom.zero_()
        opt.state.clear()
        opt.step_count = 0
        torch.manual_seed(999)
        assert rt.maybe_load_track3_checkpoint(model, [opt]) == 2
        assert opt.step_count == 10 + rank
        torch.testing.assert_close(owned._loco_rms_avg, torch.full_like(owned._loco_rms_avg, rank + .3))
        torch.testing.assert_close(owned._loco_rms_mom, torch.full_like(owned._loco_rms_mom, rank + .4))
        torch.testing.assert_close(opt.state[owned.fc.weight]["momentum_buffer"],
                                   torch.full_like(owned.fc.weight, rank + .5))
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        dist.destroy_process_group()


def test_checkpoint_restores_optimizer_rms_and_rng_for_each_rank(tmp_path):
    mp.spawn(_distributed_checkpoint_worker,
             args=(str(tmp_path / "checkpoint_rendezvous"), str(tmp_path)), nprocs=2, join=True)


def test_record_runner_infers_seed_argument_and_runs_each_trial(tmp_path):
    command = tmp_path / "torchrun"
    command.write_text('#!/bin/sh\nprintf "%s %s\\n" "$TRACK3_SEED_OFFSET" "$*" >> "$LOCO_TEST_ARGS"\n')
    command.chmod(0o755)
    args_path = tmp_path / "args.txt"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               TRACK3_SOURCE="records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt",
               TRACK3_GENERATED_SCRIPT=str(tmp_path / "generated.py"), TRACK3_NUM_TRIALS="2",
               TRACK3_SEED_BASE="1", TRACK3_SEED_OFFSET="90", LOCO_TEST_ARGS=str(args_path))
    subprocess.run(["bash", "tools/run_track3_locoprop_m.sh"], cwd=ROOT, env=env, check=True, capture_output=True)
    lines = args_path.read_text().splitlines()
    assert len(lines) == 2 and lines[0].endswith("--seed 91") and lines[1].endswith("--seed 92")
    assert lines[0].startswith("90 ") and lines[1].startswith("91 ")


@pytest.mark.parametrize("explicit_override", [False, True])
def test_modal_record_payload_preserves_source_schedule_defaults(explicit_override):
    launcher = (ROOT / "tools/launch_modal_track3_locom_3000_seed.sh").read_text()
    body = launcher.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    env = dict(os.environ, TRACK3_SEED_OFFSET="90",
               TRACK3_SOURCE="records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt")
    if explicit_override:
        env.update(TRACK3_LR_SCHEDULE="power", TRACK3_LR_POWER="1.1", TRACK3_LR_SCHEDULE_STEPS="3000")
    result = subprocess.run([sys.executable, "-c", body], cwd=ROOT, env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["TRACK3_LR_SCHEDULE"] == ("power" if explicit_override else "pr287")
    assert payload["TRACK3_LR_POWER"] == ("1.1" if explicit_override else "1.2")
    assert payload["TRACK3_LR_SCHEDULE_STEPS"] == ("3000" if explicit_override else "3105")


@pytest.mark.parametrize("overrides, expected, error", [
    ({}, (0, 125), None),
    ({"TRAIN_PROGRESS_INTERVAL": "25", "SCREEN_VAL_EVERY": "5"}, (25, 5), None),
    ({"TRAIN_PROGRESS_INTERVAL": "-1"}, None, "TRAIN_PROGRESS_INTERVAL"),
    ({"SCREEN_VAL_EVERY": "0"}, None, "SCREEN_VAL_EVERY"),
    ({"SCREEN_VAL_EVERY": "-5"}, None, "SCREEN_VAL_EVERY"),
])
def test_wr_record_honors_and_validates_logging_intervals(tmp_path, monkeypatch, overrides, expected, error):
    source = ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt"
    output = tmp_path / "generated.py"
    generator("make_wr_record_locoprop_m").generate(source, output, 125, 3105)
    names = {"TRAIN_PROGRESS_INTERVAL", "val_regular_interval"}
    # Execute the actual generated assignments and guards without starting its CUDA driver.
    nodes = [node for node in ast.parse(output.read_text()).body
             if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                                      for target in node.targets))
             or (isinstance(node, ast.If) and any(isinstance(child, ast.Name) and child.id in names
                                                  for child in ast.walk(node.test)))]
    assert len(nodes) == 4
    for name in ("TRAIN_PROGRESS_INTERVAL", "SCREEN_VAL_EVERY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    namespace = {"os": os}
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(output), "exec")
    if error:
        with pytest.raises(ValueError, match=error):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert tuple(namespace[name] for name in ("TRAIN_PROGRESS_INTERVAL", "val_regular_interval")) == expected
