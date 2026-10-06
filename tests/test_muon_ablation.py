import ast
import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt"


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    for name in ("WR_MUON_PROFILE", "WR_MUON_CONTRA", "WR_MUON_SOFT", "WR_MUON_SOAP",
                 "WR_MUON_NORMUON", "WR_MUON_UW_FLOOR", "WR_MUON_DECOUPLED_WD", "WR_MUON_POLAR_NORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WR_LOCOM_UPDATE_MODE", raising=False)
    monkeypatch.delenv("WR_LOCOM_REPLACE_NORM_CAP", raising=False)
    spec = importlib.util.spec_from_file_location("muon_ablation_runner", ROOT / "tools/run_wr_muon_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimizer_runtime(source, config=None):
    """Run real optimizer definitions while excluding the CUDA training driver."""
    import os
    source = source.split("\n====================================================================================================", 1)[0]
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id.isupper()
                and target.id not in {"SEED", "LOG_DIR"} for target in node.targets):
            # The generated LocoProp globals require its runtime; only record constants are needed here.
            if all(not isinstance(target, ast.Name) or not target.id.startswith(("WR_", "LOCO_"))
                   for target in node.targets):
                nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {
            "gram_frobenius_norm_estimate", "zeropower_via_newtonschulz5", "_soft_coefficients",
            "zeropower_frobenius_norm_like", "soft_via_newtonschulz5", "scale_to_unit_operator_norm",
            "should_soap_param", "is_attn_proj_param", "is_attn_param", "is_v_param",
            "param_matches_spec", "cosine_similarity", "trust_gate", "early_trust_floor_for_step",
            "bounded_trust_gate", "attention_soap_blend_for_step", "norm_preserving_blend",
            "soap_precondition_momentum", "soap_update_preconditioner", "soap_eigenbasis", "_linear_ramp",
            "contra_coeff_for_step", "soft_blend_for_step", "muon_update"}:
            node.decorator_list = []
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Muon":
            nodes.append(node)
    namespace = dict(torch=torch, Tensor=Tensor, os=os,
                     dist=types.SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0,
                                                all_gather=lambda *a, **kw: None))
    if config:
        exec(config, namespace)
    exec(compile(ast.Module(nodes, type_ignores=[]), "optimizer_ablation", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def generated(harness, tmp_path):
    import make_wr_record_muon_ablation as generator
    path = tmp_path / "generated.py"
    generator.generate(SOURCE, path)
    return path.read_text(), generator.ABLATION_CONFIG


def test_default_profile_preserves_native_may9_updates_and_state(harness, tmp_path):
    text, config = generated(harness, tmp_path)
    actual = optimizer_runtime(text, config)
    native = optimizer_runtime(SOURCE.read_text())
    torch.manual_seed(7)
    weights = [torch.randn(5, 3), torch.randn(3, 5)]
    ap = [torch.nn.Parameter(p.clone()) for p in weights]
    bp = [torch.nn.Parameter(p.clone()) for p in weights]
    names = ["0.mlp.fc.weight", "0.attn.v.weight"]
    a = actual.Muon(list(zip(names, ap)), lr=.0375, weight_decay=.025)
    b = native.Muon(list(zip(names, bp)), lr=.0375, weight_decay=.025)
    for step in (0, 1, 2750):
        a.step_count = b.step_count = step
        for p, q in zip(ap, bp):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        a.step()
        b.step()
        for p, q in zip(ap, bp):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            assert a.state[p].keys() == b.state[q].keys()
            for name, value in a.state[p].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, b.state[q][name], rtol=0, atol=0)
                else:
                    assert value == b.state[q][name]


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_simple_profile_matches_original_muon_weight_decay_and_momentum(harness, tmp_path, monkeypatch, shape):
    monkeypatch.setenv("WR_MUON_PROFILE", "simple")
    text, config = generated(harness, tmp_path)
    actual = optimizer_runtime(text, config)
    original = optimizer_runtime((ROOT / "records/track_3_optimization/train_gpt_simple.py").read_text())
    torch.manual_seed(8)
    p = torch.nn.Parameter(torch.randn(shape))
    q = torch.nn.Parameter(p.detach().clone())
    a = actual.Muon([("0.mlp.fc.weight", p)], lr=.0375, weight_decay=.025)
    b = original.Muon([q], lr=.0375, weight_decay=.025)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(a.state[p]["momentum"], b.state[q]["momentum"], rtol=0, atol=0)
    assert set(a.state[p]) == {"momentum"}
    assert not a.soap_params


@pytest.mark.parametrize("profile", ["contra", "soft", "soap", "normuon", "floor", "polar_norm", "no_wd"])
def test_each_optimizer_lever_changes_real_update(harness, tmp_path, monkeypatch, profile):
    text, config = generated(harness, tmp_path)
    monkeypatch.setenv("WR_MUON_PROFILE", "simple")
    simple = optimizer_runtime(text, config)
    monkeypatch.setenv("WR_MUON_PROFILE", profile)
    arm = optimizer_runtime(text, config)
    torch.manual_seed(20)
    # Large weight magnitudes activate the u/w floor; all arms share the same initial weight.
    p = torch.nn.Parameter(torch.randn(5, 3) * 100)
    q = torch.nn.Parameter(p.detach().clone())
    a = simple.Muon([("0.mlp.fc.weight", p)], lr=.0375, weight_decay=.025)
    b = arm.Muon([("0.mlp.fc.weight", q)], lr=.0375, weight_decay=.025)
    for step in range(4):
        a.step_count = b.step_count = 3000 if profile == "soft" else step
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
    assert not torch.equal(p, q), f"{profile} was silently inactive"
    if profile == "soap":
        assert b.state[q]["q_row"] is not None and b.state[q]["soap_step"] == 4
        assert not a.soap_params
    if profile == "normuon":
        assert b.state[q]["second_moment"].norm() > 0
        assert "second_moment" not in a.state[p]


def test_plan_pairs_and_single_lever_settings(harness, tmp_path):
    plan = harness.prepare(SOURCE, tmp_path / "plan", list(harness.PROFILES), [3710, 3711], native_timing=True)
    assert len(plan["arms"]) == 90
    assert plan["common"]["steps"] == 3040 and plan["common"]["schedule_steps"] == 3105
    simple = harness.profile_settings("simple")
    for profile in harness.PROFILES[1:-1]:
        arm = harness.profile_settings(profile)
        assert sum(simple[key] != arm[key] for key in simple if key != "profile") == 1
    for profile in harness.PROFILES:
        for solver in plan["solvers"]:
            control = next(arm for arm in plan["arms"] if arm["id"] == f"{profile}_alpha0_{solver}_seed3710")
            treatment = next(arm for arm in plan["arms"] if arm["id"] == f"{profile}_alpha1_{solver}_seed3710")
            assert {key for key in control["env"] if control["env"][key] != treatment["env"][key]} == {"WR_LOCOM_ALPHA"}
            assert control["env"]["WR_LOCOM_ENABLED"] == "1"
            assert control["env"]["WR_LOCOM_LOCAL_OPT"] == harness.SOLVERS[solver][0]
            assert int(control["env"]["WR_LOCOM_STEPS"]) == harness.SOLVERS[solver][1]
            assert control["env"]["WR_LOCOM_RMS_STYLE"] == harness.SOLVER_METADATA[solver]["rms_style"]
            assert int(control["env"]["WR_LOCOM_LR_DECAY"]) == harness.SOLVER_METADATA[solver]["local_lr_decay"]
        native = next(arm for arm in plan["arms"] if arm["id"] == f"{profile}_native_seed3710")
        assert native["env"]["WR_LOCOM_ENABLED"] == "0"
    saved = json.loads((tmp_path / "plan/plan.json").read_text())
    assert saved["generated_sha256"] == harness.sha256(tmp_path / "plan/train_muon_ablation.py")
    with pytest.raises(FileExistsError):
        harness.prepare(SOURCE, tmp_path / "plan", ["simple"], [3710])


def test_partial_fc_replacement_skips_only_active_selected_fc_native_update(harness, tmp_path, monkeypatch):
    monkeypatch.setenv("WR_MUON_PROFILE", "simple")
    monkeypatch.setenv("WR_LOCOM_UPDATE_MODE", "partial_fc_replacement")
    text, config = generated(harness, tmp_path)
    rt = optimizer_runtime(text, config)
    active = [True]
    rt.muon_update.__globals__.update(WR_LOCOM_CURRENT_STEP=0,
        _wr_locom_active=lambda step: active[0], _wr_locom_layer_active=lambda layer: layer == 0)
    torch.manual_seed(17)
    params = [torch.nn.Parameter(torch.randn(5, 3)) for _ in range(3)]
    names = ["0.mlp.fc.weight", "1.mlp.fc.weight", "0.attn.q.weight"]
    before = [p.detach().clone() for p in params]
    opt = rt.Muon(list(zip(names, params)), lr=.0375, weight_decay=.025)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    assert torch.equal(params[0], before[0])
    assert not torch.equal(params[1], before[1]) and not torch.equal(params[2], before[2])
    assert opt.state[params[0]]["momentum"].norm() > 0
    active[0] = False
    opt.step()
    assert not torch.equal(params[0], before[0])


def test_replacement_cap_uses_weight_norm_independently_of_base_step(harness):
    import make_wr_record_muon_ablation as generator
    namespace = dict(os=__import__("os"), torch=torch, WR_LOCOM_ALPHA=1)
    exec(generator.ABLATION_CONFIG, namespace)
    corr, reference = torch.ones(3, 2) * 100, torch.ones(3, 2)
    scale = namespace["_wr_locom_replacement_scale"](corr, reference)
    torch.testing.assert_close(scale * corr.norm(), .01 * reference.norm())
    namespace["WR_LOCOM_REPLACE_NORM_CAP"] = 0
    assert namespace["_wr_locom_replacement_scale"](corr, reference) == 1


def test_partial_fc_replacement_applies_stored_local_change_without_base_fc_step(harness, tmp_path, monkeypatch):
    from test_locoprop_generators import runtime
    monkeypatch.setenv("WR_LOCOM_SAMPLE_TOKENS", "8")
    monkeypatch.setenv("WR_LOCOM_UPDATE_MODE", "partial_fc_replacement")
    text, config = generated(harness, tmp_path)
    rt = runtime("make_wr_record_locoprop_m")
    exec(config, vars(rt))
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                and node.name == "apply_wr_locom_m")
    exec(compile(ast.Module([node], type_ignores=[]), "replacement_apply", "exec"), vars(rt))
    block = torch.nn.Module()
    block.mlp = rt.MLP(3)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    p = block.mlp.fc.weight
    p.data.fill_(.1)
    block.mlp.fc.bias.data.fill_(2)
    reference, bias_reference = p.detach().clone(), block.mlp.fc.bias.detach().clone()
    x = torch.ones(8, 3)
    pre0 = x @ reference.T + bias_reference
    dpre = torch.ones_like(pre0) * .001
    corr = torch.full_like(p, -1e-4)
    # The FC weight is frozen by the Muon hook, while native bias optimization still runs.
    block.mlp.fc.bias.data.add_(.0001)
    rt.WR_LOCOM_CORR[0] = (corr, reference, x, pre0, dpre, bias_reference, None)
    rt.apply_wr_locom_m(model, None, 0)
    assert (p - reference).norm() > 0
    assert (p - reference).norm() <= .01 * reference.norm()
    assert torch.equal(block.mlp.fc.bias, bias_reference + .0001)


def test_runner_records_effective_env_and_prevents_duplicate_launch(harness, tmp_path, monkeypatch):
    output = tmp_path / "plan"
    plan = harness.prepare(SOURCE, output, ["simple"], [3710], solvers=["sgd4"])
    monkeypatch.setenv("WR_MUON_SOAP", "1")
    monkeypatch.setenv("WR_LOCOM_UNKNOWN_INHERITED_FLAG", "1")
    monkeypatch.setenv("WR_MUON_CHECKPOINT_STOP_FILE", str(tmp_path / "inherited-stop"))
    seen = []
    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        kwargs["stdout"].write("step:3040/3040 val_loss:3.28\n")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    arm = plan["arms"][0]["id"]
    assert harness.run(output / "plan.json", arm, ROOT) == 0
    command, kwargs = seen[0]
    assert command[-2:] == ["--seed", "3710"]
    assert kwargs["env"]["WR_MUON_SOAP"] == "0"
    assert "WR_LOCOM_UNKNOWN_INHERITED_FLAG" not in kwargs["env"]
    assert "WR_MUON_CHECKPOINT_STOP_FILE" not in kwargs["env"]
    status = json.loads((output / arm / "status.json").read_text())
    assert status["exit_code"] == 0 and status["console_sha256"] == harness.sha256(output / arm / "console.log")
    with pytest.raises(RuntimeError):
        harness.run(output / "plan.json", arm, ROOT)
    assert len(seen) == 1


def test_runner_checkpoint_stop_marker_is_explicit_and_logged(harness, tmp_path, monkeypatch):
    output = tmp_path / "plan"
    plan = harness.prepare(SOURCE, output, ["simple"], [3710], solvers=["sgd4"])
    stop_file = tmp_path / "checkpoint_stop.request"
    seen = []
    def fake_run(command, **kwargs):
        seen.append(kwargs["env"])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    arm = plan["arms"][0]["id"]
    with pytest.raises(ValueError, match="requires checkpoint"):
        harness.run(output / "plan.json", arm, ROOT, checkpoint_stop_file=stop_file)
    harness.run(output / "plan.json", arm, ROOT, checkpoint=True, checkpoint_stop_file=stop_file)
    assert seen[0]["WR_MUON_CHECKPOINT_STOP_FILE"] == str(stop_file.resolve())
    status = json.loads((output / arm / "status.json").read_text())
    assert status["env"]["WR_MUON_CHECKPOINT_STOP_FILE"] == str(stop_file.resolve())


def test_runner_rejects_unreviewed_generated_script(harness, tmp_path, monkeypatch):
    output = tmp_path / "plan"
    plan = harness.prepare(SOURCE, output, ["simple"], [3710], solvers=["sgd4"])
    (output / "train_muon_ablation.py").write_text("raise RuntimeError('changed')")
    with pytest.raises(ValueError, match="checksum"):
        harness.run(output / "plan.json", plan["arms"][0]["id"], ROOT)


@pytest.mark.parametrize("name,value", [("WR_MUON_PROFILE", "typo"), ("WR_MUON_CONTRA", "maybe"), ("WR_MUON_POLAR_NORM", "spectral")])
def test_invalid_flags_fail_closed(harness, monkeypatch, name, value):
    import make_wr_record_muon_ablation as generator
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        exec(generator.ABLATION_CONFIG, {"os": __import__("os")})
