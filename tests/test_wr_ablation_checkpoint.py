import ast
import importlib.util
from pathlib import Path
import struct
import types

import pytest
import torch

from test_muon_ablation import harness, generated, optimizer_runtime


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_module():
    spec = importlib.util.spec_from_file_location("wr_checkpoint", ROOT / "tools/wr_ablation_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for index in range(4):
        values = list(range(index * 30, index * 30 + 21))
        (tmp_path / f"train_{index:02d}.bin").write_bytes(struct.pack("21H", *values))
    def load(path):
        data = Path(path).read_bytes()
        return torch.tensor(struct.unpack(f"{len(data) // 2}H", data), dtype=torch.uint16)
    return load


def test_loader_resume_preserves_native_discarded_tails_and_rng(tmp_path, monkeypatch):
    cp = checkpoint_module()
    load = shards(tmp_path, monkeypatch)
    loader = cp.WrAblationDataIterator("train_*.bin", 8, load, seq_len=4, device="cpu", verify_content=True)
    def native():
        files = sorted(Path.cwd().glob("train_*.bin"))
        file_iter = iter(files)
        tokens, pos = load(next(file_iter)), 0
        while True:
            if pos + 8 + 1 >= len(tokens):
                tokens, pos = load(next(file_iter)), 0
            buf = tokens[pos:][:9]
            pos += 8
            yield buf[:-1].to(torch.int32).view(-1, 4), buf[1:].long().view(-1, 4)
    original = native()
    for _ in range(3):
        actual, expected = next(loader), next(original)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    state = loader.state_dict()
    rng = torch.get_rng_state().clone()
    restored = cp.WrAblationDataIterator("train_*.bin", 8, load, seq_len=4, device="cpu", verify_content=True)
    restored.load_state_dict(state)
    assert torch.equal(rng, torch.get_rng_state())
    assert state["file_index"] == 1 and state["pos"] == 8
    for _ in range(4):
        for a, b in zip(next(restored), next(original)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # A same-size, same-header data replacement must be rejected by the full content hash.
    path = tmp_path / "train_03.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    changed = cp.WrAblationDataIterator("train_*.bin", 8, load, seq_len=4, device="cpu", verify_content=True)
    with pytest.raises(ValueError, match="manifest"):
        changed.load_state_dict(state)


def test_checkpoint_roundtrip_retention_identity_and_every_optimizer(tmp_path, monkeypatch):
    cp = checkpoint_module()
    load = shards(tmp_path, monkeypatch)
    model = torch.nn.Linear(4, 4)
    model.register_buffer("_loco_rms_avg", torch.randn_like(model.weight))
    model.register_buffer("_loco_rms_mom", torch.randn_like(model.weight))
    opts = [torch.optim.SGD([model.weight], lr=.1, momentum=.9), torch.optim.AdamW([model.bias], lr=.01)]
    opts[0].step_count = 7
    model(torch.randn(2, 4)).sum().backward()
    for opt in opts:
        opt.step()
    loader = cp.WrAblationDataIterator("train_*.bin", 8, load, seq_len=4, device="cpu", verify_content=True)
    identity = dict(world_size=1, train_steps=10, seed=17, generated_sha256="same", optimizer_settings="simple")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for step in (1, 2, 3):
        next(loader)
        checkpoint = cp.wr_save_ablation_checkpoint(tmp_path / "checkpoints", model, opts, loader, step, identity, 12.5)
    assert [path.name for path in sorted((tmp_path / "checkpoints").glob("*.pt"))] == ["step_000002.pt", "step_000003.pt"]
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))
    expected_rng = torch.rand(5)
    for p in model.parameters():
        p.data.zero_()
    model._loco_rms_avg.zero_()
    model._loco_rms_mom.zero_()
    opts[0].step_count = 0
    assert cp.wr_load_ablation_checkpoint(checkpoint, model, opts, loader, identity) == (3, 12.5)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert opts[0].step_count == 7
    assert opts[0].state[model.weight]["momentum_buffer"].norm() > 0
    assert opts[1].state[model.bias]["exp_avg_sq"].norm() > 0
    torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
    for key, value in {"world_size": 2, "seed": 18, "generated_sha256": "changed", "optimizer_settings": "soap"}.items():
        with pytest.raises(ValueError):
            cp.wr_load_ablation_checkpoint(checkpoint, model, opts, loader, {**identity, key: value})


def test_generated_training_loop_resume_matches_continuous_weights_soap_rms_and_rng(harness, tmp_path, monkeypatch):
    from test_locoprop_generators import runtime
    monkeypatch.setenv("WR_MUON_PROFILE", "soap")
    monkeypatch.setenv("WR_LOCOM_SAMPLE_TOKENS", "8")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    load = shards(tmp_path, monkeypatch)
    text, _ = generated(harness, tmp_path)
    import make_wr_record_muon_ablation as gen
    loop = next(node for node in ast.parse(text).body if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Call) and any(isinstance(arg, ast.Name) and arg.id == "start_step" for arg in node.iter.args))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.register_buffer("_loco_rms_avg", torch.zeros_like(self.fc.weight))
            self.register_buffer("_loco_rms_mom", torch.zeros_like(self.fc.weight))
        def forward(self, inputs, targets):
            x = torch.nn.functional.dropout(inputs.float() / 100, .2, self.training)
            return (self.fc(x) - targets.float() / 100).square().sum()
    def driver(directory, resume=None, exit_step=-1, marker_step=None):
        monkeypatch.setenv("WR_MUON_CHECKPOINT_DIR", str(directory))
        monkeypatch.setenv("WR_MUON_CHECKPOINT_EVERY", "2")
        monkeypatch.setenv("WR_MUON_CHECKPOINT_EXIT_STEP", str(exit_step))
        stop_file = directory / "checkpoint_stop.request"
        monkeypatch.setenv("WR_MUON_CHECKPOINT_STOP_FILE", str(stop_file) if marker_step else "")
        torch.manual_seed(31)
        rt = runtime("make_wr_record_locoprop_m")
        rt.FINAL_TRAIN_STEPS = 6
        exec(gen.ABLATION_CONFIG + gen.CHECKPOINT_SOURCE + gen.CHECKPOINT_RUNTIME, vars(rt))
        real = optimizer_runtime(text, gen.ABLATION_CONFIG)
        rt.model = Model()
        rt.optimizers = [torch.optim.AdamW([rt.model.fc.bias], lr=.001),
                         real.Muon([("0.mlp.fc.weight", rt.model.fc.weight)], lr=.02)]
        rt.optimizer2 = rt.optimizers[1]
        rt.dist = types.SimpleNamespace(get_world_size=lambda: 1, all_reduce=lambda *a, **kw: None,
                                        barrier=lambda: None, ReduceOp=types.SimpleNamespace(SUM=None))
        rt.train_loader = rt.WrAblationDataIterator("train_*.bin", 8, load, seq_len=4, device="cpu", verify_content=True)
        rt.val_inputs, rt.val_targets = torch.ones(2, 4, dtype=torch.int32), torch.ones(2, 4, dtype=torch.int64)
        rt.train_steps, rt.FINAL_SCHEDULE_STEPS, rt.SEED = 6, 10, 31
        rt.batch_size, rt.mbs, rt.val_tokens, rt.val_regular_interval = 8, 2, 8, 2
        rt.extra_val_steps, rt.TRAIN_PROGRESS_INTERVAL = set(), 0
        rt.code, rt.training_time, rt.start_step = text, 0.0, 0
        rt.time = __import__("time")
        rt._wr_locom_begin_step = lambda *a: None
        rt.prepare_wr_locom_m = lambda *a: None
        def apply(model, optimizer, step):
            model._loco_rms_avg.lerp_(model.fc.weight.grad.square(), .1)
            model._loco_rms_mom.lerp_(model.fc.weight.grad, .2)
            if step + 1 == marker_step:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()
        rt.apply_wr_locom_m = apply
        rt.set_hparams = lambda step: None
        if resume:
            rt.start_step, rt.training_time = rt.wr_load_ablation_checkpoint(resume, rt.model, rt.optimizers,
                rt.train_loader, rt._wr_ablation_checkpoint_identity())
        rt.t0 = rt.time.perf_counter()
        exec(compile(ast.Module([loop], type_ignores=[]), "real_generated_training_loop", "exec"), vars(rt))
        return rt, torch.rand(5)
    continuous, continuous_rng = driver(tmp_path / "continuous")
    stopped, _ = driver(tmp_path / "split", exit_step=3)
    assert stopped.optimizer2.step_count == 3
    resumed, resumed_rng = driver(tmp_path / "split", tmp_path / "split/step_000003.pt")
    assert resumed.optimizer2.step_count == continuous.optimizer2.step_count == 6
    for key, value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    for a, b in zip(continuous.optimizers, resumed.optimizers):
        for p, q in zip(a.param_groups[0]["params"], b.param_groups[0]["params"]):
            for key, value in a.state[p].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(b.state[q][key], value, rtol=0, atol=0)
                else:
                    assert b.state[q][key] == value
    assert resumed.train_loader.state_dict() == continuous.train_loader.state_dict()
    torch.testing.assert_close(resumed_rng, continuous_rng, rtol=0, atol=0)
    marker, _ = driver(tmp_path / "marker", marker_step=3)
    assert marker.optimizer2.step_count == 3 and marker.WR_MUON_CHECKPOINT_EXIT_REASON == "stop_file"
    assert (tmp_path / "marker/step_000003.pt").exists()
    assert not (tmp_path / "marker/step_000006.pt").exists()
