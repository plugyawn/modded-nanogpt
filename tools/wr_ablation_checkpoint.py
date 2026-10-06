"""Single-rank training checkpoints for the generated Muon ablation driver."""

import hashlib
import os
from pathlib import Path

import torch


class WrAblationDataIterator:
    """Native shard order/boundaries with a directly restorable next-token offset."""

    def __init__(self, pattern, batch_size, load_shard, seq_len=1024, *,
                 world_size=1, rank=0, device="cuda", verify_content=False):
        self.files = sorted(Path.cwd().glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no training shards match {pattern}")
        if batch_size % world_size:
            raise ValueError("global batch must be divisible by world size")
        self.batch_size, self.seq_len = batch_size, seq_len
        self.world_size, self.rank, self.device = world_size, rank, device
        self.load_shard = load_shard
        self.file_index, self.pos = 0, 0
        self.tokens = load_shard(self.files[0])
        self.manifest = []
        for path in self.files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                if verify_content:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                else:
                    digest.update(stream.read(1024))
            self.manifest.append((str(path.relative_to(Path.cwd())), path.stat().st_size, digest.hexdigest()))

    def __iter__(self):
        return self

    def __next__(self):
        # Preserve the native generator's >= boundary rule, including discarded tails.
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.file_index += 1
            if self.file_index >= len(self.files):
                raise StopIteration
            self.tokens, self.pos = self.load_shard(self.files[self.file_index]), 0
        local = self.batch_size // self.world_size
        buf = self.tokens[self.pos + self.rank * local:][:local + 1]
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)
        self.pos += self.batch_size
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def state_dict(self):
        return dict(file_index=self.file_index, pos=self.pos, manifest=self.manifest,
                    batch_size=self.batch_size, seq_len=self.seq_len,
                    world_size=self.world_size, rank=self.rank)

    def load_state_dict(self, state):
        for key in ("manifest", "batch_size", "seq_len", "world_size", "rank"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"training data checkpoint mismatch: {key}")
        index, pos = state["file_index"], state["pos"]
        if not 0 <= index < len(self.files) or pos < 0 or pos % self.batch_size:
            raise ValueError("invalid training shard checkpoint position")
        tokens = self.load_shard(self.files[index])
        if pos > len(tokens):
            raise ValueError("training checkpoint position exceeds its shard")
        self.file_index, self.pos, self.tokens = index, pos, tokens


def _wr_checkpoint_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _wr_checkpoint_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wr_checkpoint_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wr_checkpoint_cpu(item) for item in value)
    return value


def wr_save_ablation_checkpoint(directory, model, optimizers, loader, next_step,
                               identity, training_time=0.0):
    if identity["world_size"] != 1:
        raise ValueError("ablation checkpoints support world_size=1 only")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{next_step:06d}.pt"
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    payload = dict(schema=1, identity=identity, next_step=next_step,
        training_time=training_time, model=_wr_checkpoint_cpu(model.state_dict()),
        optimizers=[_wr_checkpoint_cpu(opt.state_dict()) for opt in optimizers],
        optimizer_step_counts=[getattr(opt, "step_count", None) for opt in optimizers],
        loader=loader.state_dict(), rng_cpu=torch.get_rng_state().clone(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    # This dedicated arm directory retains exactly the two latest recovery points.
    for stale in sorted(directory.glob("step_*.pt"))[:-2]:
        stale.unlink()
    return path


def wr_load_ablation_checkpoint(path, model, optimizers, loader, identity):
    # Checkpoints are generated by this task, not supplied as untrusted model files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1:
        raise ValueError("unsupported ablation checkpoint schema")
    if identity["world_size"] != 1 or payload["identity"].get("world_size") != 1:
        raise ValueError("ablation checkpoint resume requires world_size=1")
    if payload["identity"] != identity:
        differences = sorted(key for key in set(identity) | set(payload["identity"])
                             if identity.get(key) != payload["identity"].get(key))
        raise ValueError(f"ablation checkpoint identity mismatch: {differences}")
    if len(optimizers) != len(payload["optimizers"]):
        raise ValueError("ablation checkpoint optimizer count mismatch")
    if payload["next_step"] < 0 or payload["next_step"] > identity["train_steps"]:
        raise ValueError("invalid ablation checkpoint next step")
    loader.load_state_dict(payload["loader"])
    model.load_state_dict(payload["model"], strict=True)
    for opt, state, count in zip(optimizers, payload["optimizers"], payload["optimizer_step_counts"]):
        opt.load_state_dict(state)
        if count is not None:
            opt.step_count = count
    if payload["rng_cuda"]:
        if not torch.cuda.is_available() or len(payload["rng_cuda"]) != torch.cuda.device_count():
            raise ValueError("ablation checkpoint CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    # Restore RNG after every initialization/loading operation and before training.
    torch.set_rng_state(payload["rng_cpu"])
    return payload["next_step"], payload["training_time"]
