#!/usr/bin/env python3
"""Generate a current-record Track 3 script with additive c_fc LocoProp-M.

This targets the 2026-05-09 current-record source format under
records/track_3_optimization/results/20260509_contra_soft_muon/*.txt. That
source is a simple module-level script, not the sharded speedrun-bank script
patched by make_wr_locoprop_m.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SHARED_LOCAL_SOURCE = Path(__file__).with_name("locoprop_local.py").read_text()


LOCOM_BLOCK = r'''
def _wr_locom_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def _wr_locom_parse_set(spec: str, total_layers: int = 12) -> set[int]:
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return set(range(total_layers))
    if spec in {"", "none"}:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out

def _wr_locom_parse_windows(spec: str) -> list[tuple[int, int]]:
    if not spec.strip():
        return []
    windows = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split(":", 1)
        windows.append((int(lo), int(hi)))
    return windows

WR_LOCOM_ENABLED = _wr_locom_flag("WR_LOCOM_ENABLED", "1")
WR_LOCOM_LAYERS_SPEC = os.environ.get("WR_LOCOM_LAYERS", "all")
WR_LOCOM_LAYER_SET = _wr_locom_parse_set(WR_LOCOM_LAYERS_SPEC)
WR_LOCOM_ACTIVE_WINDOWS = _wr_locom_parse_windows(os.environ.get("WR_LOCOM_ACTIVE_WINDOWS", ""))
WR_LOCOM_START_STEP = int(os.environ.get("WR_LOCOM_START_STEP", "0"))
WR_LOCOM_END_STEP = int(os.environ.get("WR_LOCOM_END_STEP", "1000000000"))
WR_LOCOM_INTERVAL = int(os.environ.get("WR_LOCOM_INTERVAL", "1"))
WR_LOCOM_SAMPLE_TOKENS = int(os.environ.get("WR_LOCOM_SAMPLE_TOKENS", "1024"))
WR_LOCOM_STEPS = int(os.environ.get("WR_LOCOM_STEPS", "4"))
WR_LOCOM_INNER_LR = float(os.environ.get("WR_LOCOM_INNER_LR", "0.1"))
WR_LOCOM_TARGET_GAMMA = float(os.environ.get("WR_LOCOM_TARGET_GAMMA", "1.0"))
WR_LOCOM_PROX = float(os.environ.get("WR_LOCOM_PROX", "0.1"))
WR_LOCOM_ALPHA = float(os.environ.get("WR_LOCOM_ALPHA", "1.0"))
WR_LOCOM_NORM_CAP = float(os.environ.get("WR_LOCOM_NORM_CAP", "0.20"))
WR_LOCOM_NORM_TO_BASE = _wr_locom_flag("WR_LOCOM_NORM_TO_BASE", "0")
WR_LOCOM_REQUIRE_LOSS_DECREASE = _wr_locom_flag("WR_LOCOM_REQUIRE_LOSS_DECREASE", "1")
WR_LOCOM_MIN_COS_DESC = float(os.environ.get("WR_LOCOM_MIN_COS_DESC", "0.0"))
WR_LOCOM_LOG_STEPS = {
    int(x)
    for x in os.environ.get("WR_LOCOM_LOG_STEPS", "0,1,2,10,50,125,250,500,1000,1500,2000,2125,2250,2500,2750,3000").split(",")
    if x.strip()
}

WR_LOCOM_MAX_BACKTRACKS = int(os.environ.get("WR_LOCOM_MAX_BACKTRACKS", "20"))
WR_LOCOM_ACCUM_SAMPLES = _wr_locom_flag("WR_LOCOM_ACCUM_SAMPLES", "1")
WR_LOCOM_LOCAL_OPT = os.environ.get("WR_LOCOM_LOCAL_OPT", "sgd").lower()
if WR_LOCOM_LOCAL_OPT not in {"sgd", "rmsprop"}:
    raise ValueError("WR_LOCOM_LOCAL_OPT must be 'sgd' or 'rmsprop'")
WR_LOCOM_RMS_BETA1 = float(os.environ.get("WR_LOCOM_RMS_BETA1", "0.999"))
WR_LOCOM_RMS_BETA2 = float(os.environ.get("WR_LOCOM_RMS_BETA2", "0.9"))
WR_LOCOM_RMS_EPS = float(os.environ.get("WR_LOCOM_RMS_EPS", "1e-5"))
WR_LOCOM_RMS_STYLE = os.environ.get("WR_LOCOM_RMS_STYLE", "torch").lower()
if WR_LOCOM_RMS_STYLE not in {"torch", "tensorflow"}:
    raise ValueError("WR_LOCOM_RMS_STYLE must be 'torch' or 'tensorflow'")
WR_LOCOM_RESET_RMS = _wr_locom_flag("WR_LOCOM_RESET_RMS", "0")
WR_LOCOM_LR_DECAY = _wr_locom_flag("WR_LOCOM_LR_DECAY", "0")
WR_LOCOM_SCALE_MOMENTUM = _wr_locom_flag("WR_LOCOM_SCALE_MOMENTUM", "0")
WR_LOCOM_CENTER = os.environ.get("WR_LOCOM_CENTER", "pre_base")
if WR_LOCOM_CENTER not in {"pre_base", "post_base"}:
    raise ValueError("WR_LOCOM_CENTER must be pre_base or post_base")
WR_LOCOM_LINEAR_TERM = os.environ.get("WR_LOCOM_LINEAR_TERM", "sample")
if WR_LOCOM_LINEAR_TERM not in {"sample", "full"}:
    raise ValueError("WR_LOCOM_LINEAR_TERM must be sample or full")

WR_LOCOM_CURRENT_STEP = -1
WR_LOCOM_NEXT_LAYER = 0
WR_LOCOM_CORR: dict[int, tuple] = {}
WR_LOCOM_LINEAR_GRADS: dict[int, Tensor] = {}
WR_LOCOM_APPLY_STATS: list[str] = []

def _wr_locom_active(step: int) -> bool:
    if not (WR_LOCOM_ENABLED and WR_LOCOM_STEPS > 0):
        return False
    if step % max(WR_LOCOM_INTERVAL, 1) != 0:
        return False
    if WR_LOCOM_ACTIVE_WINDOWS:
        return any(start <= step < end for start, end in WR_LOCOM_ACTIVE_WINDOWS)
    return WR_LOCOM_START_STEP <= step < WR_LOCOM_END_STEP

def _wr_locom_layer_active(layer_idx: int) -> bool:
    return layer_idx in WR_LOCOM_LAYER_SET

def _wr_locom_begin_step(step: int, microbatches: int = 1, model=None) -> None:
    global WR_LOCOM_CURRENT_STEP
    WR_LOCOM_CURRENT_STEP = step
    WR_LOCOM_CORR.clear()
    WR_LOCOM_LINEAR_GRADS.clear()
    WR_LOCOM_APPLY_STATS.clear()
    if model is not None:
        for block in model.blocks:
            block.mlp.wr_locom_capture_now = _wr_locom_active(step)
    loco_begin_capture(step, microbatches, _wr_locom_active(step),
                       WR_LOCOM_LAYER_SET, WR_LOCOM_ACCUM_SAMPLES, dist.get_rank())

@torch.no_grad()
def _wr_locom_gather(t: Tensor) -> Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t.float()
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0).float()

@torch.no_grad()
def prepare_wr_locom_m(model: nn.Module, step: int) -> None:
    if not _wr_locom_active(step):
        return
    stats = []
    for layer_idx, block in enumerate(model.blocks):
        if not _wr_locom_layer_active(layer_idx):
            continue
        layer = block.mlp
        sx, sp, sg = loco_samples(layer._loco_samples, layer_idx, layer.fc.weight.shape[1])
        if sx.shape[0] == 0:
            continue
        x, pre0, dpre = _wr_locom_gather(sx), _wr_locom_gather(sp), _wr_locom_gather(sg)
        raw_grad = layer.fc.weight.grad
        linear_grad = None
        if WR_LOCOM_LINEAR_TERM == "full" and raw_grad is not None:
            # This driver sums token losses, accumulated microbatches, and
            # distributed gradients. batch_size is the global token count.
            if batch_size <= 0:
                raise ValueError("full linear term requires a positive global token count")
            linear_grad = raw_grad.detach().float() / batch_size
            WR_LOCOM_LINEAR_GRADS[layer_idx] = linear_grad
        if WR_LOCOM_CENTER == "post_base":
            # Defer the solve until the native optimizer has moved the weights.
            # Keep the captured global gradient; this remains a frozen-input
            # correction surrogate, not a second full-model backward pass.
            bias_reference = layer.fc.bias.detach().float().clone() if layer.fc.bias is not None else None
            WR_LOCOM_CORR[layer_idx] = (None, layer.fc.weight.detach().float().clone(),
                x, pre0, dpre, bias_reference, raw_grad.detach().float().clone() if raw_grad is not None else None)
            continue
        corr, diag, staged_rms = loco_solve(
            x, pre0, dpre, raw_grad, steps=WR_LOCOM_STEPS,
            inner_lr=WR_LOCOM_INNER_LR, gamma=WR_LOCOM_TARGET_GAMMA,
            prox=WR_LOCOM_PROX, min_cos=WR_LOCOM_MIN_COS_DESC,
            require_decrease=WR_LOCOM_REQUIRE_LOSS_DECREASE,
            max_backtracks=WR_LOCOM_MAX_BACKTRACKS, output_dtype=layer.fc.weight.dtype,
            local_opt=WR_LOCOM_LOCAL_OPT, rms_avg=layer._loco_rms_avg,
            rms_mom=layer._loco_rms_mom, beta1=WR_LOCOM_RMS_BETA1,
            beta2=WR_LOCOM_RMS_BETA2, rms_eps=WR_LOCOM_RMS_EPS,
            reset_rms=WR_LOCOM_RESET_RMS, lr_decay=WR_LOCOM_LR_DECAY,
            rms_style=WR_LOCOM_RMS_STYLE,
            scale_momentum=WR_LOCOM_SCALE_MOMENTUM,
            linear_grad=linear_grad,
        )
        if diag["accepted"]:
            bias_reference = layer.fc.bias.detach().float().clone() if layer.fc.bias is not None else None
            WR_LOCOM_CORR[layer_idx] = (corr, layer.fc.weight.detach().float().clone(),
                                       x, pre0, dpre, bias_reference, staged_rms)
        if step in WR_LOCOM_LOG_STEPS:
            stats.append(
                f"l{layer_idx}:loss0={diag['loss0']:.3e},lossK={diag['lossK']:.3e}"
                f",corr={float(corr.float().norm()):.3e}"
                f",grad={float(raw_grad.float().norm()) if raw_grad is not None else float('nan'):.3e}"
                f",cos={diag['cos_desc']:.3f},accepted={int(diag['accepted'])},tokens={x.size(0)}"
                f",backtracks={diag['backtracks']},local_steps={diag['local_steps']}"
                f",inner_lr={diag['inner_lr']:.3e},reason={diag['reason']}"
            )
    if step in WR_LOCOM_LOG_STEPS and stats:
        print0("wr_locom_prepare step=" + str(step) + " " + " | ".join(stats), console=True)

@torch.no_grad()
def apply_wr_locom_m(model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    if not _wr_locom_active(step):
        return
    for layer_idx, context in sorted(WR_LOCOM_CORR.items()):
        corr, reference, x, pre0, dpre, bias_reference, staged_rms = context
        layer = model.blocks[layer_idx].mlp
        p = layer.fc.weight
        linear_grad = WR_LOCOM_LINEAR_GRADS.get(layer_idx)
        # Muon has already gathered the updated weights onto every rank. Measure
        # their actual displacement; only the owner has the optimizer's state.
        base_delta = p.float() - reference
        base_step_norm = base_delta.norm()
        bias_delta = layer.fc.bias.float() - bias_reference if bias_reference is not None else None
        if WR_LOCOM_CENTER == "post_base":
            raw_grad = staged_rms
            pre0 = pre0 + x @ base_delta.mT
            if bias_delta is not None:
                pre0 = pre0 + bias_delta
            corr, diag, staged_rms = loco_solve(
                x, pre0, dpre, raw_grad, steps=WR_LOCOM_STEPS,
                inner_lr=WR_LOCOM_INNER_LR, gamma=WR_LOCOM_TARGET_GAMMA,
                prox=WR_LOCOM_PROX, min_cos=WR_LOCOM_MIN_COS_DESC,
                require_decrease=WR_LOCOM_REQUIRE_LOSS_DECREASE,
                max_backtracks=WR_LOCOM_MAX_BACKTRACKS, output_dtype=p.dtype,
                local_opt=WR_LOCOM_LOCAL_OPT, rms_avg=layer._loco_rms_avg,
                rms_mom=layer._loco_rms_mom, beta1=WR_LOCOM_RMS_BETA1,
                beta2=WR_LOCOM_RMS_BETA2, rms_eps=WR_LOCOM_RMS_EPS,
                reset_rms=WR_LOCOM_RESET_RMS, lr_decay=WR_LOCOM_LR_DECAY,
                rms_style=WR_LOCOM_RMS_STYLE, scale_momentum=WR_LOCOM_SCALE_MOMENTUM,
                linear_grad=linear_grad)
            if step in WR_LOCOM_LOG_STEPS:
                print0(f"wr_locom_prepare step={step} l{layer_idx}:loss0={diag['loss0']:.3e},lossK={diag['lossK']:.3e},corr={float(corr.float().norm()):.3e},cos={diag['cos_desc']:.3f},accepted={int(diag['accepted'])},tokens={x.size(0)},backtracks={diag['backtracks']},local_steps={diag['local_steps']},inner_lr={diag['inner_lr']:.3e},reason={diag['reason']},center=post_base", console=True)
            if not diag["accepted"]:
                continue
            # Center both the trust penalty and final stored-weight check at
            # the same native state used by the solve.
            reference = p.detach().float().clone()
            base_delta = torch.zeros_like(reference)
            bias_delta = None
        corr_norm = corr.float().norm()
        scale = loco_correction_scale(corr, base_step_norm, alpha=WR_LOCOM_ALPHA,
                                     cap=WR_LOCOM_NORM_CAP, norm_to_base=WR_LOCOM_NORM_TO_BASE)
        scale, gate = loco_post_step_scale(corr, scale, base_delta, x, pre0, dpre,
                                          gamma=WR_LOCOM_TARGET_GAMMA, prox=WR_LOCOM_PROX,
                                          max_backtracks=WR_LOCOM_MAX_BACKTRACKS,
                                          bias_delta=bias_delta, current_weight=p,
                                          reference_weight=reference, return_info=True,
                                          linear_grad=linear_grad)
        applied = gate["candidate"] is not None
        logged = step in WR_LOCOM_LOG_STEPS
        stored_norm, bf16_fraction = 0.0, 0.0
        if applied:
            candidate = gate["candidate"]
            if logged:
                stored_norm = float((candidate.float() - p.float()).norm())
                bf16_fraction = float((candidate.bfloat16() != p.bfloat16()).float().mean())
            p.copy_(candidate)
            if staged_rms is not None:
                layer._loco_rms_avg.copy_(staged_rms[0])
                layer._loco_rms_mom.copy_(staged_rms[1])
        if logged:
            WR_LOCOM_APPLY_STATS.append(
                f"l{layer_idx}:base={float(base_step_norm):.3e},corr={float(corr_norm):.3e},scale={float(scale):.3e}"
                f",accepted_apply={int(applied)},stored_corr={stored_norm:.3e},bf16_changed={bf16_fraction:.6f}"
                f",gate_attempts={gate['attempts']},gate_backtracks={gate['backtracks']}"
                f",gate_final_checks={gate['final_checks']},gate_reason={gate['reason']}"
            )
    WR_LOCOM_CORR.clear()
    WR_LOCOM_LINEAR_GRADS.clear()
    if step in WR_LOCOM_LOG_STEPS and WR_LOCOM_APPLY_STATS:
        print0("wr_locom_apply step=" + str(step) + " " + " | ".join(WR_LOCOM_APPLY_STATS), console=True)
'''


MLP_BLOCK = r'''class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        global WR_LOCOM_NEXT_LAYER
        self.layer_idx = WR_LOCOM_NEXT_LAYER
        WR_LOCOM_NEXT_LAYER += 1
        self.wr_locom_layer_enabled = WR_LOCOM_ENABLED and _wr_locom_layer_active(self.layer_idx)
        self.wr_locom_capture_now = True
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        world = dist.get_world_size() if dist.is_initialized() else 1
        samples = loco_sample_buffer(WR_LOCOM_SAMPLE_TOKENS, dim, hdim, world) if self.wr_locom_layer_enabled else None
        self.register_buffer("_loco_samples", samples, persistent=False)
        rms_enabled = self.wr_locom_layer_enabled and WR_LOCOM_LOCAL_OPT == "rmsprop"
        rms_initial = (torch.ones_like(self.fc.weight, dtype=torch.float32) if WR_LOCOM_RMS_STYLE == "tensorflow"
                       else torch.zeros_like(self.fc.weight, dtype=torch.float32)) if rms_enabled else None
        self.register_buffer("_loco_rms_avg", rms_initial)
        self.register_buffer("_loco_rms_mom", torch.zeros_like(self.fc.weight, dtype=torch.float32) if rms_enabled else None)

    def forward(self, x: Tensor):
        if self.wr_locom_layer_enabled and self.wr_locom_capture_now:
            return LocoMLPFunction.apply(x, self.fc.weight, self.fc.bias,
                                        self.proj.weight, self.proj.bias,
                                        self._loco_samples, self.layer_idx)
        return self.proj(self.fc(x).relu().square())
'''


def _code_only(text: str) -> str:
    marker = "\n===================================================================================================="
    if marker in text:
        return text.split(marker, 1)[0].rstrip() + "\n"
    return text


def replace_exact(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found:\n{old[:500]}")
    return text.replace(old, new, 1)


def generate(source: Path, output: Path, train_steps: int, schedule_steps: int | None) -> None:
    text = _code_only(source.read_text())
    text = replace_exact(
        text,
        "import torch.distributed as dist\n",
        "import torch.distributed as dist\n" + SHARED_LOCAL_SOURCE + LOCOM_BLOCK + "\n",
    )
    text = replace_exact(
        text,
        "FINAL_TRAIN_STEPS = 3040\n",
        f"FINAL_TRAIN_STEPS = int(os.environ.get(\"FINAL_TRAIN_STEPS\", \"{train_steps}\"))\n",
    )
    text = replace_exact(
        text,
        "TRAIN_PROGRESS_INTERVAL = 0\n",
        'TRAIN_PROGRESS_INTERVAL = int(os.environ.get("TRAIN_PROGRESS_INTERVAL", "0"))\n'
        'if TRAIN_PROGRESS_INTERVAL < 0:\n'
        '    raise ValueError("TRAIN_PROGRESS_INTERVAL must be nonnegative")\n',
    )
    text = replace_exact(
        text,
        "val_regular_interval = 125\n",
        'val_regular_interval = int(os.environ.get("SCREEN_VAL_EVERY", "125"))\n'
        'if val_regular_interval <= 0:\n'
        '    raise ValueError("SCREEN_VAL_EVERY must be positive")\n',
    )
    if schedule_steps is not None:
        text = replace_exact(
            text,
            "FINAL_SCHEDULE_STEPS = 3105\n",
            f"FINAL_SCHEDULE_STEPS = int(os.environ.get(\"FINAL_SCHEDULE_STEPS\", \"{schedule_steps}\"))\n",
        )
    else:
        text = replace_exact(
            text,
            "FINAL_SCHEDULE_STEPS = 3105\n",
            "FINAL_SCHEDULE_STEPS = int(os.environ.get(\"FINAL_SCHEDULE_STEPS\", \"3105\"))\n",
        )
    old_mlp = '''class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x
'''
    text = replace_exact(text, old_mlp, MLP_BLOCK)
    text = replace_exact(
        text,
        "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\nmodel.compile(dynamic=False)\n",
        "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n"
        "if WR_LOCOM_ENABLED and not _wr_locom_flag(\"WR_LOCOM_COMPILE\", \"1\"):\n"
        "    print0(\"WR LocoProp-M: eager model requested\", console=True)\n"
        "else:\n"
        "    model.compile(dynamic=False)\n",
    )
    text = replace_exact(
        text,
        "print0(\"=\"*100)\n\nval_tokens = 20 * 524288\n",
        "print0(\n"
        "    f\"WR LocoProp-M generated run enabled={WR_LOCOM_ENABLED} layers={WR_LOCOM_LAYERS_SPEC} \"\n"
        "    f\"windows={WR_LOCOM_ACTIVE_WINDOWS or [(WR_LOCOM_START_STEP, WR_LOCOM_END_STEP)]} \"\n"
        "    f\"K={WR_LOCOM_STEPS} sample_tokens={WR_LOCOM_SAMPLE_TOKENS} inner_lr={WR_LOCOM_INNER_LR} \"\n"
        "    f\"prox={WR_LOCOM_PROX} alpha={WR_LOCOM_ALPHA} norm_cap={WR_LOCOM_NORM_CAP} \"\n"
        "    f\"norm_to_base={WR_LOCOM_NORM_TO_BASE} min_cos={WR_LOCOM_MIN_COS_DESC} \"\n"
        "    f\"local_opt={WR_LOCOM_LOCAL_OPT} rms_beta1={WR_LOCOM_RMS_BETA1} rms_beta2={WR_LOCOM_RMS_BETA2} \"\n"
        "    f\"rms_eps={WR_LOCOM_RMS_EPS} rms_style={WR_LOCOM_RMS_STYLE} reset_rms={WR_LOCOM_RESET_RMS} lr_decay={WR_LOCOM_LR_DECAY} center={WR_LOCOM_CENTER} scale_momentum={WR_LOCOM_SCALE_MOMENTUM} linear_term={WR_LOCOM_LINEAR_TERM}\",\n"
        "    console=True,\n"
        ")\n"
        "print0(\"=\"*100)\n\nval_tokens = 20 * 524288\n",
    )
    text = replace_exact(
        text,
        "for step in range(train_steps + 1):\n",
        "for step in range(train_steps + 1):\n    _wr_locom_begin_step(step, model=model)\n",
    )
    text = replace_exact(
        text,
        "    # set optimization hyperparameters and take a step\n    set_hparams(step)\n    for opt in optimizers:\n        opt.step()\n    model.zero_grad(set_to_none=True)\n",
        "    # set optimization hyperparameters and take a step\n"
        "    set_hparams(step)\n"
        "    prepare_wr_locom_m(model, step)\n"
        "    for opt in optimizers:\n"
        "        opt.step()\n"
        "    apply_wr_locom_m(model, optimizer2, step)\n"
        "    model.zero_grad(set_to_none=True)\n",
    )
    text = replace_exact(
        text,
        "    for i in range(len(inputs) // mbs):\n",
        "    _wr_locom_begin_step(step, len(inputs) // mbs, model=model)\n"
        "    for i in range(len(inputs) // mbs):\n"
        "        loco_set_microbatch(i)\n",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3040)
    parser.add_argument("--schedule-steps", type=int, default=None)
    args = parser.parse_args()
    generate(Path(args.source), Path(args.output), args.steps, args.schedule_steps)


if __name__ == "__main__":
    main()
