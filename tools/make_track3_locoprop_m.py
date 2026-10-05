#!/usr/bin/env python3
"""Generate a Track 3 script with a sampled LocoProp-M local correction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SHARED_LOCAL_SOURCE = Path(__file__).with_name("locoprop_local.py").read_text()


LOCOM_CONFIG = r'''
# Modal's CUDA image can route compiled SDPA through cuDNN plans that are
# unavailable for this shape. Keep the Track 3 model unchanged, but force
# PyTorch to use the flash/math SDPA backends instead.
torch.backends.cuda.enable_cudnn_sdp(False)

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def _parse_layer_set(spec: str, total_layers: int = 12) -> set[int]:
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return set(range(total_layers))
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

LOCO_M_ENABLED = _env_flag("TRACK3_LOCOM_ENABLED", "1")
LOCO_M_LAYERS_SPEC = os.environ.get("TRACK3_LOCOM_LAYERS", "all")
LOCO_M_LAYER_SET = _parse_layer_set(LOCO_M_LAYERS_SPEC)
LOCO_M_SAMPLE_TOKENS = int(os.environ.get("TRACK3_LOCOM_SAMPLE_TOKENS", "2048"))
LOCO_M_LOCAL_STEPS = int(os.environ.get("TRACK3_LOCOM_STEPS", "4"))
LOCO_M_INNER_LR = float(os.environ.get("TRACK3_LOCOM_INNER_LR", "0.1"))
LOCO_M_TARGET_GAMMA = float(os.environ.get("TRACK3_LOCOM_TARGET_GAMMA", "1.0"))
LOCO_M_PROX = float(os.environ.get("TRACK3_LOCOM_PROX", "0.1"))
LOCO_M_ALPHA = float(os.environ.get("TRACK3_LOCOM_ALPHA", "1.0"))
LOCO_M_NORM_TO_BASE = _env_flag("TRACK3_LOCOM_NORM_TO_BASE", "0")
LOCO_M_NORM_TARGET = float(os.environ.get("TRACK3_LOCOM_NORM_TARGET", "0.0"))
LOCO_M_NORM_CAP = float(os.environ.get("TRACK3_LOCOM_NORM_CAP", "0.20"))
LOCO_M_NORM_CAP_WINDOWS_SPEC = os.environ.get("TRACK3_LOCOM_NORM_CAP_WINDOWS", "")
LOCO_M_ACTIVE_WINDOWS_SPEC = os.environ.get("TRACK3_LOCOM_ACTIVE_WINDOWS", "")
LOCO_M_START_STEP = int(os.environ.get("TRACK3_LOCOM_START_STEP", "0"))
LOCO_M_END_STEP = int(os.environ.get("TRACK3_LOCOM_END_STEP", "1000000000"))
LOCO_M_INTERVAL = int(os.environ.get("TRACK3_LOCOM_INTERVAL", "1"))
LOCO_M_GATHER_SAMPLES = _env_flag("TRACK3_LOCOM_GATHER_SAMPLES", "1")
LOCO_M_ACCUM_SAMPLES = _env_flag("TRACK3_LOCOM_ACCUM_SAMPLES", "1")
LOCO_M_MICRO_SAMPLE_TOKENS = int(os.environ.get("TRACK3_LOCOM_MICRO_SAMPLE_TOKENS", "0"))
LOCO_M_LOCAL_OPT = os.environ.get("TRACK3_LOCOM_LOCAL_OPT", "sgd").lower()
LOCO_M_TARGET_SPACE = os.environ.get("TRACK3_LOCOM_TARGET_SPACE", "post").lower()
LOCO_M_RANDOM_CORRECTION = _env_flag("TRACK3_LOCOM_RANDOM_CORRECTION", "0") or LOCO_M_LOCAL_OPT == "random"
LOCO_M_LOCAL_LR_DECAY = _env_flag("TRACK3_LOCOM_LOCAL_LR_DECAY", "0")
LOCO_M_RMS_BETA1 = float(os.environ.get("TRACK3_LOCOM_RMS_BETA1", "0.999"))
LOCO_M_RMS_BETA2 = float(os.environ.get("TRACK3_LOCOM_RMS_BETA2", "0.9"))
LOCO_M_RMS_EPS = float(os.environ.get("TRACK3_LOCOM_RMS_EPS", "1e-5"))
LOCO_M_RMS_RESET_EACH_STEP = _env_flag("TRACK3_LOCOM_RMS_RESET_EACH_STEP", "0")
LOCO_M_REQUIRE_LOSS_DECREASE = _env_flag("TRACK3_LOCOM_REQUIRE_LOSS_DECREASE", "1")
LOCO_M_MIN_COS_DESC = float(os.environ.get("TRACK3_LOCOM_MIN_COS_DESC", "0.0"))
LOCO_M_MAX_BACKTRACKS = int(os.environ.get("TRACK3_LOCOM_MAX_BACKTRACKS", "20"))
TRACK3_TARGET_LOSS = float(os.environ.get("TRACK3_TARGET_LOSS", "0"))
TRACK3_SEED_BASE = int(os.environ.get("TRACK3_SEED_BASE", "0"))
TRACK3_SEED_OFFSET = int(os.environ.get("TRACK3_SEED_OFFSET", "0"))
TRACK3_RESET_TRIAL_SEED = _env_flag("TRACK3_RESET_TRIAL_SEED", "1")
TRACK3_COOLDOWN_FRAC = float(os.environ.get("TRACK3_COOLDOWN_FRAC", "0.7"))
TRACK3_LR_SCHEDULE = os.environ.get("TRACK3_LR_SCHEDULE", "linear").lower()
TRACK3_LR_POWER = float(os.environ.get("TRACK3_LR_POWER", "1.0"))
TRACK3_LR_SCHEDULE_STEPS = int(os.environ.get("TRACK3_LR_SCHEDULE_STEPS", "0"))
TRACK3_LR_MIN_ETA = float(os.environ.get("TRACK3_LR_MIN_ETA", "0.0"))
TRACK3_LR_SWITCH_STEP = int(os.environ.get("TRACK3_LR_SWITCH_STEP", "-1"))
TRACK3_LR_SWITCH_BLEND_STEPS = int(os.environ.get("TRACK3_LR_SWITCH_BLEND_STEPS", "100"))
if TRACK3_LR_SWITCH_BLEND_STEPS < 0:
    raise ValueError("TRACK3_LR_SWITCH_BLEND_STEPS must be nonnegative")
TRACK3_LR_AFTER_SWITCH = os.environ.get("TRACK3_LR_AFTER_SWITCH", "").lower()
TRACK3_LR_AFTER_SWITCH_POWER = float(os.environ.get("TRACK3_LR_AFTER_SWITCH_POWER", str(TRACK3_LR_POWER)))
TRACK3_LR_AFTER_SWITCH_STEPS = int(os.environ.get("TRACK3_LR_AFTER_SWITCH_STEPS", "0"))
TRACK3_LR_BLEND_START = int(os.environ.get("TRACK3_LR_BLEND_START", "-1"))
TRACK3_LR_BLEND_END = int(os.environ.get("TRACK3_LR_BLEND_END", "-1"))
TRACK3_LR_BLEND_TARGET = os.environ.get("TRACK3_LR_BLEND_TARGET", "").lower()
TRACK3_LR_BLEND_TARGET_POWER = float(os.environ.get("TRACK3_LR_BLEND_TARGET_POWER", str(TRACK3_LR_POWER)))
TRACK3_LR_BLEND_TARGET_STEPS = int(os.environ.get("TRACK3_LR_BLEND_TARGET_STEPS", "0"))
TRACK3_LR_BUMP_WINDOWS_SPEC = os.environ.get("TRACK3_LR_BUMP_WINDOWS", "")
TRACK3_ADAM_EMBED_POWER_C = float(os.environ.get("TRACK3_ADAM_EMBED_POWER_C", "4.976805410800738e-05"))
TRACK3_ADAM_PROJ_POWER_C = float(os.environ.get("TRACK3_ADAM_PROJ_POWER_C", "5.184172302917436e-07"))
TRACK3_ADAM_OTHER_POWER_C = float(os.environ.get("TRACK3_ADAM_OTHER_POWER_C", "1.6589351369335795e-06"))
TRACK3_MUON_POWER_C = float(os.environ.get("TRACK3_MUON_POWER_C", "3.3169534699576625e-06"))
if TRACK3_LR_SCHEDULE not in {"linear", "power", "pr287"}:
    raise ValueError("TRACK3_LR_SCHEDULE must be 'linear', 'power', or 'pr287'")
if TRACK3_LR_AFTER_SWITCH and TRACK3_LR_AFTER_SWITCH not in {"linear", "power", "pr287"}:
    raise ValueError("TRACK3_LR_AFTER_SWITCH must be empty, 'linear', 'power', or 'pr287'")
if TRACK3_LR_BLEND_TARGET and TRACK3_LR_BLEND_TARGET not in {"linear", "power", "pr287"}:
    raise ValueError("TRACK3_LR_BLEND_TARGET must be empty, 'linear', 'power', or 'pr287'")
if TRACK3_LR_BLEND_TARGET and TRACK3_LR_BLEND_END < TRACK3_LR_BLEND_START:
    raise ValueError("TRACK3_LR_BLEND_END must be >= TRACK3_LR_BLEND_START")
if LOCO_M_TARGET_SPACE not in {"post", "pre"}:
    raise ValueError("TRACK3_LOCOM_TARGET_SPACE must be 'post' or 'pre'")
TRACK3_SOFT_MUON = _env_flag("TRACK3_SOFT_MUON", "0")
TRACK3_SOFT_MUON_BLEND = float(os.environ.get("TRACK3_SOFT_MUON_BLEND", "1.0"))
TRACK3_SOFT_MUON_NORM_RESTORE = _env_flag("TRACK3_SOFT_MUON_NORM_RESTORE", "1")
TRACK3_SOFT_MUON_START_STEP = int(os.environ.get("TRACK3_SOFT_MUON_START_STEP", "-1"))
TRACK3_SOFT_MUON_END_STEP = int(os.environ.get("TRACK3_SOFT_MUON_END_STEP", "-1"))
TRACK3_SOFT_MUON_CEIL = float(os.environ.get("TRACK3_SOFT_MUON_CEIL", "1.0"))
TRACK3_CHECKPOINT_STEPS = {
    int(x)
    for x in os.environ.get("TRACK3_CHECKPOINT_STEPS", "").split(",")
    if x.strip()
}
TRACK3_CHECKPOINT_DIR = os.environ.get("TRACK3_CHECKPOINT_DIR", "")
TRACK3_CHECKPOINT_PREFIX = os.environ.get("TRACK3_CHECKPOINT_PREFIX", "track3_locom")
TRACK3_CHECKPOINT_EXIT_AFTER = _env_flag("TRACK3_CHECKPOINT_EXIT_AFTER", "0")
TRACK3_RESUME_CHECKPOINT = os.environ.get("TRACK3_RESUME_CHECKPOINT", "")
TRACK3_RESUME_ADVANCE_DATA = _env_flag("TRACK3_RESUME_ADVANCE_DATA", "1")
TRACK3_RESUME_RESTORE_RNG = _env_flag("TRACK3_RESUME_RESTORE_RNG", "1")
TRACK3_RESUME_LOAD_OPTIMIZERS = _env_flag("TRACK3_RESUME_LOAD_OPTIMIZERS", "1")
TRACK3_CHECKPOINT_SAVED_STEPS: set[int] = set()
LOCO_M_LOG_STEPS = {
    int(x)
    for x in os.environ.get("TRACK3_LOCOM_LOG_STEPS", "0,1,2,10,50,125,250,500").split(",")
    if x.strip()
}
LOCO_M_OWNED_LAYER_SET: set[int] = set()
LOCO_M_APPLY_STATS: list[str] = []
LOCO_M_CURRENT_STEP = -1

def _parse_step_value_windows(spec: str) -> list[tuple[int, int, float]]:
    windows: list[tuple[int, int, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError("TRACK3_LOCOM_NORM_CAP_WINDOWS entries must be start:end:value")
        start, end, value = int(fields[0]), int(fields[1]), float(fields[2])
        if end < start:
            raise ValueError("TRACK3_LOCOM_NORM_CAP_WINDOWS end must be >= start")
        windows.append((start, end, value))
    return windows

def _parse_step_windows(spec: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 2:
            raise ValueError("TRACK3_LOCOM_ACTIVE_WINDOWS entries must be start:end")
        start, end = int(fields[0]), int(fields[1])
        if end < start:
            raise ValueError("TRACK3_LOCOM_ACTIVE_WINDOWS end must be >= start")
        windows.append((start, end))
    return windows

LOCO_M_NORM_CAP_WINDOWS = _parse_step_value_windows(LOCO_M_NORM_CAP_WINDOWS_SPEC)
LOCO_M_ACTIVE_WINDOWS = _parse_step_windows(LOCO_M_ACTIVE_WINDOWS_SPEC)

def _parse_lr_bump_windows(spec: str) -> list[tuple[int, int, int, int, float]]:
    windows: list[tuple[int, int, int, int, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 5:
            raise ValueError("TRACK3_LR_BUMP_WINDOWS entries must be start:ramp_end:hold_end:fade_end:mult")
        start, ramp_end, hold_end, fade_end = map(int, fields[:4])
        mult = float(fields[4])
        if not (start <= ramp_end <= hold_end <= fade_end):
            raise ValueError("TRACK3_LR_BUMP_WINDOWS requires start <= ramp_end <= hold_end <= fade_end")
        windows.append((start, ramp_end, hold_end, fade_end, mult))
    return windows

TRACK3_LR_BUMP_WINDOWS = _parse_lr_bump_windows(TRACK3_LR_BUMP_WINDOWS_SPEC)

def _track3_lr_multiplier(step: int) -> float:
    multiplier = 1.0
    for start, ramp_end, hold_end, fade_end, mult in TRACK3_LR_BUMP_WINDOWS:
        if step < start or step > fade_end:
            continue
        if ramp_end <= start or step >= ramp_end:
            if step <= hold_end:
                t = 1.0
            elif fade_end <= hold_end:
                t = 0.0
            else:
                t = max(0.0, (fade_end - step) / (fade_end - hold_end))
        else:
            t = (step - start) / (ramp_end - start)
        multiplier *= 1.0 + (mult - 1.0) * t
    return multiplier

def _track3_soft_muon_blend_for_step(step: int) -> float:
    if not TRACK3_SOFT_MUON:
        return 0.0
    if TRACK3_SOFT_MUON_START_STEP < 0:
        return min(TRACK3_SOFT_MUON_CEIL, TRACK3_SOFT_MUON_BLEND)
    if step < TRACK3_SOFT_MUON_START_STEP:
        return 0.0
    if TRACK3_SOFT_MUON_END_STEP <= TRACK3_SOFT_MUON_START_STEP:
        ramp = 1.0
    else:
        ramp = (step - TRACK3_SOFT_MUON_START_STEP) / (TRACK3_SOFT_MUON_END_STEP - TRACK3_SOFT_MUON_START_STEP)
        ramp = max(0.0, min(1.0, ramp))
    return min(TRACK3_SOFT_MUON_CEIL, TRACK3_SOFT_MUON_BLEND * ramp)

def _locom_current_norm_cap(step: int) -> float:
    cap = LOCO_M_NORM_CAP
    for start, end, value in LOCO_M_NORM_CAP_WINDOWS:
        if start <= step <= end:
            cap = value
    return cap
'''


LOCOM_MLP = r'''class MLP(nn.Module):
    def __init__(self, dim: int, layer_idx: int):
        super().__init__()
        hdim = 4 * dim
        self.layer_idx = layer_idx
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.register_buffer("fc_xtx", torch.zeros(dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("fc_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("proj_xtx", torch.zeros(4, dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("proj_count", torch.zeros((), dtype=torch.float32), persistent=False)
        world = dist.get_world_size() if dist.is_initialized() and LOCO_M_GATHER_SAMPLES else 1
        rows = LOCO_M_SAMPLE_TOKENS if LOCO_M_ENABLED and layer_idx in LOCO_M_LAYER_SET else 1
        self.register_buffer("_loco_samples", loco_sample_buffer(rows, dim, hdim, world), persistent=False)
        self.register_buffer("_loco_rms_avg", torch.zeros(hdim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("_loco_rms_mom", torch.zeros(hdim, dim, dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor):
        if LOCO_M_ENABLED and not LOCO_M_RANDOM_CORRECTION and self.layer_idx in LOCO_M_LAYER_SET:
            if self.fc._forward_hooks or self.proj._forward_hooks:
                pre = self.fc(x)
                post = LocoReLUSquareFunction.apply(x, pre, self._loco_samples, self.layer_idx)
                return self.proj(post)
            return LocoMLPFunction.apply(x, self.fc.weight, self.fc.bias,
                                        self.proj.weight, self.proj.bias,
                                        self._loco_samples, self.layer_idx)
        return self.proj(self.fc(x).relu().square())
'''


LOCOM_HELPERS = r'''
@torch.no_grad()
def _locom_active(step: int) -> bool:
    if not (LOCO_M_ENABLED and step % max(LOCO_M_INTERVAL, 1) == 0):
        return False
    if not LOCO_M_RANDOM_CORRECTION and LOCO_M_LOCAL_STEPS <= 0:
        return False
    if LOCO_M_ACTIVE_WINDOWS:
        return any(start <= step < end for start, end in LOCO_M_ACTIVE_WINDOWS)
    return step >= LOCO_M_START_STEP and step < LOCO_M_END_STEP

@torch.no_grad()
def set_locoprop_m_current_step(step: int):
    global LOCO_M_CURRENT_STEP
    LOCO_M_CURRENT_STEP = step

@torch.no_grad()
def attach_locoprop_m_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    global LOCO_M_OWNED_LAYER_SET
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    fc_to_layer = {
        id(model.blocks[i].mlp.fc.weight): i
        for i in range(len(model.blocks))
        if i in LOCO_M_LAYER_SET
    }
    owned = set()
    for group in optimizer.param_groups:
        for idx, p in enumerate(group["params"]):
            layer_idx = fc_to_layer.get(id(p))
            if layer_idx is not None and idx % world == rank:
                owned.add(layer_idx)
    LOCO_M_OWNED_LAYER_SET = owned
    print0(
        f"locoprop_m_owner rank={rank} world={world} owned_layers={sorted(owned)} "
        f"sample_tokens={LOCO_M_SAMPLE_TOKENS} gather_samples={LOCO_M_GATHER_SAMPLES} "
        f"accum_samples={LOCO_M_ACCUM_SAMPLES} micro_sample_tokens={LOCO_M_MICRO_SAMPLE_TOKENS}",
        console=True,
    )

@torch.no_grad()
def _locom_gather_sample(t: Tensor) -> Tensor:
    if not LOCO_M_GATHER_SAMPLES or not dist.is_initialized() or dist.get_world_size() == 1:
        return t.float()
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0).float()

@torch.no_grad()
def prepare_locoprop_m(model: nn.Module, step: int):
    if not _locom_active(step):
        return
    stats = []
    if LOCO_M_RANDOM_CORRECTION:
        for layer_idx in sorted(LOCO_M_LAYER_SET):
            if layer_idx < 0 or layer_idx >= len(model.blocks):
                continue
            if layer_idx not in LOCO_M_OWNED_LAYER_SET:
                continue
            mlp = model.blocks[layer_idx].mlp
            if mlp.fc.weight.grad is None:
                continue
            corr_f = torch.randn(
                mlp.fc.weight.shape,
                device=mlp.fc.weight.device,
                dtype=torch.float32,
            )
            raw_grad = mlp.fc.weight.grad.float()
            raw_desc = -raw_grad
            denom = corr_f.norm().mul(raw_desc.norm()).clamp_min(1e-12)
            cosine = corr_f.flatten().dot(raw_desc.flatten()) / denom
            corr = corr_f.to(mlp.fc.weight.dtype)
            mlp._loco_corr = corr
            mlp.fc.weight._loco_corr = corr
            mlp.fc.weight._loco_step = step
            if step in LOCO_M_LOG_STEPS and len(stats) < 4:
                stats.append(
                    f"l{layer_idx}:random=1"
                    f",corr_norm={float(corr_f.norm()):.3e}"
                    f",grad_norm={float(raw_grad.norm()):.3e}"
                    f",cos_desc={float(cosine):.3f}"
                    f",accepted=1"
                    f",opt={LOCO_M_LOCAL_OPT}"
                )
        if step in LOCO_M_LOG_STEPS and stats:
            print0("locoprop_m_prepare step=" + str(step) + " " + " | ".join(stats), console=True)
        return

    for layer_idx in sorted(LOCO_M_LAYER_SET):
        if layer_idx < 0 or layer_idx >= len(model.blocks):
            continue
        mlp = model.blocks[layer_idx].mlp
        x_local, pre_local, dpre_local = loco_samples(mlp._loco_samples, layer_idx, mlp.fc.weight.shape[1])
        if x_local.shape[0] == 0 or mlp.fc.weight.grad is None:
            continue
        # All ranks gather the same layers, including those owned elsewhere.
        x = _locom_gather_sample(x_local)
        pre0 = _locom_gather_sample(pre_local)
        dpre = _locom_gather_sample(dpre_local)
        if layer_idx not in LOCO_M_OWNED_LAYER_SET:
            continue
        corr, diag, rms_state = loco_solve(
            x, pre0, dpre, mlp.fc.weight.grad,
            steps=LOCO_M_LOCAL_STEPS, inner_lr=LOCO_M_INNER_LR,
            gamma=LOCO_M_TARGET_GAMMA, prox=LOCO_M_PROX,
            target_space=LOCO_M_TARGET_SPACE, min_cos=LOCO_M_MIN_COS_DESC,
            require_decrease=LOCO_M_REQUIRE_LOSS_DECREASE,
            max_backtracks=LOCO_M_MAX_BACKTRACKS, output_dtype=mlp.fc.weight.dtype,
            local_opt=LOCO_M_LOCAL_OPT, rms_avg=mlp._loco_rms_avg,
            rms_mom=mlp._loco_rms_mom, beta1=LOCO_M_RMS_BETA1,
            beta2=LOCO_M_RMS_BETA2, rms_eps=LOCO_M_RMS_EPS,
            reset_rms=LOCO_M_RMS_RESET_EACH_STEP, lr_decay=LOCO_M_LOCAL_LR_DECAY,
        )
        mlp.fc.weight._loco_corr = corr if diag["accepted"] else None
        mlp.fc.weight._loco_step = step
        mlp.fc.weight._loco_apply_context = (
            mlp.fc.weight.detach().float().clone(), x, pre0, dpre, rms_state, mlp,
        ) if diag["accepted"] else None
        if step in LOCO_M_LOG_STEPS and len(stats) < 4:
            stats.append(
                f"l{layer_idx}:loss0={diag['loss0']:.3e},lossK={diag['lossK']:.3e}"
                f",corr_norm={float(corr.float().norm()):.3e}"
                f",grad_norm={float(mlp.fc.weight.grad.float().norm()):.3e}"
                f",cos_desc={diag['cos_desc']:.3f},accepted={int(diag['accepted'])}"
                f",tokens={x.size(0)},opt={LOCO_M_LOCAL_OPT}"
                f",backtracks={diag['backtracks']},local_steps={diag['local_steps']}"
                f",inner_lr={diag['inner_lr']:.3e},reason={diag['reason']}"
            )

    if step in LOCO_M_LOG_STEPS and stats:
        print0("locoprop_m_prepare step=" + str(step) + " " + " | ".join(stats), console=True)

@torch.no_grad()
def _locom_apply_owned_param_(p: nn.Parameter, update: Tensor, lr: float):
    corr = getattr(p, "_loco_corr", None)
    if corr is None:
        return
    step = getattr(p, "_loco_step", -1)
    base_step_norm = update.float().norm().mul(lr)
    corr_f = corr.float()
    corr_norm = corr_f.norm().clamp_min(1e-12)
    norm_cap = _locom_current_norm_cap(step)
    scale = loco_correction_scale(corr, base_step_norm, alpha=LOCO_M_ALPHA,
                                 cap=norm_cap, norm_target=LOCO_M_NORM_TARGET,
                                 norm_to_base=LOCO_M_NORM_TO_BASE)
    context = getattr(p, "_loco_apply_context", None)
    if context is not None:
        reference, x, pre0, dpre, rms_state, mlp = context
        scale = loco_post_step_scale(corr, scale, p.float() - reference, x, pre0, dpre,
                                    gamma=LOCO_M_TARGET_GAMMA, prox=LOCO_M_PROX,
                                    target_space=LOCO_M_TARGET_SPACE,
                                    max_backtracks=LOCO_M_MAX_BACKTRACKS)
        if float(scale) > 0 and rms_state is not None:
            mlp._loco_rms_avg.copy_(rms_state[0])
            mlp._loco_rms_mom.copy_(rms_state[1])
    if float(scale) > 0:
        p.add_(corr, alpha=float(scale))
    p._loco_apply_context = None
    if step in LOCO_M_LOG_STEPS and len(LOCO_M_APPLY_STATS) < 8:
        LOCO_M_APPLY_STATS.append(
            f"shape={tuple(p.shape)}:base_step={float(base_step_norm):.3e}"
            f",corr_norm={float(corr_norm):.3e},cap={norm_cap:.3f},scale={float(scale):.3e},accepted_apply={int(float(scale) > 0)}"
        )
    p._loco_corr = None

@torch.no_grad()
def flush_locoprop_m_apply_stats(step: int):
    if step in LOCO_M_LOG_STEPS and LOCO_M_APPLY_STATS:
        print0("locoprop_m_apply step=" + str(step) + " " + " | ".join(LOCO_M_APPLY_STATS[:8]), console=True)
    LOCO_M_APPLY_STATS.clear()

@torch.no_grad()
def maybe_save_track3_checkpoint(model: nn.Module, optimizers: list[torch.optim.Optimizer], step: int, train_steps: int, trial_idx: int, val_loss: Tensor | None):
    if step not in TRACK3_CHECKPOINT_STEPS:
        return False
    if step in TRACK3_CHECKPOINT_SAVED_STEPS:
        return False
    if not TRACK3_CHECKPOINT_DIR:
        raise ValueError("TRACK3_CHECKPOINT_STEPS is set but TRACK3_CHECKPOINT_DIR is empty")
    TRACK3_CHECKPOINT_SAVED_STEPS.add(step)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank_state = {
        "optimizers": torch.utils._pytree.tree_map(
            lambda value: value.detach().cpu() if isinstance(value, Tensor) else value,
            [opt.state_dict() for opt in optimizers],
        ),
        "optimizer_step_counts": [getattr(opt, "step_count", None) for opt in optimizers],
        "optimizer_runtime": [
            {name: getattr(opt, name) for name in ("global_step", "_precond_has_inv") if hasattr(opt, name)}
            for opt in optimizers
        ],
        "loco_local_state": {
            i: {"avg": block.mlp._loco_rms_avg.cpu(), "mom": block.mlp._loco_rms_mom.cpu()}
            for i, block in enumerate(model.blocks)
            if world == 1 or i in LOCO_M_OWNED_LAYER_SET
        } if LOCO_M_LOCAL_OPT == "rmsprop" else {},
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }
    rank_states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(rank_state, rank_states, dst=0)
    else:
        rank_states[0] = rank_state
    if rank == 0:
        checkpoint_dir = Path(TRACK3_CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        seed = TRACK3_SEED_BASE + TRACK3_SEED_OFFSET + trial_idx
        if not TRACK3_RESET_TRIAL_SEED:
            seed = globals().get("SEED", seed)
        path = checkpoint_dir / f"{TRACK3_CHECKPOINT_PREFIX}_seed{seed}_step{step}.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "step": step,
            "train_steps": train_steps,
            "trial_idx": trial_idx,
            "seed": seed,
            "val_loss": None if val_loss is None else float(val_loss),
            "model": model.state_dict(),
            # Keep the legacy keys for model-only handoff tools. Exact resumes
            # select the optimizer, local RMS and RNG state for their own rank.
            **rank_states[0],
            "rank_states": rank_states,
            "world_size": world,
            "config": {
                "source": "track3_locom_generated",
                "loco_m_enabled": LOCO_M_ENABLED,
                "loco_m_layers": LOCO_M_LAYERS_SPEC,
                "loco_m_steps": LOCO_M_LOCAL_STEPS,
                "loco_m_sample_tokens": LOCO_M_SAMPLE_TOKENS,
                "loco_m_inner_lr": LOCO_M_INNER_LR,
                "loco_m_target_gamma": LOCO_M_TARGET_GAMMA,
                "loco_m_prox": LOCO_M_PROX,
                "loco_m_alpha": LOCO_M_ALPHA,
                "loco_m_norm_to_base": LOCO_M_NORM_TO_BASE,
                "loco_m_norm_target": LOCO_M_NORM_TARGET,
                "loco_m_norm_cap": LOCO_M_NORM_CAP,
                "loco_m_norm_cap_windows": LOCO_M_NORM_CAP_WINDOWS_SPEC,
                "loco_m_active_windows": LOCO_M_ACTIVE_WINDOWS_SPEC,
                "loco_m_start_step": LOCO_M_START_STEP,
                "loco_m_end_step": LOCO_M_END_STEP,
                "loco_m_interval": LOCO_M_INTERVAL,
                "loco_m_gather_samples": LOCO_M_GATHER_SAMPLES,
                "loco_m_accum_samples": LOCO_M_ACCUM_SAMPLES,
                "loco_m_local_opt": LOCO_M_LOCAL_OPT,
                "loco_m_target_space": LOCO_M_TARGET_SPACE,
                "loco_m_random_correction": LOCO_M_RANDOM_CORRECTION,
                "loco_m_local_lr_decay": LOCO_M_LOCAL_LR_DECAY,
                "loco_m_rms_beta1": LOCO_M_RMS_BETA1,
                "loco_m_rms_beta2": LOCO_M_RMS_BETA2,
                "loco_m_rms_eps": LOCO_M_RMS_EPS,
                "loco_m_rms_reset_each_step": LOCO_M_RMS_RESET_EACH_STEP,
                "loco_m_require_loss_decrease": LOCO_M_REQUIRE_LOSS_DECREASE,
                "loco_m_min_cos_desc": LOCO_M_MIN_COS_DESC,
                "loco_m_max_backtracks": LOCO_M_MAX_BACKTRACKS,
                "track3_lr_switch_blend_steps": TRACK3_LR_SWITCH_BLEND_STEPS,
                "track3_cooldown_frac": TRACK3_COOLDOWN_FRAC,
                "track3_lr_schedule": TRACK3_LR_SCHEDULE,
                "track3_lr_power": TRACK3_LR_POWER,
                "track3_lr_schedule_steps": TRACK3_LR_SCHEDULE_STEPS,
                "track3_lr_min_eta": TRACK3_LR_MIN_ETA,
                "track3_lr_bump_windows": TRACK3_LR_BUMP_WINDOWS_SPEC,
                "track3_lr_blend_start": TRACK3_LR_BLEND_START,
                "track3_lr_blend_end": TRACK3_LR_BLEND_END,
                "track3_lr_blend_target": TRACK3_LR_BLEND_TARGET,
                "track3_lr_blend_target_power": TRACK3_LR_BLEND_TARGET_POWER,
                "track3_lr_blend_target_steps": TRACK3_LR_BLEND_TARGET_STEPS,
                "track3_reset_trial_seed": TRACK3_RESET_TRIAL_SEED,
            },
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        print0(f"track3_checkpoint_saved step:{step} path:{path} val_loss:{payload['val_loss']}", console=True)
    if dist.is_initialized():
        dist.barrier()
    return TRACK3_CHECKPOINT_EXIT_AFTER

@torch.no_grad()
def maybe_load_track3_checkpoint(model: nn.Module, optimizers: list[torch.optim.Optimizer]) -> int:
    if not TRACK3_RESUME_CHECKPOINT:
        return 0
    # Loading all ranks' optimizer state onto one GPU would defeat sharding.
    checkpoint = torch.load(TRACK3_RESUME_CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    rank_state = checkpoint
    if "rank_states" in checkpoint:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        if (TRACK3_RESUME_LOAD_OPTIMIZERS or TRACK3_RESUME_RESTORE_RNG) and world != checkpoint["world_size"]:
            raise ValueError("exact checkpoint resume requires the saved world size; disable optimizer/RNG restore for a model-only handoff")
        rank_state = checkpoint["rank_states"][rank] if rank < checkpoint["world_size"] else checkpoint
    if TRACK3_RESUME_LOAD_OPTIMIZERS:
        saved_optimizers = rank_state.get("optimizers", [])
        if len(saved_optimizers) != len(optimizers):
            raise ValueError(
                f"checkpoint has {len(saved_optimizers)} optimizer states, expected {len(optimizers)}"
            )
        for optimizer, optimizer_state in zip(optimizers, saved_optimizers):
            optimizer.load_state_dict(optimizer_state)
        for index, optimizer in enumerate(optimizers):
            if hasattr(optimizer, "step_count"):
                counts = rank_state.get("optimizer_step_counts", [])
                count = counts[index] if index < len(counts) else None
                optimizer.step_count = int(checkpoint["step"] if count is None else count)
            runtime_states = rank_state.get("optimizer_runtime", [])
            runtime = runtime_states[index] if index < len(runtime_states) else {}
            if hasattr(optimizer, "global_step"):
                optimizer.global_step = runtime.get("global_step", int(checkpoint["step"]))
            if getattr(optimizer, "_precond_attached", False):
                # load_state_dict replaces dictionaries/tensors and breaks the
                # Newton-Muon inverse views used by future covariance refreshes.
                refs = {id(stat["accum"]): stat for stat in optimizer._precond_stats}
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        saved_stat = optimizer.state[p]["precond"]
                        stat = refs[id(p._stats_ref["accum"])]
                        for key in ("accum", "count", "cov", "inv"):
                            stat[key].copy_(saved_stat[key])
                        optimizer.state[p]["precond"] = stat
                optimizer._precond_has_inv = runtime.get(
                    "_precond_has_inv", int(checkpoint["step"]) >= optimizer.refresh_interval,
                )
    if TRACK3_RESUME_LOAD_OPTIMIZERS and LOCO_M_LOCAL_OPT == "rmsprop":
        for layer_idx, state in rank_state.get("loco_local_state", {}).items():
            mlp = model.blocks[int(layer_idx)].mlp
            mlp._loco_rms_avg.copy_(state["avg"])
            mlp._loco_rms_mom.copy_(state["mom"])
    if TRACK3_LR_SCHEDULE == "pr287" or TRACK3_LR_AFTER_SWITCH == "pr287" or TRACK3_LR_BLEND_TARGET == "pr287":
        optimizers[0].param_groups[0]["power_c"] = TRACK3_ADAM_EMBED_POWER_C
        optimizers[0].param_groups[1]["power_c"] = TRACK3_ADAM_PROJ_POWER_C
        optimizers[0].param_groups[2]["power_c"] = TRACK3_ADAM_OTHER_POWER_C
        optimizers[1].param_groups[0]["power_c"] = TRACK3_MUON_POWER_C
    if TRACK3_RESUME_RESTORE_RNG:
        if "rng_cpu" in rank_state:
            torch.set_rng_state(rank_state["rng_cpu"].detach().cpu().to(torch.uint8))
        if "rng_cuda" in rank_state:
            torch.cuda.set_rng_state_all([
                state.detach().cpu().to(torch.uint8)
                for state in rank_state["rng_cuda"]
            ])
    step = int(checkpoint["step"])
    print0(
        f"track3_checkpoint_loaded path:{TRACK3_RESUME_CHECKPOINT} step:{step} "
        f"seed:{checkpoint.get('seed')} val_loss:{checkpoint.get('val_loss')} "
        f"load_optimizers:{TRACK3_RESUME_LOAD_OPTIMIZERS}",
        console=True,
    )
    return step
'''


def replace_exact(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found:\n{old[:240]}")
    return text.replace(old, new, 1)


def generate(source: Path, output: Path, train_steps: int) -> None:
    source_label = str(source)
    text = source.read_text()
    if source.suffix == ".txt" and "\n====================================================================================================\n" in text:
        text = text.split("\n====================================================================================================\n", 1)[0]
    text = replace_exact(
        text,
        "import torch.distributed as dist\n",
        "import torch.distributed as dist\n" + SHARED_LOCAL_SOURCE + LOCOM_CONFIG,
    )
    plain_mlp = """class MLP(nn.Module):
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
"""
    nm_mlp = """class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.register_buffer("fc_xtx", torch.zeros(dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("fc_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("proj_xtx", torch.zeros(4, dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("proj_count", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x
"""
    if plain_mlp in text:
        text = text.replace(plain_mlp, LOCOM_MLP, 1)
    elif nm_mlp in text:
        text = text.replace(nm_mlp, LOCOM_MLP, 1)
    else:
        raise RuntimeError("MLP class pattern not found")
    text = replace_exact(
        text,
        """class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
""",
        """class Block(nn.Module):
    def __init__(self, dim: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim, layer_idx)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
""",
    )
    text = replace_exact(
        text,
        "        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])\n",
        "        self.blocks = nn.ModuleList([Block(model_dim, i) for i in range(num_layers)])\n",
    )
    simple_muon_update = """@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update
"""
    if simple_muon_update in text:
        text = text.replace(
            simple_muon_update,
            """def _track3_gram_frobenius_norm_estimate(x: torch.Tensor, keepdim: bool = False, eps: float = 1e-7):
    gram = x.mT @ x if x.size(-2) > x.size(-1) else x @ x.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)

def _track3_soft_muon_pr291_from_operand(g: torch.Tensor):
    x = g.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / _track3_gram_frobenius_norm_estimate(x, keepdim=True, eps=1e-7).to(x.dtype)
    coeffs = (
        0.1091613623,
        0.07085664498,
        0.05210528973,
        0.05457295795,
        0.05011334061,
        0.03334622198,
        0.05022104481,
        0.1053727358,
        0.1187323776,
        0.1185061091,
        0.1185059576,
        0.1185059576,
    )
    a, b, c = 2.0, -1.5, 0.5
    out = torch.zeros_like(x)
    for coeff in coeffs:
        out = out + coeff * x
        gram = x @ x.mT
        basis = b * gram + c * (gram @ gram)
        x = a * x + basis @ x
    if transposed:
        out = out.mT
    return out

@torch.compile
def _track3_muon_update_soft(grad, momentum, soft_blend, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    operand = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(operand)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    soft_update = _track3_soft_muon_pr291_from_operand(operand)
    soft_update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    if TRACK3_SOFT_MUON_NORM_RESTORE:
        soft_update = soft_update * update.float().norm().div(soft_update.float().norm().clamp_min(1e-12))
    update = update + soft_blend.to(update.dtype) * (soft_update - update)
    return update

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update
""" + LOCOM_HELPERS,
            1,
        )
    else:
        text = replace_exact(
            text,
            "\nclass Muon(torch.optim.Optimizer):\n",
            "\n" + LOCOM_HELPERS + "\nclass Muon(torch.optim.Optimizer):\n",
        )
    if "                    update = muon_update(p.grad, state[\"momentum\"], mu=group[\"mu\"])\n" in text:
        text = replace_exact(
            text,
            "                    update = muon_update(p.grad, state[\"momentum\"], mu=group[\"mu\"])\n",
            "                    soft_blend = _track3_soft_muon_blend_for_step(LOCO_M_CURRENT_STEP)\n"
            "                    if soft_blend > 0.0:\n"
            "                        update = _track3_muon_update_soft(p.grad, state[\"momentum\"], p.grad.new_tensor(soft_blend), mu=group[\"mu\"])\n"
            "                    else:\n"
            "                        update = muon_update(p.grad, state[\"momentum\"], mu=group[\"mu\"])\n"
            "                    state[\"last_update_norm\"] = update.float().norm()\n",
        )
    if (
        "                    p.mul_(1 - group[\"lr\"] * group[\"weight_decay\"])\n"
        "                    p.add_(update, alpha=-group[\"lr\"])\n"
    ) in text:
        text = replace_exact(
            text,
            "                    p.mul_(1 - group[\"lr\"] * group[\"weight_decay\"])\n"
            "                    p.add_(update, alpha=-group[\"lr\"])\n",
            "                    p.mul_(1 - group[\"lr\"] * group[\"weight_decay\"])\n"
            "                    p.add_(update, alpha=-group[\"lr\"])\n"
            "                    _locom_apply_owned_param_(p, update, group[\"lr\"])\n",
        )
    else:
        text = replace_exact(
            text,
            "                    p.add_(update, alpha=-group[\"lr\"])\n",
            "                    p.add_(update, alpha=-group[\"lr\"])\n"
            "                    _locom_apply_owned_param_(p, update, group[\"lr\"])\n",
        )
    if "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\nattach_precond_stats(model)\nmodel.compile(dynamic=False)\n" in text:
        text = replace_exact(
            text,
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\nattach_precond_stats(model)\nmodel.compile(dynamic=False)\n",
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n"
            "attach_precond_stats(model)\n"
            "if not LOCO_M_ENABLED or _env_flag(\"TRACK3_LOCOM_COMPILE\", \"1\"):\n"
            "    model.compile(dynamic=False)\n"
            "compiled_model = model\n",
        )
    else:
        text = replace_exact(
            text,
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\nmodel.compile(dynamic=False)\n",
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n"
            "if not LOCO_M_ENABLED or _env_flag(\"TRACK3_LOCOM_COMPILE\", \"1\"):\n"
            "    model.compile(dynamic=False)\n"
            "compiled_model = model\n",
        )
    text = replace_exact(
        text,
        'print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"',
        f'print0("Track3 LocoProp-M generated run: source={source_label} train_steps={train_steps}")\n'
        'print0(f"LocoM enabled={LOCO_M_ENABLED} layers={LOCO_M_LAYERS_SPEC} steps={LOCO_M_LOCAL_STEPS} sample_tokens={LOCO_M_SAMPLE_TOKENS} gather={LOCO_M_GATHER_SAMPLES} accum={LOCO_M_ACCUM_SAMPLES} micro_sample_tokens={LOCO_M_MICRO_SAMPLE_TOKENS} local_opt={LOCO_M_LOCAL_OPT} target_space={LOCO_M_TARGET_SPACE} random_correction={LOCO_M_RANDOM_CORRECTION} lr_decay={LOCO_M_LOCAL_LR_DECAY} rms_beta1={LOCO_M_RMS_BETA1} rms_beta2={LOCO_M_RMS_BETA2} rms_eps={LOCO_M_RMS_EPS} rms_reset_each_step={LOCO_M_RMS_RESET_EACH_STEP} require_loss_decrease={LOCO_M_REQUIRE_LOSS_DECREASE} min_cos_desc={LOCO_M_MIN_COS_DESC} inner_lr={LOCO_M_INNER_LR} target_gamma={LOCO_M_TARGET_GAMMA} prox={LOCO_M_PROX} alpha={LOCO_M_ALPHA} norm_to_base={LOCO_M_NORM_TO_BASE} norm_target={LOCO_M_NORM_TARGET} norm_cap={LOCO_M_NORM_CAP} norm_cap_windows={LOCO_M_NORM_CAP_WINDOWS_SPEC} active_windows={LOCO_M_ACTIVE_WINDOWS_SPEC} start_step={LOCO_M_START_STEP} end_step={LOCO_M_END_STEP} interval={LOCO_M_INTERVAL} target_loss={TRACK3_TARGET_LOSS} seed_base={TRACK3_SEED_BASE} seed_offset={TRACK3_SEED_OFFSET} cooldown_frac={TRACK3_COOLDOWN_FRAC} lr_schedule={TRACK3_LR_SCHEDULE} lr_power={TRACK3_LR_POWER} lr_schedule_steps={TRACK3_LR_SCHEDULE_STEPS} lr_min_eta={TRACK3_LR_MIN_ETA} lr_bump_windows={TRACK3_LR_BUMP_WINDOWS_SPEC} lr_switch_step={TRACK3_LR_SWITCH_STEP} lr_after_switch={TRACK3_LR_AFTER_SWITCH} lr_after_switch_power={TRACK3_LR_AFTER_SWITCH_POWER} lr_after_switch_steps={TRACK3_LR_AFTER_SWITCH_STEPS} lr_blend_start={TRACK3_LR_BLEND_START} lr_blend_end={TRACK3_LR_BLEND_END} lr_blend_target={TRACK3_LR_BLEND_TARGET} lr_blend_target_power={TRACK3_LR_BLEND_TARGET_POWER} lr_blend_target_steps={TRACK3_LR_BLEND_TARGET_STEPS} soft_muon={TRACK3_SOFT_MUON} soft_blend={TRACK3_SOFT_MUON_BLEND} soft_start={TRACK3_SOFT_MUON_START_STEP} soft_end={TRACK3_SOFT_MUON_END_STEP} soft_ceil={TRACK3_SOFT_MUON_CEIL} soft_norm_restore={TRACK3_SOFT_MUON_NORM_RESTORE} resume_checkpoint={TRACK3_RESUME_CHECKPOINT} resume_load_optimizers={TRACK3_RESUME_LOAD_OPTIMIZERS}")\n'
        'print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"',
    )
    if "for _ in range(num_trials):\n" in text:
        text = replace_exact(
            text,
            "for _ in range(num_trials):\n",
            "for trial_idx in range(num_trials):\n"
            "    track3_trial_seed = TRACK3_SEED_BASE + TRACK3_SEED_OFFSET + trial_idx\n"
            "    if TRACK3_RESET_TRIAL_SEED:\n"
            "        torch.manual_seed(track3_trial_seed)\n"
            "        torch.cuda.manual_seed_all(track3_trial_seed)\n"
            "    print0(f\"track3_trial_seed={track3_trial_seed} trial={trial_idx}\", console=True)\n",
        )
    else:
        text = replace_exact(
            text,
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n",
            "trial_idx = 0\n"
            "track3_trial_seed = TRACK3_SEED_BASE + TRACK3_SEED_OFFSET + trial_idx\n"
            "if TRACK3_RESET_TRIAL_SEED:\n"
            "    torch.manual_seed(track3_trial_seed)\n"
            "    torch.cuda.manual_seed_all(track3_trial_seed)\n"
            "print0(f\"track3_trial_seed={track3_trial_seed} trial={trial_idx}\", console=True)\n"
            "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n",
        )
    if "    train_steps = 3350\n" in text:
        text = replace_exact(
            text,
            "    train_steps = 3350\n",
            f"    train_steps = {train_steps}  # generated by tools/make_track3_locoprop_m.py\n",
        )
    elif "    train_steps = 3300\n" in text:
        text = replace_exact(
            text,
            "    train_steps = 3300\n",
            f"    train_steps = {train_steps}  # generated by tools/make_track3_locoprop_m.py\n",
        )
    elif "train_steps = FINAL_TRAIN_STEPS\n" in text:
        text = replace_exact(
            text,
            "train_steps = FINAL_TRAIN_STEPS\n",
            f"train_steps = {train_steps}  # generated by tools/make_track3_locoprop_m.py\n",
        )
    else:
        text, count = re.subn(
            r"(?m)^train_steps = \d+\s*$",
            f"train_steps = {train_steps}  # generated by tools/make_track3_locoprop_m.py",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("train_steps pattern not found")
    text = replace_exact(
        text,
        "mbs = 64\n",
        "mbs = int(os.environ.get(\"TRACK3_MBS\", \"64\"))\n",
    )
    fixed_power_set_hparams = """def set_hparams(step):
    progress = step / FINAL_SCHEDULE_STEPS
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)
"""
    if "    def set_hparams(step, cooldown_frac=0.7):\n" in text:
        fn_indent = "    "
        set_hparams_style = "cooldown"
    elif "def set_hparams(step, cooldown_frac=0.7):\n" in text:
        fn_indent = ""
        set_hparams_style = "cooldown"
    elif fixed_power_set_hparams in text:
        fn_indent = ""
        set_hparams_style = "fixed_power"
    else:
        raise RuntimeError("set_hparams pattern not found")
    if set_hparams_style == "fixed_power":
        # Keep source-owned defaults when adapting the record script. Explicit
        # environment overrides still select experimental schedules/seeds.
        source_schedule = re.search(r"(?m)^FINAL_SCHEDULE_STEPS = (\d+)", text).group(1)
        source_power = re.search(r"(?m)^FINAL_LR_POWER = ([0-9.]+)", text).group(1)
        text = text.replace('os.environ.get("TRACK3_LR_SCHEDULE", "linear")',
                            'os.environ.get("TRACK3_LR_SCHEDULE", "pr287")')
        text = text.replace('os.environ.get("TRACK3_LR_SCHEDULE_STEPS", "0")',
                            f'os.environ.get("TRACK3_LR_SCHEDULE_STEPS", "{source_schedule}")')
        text = text.replace('os.environ.get("TRACK3_LR_POWER", "1.0")',
                            f'os.environ.get("TRACK3_LR_POWER", "{source_power}")')
        text = text.replace('_env_flag("TRACK3_RESET_TRIAL_SEED", "1")',
                            '_env_flag("TRACK3_RESET_TRIAL_SEED", "0")')
    body_indent = fn_indent + "    "
    schedule_helpers = (
        f"{fn_indent}def _track3_active_lr_schedule(step):\n"
        f"{body_indent}if TRACK3_LR_SWITCH_STEP >= 0 and step >= TRACK3_LR_SWITCH_STEP and TRACK3_LR_AFTER_SWITCH:\n"
        f"{body_indent}    return TRACK3_LR_AFTER_SWITCH, TRACK3_LR_AFTER_SWITCH_STEPS, TRACK3_LR_AFTER_SWITCH_POWER\n"
        f"{body_indent}return TRACK3_LR_SCHEDULE, TRACK3_LR_SCHEDULE_STEPS, TRACK3_LR_POWER\n\n"
        f"{fn_indent}def _track3_pr287_lr(step, initial_lr, power_c, schedule_steps, lr_power):\n"
        f"{body_indent}schedule_steps = schedule_steps if schedule_steps > 0 else train_steps\n"
        f"{body_indent}downward_lr = power_c * max(0.0, schedule_steps - step) ** lr_power\n"
        f"{body_indent}return min(initial_lr, downward_lr)\n\n"
        f"{fn_indent}def _track3_schedule_lr(step, group, lr_schedule, schedule_steps, lr_power):\n"
        f"{body_indent}schedule_steps = schedule_steps if schedule_steps > 0 else train_steps\n"
        f"{body_indent}if lr_schedule == \"pr287\":\n"
        f"{body_indent}    return _track3_pr287_lr(step, group[\"initial_lr\"], group[\"power_c\"], schedule_steps, lr_power)\n"
        f"{body_indent}progress = step / schedule_steps\n"
        f"{body_indent}if progress < 1 - TRACK3_COOLDOWN_FRAC:\n"
        f"{body_indent}    eta = 1.0\n"
        f"{body_indent}else:\n"
        f"{body_indent}    eta = max(0.0, (1 - progress) / TRACK3_COOLDOWN_FRAC)\n"
        f"{body_indent}    if lr_schedule == \"power\":\n"
        f"{body_indent}        eta = eta ** lr_power\n"
        f"{body_indent}    eta = max(eta, TRACK3_LR_MIN_ETA)\n"
        f"{body_indent}return group[\"initial_lr\"] * eta\n\n"
        f"{fn_indent}def _track3_blend_lr(step, group, base_lr, lr_mult):\n"
        f"{body_indent}if TRACK3_LR_AFTER_SWITCH and TRACK3_LR_SWITCH_STEP >= 0 and TRACK3_LR_SWITCH_BLEND_STEPS > 0:\n"
        f"{body_indent}    t = (step - TRACK3_LR_SWITCH_STEP) / TRACK3_LR_SWITCH_BLEND_STEPS\n"
        f"{body_indent}    if 0 <= t < 1:\n"
        f"{body_indent}        old_lr = _track3_schedule_lr(step, group, TRACK3_LR_SCHEDULE, TRACK3_LR_SCHEDULE_STEPS, TRACK3_LR_POWER) * lr_mult\n"
        f"{body_indent}        t = t * t * (3 - 2 * t)\n"
        f"{body_indent}        base_lr = old_lr + (base_lr - old_lr) * t\n"
        f"{body_indent}if not TRACK3_LR_BLEND_TARGET or TRACK3_LR_BLEND_START < 0:\n"
        f"{body_indent}    return base_lr\n"
        f"{body_indent}if step < TRACK3_LR_BLEND_START:\n"
        f"{body_indent}    return base_lr\n"
        f"{body_indent}target_lr = _track3_schedule_lr(step, group, TRACK3_LR_BLEND_TARGET, TRACK3_LR_BLEND_TARGET_STEPS, TRACK3_LR_BLEND_TARGET_POWER) * lr_mult\n"
        f"{body_indent}if TRACK3_LR_BLEND_END <= TRACK3_LR_BLEND_START or step >= TRACK3_LR_BLEND_END:\n"
        f"{body_indent}    t = 1.0\n"
        f"{body_indent}else:\n"
        f"{body_indent}    t = (step - TRACK3_LR_BLEND_START) / (TRACK3_LR_BLEND_END - TRACK3_LR_BLEND_START)\n"
        f"{body_indent}    t = max(0.0, min(1.0, t))\n"
        f"{body_indent}    t = t * t * (3.0 - 2.0 * t)\n"
        f"{body_indent}return base_lr + (target_lr - base_lr) * t\n\n"
    )
    if set_hparams_style == "fixed_power":
        text = replace_exact(
            text,
            fixed_power_set_hparams,
            schedule_helpers
            + f"{fn_indent}def set_hparams(step):\n"
            + f"{body_indent}lr_schedule, lr_schedule_steps, lr_power = _track3_active_lr_schedule(step)\n"
            + f"{body_indent}if lr_schedule not in {{\"pr287\", \"power\"}}:\n"
            + f"{body_indent}    lr_schedule = \"pr287\"\n"
            + f"{body_indent}lr_mult = _track3_lr_multiplier(step)\n"
            + f"{body_indent}for opt in optimizers:\n"
            + f"{body_indent}    for group in opt.param_groups:\n"
            + f"{body_indent}        base_lr = _track3_schedule_lr(step, group, lr_schedule, lr_schedule_steps, lr_power) * lr_mult\n"
            + f"{body_indent}        group[\"lr\"] = _track3_blend_lr(step, group, base_lr, lr_mult)\n",
        )
    else:
        text = replace_exact(
            text,
            f"{fn_indent}def set_hparams(step, cooldown_frac=0.7):\n",
            schedule_helpers
            + f"{fn_indent}def set_hparams(step, cooldown_frac=TRACK3_COOLDOWN_FRAC):\n",
        )
        text = replace_exact(
            text,
            f"{body_indent}progress = step / train_steps\n",
            f"{body_indent}lr_schedule, lr_schedule_steps, lr_power = _track3_active_lr_schedule(step)\n"
            f"{body_indent}if lr_schedule == \"pr287\":\n"
            f"{body_indent}    lr_mult = _track3_lr_multiplier(step)\n"
            f"{body_indent}    for opt in optimizers:\n"
            f"{body_indent}        for group in opt.param_groups:\n"
            f"{body_indent}            base_lr = _track3_pr287_lr(step, group[\"initial_lr\"], group[\"power_c\"], lr_schedule_steps, lr_power) * lr_mult\n"
            f"{body_indent}            group[\"lr\"] = _track3_blend_lr(step, group, base_lr, lr_mult)\n"
            f"{body_indent}    return\n"
            f"{body_indent}schedule_steps = lr_schedule_steps if lr_schedule_steps > 0 else train_steps\n"
            f"{body_indent}progress = step / schedule_steps\n",
        )
        text = replace_exact(
            text,
            f"{body_indent}else:\n{body_indent}    eta = (1 - progress) / cooldown_frac\n",
            f"{body_indent}else:\n"
            f"{body_indent}    eta = max(0.0, (1 - progress) / cooldown_frac)\n"
            f"{body_indent}    if lr_schedule == \"power\":\n"
            f"{body_indent}        eta = eta ** lr_power\n"
            f"{body_indent}    eta = max(eta, TRACK3_LR_MIN_ETA)\n",
        )
        text = replace_exact(
            text,
            f"{body_indent}        group[\"lr\"] = group[\"initial_lr\"] * eta\n",
            f"{body_indent}        lr_mult = _track3_lr_multiplier(step)\n"
            f"{body_indent}        base_lr = group[\"initial_lr\"] * eta * lr_mult\n"
            f"{body_indent}        group[\"lr\"] = _track3_blend_lr(step, group, base_lr, lr_mult)\n",
        )
    if "        val_step_freq = 125 if step / train_steps < 0.9 else 25\n" in text:
        text = replace_exact(
            text,
            "        val_step_freq = 125 if step / train_steps < 0.9 else 25\n",
            "        val_step_freq = int(os.environ.get(\"SCREEN_VAL_EVERY\", \"125\")) if step / train_steps < 0.9 else 25\n",
        )
    elif "val_regular_interval = 125\n" in text:
        text = replace_exact(
            text,
            "val_regular_interval = 125\n",
            "val_regular_interval = int(os.environ.get(\"SCREEN_VAL_EVERY\", \"125\"))\n",
        )
    elif "if step == train_steps or step % 125 == 0:\n" in text:
        text = replace_exact(
            text,
            "if step == train_steps or step % 125 == 0:\n",
            "if step == train_steps or step % int(os.environ.get(\"SCREEN_VAL_EVERY\", \"125\")) == 0:\n",
        )
    else:
        raise RuntimeError("validation interval pattern not found")
    text = text.replace(
        "            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float(\"nan\")\n",
        "            step_avg = time_since_last_val / (step - last_val_step) if step > last_val_step else float(\"nan\")\n",
        1,
    )
    if "    train_loader = distributed_data_generator(\"data/fineweb10B/fineweb_train_*.bin\", batch_size)\n" in text:
        run_indent = "    "
    elif "train_loader = distributed_data_generator(\"data/fineweb10B/fineweb_train_*.bin\", batch_size)\n" in text:
        run_indent = ""
    else:
        raise RuntimeError("train_loader pattern not found")
    if run_indent:
        text = replace_exact(
            text,
            f"{run_indent}train_loader = distributed_data_generator(\"data/fineweb10B/fineweb_train_*.bin\", batch_size)\n",
            f"{run_indent}train_loader = distributed_data_generator(\"data/fineweb10B/fineweb_train_*.bin\", batch_size)\n"
            f"{run_indent}start_step = maybe_load_track3_checkpoint(model, optimizers)\n"
            f"{run_indent}if start_step > 0 and TRACK3_RESUME_ADVANCE_DATA:\n"
            f"{run_indent}    for _ in range(start_step):\n"
            f"{run_indent}        next(train_loader)\n"
            f"{run_indent}    print0(f\"track3_resume_advanced_data steps:{{start_step}}\", console=True)\n",
        )
    else:
        text = replace_exact(
            text,
            'for opt in optimizers:\n    for group in opt.param_groups:\n        group["initial_lr"] = group["lr"]\n',
            'for opt in optimizers:\n    for group in opt.param_groups:\n        group["initial_lr"] = group["lr"]\n'
            'start_step = maybe_load_track3_checkpoint(model, optimizers)\n'
            'if start_step > 0 and TRACK3_RESUME_ADVANCE_DATA:\n'
            '    for _ in range(start_step):\n'
            '        next(train_loader)\n'
            '    print0(f"track3_resume_advanced_data steps:{start_step}", console=True)\n',
        )
    if f"{run_indent}last_val_step = 0\n" in text:
        text = replace_exact(
            text,
            f"{run_indent}last_val_step = 0\n",
            f"{run_indent}last_val_step = start_step\n",
        )
    text = replace_exact(
        text,
        f"{run_indent}for step in range(train_steps + 1):\n",
        f"{run_indent}for step in range(start_step, train_steps + 1):\n",
    )
    target_inserted = False
    for indent in ("            ", "        "):
        old = f"{indent}model.train()\n{indent}# start the clock again\n"
        if old in text:
            text = text.replace(
                old,
                f"{indent}if maybe_save_track3_checkpoint(model, optimizers, step, train_steps, trial_idx, val_loss):\n"
                f"{indent}    break\n"
                f"{indent}if TRACK3_TARGET_LOSS > 0 and float(val_loss) <= TRACK3_TARGET_LOSS:\n"
                f"{indent}    print0(f\"target_loss_reached step:{{step}} val_loss:{{val_loss:.5f}} target:{{TRACK3_TARGET_LOSS:.5f}}\", console=True)\n"
                f"{indent}    break\n"
                f"{indent}model.train()\n{indent}# start the clock again\n",
                1,
            )
            target_inserted = True
            break
    if not target_inserted:
        raise RuntimeError("validation target-loss insertion pattern not found")
    train_section_inserted = False
    for indent in ("        ", "    "):
        old = f"{indent}# --------------- TRAINING SECTION -----------------\n{indent}inputs, targets = next(train_loader)\n"
        if old in text:
            text = text.replace(
                old,
                f"{indent}# --------------- TRAINING SECTION -----------------\n"
                f"{indent}set_locoprop_m_current_step(step)\n"
                f"{indent}inputs, targets = next(train_loader)\n",
                1,
            )
            train_section_inserted = True
            break
    if not train_section_inserted:
        raise RuntimeError("training section insertion pattern not found")
    val_model_inserted = False
    for indent in ("                    ", "                "):
        old = f"{indent}val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])\n"
        if old in text:
            text = text.replace(
                old,
                f"{indent}val_loss += compiled_model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])\n",
                1,
            )
            val_model_inserted = True
            break
    if not val_model_inserted:
        raise RuntimeError("validation model call replacement pattern not found")
    microbatch_inserted = False
    forward_call = "(model.forward if precond_hooks else compiled_model)" if "def enable_precond_hooks(" in text else "compiled_model"
    simple_microbatch = (
        "        for i in range(len(inputs) // mbs):\n"
        "            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()\n"
    )
    guarded_microbatch = (
        "    for i in range(len(inputs) // mbs):\n"
        "        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])\n"
        "        # NaN guard: catch divergence the step it happens, not 750 steps later.\n"
        "        if not torch.isfinite(loss).all():\n"
        "            raise RuntimeError(f\"non-finite train loss at step {step} mb {i}: {loss.item()}\")\n"
        "        loss.backward()\n"
    )
    if simple_microbatch in text:
        text = text.replace(
            simple_microbatch,
            "        num_microbatches = len(inputs) // mbs\n"
            "        loco_begin_capture(step, num_microbatches, _locom_active(step), LOCO_M_LAYER_SET, LOCO_M_ACCUM_SAMPLES, dist.get_rank(), LOCO_M_MICRO_SAMPLE_TOKENS)\n"
            "        for i in range(num_microbatches):\n"
            "            loco_set_microbatch(i)\n"
            f"            {forward_call}(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()\n",
            1,
        )
        microbatch_inserted = True
    elif guarded_microbatch in text:
        text = text.replace(
            guarded_microbatch,
            "    num_microbatches = len(inputs) // mbs\n"
            "    loco_begin_capture(step, num_microbatches, _locom_active(step), LOCO_M_LAYER_SET, LOCO_M_ACCUM_SAMPLES, dist.get_rank(), LOCO_M_MICRO_SAMPLE_TOKENS)\n"
            "    for i in range(num_microbatches):\n"
            "        loco_set_microbatch(i)\n"
            f"        loss = {forward_call}(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])\n"
            "        if not torch.isfinite(loss).all():\n"
            "            raise RuntimeError(f\"non-finite train loss at step {step} mb {i}: {loss.item()}\")\n"
            "        loss.backward()\n",
            1,
        )
        microbatch_inserted = True
    if not microbatch_inserted:
        raise RuntimeError("microbatch backward insertion pattern not found")
    step_block_inserted = False
    for indent in ("        ", "    "):
        old = (
            f"{indent}set_hparams(step)\n"
            f"{indent}for opt in optimizers:\n"
            f"{indent}    opt.step()\n"
            f"{indent}model.zero_grad(set_to_none=True)\n"
        )
        if old in text:
            text = text.replace(
                old,
                f"{indent}set_hparams(step)\n"
                f"{indent}prepare_locoprop_m(model, step)\n"
                f"{indent}for opt in optimizers:\n"
                f"{indent}    opt.step()\n"
                f"{indent}flush_locoprop_m_apply_stats(step)\n"
                f"{indent}model.zero_grad(set_to_none=True)\n"
                f"{indent}if maybe_save_track3_checkpoint(model, optimizers, step + 1, train_steps, trial_idx, None):\n"
                f"{indent}    break\n",
                1,
            )
            step_block_inserted = True
            break
    if not step_block_inserted:
        raise RuntimeError("optimizer step insertion pattern not found")
    text = replace_exact(
        text,
        "assert 8 % dist.get_world_size() == 0\n",
        "assert 8 % dist.get_world_size() == 0\n",
    )
    if "    optimizer2.attach_preconditioner()\n    optimizers = [optimizer1, optimizer2]\n" in text:
        text = replace_exact(
            text,
            "    optimizer2.attach_preconditioner()\n    optimizers = [optimizer1, optimizer2]\n",
            "    optimizer2.attach_preconditioner()\n"
            "    attach_locoprop_m_optimizer(model, optimizer2)\n"
            "    optimizers = [optimizer1, optimizer2]\n",
        )
    elif (
        "    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],\n"
        "                      lr=0.035, weight_decay=0.025)\n"
    ) in text:
        text = replace_exact(
            text,
            "    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],\n"
            "                      lr=0.035, weight_decay=0.025)\n",
            "    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],\n"
            "                      lr=0.035, weight_decay=0.025)\n"
            "    attach_locoprop_m_optimizer(model, optimizer2)\n",
        )
    elif "optimizers = [optimizer1, optimizer2]\n" in text:
        text = replace_exact(
            text,
            "optimizers = [optimizer1, optimizer2]\n",
            "attach_locoprop_m_optimizer(model, optimizer2)\n"
            "optimizers = [optimizer1, optimizer2]\n",
        )
    else:
        raise RuntimeError("optimizer2 attach pattern not found")
    assert_inserted = False
    for indent in ("    ", ""):
        old = (
            f"{indent}assert set(p for opt in optimizers for group in opt.param_groups\n"
            f"{indent}           for p in group[\"params\"]) == set(model.parameters())\n"
        )
        if old in text:
            text = text.replace(
                old,
                f"{indent}if TRACK3_LR_SCHEDULE == \"pr287\" or TRACK3_LR_AFTER_SWITCH == \"pr287\":\n"
                f"{indent}    optimizer1.param_groups[0][\"power_c\"] = TRACK3_ADAM_EMBED_POWER_C\n"
                f"{indent}    optimizer1.param_groups[1][\"power_c\"] = TRACK3_ADAM_PROJ_POWER_C\n"
                f"{indent}    optimizer1.param_groups[2][\"power_c\"] = TRACK3_ADAM_OTHER_POWER_C\n"
                f"{indent}    optimizer2.param_groups[0][\"power_c\"] = TRACK3_MUON_POWER_C\n"
                f"{indent}assert set(p for opt in optimizers for group in opt.param_groups\n"
                f"{indent}           for p in group[\"params\"]) == set(model.parameters())\n",
                1,
            )
            assert_inserted = True
            break
    if not assert_inserted:
        raise RuntimeError("optimizer assert/power_c insertion pattern not found")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="records/track_3_optimization/train_gpt_simple.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    generate(Path(args.source), Path(args.output), args.steps)


if __name__ == "__main__":
    main()
