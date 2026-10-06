#!/usr/bin/env python3
"""Generate matched Muon/LocoProp ablations from the May 9 record architecture.

The default ``may9`` profile reproduces the record optimizer. ``simple`` uses
the original simplified Track 3 Muon: Frobenius-normalized polar iteration,
Nesterov momentum, aspect-ratio scaling, and decoupled weight decay. Individual
profiles add exactly one mechanism to that simple optimizer. Architecture,
initialization, Adam groups, and the May 9 learning-rate schedule stay common.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from make_wr_record_locoprop_m import generate as generate_locom, replace_exact

CHECKPOINT_SOURCE = Path(__file__).with_name("wr_ablation_checkpoint.py").read_text()


CHECKPOINT_RUNTIME = r'''
WR_MUON_CHECKPOINT_DIR = os.environ.get("WR_MUON_CHECKPOINT_DIR", "")
WR_MUON_CHECKPOINT_EVERY = int(os.environ.get("WR_MUON_CHECKPOINT_EVERY", "500"))
WR_MUON_RESUME_CHECKPOINT = os.environ.get("WR_MUON_RESUME_CHECKPOINT", "")
WR_MUON_CHECKPOINT_EXIT_STEP = int(os.environ.get("WR_MUON_CHECKPOINT_EXIT_STEP", "-1"))
WR_MUON_CHECKPOINT_STOP_FILE = os.environ.get("WR_MUON_CHECKPOINT_STOP_FILE", "")
WR_MUON_CHECKPOINT_EXIT_REASON = ""
WR_MUON_CHECKPOINT_ENABLED = bool(WR_MUON_CHECKPOINT_DIR or WR_MUON_RESUME_CHECKPOINT)
if WR_MUON_CHECKPOINT_EVERY <= 0:
    raise ValueError("WR_MUON_CHECKPOINT_EVERY must be positive")
if WR_MUON_CHECKPOINT_EXIT_STEP >= 0 and not WR_MUON_CHECKPOINT_DIR:
    raise ValueError("WR_MUON_CHECKPOINT_EXIT_STEP requires WR_MUON_CHECKPOINT_DIR")
if WR_MUON_CHECKPOINT_EXIT_STEP != -1 and not 0 < WR_MUON_CHECKPOINT_EXIT_STEP <= FINAL_TRAIN_STEPS:
    raise ValueError("WR_MUON_CHECKPOINT_EXIT_STEP must be -1 or a positive update within the training horizon")
if WR_MUON_CHECKPOINT_STOP_FILE and not WR_MUON_CHECKPOINT_DIR:
    raise ValueError("WR_MUON_CHECKPOINT_STOP_FILE requires WR_MUON_CHECKPOINT_DIR")

def _wr_ablation_checkpoint_identity():
    loco_keys = ("ENABLED", "LAYERS_SPEC", "ACTIVE_WINDOWS", "START_STEP", "END_STEP",
                "INTERVAL", "SAMPLE_TOKENS", "STEPS", "INNER_LR", "TARGET_GAMMA",
                "PROX", "ALPHA", "NORM_CAP", "NORM_TO_BASE", "REQUIRE_LOSS_DECREASE",
                "MIN_COS_DESC", "MAX_BACKTRACKS", "ACCUM_SAMPLES", "LOCAL_OPT",
                "RMS_BETA1", "RMS_BETA2", "RMS_EPS", "RMS_STYLE", "RESET_RMS", "LR_DECAY",
                "UPDATE_MODE", "REPLACE_NORM_CAP", "CENTER", "SCALE_MOMENTUM", "LINEAR_TERM")
    return dict(generated_sha256=hashlib.sha256(code.encode()).hexdigest(),
        seed=SEED, world_size=dist.get_world_size(), train_steps=train_steps,
        schedule_steps=FINAL_SCHEDULE_STEPS, batch_size=batch_size, mbs=mbs,
        optimizer_settings=WR_MUON_SETTINGS,
        loco_settings={key: globals()["WR_LOCOM_" + key] for key in loco_keys},
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        dataset_revision="889765ea1f903759787add96995d81171b632d0c")

def _wr_ablation_maybe_checkpoint(next_step):
    global training_time, t0, WR_MUON_CHECKPOINT_EXIT_REASON
    if not WR_MUON_CHECKPOINT_DIR:
        return False
    stop_requested = bool(WR_MUON_CHECKPOINT_STOP_FILE and Path(WR_MUON_CHECKPOINT_STOP_FILE).exists())
    should_save = (stop_requested or next_step % WR_MUON_CHECKPOINT_EVERY == 0
                   or next_step == train_steps or next_step == WR_MUON_CHECKPOINT_EXIT_STEP)
    if not should_save:
        return False
    WR_MUON_CHECKPOINT_EXIT_REASON = ("stop_file" if stop_requested else
        "configured_step" if next_step == WR_MUON_CHECKPOINT_EXIT_STEP else "")
    torch.cuda.synchronize()
    training_time += time.perf_counter() - t0
    checkpoint = wr_save_ablation_checkpoint(WR_MUON_CHECKPOINT_DIR, model, optimizers,
        train_loader, next_step, _wr_ablation_checkpoint_identity(), training_time)
    print0(f"wr_ablation_checkpoint path={checkpoint} next_step={next_step} reason={WR_MUON_CHECKPOINT_EXIT_REASON or 'periodic_or_final'}", console=True)
    # Exclude checkpoint IO from the same training-time metric used by the record.
    t0 = time.perf_counter()
    return bool(WR_MUON_CHECKPOINT_EXIT_REASON)

'''


ABLATION_CONFIG = r'''
def _muon_ablation_bool(name, default):
    value = os.environ.get(name, str(int(default))).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")

WR_MUON_PROFILE = os.environ.get("WR_MUON_PROFILE", "may9")
_muon_profile_names = {"simple", "contra", "soft", "soap", "normuon", "floor", "polar_norm", "no_wd", "may9"}
if WR_MUON_PROFILE not in _muon_profile_names:
    raise ValueError(f"unknown WR_MUON_PROFILE={WR_MUON_PROFILE!r}")
WR_MUON_CONTRA = _muon_ablation_bool("WR_MUON_CONTRA", WR_MUON_PROFILE in {"contra", "may9"})
WR_MUON_SOFT = _muon_ablation_bool("WR_MUON_SOFT", WR_MUON_PROFILE in {"soft", "may9"})
WR_MUON_SOAP = _muon_ablation_bool("WR_MUON_SOAP", WR_MUON_PROFILE in {"soap", "may9"})
WR_MUON_NORMUON = _muon_ablation_bool("WR_MUON_NORMUON", WR_MUON_PROFILE in {"normuon", "may9"})
WR_MUON_UW_FLOOR = _muon_ablation_bool("WR_MUON_UW_FLOOR", WR_MUON_PROFILE in {"floor", "may9"})
WR_MUON_DECOUPLED_WD = _muon_ablation_bool("WR_MUON_DECOUPLED_WD", WR_MUON_PROFILE not in {"no_wd", "may9"})
WR_MUON_POLAR_NORM = os.environ.get("WR_MUON_POLAR_NORM", "gram" if WR_MUON_PROFILE in {"polar_norm", "may9"} else "frobenius")
if WR_MUON_POLAR_NORM not in {"frobenius", "gram"}:
    raise ValueError("WR_MUON_POLAR_NORM must be frobenius or gram")
WR_MUON_SETTINGS = dict(profile=WR_MUON_PROFILE, contra=WR_MUON_CONTRA,
    soft=WR_MUON_SOFT, soap=WR_MUON_SOAP, normuon=WR_MUON_NORMUON,
    uw_floor=WR_MUON_UW_FLOOR, decoupled_wd=WR_MUON_DECOUPLED_WD,
    polar_norm=WR_MUON_POLAR_NORM)
WR_LOCOM_UPDATE_MODE = os.environ.get("WR_LOCOM_UPDATE_MODE", "additive")
if WR_LOCOM_UPDATE_MODE not in {"additive", "partial_fc_replacement"}:
    raise ValueError("WR_LOCOM_UPDATE_MODE must be additive or partial_fc_replacement")
WR_LOCOM_REPLACE_NORM_CAP = float(os.environ.get("WR_LOCOM_REPLACE_NORM_CAP", "0.01"))
if not 0 <= WR_LOCOM_REPLACE_NORM_CAP < float("inf"):
    raise ValueError("WR_LOCOM_REPLACE_NORM_CAP must be finite and nonnegative; 0 disables this cap")

def _wr_locom_replacement_scale(corr, reference):
    if not 0 <= WR_LOCOM_ALPHA < float("inf"):
        raise ValueError("partial FC replacement requires finite nonnegative WR_LOCOM_ALPHA")
    scale = torch.as_tensor(WR_LOCOM_ALPHA, dtype=torch.float32, device=corr.device)
    if WR_LOCOM_REPLACE_NORM_CAP > 0:
        maximum = WR_LOCOM_REPLACE_NORM_CAP * reference.float().norm() / corr.float().norm().clamp_min(1e-20)
        scale = torch.minimum(scale, maximum)
    return scale

'''


MUON_UPDATE = r'''def muon_update(update, second_moment, step, beta2=NOR_BETA2, use_contra=True, use_soft=True):
    ns_update = zeropower_via_newtonschulz5(update)
    if use_contra or use_soft:
        # Preserve the record's interpolation and normalization when enabled.
        normalized_grad = scale_to_unit_operator_norm(update.clone())
        update_norm_estimate = gram_frobenius_norm_estimate(ns_update)
        contra_coeff = contra_coeff_for_step(step) if use_contra else 0.0
        contra_update = ns_update + contra_coeff * normalized_grad
        contra_update = contra_update * update_norm_estimate / gram_frobenius_norm_estimate(contra_update)
        if use_soft:
            soft_update = soft_via_newtonschulz5(update, SOFT_MUON_P, SOFT_MUON_SCALE, SOFT_MUON_INPUT_NORM)
            soft_update = soft_update * update_norm_estimate / gram_frobenius_norm_estimate(soft_update)
            blend = soft_blend_for_step(step)
        else:
            soft_update = contra_update
            blend = 0.0
        update = contra_update + (soft_update - contra_update) * blend
        update = update * update_norm_estimate / gram_frobenius_norm_estimate(update)
    else:
        # Avoid hidden Gram renormalizations in the simple-Muon control.
        update = ns_update
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    if WR_MUON_NORMUON:
        if second_moment is None:
            raise RuntimeError("NorMuon enabled without its second-moment buffer")
        if update.size(-2) >= update.size(-1):
            per_row_var = (update * update).mean(dim=-1, keepdim=True)
        else:
            per_row_var = (update * update).mean(dim=-2, keepdim=True)
        second_moment.lerp_(per_row_var.float(), 1 - beta2)
        vnorm = gram_frobenius_norm_estimate(update)
        update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
        vnorm_new = gram_frobenius_norm_estimate(update)
        update = update * (vnorm / vnorm_new)
    return update
'''


def _replace_function(text: str, name: str, replacement: str) -> str:
    nodes = [node for node in ast.parse(text).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one {name} definition")
    lines = text.splitlines(keepends=True)
    node = nodes[0]
    return "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])


def generate(source: Path, output: Path, train_steps: int = 3040,
             schedule_steps: int | None = 3105) -> None:
    generate_locom(source, output, train_steps, schedule_steps)
    text = output.read_text()
    text = replace_exact(text, "def gram_frobenius_norm_estimate(",
                         ABLATION_CONFIG + CHECKPOINT_SOURCE + CHECKPOINT_RUNTIME + "def gram_frobenius_norm_estimate(")
    text = replace_exact(text,
        'train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)\n',
        'if WR_MUON_CHECKPOINT_ENABLED and dist.get_world_size() != 1:\n'
        '    raise ValueError("ablation checkpointing requires world_size=1")\n'
        'train_loader = WrAblationDataIterator("data/fineweb10B/fineweb_train_*.bin", batch_size,\n'
        '    _load_data_shard, world_size=dist.get_world_size(), rank=dist.get_rank(),\n'
        '    verify_content=WR_MUON_CHECKPOINT_ENABLED)\n')
    text = replace_exact(text,
        '# start the clock\ntraining_time = 0\ndist.barrier()\nt0 = time.perf_counter()\nfor step in range(train_steps + 1):\n',
        'training_time, start_step = 0.0, 0\n'
        'if WR_MUON_RESUME_CHECKPOINT:\n'
        '    start_step, training_time = wr_load_ablation_checkpoint(WR_MUON_RESUME_CHECKPOINT,\n'
        '        model, optimizers, train_loader, _wr_ablation_checkpoint_identity())\n'
        '    print0(f"wr_ablation_resumed next_step={start_step} path={WR_MUON_RESUME_CHECKPOINT}", console=True)\n'
        '# Start after native initialization and checkpoint restore.\n'
        'dist.barrier()\nt0 = time.perf_counter()\nfor step in range(start_step, train_steps + 1):\n')
    text = replace_exact(text,
        '    model.zero_grad(set_to_none=True)\n    if TRAIN_PROGRESS_INTERVAL > 0',
        '    model.zero_grad(set_to_none=True)\n'
        '    if _wr_ablation_maybe_checkpoint(step + 1):\n'
        '        print0(f"wr_ablation_checkpoint_exit next_step={step + 1} reason={WR_MUON_CHECKPOINT_EXIT_REASON}", console=True)\n'
        '        break\n'
        '    if TRAIN_PROGRESS_INTERVAL > 0')
    text = replace_exact(text,
        "    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)\n",
        '    if WR_MUON_POLAR_NORM == "gram":\n'
        '        X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)\n'
        '    else:\n'
        '        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)\n')
    text = _replace_function(text, "muon_update", MUON_UPDATE)
    text = text.replace("if should_soap_param(n)", "if WR_MUON_SOAP and should_soap_param(n)")
    text = replace_exact(text,
        '                        if p.size(-2) >= p.size(-1):\n'
        '                            state["second_moment"] = torch.zeros((*p.shape[:-1], 1),\n'
        '                                dtype=torch.float32, device=p.device)\n'
        '                        else:\n'
        '                            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),\n'
        '                                dtype=torch.float32, device=p.device)\n',
        '                        if WR_MUON_NORMUON:\n'
        '                            if p.size(-2) >= p.size(-1):\n'
        '                                state["second_moment"] = torch.zeros((*p.shape[:-1], 1),\n'
        '                                    dtype=torch.float32, device=p.device)\n'
        '                            else:\n'
        '                                state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),\n'
        '                                    dtype=torch.float32, device=p.device)\n')
    text = replace_exact(text, '                        state["second_moment"],\n',
                         '                        state.get("second_moment"),\n')
    text = replace_exact(text, "                        use_contra=p not in self.no_contra_params,\n",
                         "                        use_contra=WR_MUON_CONTRA and p not in self.no_contra_params,\n")
    text = replace_exact(text, "                        use_soft=p not in self.no_soft_params,\n",
                         "                        use_soft=WR_MUON_SOFT and p not in self.no_soft_params,\n")
    text = replace_exact(text, "        self.step_count = 0\n",
        '        self.loco_fc_layers = {p: int(n.split(".", 1)[0]) for n, p in named_params\n'
        '                               if n.endswith(".mlp.fc.weight")}\n'
        '        self.step_count = 0\n')
    old_floor = '''                    # u/w-floor. SOAP and non-SOAP params can use different floors.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    target_uw = SOAP_TARGET_UW if use_soap else NONSOAP_TARGET_UW
                    scale = torch.where(cur_uw < target_uw, target_uw * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    # WD set to 0 — u/w target replaces wd's role (smaller updates as p grows).
'''
    new_floor = "                    if WR_MUON_UW_FLOOR:\n" + "".join(
        "    " + line for line in old_floor.splitlines(keepends=True)[:-1])
    new_floor += ('                    if WR_MUON_DECOUPLED_WD:\n'
                  '                        p.mul_(1 - group["lr"] * group["weight_decay"])\n')
    text = replace_exact(text, old_floor, new_floor)
    text = replace_exact(text,
        '                    if WR_MUON_DECOUPLED_WD:\n'
        '                        p.mul_(1 - group["lr"] * group["weight_decay"])\n'
        '                    p.add_(update, alpha=-group["lr"])\n',
        '                    replace_fc = (WR_LOCOM_UPDATE_MODE == "partial_fc_replacement"\n'
        '                        and p in self.loco_fc_layers\n'
        '                        and _wr_locom_active(WR_LOCOM_CURRENT_STEP)\n'
        '                        and _wr_locom_layer_active(self.loco_fc_layers[p]))\n'
        '                    if not replace_fc:\n'
        '                        if WR_MUON_DECOUPLED_WD:\n'
        '                            p.mul_(1 - group["lr"] * group["weight_decay"])\n'
        '                        p.add_(update, alpha=-group["lr"])\n')
    # The replacement cap is relative to the reference weight. A skipped native
    # c_fc step has zero displacement and cannot supply an additive-step cap.
    tree = ast.parse(text)
    apply = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "apply_wr_locom_m")
    assignments = [node for node in ast.walk(apply) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == "loco_correction_scale"]
    if len(assignments) != 1:
        raise RuntimeError("expected one additive correction scale assignment")
    node = assignments[0]
    lines = text.splitlines(keepends=True)
    old_scale = "".join(lines[node.lineno - 1:node.end_lineno])
    replacement = ('        if WR_LOCOM_UPDATE_MODE == "partial_fc_replacement":\n'
                   '            scale = _wr_locom_replacement_scale(corr, reference)\n'
                   '        else:\n') + "".join("    " + line for line in old_scale.splitlines(keepends=True))
    text = text.replace(old_scale, replacement, 1)
    text = replace_exact(text,
        'print0("Muon update linearly changes from Contra Muon to normal Muon, then normal Muon to soft Muon.")\n'
        'print0("This script retains the PR 274 NorMuon-lite row/column variance normalization and u/w-floor postprocessing.")\n'
        'print0("MLP and V SOAP stay on all run; V SOAP blend is fixed at 0.95.")\n',
        'print0(f"Muon ablation effective settings={WR_MUON_SETTINGS}", console=True)\n'
        'print0(f"LocoProp update mode={WR_LOCOM_UPDATE_MODE} replacement_weight_norm_cap={WR_LOCOM_REPLACE_NORM_CAP}", console=True)\n')
    text = text.replace("len(stats) < 8", "len(stats) < 12").replace("len(WR_LOCOM_APPLY_STATS) < 8", "len(WR_LOCOM_APPLY_STATS) < 12")
    text = replace_exact(text, "dist.destroy_process_group()",
        'print0(f"GPU_PEAK allocated={torch.cuda.max_memory_allocated()} reserved={torch.cuda.max_memory_reserved()}", console=True)\n'
        'dist.destroy_process_group()')
    compile(text, str(output), "exec")
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3040)
    parser.add_argument("--schedule-steps", type=int, default=3105)
    args = parser.parse_args()
    generate(args.source, args.output, args.steps, args.schedule_steps)


if __name__ == "__main__":
    main()
