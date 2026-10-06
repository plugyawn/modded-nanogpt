#!/usr/bin/env python3
"""Prepare a matched ablation manifest or run one explicit arm on existing GPUs.

Example (preparation performs no training or provisioning)::

    python tools/run_wr_muon_ablation.py prepare --output /tmp/muon-plan \
        --profiles simple,contra,soft,soap,normuon,floor,polar_norm,no_wd,may9 \
        --seeds 3710,3711,3712
    python tools/run_wr_muon_ablation.py run --plan /tmp/muon-plan/plan.json \
        --arm simple_alpha0_sgd4_seed3710 --gpus 1

All alpha0 arms retain capture/solve overhead; optional native timing arms use
WR_LOCOM_ENABLED=0. A plan is a list of experiments, not a request to launch all
of them. The runner never provisions or deletes resources and never retries a
running or failed arm automatically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from make_wr_record_muon_ablation import generate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt"
PROFILES = ("simple", "contra", "soft", "soap", "normuon", "floor", "polar_norm", "no_wd", "may9")
SOLVERS = {"sgd4": ("sgd", 4), "rmsprop10": ("rmsprop", 10),
           "rmsprop10_safe": ("rmsprop", 10), "rmsprop10_post": ("rmsprop", 10),
           "rmsprop10_fullgrad": ("rmsprop", 10)}
SOLVER_METADATA = {
    "sgd4": dict(local_opt="sgd", max_local_steps=4, rms_style="torch", local_lr_decay=False),
    "rmsprop10": dict(local_opt="rmsprop", max_local_steps=10, rms_style="tensorflow",
        local_lr_decay=True, label="TensorFlow-style guarded RMSProp10",
        momentum_beta=.999, squared_gradient_beta=.9, epsilon=1e-5,
        limitations="Sampled c_fc-only hybrid; RMS state commits only after an accepted stored weight update. Local descent/backtracking/norm gates remain enabled; not a fully paper-faithful implementation."),
}
for _name, _center in (("rmsprop10_safe", "pre_base"), ("rmsprop10_post", "post_base")):
    SOLVER_METADATA[_name] = dict(SOLVER_METADATA["rmsprop10"],
        label="Guarded RMSProp10 with complete-displacement backtracking",
        scale_momentum=True, center=_center, sample_tokens=1024,
        limitations="Sampled FC hybrid, with full momentum displacement shrunk during line search. The post_base mode centers its frozen-input correction after the native optimizer step; it is not the paper's standalone layerwise optimizer.")
SOLVER_METADATA["rmsprop10_fullgrad"] = dict(SOLVER_METADATA["rmsprop10_safe"],
    label="Full-gradient linear term with sampled M curvature", linear_term="full",
    limitations="Experimental FC-only hybrid. Preserves the full mean-token gradient as the linear term, but still uses activation-matching curvature and an additive outer update. Not a validated transformer optimizer.")
DATASET_REVISION = "889765ea1f903759787add96995d81171b632d0c"
CONTROLLED_PREFIXES = ("WR_MUON_", "WR_LOCOM_", "FINAL_")
CONTROLLED_NAMES = {"SCREEN_VAL_EVERY", "TRAIN_PROGRESS_INTERVAL"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_settings(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    return dict(profile=profile, contra=profile in {"contra", "may9"},
                soft=profile in {"soft", "may9"}, soap=profile in {"soap", "may9"},
                normuon=profile in {"normuon", "may9"}, uw_floor=profile in {"floor", "may9"},
                decoupled_wd=profile not in {"no_wd", "may9"},
                polar_norm="gram" if profile in {"polar_norm", "may9"} else "frobenius")


def arm_environment(profile: str, alpha: int, *, native: bool = False,
                    steps: int = 3040, schedule_steps: int = 3105,
                    solver: str = "sgd4", replacement: bool = False) -> dict[str, str]:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    local_opt, local_steps = SOLVERS[solver]
    settings = profile_settings(profile)
    return {
        "WR_MUON_PROFILE": profile,
        **{"WR_MUON_" + key.upper(): str(int(settings[key]))
           for key in ("contra", "soft", "soap", "normuon", "uw_floor", "decoupled_wd")},
        "WR_MUON_POLAR_NORM": settings["polar_norm"],
        "FINAL_TRAIN_STEPS": str(steps), "FINAL_SCHEDULE_STEPS": str(schedule_steps),
        "TRAIN_PROGRESS_INTERVAL": "25", "SCREEN_VAL_EVERY": "125",
        "WR_LOCOM_ENABLED": str(int(not native)), "WR_LOCOM_COMPILE": "1",
        "WR_LOCOM_LAYERS": "all", "WR_LOCOM_ACTIVE_WINDOWS": "",
        "WR_LOCOM_START_STEP": "0", "WR_LOCOM_END_STEP": str(steps),
        "WR_LOCOM_INTERVAL": "1", "WR_LOCOM_SAMPLE_TOKENS": str(SOLVER_METADATA[solver].get("sample_tokens", 8192)),
        "WR_LOCOM_STEPS": str(local_steps), "WR_LOCOM_INNER_LR": "0.1",
        "WR_LOCOM_LOCAL_OPT": local_opt,
        "WR_LOCOM_RMS_BETA1": "0.999", "WR_LOCOM_RMS_BETA2": "0.9",
        "WR_LOCOM_RMS_EPS": "1e-5", "WR_LOCOM_RESET_RMS": "0",
        "WR_LOCOM_RMS_STYLE": SOLVER_METADATA[solver]["rms_style"],
        "WR_LOCOM_LR_DECAY": str(int(SOLVER_METADATA[solver]["local_lr_decay"])),
        "WR_LOCOM_SCALE_MOMENTUM": str(int(SOLVER_METADATA[solver].get("scale_momentum", False))),
        "WR_LOCOM_CENTER": SOLVER_METADATA[solver].get("center", "pre_base"),
        "WR_LOCOM_LINEAR_TERM": SOLVER_METADATA[solver].get("linear_term", "sample"),
        "WR_LOCOM_UPDATE_MODE": "partial_fc_replacement" if replacement else "additive",
        "WR_LOCOM_REPLACE_NORM_CAP": "0.01",
        "WR_LOCOM_TARGET_GAMMA": "100", "WR_LOCOM_PROX": "0.1",
        "WR_LOCOM_ALPHA": str(alpha), "WR_LOCOM_NORM_CAP": "0.05",
        "WR_LOCOM_NORM_TO_BASE": "0", "WR_LOCOM_REQUIRE_LOSS_DECREASE": "1",
        "WR_LOCOM_MIN_COS_DESC": "0", "WR_LOCOM_MAX_BACKTRACKS": "20",
        "WR_LOCOM_ACCUM_SAMPLES": "1",
        "WR_LOCOM_LOG_STEPS": "0,1,2,10,50,125,250,500,1000,1500,2000,2500,2750,3000,3039",
    }


def prepare(source: Path, output: Path, profiles: list[str], seeds: list[int],
            steps: int = 3040, schedule_steps: int = 3105, native_timing: bool = False,
            solvers: list[str] | None = None, replacement: bool = False) -> dict:
    solvers = ["sgd4", "rmsprop10"] if solvers is None else solvers
    if not 0 < steps < schedule_steps:
        raise ValueError("require 0 < steps < schedule_steps to preserve the LR horizon")
    if len(set(profiles)) != len(profiles) or len(set(seeds)) != len(seeds) or len(set(solvers)) != len(solvers):
        raise ValueError("profiles, seeds and solvers must be unique")
    if not profiles or not seeds or not solvers:
        raise ValueError("at least one profile, seed and solver are required")
    for solver in solvers:
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r}")
    for profile in profiles:
        profile_settings(profile)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("plan.json already exists; use a new output directory")
    generated = output / "train_muon_ablation.py"
    generate(source, generated, steps, schedule_steps)
    arms = []
    for seed in seeds:
        for profile in profiles:
            for solver in solvers:
                for alpha in (0, 1):
                    arms.append(dict(id=f"{profile}_alpha{alpha}_{solver}_seed{seed}", seed=seed,
                        settings=profile_settings(profile),
                        env=arm_environment(profile, alpha, steps=steps, schedule_steps=schedule_steps, solver=solver)))
            if replacement:
                arms.append(dict(id=f"{profile}_partial_fc_replacement_rmsprop10_seed{seed}", seed=seed,
                    settings=profile_settings(profile),
                    env=arm_environment(profile, 1, steps=steps, schedule_steps=schedule_steps,
                                        solver="rmsprop10", replacement=True)))
            if native_timing:
                arms.append(dict(id=f"{profile}_native_seed{seed}", seed=seed,
                    settings=profile_settings(profile),
                    env=arm_environment(profile, 0, native=True, steps=steps, schedule_steps=schedule_steps)))
    manifest = dict(schema=1, generated_file=generated.name, generated_sha256=sha256(generated),
        source_file=str(source.resolve()), source_sha256=sha256(source),
        generator_sha256=sha256(Path(__file__).with_name("make_wr_record_muon_ablation.py")),
        base_generator_sha256=sha256(Path(__file__).with_name("make_wr_record_locoprop_m.py")),
        local_solver_sha256=sha256(Path(__file__).with_name("locoprop_local.py")),
        checkpoint_helper_sha256=sha256(Path(__file__).with_name("wr_ablation_checkpoint.py")),
        solvers={name: SOLVER_METADATA[name] for name in solvers},
        dataset=dict(repo="kjj0/fineweb10B-gpt2", revision=DATASET_REVISION,
                     train_chunks=list(range(1, 17)), validation_chunks=[0]),
        common=dict(model_layers=12, model_dim=768, vocab_size=50304, global_tokens=524288,
                    microbatch_sequences=64, sequence_length=1024, steps=steps,
                    schedule_steps=schedule_steps, schedule="May9 PR287 power=1.2",
                    muon_lr=0.0375, muon_momentum=0.95, muon_weight_decay=0.025),
        interpretation="Individual profiles add one optimizer mechanism to simple Muon; common May9 schedule is not ablated. Alpha0 retains matching local-solver overhead. Partial FC replacement freezes only the selected native c_fc weight update; biases and other parameters remain native, and replacement uses a separate weight-relative norm cap. One seed is exploratory.",
        arms=arms)
    (output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(plan_path: Path, arm_id: str, workdir: Path, gpus: int = 1,
        retry_failed: bool = False, checkpoint: bool = False, checkpoint_every: int = 500,
        resume_checkpoint: Path | None = None, checkpoint_exit_step: int = -1,
        checkpoint_stop_file: Path | None = None) -> int:
    if gpus not in {1, 2, 4, 8}:
        raise ValueError("gpus must be 1, 2, 4, or 8")
    if (checkpoint or resume_checkpoint is not None) and gpus != 1:
        raise ValueError("checkpointing and resume support one GPU only")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if checkpoint_exit_step >= 0 and not checkpoint:
        raise ValueError("checkpoint_exit_step requires checkpoint=True")
    if checkpoint_stop_file is not None and not checkpoint:
        raise ValueError("checkpoint_stop_file requires checkpoint=True")
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    if checkpoint_exit_step != -1 and not 0 < checkpoint_exit_step <= plan["common"]["steps"]:
        raise ValueError("checkpoint_exit_step must be -1 or a positive update within the training horizon")
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    matches = [arm for arm in plan["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError(f"expected one arm named {arm_id!r}")
    arm = matches[0]
    generated = plan_path.parent / plan["generated_file"]
    if sha256(generated) != plan["generated_sha256"]:
        raise ValueError("generated script checksum differs from the manifest")
    output = plan_path.parent / arm_id
    status_path = output / "status.json"
    if status_path.exists():
        old_status = json.loads(status_path.read_text())
        finished_resume = resume_checkpoint is not None and "exit_code" in old_status
        if not (finished_resume or (retry_failed and old_status.get("exit_code", 0) != 0)):
            raise RuntimeError("arm already started; inspect its status before any retry")
    output.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if key not in CONTROLLED_NAMES and not key.startswith(CONTROLLED_PREFIXES)}
    env.update(arm["env"])
    runtime_env = dict(arm["env"])
    if checkpoint:
        runtime_env.update(WR_MUON_CHECKPOINT_DIR=str(output / "checkpoints"),
                           WR_MUON_CHECKPOINT_EVERY=str(checkpoint_every),
                           WR_MUON_CHECKPOINT_EXIT_STEP=str(checkpoint_exit_step))
    if resume_checkpoint is not None:
        runtime_env["WR_MUON_RESUME_CHECKPOINT"] = str(resume_checkpoint.resolve())
    if checkpoint_stop_file is not None:
        runtime_env["WR_MUON_CHECKPOINT_STOP_FILE"] = str(checkpoint_stop_file.resolve())
    env.update(runtime_env)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={gpus}", str(generated), "--seed", str(arm["seed"])]
    now = lambda: datetime.now(timezone.utc).isoformat()
    status = dict(arm=arm_id, started_utc=now(), runner_pid=os.getpid(), command=command,
                  gpus=gpus, effective_settings=arm["settings"], env=runtime_env,
                  generated_sha256=plan["generated_sha256"])
    if status_path.exists():
        status["previous_attempt"] = old_status
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Starting {arm_id}; output={output}", flush=True)
    with (output / "console.log").open("a" if retry_failed or resume_checkpoint else "w") as console:
        result = subprocess.run(command, cwd=workdir, env=env, stdout=console, stderr=subprocess.STDOUT)
    status.update(finished_utc=now(), exit_code=result.returncode,
                  console_sha256=sha256(output / "console.log"))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Finished {arm_id}; exit_code={result.returncode}", flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--profiles", default="simple,contra,soft,soap,normuon,floor,polar_norm,no_wd,may9")
    create.add_argument("--seeds", default="3710")
    create.add_argument("--steps", type=int, default=3040)
    create.add_argument("--schedule-steps", type=int, default=3105)
    create.add_argument("--native-timing", action="store_true")
    create.add_argument("--solvers", default="sgd4,rmsprop10")
    create.add_argument("--replacement", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--arm", required=True)
    execute.add_argument("--workdir", type=Path, default=ROOT)
    execute.add_argument("--gpus", type=int, default=1)
    execute.add_argument("--retry-failed", action="store_true")
    execute.add_argument("--checkpoint", action="store_true")
    execute.add_argument("--checkpoint-every", type=int, default=500)
    execute.add_argument("--resume-checkpoint", type=Path)
    execute.add_argument("--checkpoint-exit-step", type=int, default=-1)
    execute.add_argument("--checkpoint-stop-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.source, args.output, args.profiles.split(","),
                           [int(seed) for seed in args.seeds.split(",")],
                           args.steps, args.schedule_steps, args.native_timing,
                           args.solvers.split(","), args.replacement)
        print(f"Prepared {len(manifest['arms'])} arms at {args.output.resolve() / 'plan.json'}")
    else:
        sys.exit(run(args.plan, args.arm, args.workdir, args.gpus, args.retry_failed,
                     args.checkpoint, args.checkpoint_every, args.resume_checkpoint,
                     args.checkpoint_exit_step, args.checkpoint_stop_file))


if __name__ == "__main__":
    main()
