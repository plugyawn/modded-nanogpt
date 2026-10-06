# Track 3 LocoProp-M Runbook

This branch contains the Track 3 LocoProp-M experiment harness used for the
recent optimizer-schedule sweeps. It is experimental code, not a record claim.
The historical, pre-fix tail comparison ends with the best new lane at
`3.30262 @3000`, well outside the current WR/reference band.

Latest plot artifacts:

- `.opencode/plots/track3_oldtail_vs_live_blends_20260603_refresh3_full.png`
- `.opencode/plots/track3_oldtail_vs_live_blends_20260603_refresh3_midlate.png`
- `.opencode/plots/track3_oldtail_vs_live_blends_20260603_refresh3_terminal.png`
- `.opencode/plots/track3_oldtail_vs_live_blends_20260603_refresh3.tsv`

## What The Harness Does

`tools/make_track3_locoprop_m.py` generates a Track 3 training script from
`records/track_3_optimization/train_gpt_simple.py` (or the Newton-Muon/May 9
record source) and adds:

- MLP `c_fc` LocoProp-M local correction probes.
- Norm-capped additive correction application.
- checkpoint save/resume with optimizer/RNG restore controls.
- linear, power, PR287-style, blended, and temporary LR-bump schedules.
- optional Soft-Muon ramp for late-stage suffix tests.
- diagnostics for correction norms, loss decrease, and descent cosine.

`tools/run_track3_locoprop_m.sh` is the local runner. It prints source,
generator, and generated-script SHA256 hashes before training so checkpoint
provenance can be reconstructed.

The three generators embed the same numerical/capture implementation from
`tools/locoprop_local.py`. Generated scripts remain self-contained. The separate
`make_wr_record_locoprop_m.py` targets the May 9 module-level record, while
`make_wr_locoprop_m.py` targets the sharded `train_gpt.py` bank.

## Solver and Capture Fixes

The local replay starts at the captured BF16 preactivation and models a weight
displacement as `pre0 + X @ delta.T`. This makes a zero target gradient produce
exactly zero correction, eliminating the previous BF16-forward/FP32-replay
residual. It is a local surrogate anchored to the forward pass; it is not a
bit-exact simulation of every later BF16 weight update.

For squared ReLU, the matching potential is `relu(z)^3 / 3`. The solver now
checks its regularized Bregman objective, including the proximal term, rather
than output MSE. Armijo backtracking stabilizes inner steps, and the final
stored correction is checked after the last update and dtype conversion.
The default gates require finite values, a decrease in that objective, and
nonnegative alignment with the full weight-gradient descent direction. The
sharded bank checks the reduced full gradient inside the optimizer, after
reduce-scatter; preparation uses the globally gathered sample gradient.

The correction is capped with alpha included, then checked again at the weights
the outer optimizer actually reached. A correction that no longer decreases the
local objective is shrunk or skipped. Norm targets are upper limits and never
amplify a small correction to fill a requested norm. RMS state commits only
when the correction is applied.

Sample capture writes paired input/preactivation/gradient rows into fixed FP32
buffers through an opaque backward custom op. It retains no strided views of
full activations. The global sample budget is spread across all microbatches
by default; each rank gets `ceil(sample_tokens / world_size)` rows. Stratified
sampling uses a separate RNG, preserving training RNG. Set `ACCUM_SAMPLES=0`
explicitly for a last-microbatch control. Compilation is enabled by default and
active-window changes do not require recompilation. The Python-backed capture
op is tagged `cudagraph_unsafe` so its schedule/RNG state is read on every call.
Newton-Muon covariance refresh steps retain the Linear module hooks and call
the eager root forward; ordinary steps use the compiled MLP capture path.

Multi-rank sample collectives run in the same layer order before owner filtering.
The module-level WR path caps against the actual gathered outer-step displacement,
so correction application does not depend on owner-only optimizer state.

Diagnostics retain `loss0`/`lossK` field names, but now report the matching
objective **relative to the initial iterate**: `loss0=0`, and a negative `lossK`
means improvement. Historical MSE values are not comparable. `accepted` reports
solver acceptance; `accepted_apply` reports the final application gate. Inspect
`backtracks`, `local_steps`, `inner_lr`, and `reason` alongside correction norms.

Source schedule/seed defaults are preserved when adapting the May 9 record
(`pr287`, horizon 3105, power 1.2; its native seed initialization is not run a
second time). The runner's seed controls select the CLI seed. Explicit
environment overrides still apply. Automatic LR switches blend for 100 steps;
set `TRACK3_LR_SWITCH_BLEND_STEPS=0` to reproduce the old abrupt switch.
The runner detects positional trial counts versus `--seed` and runs each
requested record trial with a distinct seed.

New Track 3 checkpoints include each rank's optimizer state, optimizer step
counter, local RMS state and RNG. Exact resumes require the saved world size.
Newton-Muon covariance/inverse state is restored with its shared tensor views
rebound, so later refreshes continue updating the inverse used by the optimizer.
Legacy single-rank checkpoints remain readable; legacy multi-rank checkpoints
cannot recover optimizer/RMS/RNG state that was never saved.

## Local 1x Run

Install the normal repo requirements and cache FineWeb as usual:

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 9
```

Then run a 1x Track 3 LocoProp-M screen:

```bash
TRACK3_TRAIN_STEPS=3000 \
TRACK3_NUM_TRIALS=1 \
TRACK3_SEED_OFFSET=3710 \
TRACK3_LOCOM_ENABLED=1 \
TRACK3_LOCOM_LAYERS=all \
TRACK3_LOCOM_STEPS=4 \
TRACK3_LOCOM_NORM_CAP=0.20 \
SCREEN_VAL_EVERY=25 \
bash tools/run_track3_locoprop_m.sh
```

Use `TRACK3_DRY_RUN=1` to only generate and compile-check the script.

## Modal Single-Lane Run

For Modal, use the launcher wrapper. Keep the protobuf env var; it avoids local
Modal CLI descriptor failures on this machine.

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
MODAL_DETACH=1 \
NANOGPT_MODAL_GPU=H100 \
MODAL_RUN_NAME=track3-locom-screen-seed3710 \
TRACK3_TRAIN_STEPS=3000 \
TRACK3_SEED_OFFSET=3710 \
SCREEN_VAL_EVERY=25 \
bash tools/launch_modal_track3_locom_3000_seed.sh
```

## Resume From A 1600-Step Checkpoint

The recent suffix tests resumed from a 1600 checkpoint and preserved model,
optimizer, RNG, and data-stream provenance. The checkpoint path must exist on
the Modal/remote machine:

```bash
export TRACK3_RESUME_CHECKPOINT=/root/.cache/track3_checkpoints/track3_cd500red_softmerge_pr2872000_p110_ckpt1600_seed3710_step1600.pt
export TRACK3_RESUME_ADVANCE_DATA=1
export TRACK3_RESUME_RESTORE_RNG=1
export TRACK3_RESUME_LOAD_OPTIMIZERS=1
export TRACK3_SEED_OFFSET=3710
export TRACK3_TRAIN_STEPS=3000
export TRACK3_CHECKPOINT_STEPS=2000,2400
export TRACK3_CHECKPOINT_DIR=/root/.cache/track3_checkpoints
export TRACK3_CHECKPOINT_PREFIX=track3_loco2400_h3075ramp2350x150

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
MODAL_DETACH=1 \
NANOGPT_MODAL_GPU=H100 \
MODAL_RUN_NAME=track3-redckpt1600-loco2400-h3075ramp2350x150 \
TRACK3_LOCOM_ACTIVE_WINDOWS=0:1800,2400:3000 \
TRACK3_LOCOM_END_STEP=3000 \
TRACK3_LR_BLEND_START=1800 \
TRACK3_LR_BLEND_END=2000 \
TRACK3_LR_BLEND_TARGET=pr287 \
TRACK3_LR_BLEND_TARGET_POWER=1.10 \
TRACK3_LR_BLEND_TARGET_STEPS=3075 \
TRACK3_LR_BUMP_WINDOWS=2350:2600:3000:3000:1.50 \
bash tools/launch_modal_track3_locom_3000_seed.sh
```

That exact family is the one plotted in the refresh3 artifacts. It did not land
near the WR tail, but it is useful as a reproducible negative control.

## Fanout Helpers

The helper scripts below launch small suffix grids. They assume the checkpoint
path exists on the remote host or Modal volume.

```bash
bash tools/launch_modal_track3_locom_cd500_suffix_array.sh
bash tools/launch_modal_track3_locom_resume1600_lr_array.sh
bash tools/launch_modal_track3_locom_blend_tail3.sh
```

## Important Env Knobs

Core LocoProp-M:

```text
TRACK3_LOCOM_ENABLED=1
TRACK3_LOCOM_LAYERS=all
TRACK3_LOCOM_STEPS=4
TRACK3_LOCOM_SAMPLE_TOKENS=1024
TRACK3_LOCOM_ACCUM_SAMPLES=1
TRACK3_LOCOM_MICRO_SAMPLE_TOKENS=0
TRACK3_LOCOM_COMPILE=1
TRACK3_LOCOM_REQUIRE_LOSS_DECREASE=1
TRACK3_LOCOM_MIN_COS_DESC=0.0
TRACK3_LOCOM_MAX_BACKTRACKS=20
TRACK3_LOCOM_NORM_CAP=0.20
TRACK3_LOCOM_ACTIVE_WINDOWS=0:1800
TRACK3_LOCOM_END_STEP=1800
```

Schedule controls:

```text
TRACK3_LR_SCHEDULE=linear|power|pr287
TRACK3_LR_POWER=1.0
TRACK3_LR_SCHEDULE_STEPS=3000
TRACK3_LR_SWITCH_BLEND_STEPS=100
TRACK3_LR_BLEND_START=1800
TRACK3_LR_BLEND_END=2000
TRACK3_LR_BLEND_TARGET=pr287
TRACK3_LR_BLEND_TARGET_POWER=1.10
TRACK3_LR_BLEND_TARGET_STEPS=3075
TRACK3_LR_BUMP_WINDOWS=start:ramp_end:hold_end:fade_end:mult
```

Checkpoint controls:

```text
TRACK3_CHECKPOINT_STEPS=1600,2000,2400
TRACK3_CHECKPOINT_DIR=/root/.cache/track3_checkpoints
TRACK3_RESUME_CHECKPOINT=/path/to/checkpoint.pt
TRACK3_RESUME_LOAD_OPTIMIZERS=1
TRACK3_RESUME_RESTORE_RNG=1
TRACK3_RESUME_ADVANCE_DATA=1
```

Soft-Muon suffix controls:

```text
TRACK3_SOFT_MUON=1
TRACK3_SOFT_MUON_BLEND=1.0
TRACK3_SOFT_MUON_START_STEP=2500
TRACK3_SOFT_MUON_END_STEP=3010
TRACK3_SOFT_MUON_CEIL=0.80
```

## Historical Read

The historical evidence suggested that LocoProp-M created early/mid-run
loss improvements, but the later handoff/suffix has not preserved those gains.
The refresh3 plot shows the failure mode clearly: the LocoProp suffixes cluster
around `3.302-3.306 @3000`, while the WR/reference means continue descending to
about `3.281 @3000`.

## Historical Model-State Handoff

The historical direct current-record WR + LocoProp-M hook integration OOMed
before step 0 on Prime. That motivated a model-state handoff:

```text
simple Track 3 + LocoProp-M prefix checkpoint at step 1600
-> current-record WR source suffix with no LocoProp hooks
```

Use:

```bash
WR_RESUME_DRY_RUN=1 bash tools/run_wr_record_resume.sh
```

for local generation/compile, and:

```bash
RUN_LABEL=wr_resume_locom1600_modelonly_seed3710 \
WR_RESUME_CHECKPOINT=/root/.cache/track3_checkpoints/track3_cd500red_softmerge_pr2872000_p110_ckpt1600_seed3710_step1600.pt \
WR_RESUME_LOAD_ADAM=0 \
WR_RESUME_RESTORE_RNG=1 \
WR_RESUME_ADVANCE_DATA=1 \
WR_TRAIN_STEPS=3040 \
WR_SCHEDULE_STEPS=3105 \
WR_SEED=3710 \
WR_TARGET_LOSS=3.28 \
NPROC_PER_NODE=1 \
bash tools/prime_wr_record_resume_remote.sh
```

on a synced Prime H100/GH200 pod. The rationale and launch gates are recorded
in `.opencode/track3_wr_resume_handoff_plan_20260603.md`.

## Validate the Fixed Integration

Local correctness checks require pytest and the repository's PyTorch version:

```bash
python -m pytest -q tests
TRACK3_DRY_RUN=1 bash tools/run_track3_locoprop_m.sh
WR_LOCOM_DRY_RUN=1 bash tools/run_wr_record_locoprop_m.sh
```

The regression suite covers zero-gradient BF16 replay, matching-objective
gradients, backtracking, final-iterate acceptance, post-outer-step rejection,
bounded capture, native MLP gradients, fullgraph compilation (including CPU
Inductor), schedule preservation, and two-rank Gloo collectives/checkpoint restore.
The local sharded fused-MLP test uses a CPU substitute for the Triton kernel.
The October 6 GPU checks below additionally exercise the actual CUDA kernels.
NCCL and end-to-end sharded-bank training still require multi-GPU validation.

On an already provisioned GPU, first run a short May 9 record smoke test:

```bash
WR_TRAIN_STEPS=10 WR_SCHEDULE_STEPS=3105 WR_SEED=3710 \
WR_LOCOM_COMPILE=1 SCREEN_VAL_EVERY=5 \
bash tools/run_wr_record_locoprop_m.sh
```

Then compare a 125-step screen against `WR_LOCOM_ALPHA=0` using the same source,
seed, microbatch size, sample capture, compilation and schedule horizon. Keep
LocoProp enabled in both arms so the control includes capture/solve overhead.
Check finite training loss, peak memory, step time, acceptance rates, and validation
loss before extending to full runs and multiple matched seeds. Local objective
decrease establishes a solver invariant; it does not establish a final validation
loss advantage over the record optimizer.

The May 9 WR record adapter honors `TRAIN_PROGRESS_INTERVAL` (default 0,
nonnegative) and `SCREEN_VAL_EVERY` (default 125, positive). These previously
remained fixed constants despite the environment configuration. The Prime WR
launchers default to PyTorch 2.11 with CUDA 12.8 wheels. CUDA 12.6 NVRTC cannot
compile the root fused cross-entropy kernel's `__tanhf` intrinsic; the full
`triton_kernels.py` import and fused probes passed with CUDA 12.8 on driver
570.148.08. Preserve that driver/runtime compatibility when changing the image.

## October 6 GPU Validation

The repaired implementation passed actual H100 checks for BF16 native-equivalent
MLP forward/backward, exact zero correction for a zero target gradient, fullgraph
compiled capture across changing active windows, finite high-curvature
backtracking, Newton-Muon covariance hooks across eager refresh/compiled ordinary
steps, the flattened Triton fused MLP, and BF16 mantissa-preserving optimizer
application. CPU tests additionally cover two-rank collectives and checkpoint
restore. These are correctness checks, not evidence of a final loss advantage.

Short May 9 screens used seed 3710, 524,288 global tokens/update, microbatch 64,
the source PR287 horizon 3105, four local steps, proximal coefficient 0.1, and
PyTorch 2.11/CUDA 12.6 on one H100 SXM5. Both arms enabled capture and solving;
alpha 0 disabled application in the control. All completed without OOM or
nonfinite loss. The 10-step smoke peaked at 28.94 GB allocated. Selected
125-step results:

| Alpha | Target gamma | Samples | Norm cap | Validation loss | Training seconds | Peak allocated GB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1024 | 0.20 | 4.51579 | 208.848 | 28.96 |
| 1 | 1 | 1024 | 0.20 | 4.52038 | 225.781 | 28.96 |
| 1 | 100 | 1024 | 0.20 | 4.53044 | 211.293 | 28.96 |
| 1 | 300 | 1024 | 0.20 | 4.54275 | 210.461 | 28.96 |
| 1 | 100 | 8192 | 0.20 | 4.53975 | 308.458 | 31.31 |
| 1 | 100 | 8192 | 0.05 | 4.51961 | 293.355 | 31.31 |

Times include compilation and are not warmed throughput measurements. The
larger sample budget improved diagnostic gradient alignment but did not establish
a validation gain. Gamma 1 produced mostly tiny or rejected corrections; larger
gamma made corrections active and sometimes degraded early validation. The outer
optimizer can already move beyond the local target, causing the application gate
to reject an otherwise accepted inner solve. A stable local solve alone does not
justify larger additive corrections.

A full 3040-step comparison completed with the same seed/source/data revision
and schedule, CUDA 12.8, samples 8192, gamma 100, cap 0.05, and alpha 0 versus 1.
Two independent H100 SXM5 allocations ran concurrently. Both exited successfully,
without OOM or nonfinite loss, and peaked at 31.31 GB allocated / 34.38 GB reserved.

| Arm | Final validation loss | Training seconds | Average step ms |
| --- | --- | --- | --- |
| Alpha 0 control, capture/solve enabled | 3.27925 | 6190.031 | 2036.19 |
| Alpha 1 repaired correction | 3.27894 | 8026.607 | 2640.33 |

The correction finished 0.00031 lower in loss, with 29.67% more training time.
This one-seed difference is too small to establish a useful quality advantage.
The result validates full-run numerical stability for this configuration; it does
not establish multi-GPU stability, statistical significance, or a record. The
control includes capture/solver overhead, so these times do not measure total
overhead against the unmodified record optimizer.

Results and provenance are summarized in `docs/locoprop_gpu_validation_20261006.json`.
The supervisor collected SHA256-verified logs/generated scripts and confirmed both
task-owned allocations were `TERMINATED`. Estimated total compute cost, conservatively
using creation-to-termination time, was $22.05; this is not an invoice. No resumable
checkpoint or model export was configured for the full comparison. Controller
artifacts live outside the repository in `../experiments`, including the full
configuration, resource ledger, result archives, and termination confirmations.
