# Matched Muon and LocoProp experiments

`tools/make_wr_record_muon_ablation.py` generates every arm from the repaired
May 9 record source. Architecture, initialization, Adam parameter groups,
data order, and the May 9 learning-rate schedule remain common. The simple
profile changes the hidden-matrix optimizer to the original simplified Track 3
Muon update; it does not recreate that older script's initialization or schedule.

| Profile | Change from `simple` |
|---|---|
| `simple` | Frobenius-normalized 12-step NS polar iteration, Nesterov momentum, aspect-ratio scaling, decoupled weight decay |
| `contra` | Adds the record's Contra-Muon schedule |
| `soft` | Adds the record's late Soft-Muon schedule |
| `soap` | Adds SOAP for MLP and attention V weights |
| `normuon` | Adds row/column second-moment normalization |
| `floor` | Adds the update/weight norm floor, while retaining decoupled weight decay |
| `polar_norm` | Uses the record's Gram-Frobenius polar input normalization |
| `no_wd` | Disables decoupled weight decay |
| `may9` | All five record mechanisms, Gram normalization, and disabled decoupled weight decay |

The last two switches are necessary to make the control honest: the original
simple optimizer and May 9 record use different polar input normalization, and
the record stores a weight-decay parameter without applying it. The `may9`
profile reproduces the native record's weights and optimizer state. Each
individual profile changes one setting from `simple`.

Prepare an explicit experiment list without launching training:

```bash
python tools/run_wr_muon_ablation.py prepare \
  --output /absolute/path/to/ablation-plan \
  --profiles simple,may9 --seeds 3710 \
  --solvers sgd4,rmsprop10 --native-timing
```

This produces a self-contained generated training script and a manifest with
source hashes, the pinned dataset revision, and ten arms. Each solver has an
`alpha=0` control and `alpha=1` additive-correction arm. Controls retain capture
and local-solver overhead. Native timing arms disable capture, local solvers,
their buffers, and custom MLP autograd; the MLP uses its original native forward
and backward. Native timing therefore measures the additional capture and solve
cost omitted by an `alpha=0` comparison.

Run exactly one arm on already provisioned GPUs, with the Python environment
containing the desired PyTorch installation:

```bash
python tools/run_wr_muon_ablation.py run \
  --plan /absolute/path/to/ablation-plan/plan.json \
  --arm simple_alpha1_rmsprop10_seed3710 \
  --workdir /absolute/path/to/modded-nanogpt --gpus 1 --checkpoint
```

The runner verifies the generated script hash, clears inherited experiment flags,
records the effective environment and console checksum, and refuses duplicate
launches. It neither provisions resources nor automatically retries failed jobs.
Dataset files must already be staged in the working directory. `--checkpoint`
enables an atomic recovery point every 500 updates and at the final update,
retaining the last two checkpoints. It saves the model, persistent RMS buffers,
every optimizer state and custom step counter, CPU/CUDA RNG, and the next data
shard/token position. Full shard hashes, the generated script hash, seed,
optimizer/local-solver settings, schedule, horizon, and PyTorch/CUDA versions
must match on resume. Checkpointing currently requires one GPU.

Use `--resume-checkpoint /absolute/path/to/step_000500.pt` with the same generated
artifact and arm to continue after initialization, before the training loop. Data
position restores directly, preserving the native shard-tail rule without
replaying or transferring old batches. Model compilation may run again after a
restart; this mechanism preserves state and RNG but does not promise bitwise
CUDA replay on nondeterministic kernels or across runtime/hardware changes.
Checkpoint IO is excluded from the record's training-time metric. Before deleting
an ephemeral GPU allocation, preserve requested recovery checkpoints or explicitly
record their intentional discard under the agreed retention policy. The October 6
pool keeps the last two checkpoints on active GPUs only; its local archives retain
metrics, configurations, generated code, provenance, and checkpoint checksums.

`--checkpoint-stop-file /absolute/path/to/checkpoint_stop.request` lets a budget
supervisor request a current recovery point. After observing that marker at the
end of an outer update, the driver atomically saves the next step and exits with
an explicit `reason=stop_file`. This is an interrupted run, rather than a final
validation result. The option requires `--checkpoint`.

For a short replay check, prepare one 20-step artifact, then run with
`--checkpoint --checkpoint-every 10 --checkpoint-exit-step 10`. Resume its step-10
checkpoint without the exit flag and compare the resulting state to a continuous
20-step run of the identical artifact. The scheduled horizon remains unchanged.

Full-run defaults are 3040 updates, an unchanged 3105-update schedule horizon,
524288 global tokens per update, sequence length 1024, and 64 sequences per
microbatch. The guarded RMSProp10 arm uses at most ten local steps, inner learning rate 0.1,
gamma 100, proximal strength 0.1, beta1 0.999, beta2 0.9, epsilon 1e-5, persistent
RMS state, TensorFlow-style RMS semantics, and linear local learning-rate decay.
Its squared-gradient estimate starts at one, epsilon sits inside the square root,
and the momentum accumulates learning-rate-weighted updates. State commits only
after a stored correction is accepted. The SGD4 arm uses at most four local
steps with the same target, learning rate, and proximal strength. Both use 8192
sample tokens and a 5% cap relative to the actual native FC weight displacement.
These are controlled initial settings, rather than a claim of optimal tuning.

`--replacement` adds a guarded `partial_fc_replacement` pilot for each profile.
It skips the native weight and weight-decay update for selected active `c_fc`
matrices, while retaining their global gradients and optimizer state updates.
The solved local displacement supplies their weight update. Native FC biases
and every other parameter keep their native updates. The gate includes the FC
bias displacement and verifies the actual stored weight candidate. Replacement
uses a separate cap of 1% of the reference FC weight norm; setting
`WR_LOCOM_REPLACE_NORM_CAP=0` disables that additional cap. The additive cap is
inapplicable because a skipped native FC step has zero displacement.

This partial replacement is not a complete paper implementation of LocoProp.
An additive result tests an extra local correction on top of Muon; a replacement
pilot tests one subset of local weight updates. Multiple matched seeds are needed
before treating small validation-loss differences as evidence of a benefit.

## October 6 admission evidence

All twelve 125-update pilots completed with finite validation loss on a pinned
single-H100, PyTorch 2.11/CUDA 12.8 runtime. Against their matching capture-and-solve
controls, RMSProp10 additive corrections changed validation loss by -0.00381 for
simple Muon and -0.00491 for May9; SGD4 changed it by +0.00267 and +0.00765. RMSProp
additive arms remained slightly worse than their native timing arms at this early
step. These are exploratory single-seed observations, not evidence of a full-run
quality gain. Exact configurations, timings and verified archive checksums are in
[`locoprop_pilots_20261006.json`](locoprop_pilots_20261006.json).

A fresh CUDA process restored all 586 model/optimizer/local-solver state tensors
bitwise, together with loader position, RNG and counters. Independent compiled
training trajectories already differed before a resume, so those trajectories
do not provide an isolated checkpoint serialization test. See the
[`method audit`](locoprop_method_audit.md#cuda-trajectory-differences-and-checkpoint-restoration-are-separate-checks).

The original frozen full queue contained twelve 3040-update initial arms, eight additional
RMSProp/control arms for seeds 3711 and 3712, and fourteen single-component
RMSProp/control arms. These retain the 3105-update learning-rate schedule. Seed
replications are descriptive; the seven component comparisons remain exploratory.
That broad queue was superseded by the targeted work below. Interrupted and
unstarted entries retain their actual status and cannot become full-run results.

## Targeted repair after the unfavorable pilot

The broad queue was superseded after the user requested a repair of the
quality/time tradeoff. Five experimental full runs were checkpointed and stopped;
their partial metrics are not full-run results. The native simple-Muon baseline
was retained. The targeted 500-step comparison completed native May9 and four
repaired configurations with finite final validation:

| Configuration (seed 3710) | Validation loss | Recorded training seconds | Overhead vs native |
| --- | ---: | ---: | ---: |
| Native May9 | 3.82535 | 737.770 | — |
| Safe RMS10, pre-native center | 3.82082 | 863.052 | 17.0% |
| Safe RMS10, post-native center | 3.82557 | 871.788 | 18.2% |
| Post-native, every 4 updates, 5% cap | 3.82490 | 790.460 | 7.1% |
| Post-native, every 4 updates, 20% cap | 3.82334 | 790.248 | 7.1% |

These single-seed equal-token differences do not establish a reliable quality
benefit or a win at equal wall time. Recorded training time excludes validation
and checkpoint IO. The 500-step comparison must not be pooled with the old
125-step timings. Configurations and verified archive checksums are in
[`locoprop_repair_20261006.json`](locoprop_repair_20261006.json).

The logged May9 RMS pilot applied only 12 of 36 proposed corrections at the
recorded nonzero-gradient snapshots. Some inner solves also exhausted 21 trials
because shrinking the new gradient contribution did not shrink carried
TensorFlow momentum. The following changes address these observed mechanisms:

- `WR_LOCOM_SCALE_MOMENTUM=1` shrinks the complete proposed momentum displacement
  during backtracking and commits the accepted displacement to its momentum slot.
  It preserves legacy TensorFlow arithmetic when no shrinkage is needed, while
  making a rejected proposal approach the current iterate as its scale tends to
  zero. This is an explicit safeguard, not a claim of exact paper arithmetic.
- The outer gate computes its directional derivative. Convexity of the matching
  objective makes a nonnegative derivative sufficient to reject every positive
  scale, avoiding twenty futile objective evaluations.
- `WR_LOCOM_CENTER=post_base` defers the local solve until after the native
  optimizer update. Both the solve and gate center their activation and proximal
  terms at that same affine state, including the native bias displacement.
  Captured inputs and global gradients remain frozen; this is an additional
  correction surrogate, not a new full-model backward pass or standalone LocoProp.
- On inactive scheduled updates, generated MLPs execute their native forward and
  backward graph. Sparse correction schedules reuse two compiled graph variants,
  avoiding capture machinery on every inactive update.

The `rmsprop10_safe` and `rmsprop10_post` solver presets use complete-displacement
backtracking and 1024 sampled tokens. The former keeps the original pre-update
center; the latter uses the post-native center. The targeted screen also compares
post-native corrections every fourth step at norm caps 0.05 and 0.20. Baseline
quality, actual applied updates and recorded training time must decide whether
any of these configurations should be promoted. No speed or quality win is
assumed from the code changes alone.

The next bounded screen compares full-gradient and sampled-linear-term versions
of the same pre-native objective for seeds 3710 and 3711, using the first screen's
seed-3710 sampled arm. It adds a native seed-3711 control and a seed-3711 repeat
of the sparse 20% post-native correction. The five new 500-step jobs reuse the
existing GPUs; each lane is configured to archive results and terminate its
resource when finished. No full-length follow-on candidate is admitted on the
basis of the small single-seed differences above.

`rmsprop10_fullgrad` sets `WR_LOCOM_LINEAR_TERM=full`: preserve the full mean-token
gradient as the objective's linear term and sample only its matching curvature.
This is an experimental change of objective, not a proven improvement. Its
derivation, limitations, and the larger residual-block redesign are in the
[`method audit`](locoprop_method_audit.md#rethinking-the-construction-for-transformers).
