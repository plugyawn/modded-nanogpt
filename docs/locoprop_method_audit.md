# LocoProp method audit — October 6, 2026

This audit describes the repaired implementation at commit
`4ad8e7ea311ed43cf1c6264204c2b9bbd5950d6d` and the completed May-record comparison.
Later ablations must record their additional changes separately. The existing
method is a sampled, additive correction to MLP input weights. It has not yet
tested LocoProp as the optimizer for the whole transformer.

Primary references are the [AISTATS 2022 paper](https://proceedings.mlr.press/v151/amid22a/amid22a.pdf)
and the authors' [official training notebook](https://github.com/google-research/google-research/blob/master/locoprop/locoprop_training.ipynb).
The paper's Algorithms 1–2 define local updates; Figure 3 studies iteration count
and local optimizer; Appendices G and I discuss other activations and test loss.

## The local target and matching objective are correct

For captured layer inputs `X`, preactivations `a0`, and backpropagated
`g = dL/da`, the M variant sets a postactivation target

\[
y^* = f(a_0) - \gamma g.
\]

For this model, \(f(a)=\operatorname{ReLU}(a)^2\) and its convex integral is
\(F(a)=\operatorname{ReLU}(a)^3/3\). With
\(a(\Delta)=a_0+X\Delta^\top\), the implemented relative objective is

\[
J(\Delta)=\frac{1}{N}\sum_{i=1}^{N}
\left[F(a_i(\Delta))-F(a_{0,i})
-f(a_{0,i})^\top(a_i(\Delta)-a_{0,i})
+\gamma g_i^\top(a_i(\Delta)-a_{0,i})\right]
+\frac{\lambda}{2}\|\Delta\|_F^2.
\]

Here `F` sums its scalar potential over hidden features. Its gradient is

\[
\nabla_\Delta J=
\frac{[f(a_0+X\Delta^\top)-f(a_0)+\gamma g]^\top X}{N}
+\lambda\Delta.
\]

At the initial iterate, one SGD step gives
\(\Delta_1=-h\gamma g^\top X/N\), where `h` is the inner solver learning rate.
This recovers a scaled sampled weight-gradient step. The use of the gradient
with respect to **preactivation** to form a **postactivation** target is
intentional. Multiplying the local residual by another derivative of squared
ReLU would change the M method.

See [`loco_matching_objective`](../tools/locoprop_local.py) and `loco_solve`.
Anchoring at captured BF16 preactivation avoids an artificial replay residual
when the target gradient is zero. It remains a local displacement surrogate,
not an exact simulation of BF16 rounding in the subsequent full-model forward.

## What the completed experiment actually optimizes

The [May-record adapter](../tools/make_wr_record_locoprop_m.py) solves the local
problem at the pre-update weights, takes the full outer optimizer step, and then
adds a capped local displacement:

\[
W_{t+1}=W_t+\Delta_{\rm base}+s\Delta_{\rm local}.
\]

The completed run used a norm cap of 0.05, so
\(s\|\Delta_{\rm local}\|_F\leq0.05\|\Delta_{\rm base}\|_F\).
The post-step gate can shrink or reject that displacement. This is a conservative
hybrid with the May stack, rather than replacing the outer update with the local
solution. Muon/SOAP can already move beyond the frozen local target, making an
otherwise descending local correction unhelpful after the outer step.

Only the twelve `mlp.fc.weight` matrices receive the correction. They contain
\(12\times768\times3072=28,311,552\) weights: **17.44%** of this model's
162,354,816 parameters. MLP output projections and biases, attention Q/K/V/output
matrices, embeddings, readout, and normalization parameters keep their original
optimizers. The original layerwise construction and official autoencoder code
optimize all affine layers, including biases. Adding the correction to the
remaining matrices would require defining and validating their local problems;
the same elementwise activation objective cannot simply be applied to an entire
attention block.

The full comparison is therefore evidence about a narrow hybrid. It cannot
answer whether full LocoProp replaces most of the May stack effectively.

## Four SGD iterations leave an important experiment missing

At the audited commit, `WR_LOCOM_STEPS` defaults to four and can be overridden.
The May adapter omits `local_opt` and `lr_decay` when calling `loco_solve`, so it
always uses the shared solver's **SGD** choice and disables its local decay.
The shared solver already implements RMSProp; the Track3 generator exposes
RMSProp, persistent state, and local decay, but the May adapter does not.

The supplied image reproduces the paper's Figure 3(b,c): the left panel compares
M/S variants and iteration counts; the right panel shows RMSProp clearly ahead
of SGD for that autoencoder. Both plot **training loss against epochs**. They
support testing the inner optimizer and sufficient local iterations. They do
not establish a wall-clock advantage, test-loss advantage, or the same ordering
for this transformer. The paper selects ten local iterations and RMSProp for
its principal experiments after extensive tuning.

Our four-step SGD solve has no residual-norm convergence criterion. Backtracking
establishes descent of its local objective, but does not establish proximity to
the local minimizer. `inner_lr` is a numerical solver step; `prox` determines the
objective's trust penalty. For the averaged objective above, `prox=0.1`
corresponds to an inverse trust parameter of 10. Those are distinct levers.
A useful next experiment compares SGD4 with RMSProp10 while holding source,
seed, global batch, target scaling, and base optimizer fixed, and records local
solve residual, applied norm, acceptance, time, and held-out loss.

### TensorFlow arithmetic is now an explicit opt-in

The official notebook defaults to momentum **0.999**, squared-gradient decay
**0.9**, epsilon **1e-5**, and ten local iterations. These numerical labels match
both local RMS styles. See its [flag defaults](https://github.com/google-research/google-research/blob/master/locoprop/locoprop_training.ipynb#L85-L96),
[parameter mapping](https://github.com/google-research/google-research/blob/master/locoprop/locoprop_training.ipynb#L165-L170),
and [RMSProp configuration](https://github.com/google-research/google-research/blob/master/locoprop/locoprop_training.ipynb#L465-L471).

Its optimizer is created once per layer, not once per minibatch. The local
iteration counter resets, but slots persist: `reset_optimizers=False` and its
reset call is guarded. See notebook JSON source lines 310, 325, 500, and 558–559.
The graph receives initial variable initialization at line 523.

The [legacy TensorFlow implementation](https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/python/training/rmsprop.py#L170-L181)
initializes the squared-gradient slot to one and momentum to zero. Its
[update formula](https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/python/training/rmsprop.py#L22-L24)
places epsilon inside the square root and accumulates learning-rate-weighted
momentum. The resulting differences are material:

| Choice | Official notebook / legacy TensorFlow | Retained `torch` style | New `tensorflow` style |
| --- | --- | --- | --- |
| Initial squared-gradient slot | One | Zero | One |
| Initial momentum slot | Zero | Zero | Zero |
| Denominator | `sqrt(v + epsilon)` | `sqrt(v) + epsilon` | `sqrt(v + epsilon)` |
| Momentum recurrence | `m = beta1*m + h*g/denominator`; update `-m` | `m = beta1*m + g/denominator`; update `-h*m` | LR-weighted as in TensorFlow, recomputed from the original slot at each trial LR |
| Local learning-rate schedule | `h_base * max(1-j/T, 0.25)` | Legacy behavior retained | Reference LR times that fraction, with a separate backtracking multiplier |
| State across minibatches | Persistent | Accepted application only | Accepted application only |

Select `WR_LOCOM_RMS_STYLE=tensorflow`, `WR_LOCOM_LOCAL_OPT=rmsprop`,
`WR_LOCOM_STEPS=10`, and `WR_LOCOM_LR_DECAY=1` for the new guarded arm. Default
style remains `torch` to preserve existing solver behavior. Tests compare the
TensorFlow style against independent double-precision recurrences, including
backtracking, variable learning rates, persistent state, and reset to ones.

This matches the reference RMS arithmetic, not the entire notebook experiment.
Armijo backtracking, a gradient-descent fallback for non-descending momentum,
exactly zero correction on a zero gradient, the explicit proximal penalty,
sampled FC-only coverage, caps, and accepted-application-only state commitment
are our additional choices. The official notebook's local gradient loops contain
no explicit proximal-gradient term. Targets and learning rates also use different
loss-reduction conventions, as discussed below. Neither local RMS style inherits
the supplied figure's measured performance.

## Target scaling depends on the source driver's loss convention

The [May source](../records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt)
uses `cross_entropy(..., reduction="sum")`, calls `loss.backward()` directly,
and sums parameter gradients across ranks. Its captured `g` has no global
token-count divisor. Averaging rows in the local objective consequently gives
an estimate of the per-token mean gradient. There is no additional hidden
division of captured gradients by 524,288 in this adapter.

The official notebook instead averages its global loss over examples and sums
the local residual gradients. Its numerical activation learning rate and
inner learning rate therefore cannot be copied without translating reductions.
For an official batch of `B` examples, its captured activation gradient carries
a `1/B` factor; using a summed global loss requires adjusting the target step
and the local-gradient reduction to preserve the same local problem and step.
Sample budget alone should not rescale our local objective because it averages
rows, although changing the samples changes estimation noise.

There is a separate portability gap in the sharded root driver:
[`train_gpt.py`](../train_gpt.py) multiplies the summed training loss by
`grad_scale = 1 / grad_accum_steps`. The
[bank adapter](../tools/make_wr_locoprop_m.py) captures that scaled `dpre` and
does not undo the factor before setting the local target. Changing accumulation
therefore changes the effective target strength unless gamma compensates.
This requires an explicit target-normalization contract and a regression check.
It is not an explanation for the completed single-GPU May-source comparison.

## The post-step gate has an affine-offset approximation

The audited May gate measures the actual FC **weight** displacement from its
reference but omits the FC **bias** displacement taken by AdamW in the same
outer step. It evaluates `pre0 + X @ base_delta.T`, rather than including
`base_bias_delta`. Thus it is a frozen-bias surrogate gate, not a check of the
entire affine layer at its new state. Capturing the reference bias and including
its offset in baseline and combined objective evaluation would fix this gap.
Bias should remain an offset when evaluating a weight-only correction; it
should not be included in the correction's weight norm cap.

Even with that fix, upstream weights alter the next forward's inputs, other
blocks move concurrently, and BF16 casting changes effective weights. A local
decrease or nonnegative cosine with the full gradient is not a global
cross-entropy descent guarantee. Global train/validation loss remains the
necessary outcome check.

## A large gain is a hypothesis to test

The paper's evidence is centered on tuned autoencoder optimization, including
strong tanh results. Its ReLU appendix also contains cases where early gains
do not yield a large final gap. It discusses test loss separately and does not
establish transformer generalization gains.

The May optimizer already includes substantial conditioning machinery. Whether
LocoProp adds complementary curvature information is an empirical question.
Current sampling, coverage, solver choice, cap, and outer-step interaction all
limit that question. The repaired full run established numerical stability for
one configuration; its 0.00031 validation-loss advantage over a solver-enabled
control does not establish a useful quality improvement.

The most informative sequence is a matched simple-Muon comparison, an inner
SGD/RMSProp ablation, then selected optimizer-replacement and broader-coverage
experiments. A claim about LocoProp functioning independently needs replacement
experiments, not only additive improvements over Muon.
