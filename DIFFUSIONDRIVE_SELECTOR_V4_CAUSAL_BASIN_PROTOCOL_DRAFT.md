# DiffusionDrive V4 Causal-Basin Selector Post-Training

Status: **DRAFT — AWAITING ASTRA REVIEW; METHOD OUTCOMES FORBIDDEN**

## Hypothesis

Scalar V3 fails for two coupled reasons: local candidate PDM is not the
closed-loop action value, and probability-weighted exact group-GRPO gives a
vanishing logit gradient to useful candidates already suppressed by V3.
Diffusion candidates also form correlated trajectory basins rather than 20
independent classes.

V4 tests the minimal claim that policy-relative causal value plus a top-1
consistent non-vanishing update improves selection while preserving a frozen
V3 incumbent.  Basin-aware encoding is a separately gated extension, not a
required part of the claim.

## Frozen deployment contract

```text
z_final = z_base + delta_V3 + delta_V4
action  = argmax over the same original 20 candidates
```

All V3 tensors are frozen.  The V4 final layer is zero initialized, so step-0
logits and actions exactly equal V3.  No reward, Q, component, source label or
simulator value is accepted at inference.

## Incumbent-Preserving Causal Top-1 objective

For candidate causal values `Q`, V3 logits `z3`, and incumbent
`b=argmax(z3)`, fix `epsilon=0.01`.

1. `O = {i | Q_i >= max(Q)-epsilon}`.
2. If `b in O`, choose target `t=b`; otherwise choose the member of `O` with
   greatest frozen `z3`, breaking exact ties by candidate index.
3. Let `J={j | Q_t-Q_j>epsilon}` and
   `w_j=(Q_t-Q_j)/sum_{k in J}(Q_t-Q_k)`.
4. Optimize `L_IPCT=sum_{j in J} w_j softplus(z_j-z_t)`.  Empty `J` gives
   exact zero loss.

The target receives a near-constant correcting gradient when its current
logit is far below an inferior candidate.  If V3 is already causally optimal,
the same loss teaches preservation against causally worse alternatives.

## Selector variants

- Unary control: frozen V3 candidate/context tokens followed by a
  capacity-matched residual MLP.
- Basin variant: permutation-equivariant aggregation over trajectory-nearest
  neighbors `k={2,4,8}`, using only relative XY/yaw/path descriptors and
  density-normalized messages.  Q/reward never defines graph edges or message
  weights.

The basin model must pass candidate-permutation equivariance and have a unary
control with parameter count within 5%.

## Fixed attribution matrix

| Arm | Label | Objective | New relation block | Role |
|---|---|---|---|---|
| A0 | none | none | none | frozen V3 |
| A1 | local PDM | exact group-GRPO | no | matched local baseline |
| A2 | local PDM | IPCT | no | non-vanishing update with old label |
| A3 | causal Q | exact group-GRPO | no | causal-label-only effect |
| A4 | causal Q | IPCT | no | core causal top-1 method |
| A5 | causal Q | IPCT | yes | full causal-basin method |

Training examples, optimizer steps, fold exposure and checkpoint selection are
identical.  Local controls freeze the first A-baseline local rewards even when
the known progress-normalization diagnostic drifts.

## Train-only gates

### Pilot: first 64 targets, four origin-log folds

At least one of A3/A4/A5 must achieve, relative to A0:

- mean selected-Q gain at least `0.01`;
- positive gain in at least three of four folds;
- failed-stratum gain at least `0.03`;
- solved mean gain at least `-0.01`;
- at most 5% solved targets with gain below `-0.1`.

Otherwise stop the method route and audit temporal observability.

### Full: 256 targets, five origin-log folds

An arm is eligible only with overall gain `>=0.02`, failed gain `>=0.05`,
solved gain `>=-0.005`, positive gain in at least four folds, origin-log
bootstrap 95% lower bound above zero, and at most 2% catastrophic solved harm.

Attribution gates:

- causal label: A3-A1 or A4-A2 `>=0.01` with bootstrap lower bound above zero;
- non-vanishing top-1: A4-A3 `>=0.005` and positive in four folds;
- basin: A5-A4 `>=0.005`, positive in four folds, bootstrap lower bound above
  zero.

If A5 fails the basin gate, select A4.  If A3 is the only eligible arm, select
A3 and drop the new objective claim.  Eligible arms within `0.005` use the
simpler order A3, A4, A5.

## Sealed development and formal route

Only the train-CV winner is retrained on all 256 rows and evaluated once on the
sealed 64-target causal cache.  It requires overall gain and cluster-bootstrap
lower bound above zero, failed gain `>=0.03`, solved gain `>=-0.01`, and at
most 5% catastrophic solved harm.

Only then run closed-loop development.  Relative to rare-tuned V3, the mean of
CL-NR/CL-R must improve by `>=0.005`, Success Rate may not fall by more than
`0.01`, and OpenLoop navtest/rare may not fall by more than `0.005/0.01`.

Offline-Q success followed by closed-loop failure is classified as repeated
deployment distribution shift.  It stops this protocol; it does not authorize
another loss sweep on the same development data.

After development success, freeze the method and run three optimizer seeds on
the standard four evaluation blocks.  Certification/test is consumed once.
Report Base, ordinary V3, rare-tuned V3, gate-conditioned, A1--A5 attribution,
candidate-change rate, V3-solved retention and causal-oracle regret.

## Astra review checklist

Before changing this document to `FROZEN_BEFORE_V4_OUTCOMES`, Astra must review:

1. whether one-step `Q^V3` supports the claimed policy-improvement semantics;
2. whether IPCT is the minimal top-1/non-vanishing mechanism;
3. whether basin aggregation adds a diffusion-specific contribution beyond
   generic relation attention;
4. whether target selection, folds, sealing and gates admit leakage;
5. novelty boundaries against critic-guided diffusion and sampled-action value
   methods;
6. the strongest likely reviewer counterexamples and required ablations.

Every accepted/rejected review item and its rationale must be added to the
decision ledger.  No A1--A5 result may be generated before that transition.
