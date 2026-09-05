# V3-IVPS: Incumbent-Verified Preference Selection

## 1. Frozen research question

The fixed generator already provides 20 useful trajectories, but post-trained
selectors can damage the strong frozen selector while trying to exploit rare
candidate headroom.  RAPG V1 isolated the failure: scalar tie-aware preference
training was beneficial, whereas the graph was not; the best learned selector
still changed 82.9% of common decisions and lost 0.0542 common PDM.

IVPS asks a more precise question:

> Can a selector separately learn which candidate to propose and whether that
> proposal has enough scalar-reward evidence to replace the frozen incumbent?

The generator, 20-candidate action set, base selector, official PDM definition,
and all inference inputs remain unchanged.  Reward values and reward components
are unavailable at inference.

## 2. Method

Let `b = argmax(l_ref)` be the frozen selector's incumbent.  A proposal head is
trained with the successful scalar tie-aware preference objective and produces
`p = argmax(l_ref + delta_proposal)`.  A separate verifier receives the proposal
and incumbent tokens, their directed trajectory relation, the frozen-logit gap,
and the proposal-logit gap.  It estimates whether candidate `i` improves over
`b`:

```text
t_i = sigmoid((PDM_i - PDM_b - margin) / reward_temperature)
L_verify = weighted_BCE(v_i, t_i),  i != b

final action = p    if p != b and v_p > 0
               b    otherwise
```

The verifier is trained on all candidate-versus-incumbent comparisons, weighted
toward high-reward and proposal-likely candidates.  It uses only official scalar
PDM.  Ties are neutral when `margin=0`; positive margins teach a minimum material
improvement.  The proposal is frozen while the verifier is trained, explicitly
separating candidate discovery from incumbent protection.

Both heads are zero initialized.  A fresh proposal reproduces the frozen base;
after loading a trained proposal, the zero verifier still rejects every
override because deployment uses the strict condition `v_p > 0`.  IVPS therefore
has an exact reference-preserving initialization without KL projection or a
component-specific safety rule.

The framing is related to learning-to-defer and selective prediction, but the
paper claim is narrower: scalar-reward post-training for a fixed learned
trajectory incumbent and a finite generated action set.  Relevant foundations
include calibrated learning to defer (ICML 2022) and decision-focused learning
as ranking (ICML 2022); neither is claimed as an IVPS invention.

## 3. Frozen development search

All architecture and margin choices are fixed before IVPS development results
are read.

### Stage A: proposal

- Architecture: `trajectory_set_reasoner`, `relational_only`.
- Objective: exact scalar GRPO + scalar tie-aware preference, weight `1.0`.
- Budget: 16 epochs, LR `1e-4`, batch 64, KL `1e-3`, train split only.

### Stage B: verifier

- Load and freeze the Stage-A proposal.
- Train only the verifier for 16 epochs on the identical 50/50 common-hard
  schedule.
- Raw scalar-PDM reward temperature: `0.05`.
- Predeclared improvement margins: `{0.00, 0.01, 0.03, 0.05}`.
- Deployment verifier threshold: fixed at logit `0`; it is not tuned on dev.

### Controls

- Frozen V3 incumbent.
- Stage-A proposal without verification (`B01-l1.00`).
- A 32-epoch proposal-only compute control.
- Oracle proposal verification, computed offline only as a diagnostic ceiling;
  it is forbidden in a deployable checkpoint.

## 4. Frozen offline gate

Evaluation uses the existing log-disjoint development common, real-rare, and
synthetic strata.  Certification remains unread.  An IVPS candidate is eligible
only if all conditions hold:

1. equal-stratum selected PDM is at least proposal-only `B01 + 0.005`;
2. common selected PDM is at least frozen reference minus `0.005`;
3. real-rare and synthetic selected PDM are each at least `B01 - 0.020`;
4. common degraded fraction is at most `0.10`;
5. common override precision is at least `0.50` whenever any override occurs.

Candidates are ranked by equal-stratum selected PDM only after all gates pass.
If no margin passes, stop IVPS; do not tune the threshold, reward temperature,
data mixture, or gates after observing development.

## 5. Formal boundary

Only a development-eligible IVPS margin may enter CL-dev.  The architecture and
margin are then locked.  Certification is consumed once, followed by paired
three-seed formal training/evaluation.  The formal CQR comparison and promotion
gate remain exactly those frozen in `REFERENCE_ANCHORED_PREFERENCE_GRAPH_V1.md`:
mean CL at least 0.800, mean CQR gain at least 0.005, 2/3 non-negative seeds,
per-seed floor -0.020, success at least 0.862, navtest at least 0.846, and rare
PDM at least 0.607.

This protocol does not authorize reward-component losses, common/rare labels as
model inputs, generator changes, more candidates, or certification tuning.

## 6. Post-freeze pipeline and ceiling check

After the protocol and gates above were frozen, the real B01-l1.00 proposal was
loaded into a one-epoch verifier smoke checkpoint.  This is not an IVPS result:
the verifier saw only 48 smoke examples and mostly retained the reference.  Its
purpose was to validate checkpoint loading, frozen-parameter isolation, offline
diagnostics, materialization, and full config loading.

The evaluator also computed a forbidden-at-deployment oracle: accept the B01
proposal exactly when its observed PDM exceeds the incumbent PDM.  It establishes
whether proposal verification has enough headroom to justify the full search.

| Development stratum | Frozen reference | B01 proposal | Oracle-verified B01 | Beneficial proposal fraction |
|---|---:|---:|---:|---:|
| common | 0.927024 | 0.872783 | 0.946521 | 0.340259 |
| real-rare | 0.285395 | 0.737103 | 0.765673 | 0.651927 |
| synthetic | 0.091691 | 0.803664 | 0.813977 | 0.834112 |
| equal-stratum mean | 0.434703 | 0.804517 | 0.842057 | — |

Thus the proposal and incumbent are complementary even on common data.  The
common failure is not absence of better B01 proposals; it is accepting the wrong
ones.  The oracle improves equal-stratum PDM by 0.037540 over proposal-only,
which makes the frozen `+0.005` IVPS gate meaningful without altering it.

Audit artifact:
`experiments/diffusiondrive/selector_ivps_v1/smoke/implementation_smoke_20260829/development_evaluation.json`.
