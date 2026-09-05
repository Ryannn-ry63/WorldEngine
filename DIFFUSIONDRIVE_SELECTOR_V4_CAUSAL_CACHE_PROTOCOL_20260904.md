# DiffusionDrive Selector V4 Causal Cache Protocol

Status: **FROZEN BEFORE V4 CAUSAL-CACHE OUTCOMES**

## Question and estimand

This protocol creates method-independent training labels for frozen-generator,
fixed-20 selector post-training.  For frozen scalar V3 behavior policy
`pi_3`, target state `s`, and original candidate `a_i`:

```text
Q^pi3(s,a_i) = final official reactive closed-loop PDM after executing a_i
               exactly once at s and returning immediately to frozen pi_3.
```

The intervention never changes perception, generator, candidate noise, base
selector, V3 selector, candidate count, or any later policy decision.

## Source contract

- Source root: `experiments/grpo_sources/diffusiondrive_v4_split0` in the
  canonical WorldEngine workspace.
- Allowed families: original `rare_union` and original `matched_common` only.
- Allowed splits: train for training labels; validation for sealed development.
- Forbidden: test, broad common, every augmented source, BWM and the prior 33
  CCV diagnostic targets.
- The existing whole-log split audit is authoritative.  Train, validation and
  test log sets must be pairwise disjoint.
- Behavior policy is rare-tuned scalar V3 train seed 0 with checkpoint SHA256
  `bd35f0293c878c6cddb985ac0b8aaae91d4f7f8e0bf2c3bf70ba49ba329c02a3`.
- Candidate noise namespace is frozen for collection and differs from later
  formal-evaluation namespaces.

## Outcome-blind target design

Two observe-only collections, A and B, are run before any treatment manifest
is created.  A scene is target-eligible only when candidate trajectories and
logits agree within `1e-5`, deployed actions pass the per-run `1e-4` audit,
categorical outcomes agree exactly, and continuous closed-loop score/ego
progress agree within `1e-3`.  Rejected-scene reasons and counts are retained;
they may not be replaced from another data family.

Local candidate reward is deliberately not part of this repeatability filter.
Its known pairwise-progress drift is reported after reward/Q-blind target
selection, and baseline-A local reward is frozen for the matched local-label
controls.  A different target may never be chosen because its local reward is
more repeatable.

Target selection may use behavior success/failure and first-violation timing,
but may not use local candidate reward, reward components, candidate causal
outcomes, or a learned score.

- At most one target per scenario and two targets per origin log.
- Failed scene: take the last three available decisions strictly before the
  first violation and choose one by SHA256 of the scene identifier.
- Solved scene: choose a decision by deterministically matching the failed
  decision-step histogram.
- Train target count is 256.  Select up to 128 failed targets, require at least
  64, and fill the remainder with solved targets.
- Sealed-development target count is 64.  Select up to 32 failed targets,
  require at least 8, and fill the remainder with solved targets.
- Within each outcome stratum, allocate equally to rare_union and
  matched_common when available.  A deficit may be filled only by the other
  allowed source in the same outcome stratum.
- Selection and pilot membership use fixed SHA256 salts.  The first 64 train
  targets are an immutable prefix of the full 256.

The source pool is deterministically capped at 512 train scenarios per source
and 128 validation scenarios per source before A/B rollout.  If the minimum
target coverage is unavailable, the experiment stops; no source substitution
is allowed.

## Collection stages

1. `preflight`: source/split/provenance audit and exact eight-Hopper contract.
2. `baseline`: A/B observe-only collection for train and validation pools.
3. `freeze_targets`: write immutable target and stage manifests.
4. `sentinel`: policy-index intervention on eight targets; reproduce baseline.
5. `pilot64`: collect all 20 actions for the first 64 train targets.
6. `pilot_audit`: assemble the causal surface and apply the method-independent
   information gate.
7. `expand192`: collect all 20 actions for the remaining train targets.
8. `dev64`: collect all 20 actions, assemble the cache, then seal it without
   model evaluation.

Every stage is resume-safe under the same run ID.  A completed artifact is
accepted only after its manifest SHA and collection audit pass.  Partial
artifacts with a different code or implementation hash are never mixed.

## Pilot information gate

Expansion is authorized only when:

- all 64 targets have exactly 20 successful candidate outcomes;
- all action/context/provenance checks pass;
- at least 32 targets have `max(Q)-min(Q) > 0.02`;
- at least 16 targets have `max(Q)-Q(policy) > 0.02`.

This gate may inspect the causal-label surface but may not train a selector or
change the V4 method draft.  Failure produces `STOP_V4_CAUSAL_CACHE`.

## Cache schema

Each row contains:

- candidate features `(20,256)`, trajectories `(20,8,3)`, route-BEV features
  `(20,8,256)`, status `(1,256)`, ego `(1,256)`, agents `(30,256)`;
- base/reference logits `(20)`, frozen V3 logits `(20)`, policy index;
- local official PDM `(20)`, diagnostic components `(20,6)`, causal Q `(20)`;
- scene, origin log, source family, split, target step, behavior outcome;
- checkpoint, selector state, candidate-noise, code, implementation, source,
  target and treatment SHA256 provenance.

Train and sealed-development are separate files.  The prior CCV surface is
never concatenated.  `development_consumed` remains false until the method
protocol is frozen and a single winner has been selected on train-only CV.
