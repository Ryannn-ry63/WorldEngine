# Rare-V3 Hard-Set Selective Reranking V1

## Revised diagnosis

The current evidence does not justify saying that the selector is simply too
weak, nor that an additional interaction network is already the answer.
Rare-tuned V3 has two distinct properties:

1. It retrieves useful candidates well.  On the existing seed-0 train cache,
   the selected PDM is `0.879732`; the top-5 oracle is `0.936275`, which retains
   about `72.1%` of the gap from V3 to the 20-candidate oracle (`0.958155`).
   Top-8 raises this ceiling to `0.946094` (`84.6%` of that gap).
2. It often fails to choose the best candidate inside its own shortlist.  The
   selected trajectory exactly matches the 20-candidate oracle only about
   `25.3%` of the time, while the top-5 contains that oracle about `59.3%` of
   the time.

Therefore the next question is narrower and falsifiable: **inside the fixed
rare-V3 top-5, is there log-generalizing information that can safely override
V3's rank-1 choice?**  This is an information/objective audit before another
full selector is trained.

## Locked experiment

This audit consumes only the already materialized schema-v4 train caches for
noise seeds 0, 1, and 2.  It does not rebuild data, and it does not read any
development or certification result.

For every token/noise view, frozen rare-tuned V3 first ranks the valid candidates.
Its top-5 is then locked without consulting PDM reward.  All ten unordered
pairs in that set become training examples; PDM is used only as the pair label,
regret weight, and final metric.

Five arms use the same antisymmetric comparison MLP and the same optimization
budget:

| Arm | Candidate input | Relation input | Interaction input | Role |
| --- | --- | --- | --- | --- |
| H0 | frozen V3 token + V3 logit | zero | zero | simplest evaluator |
| H1 | same | aligned trajectory relation | zero | relational increment |
| H2 | same | aligned trajectory relation | aligned frozen-track summary | interaction increment |
| S1 | same | within-example shuffled relation | zero | matched relation control |
| S2 | same | aligned relation | same-stratum donor-track interaction | matched interaction control |

The comparator is antisymmetric by construction.  Its complete top-5 pair
graph nominates one challenger by Borda score.  V3 remains selected unless an
affine monotonic calibrator says the challenger-vs-incumbent evidence is
positive.

## Leakage and statistics contract

- Outer evaluation is five-fold by audited log.
- For each outer fold, the following fold is calibration-only and the other
  three folds train the comparator.
- A token's three noise views always share a fold.
- Selection PDM, pair AUC, and all promotion decisions are out-of-fold.
- Uncertainty is a paired bootstrap over logs, never over individual pairs.
- Top-5 is the method budget.  Top-8 is reported only as a ceiling.
- PDM is primary.  Reward components are not model inputs and do not become a
  new scalar training target.

## Pre-registered gate

An evaluator arm is eligible only if all conditions hold:

- equal-weight common/rare PDM improves over V3 by at least `0.005`;
- common PDM is no worse than V3 minus `0.005`;
- rare PDM is no worse than V3;
- the cached hard-safety proxy (no-at-fault collision score times drivable-area
  compliance) loses no more than `0.002` on either common or rare;
- at most `10%` of common selections are degraded;
- the log-bootstrap lower 95% bound of PDM gain is positive;
- noise-view selection agreement is no worse than V3 minus `0.02`.

Relations count as new information only if H1 gains at least `0.01` pair AUC
over both H0 and S1, the paired-log lower bound is positive, and neither common
nor rare AUC loses more than `0.01`.  Interactions use the identical rule for
H2 over both H1 and S2.

The simplest eligible arm wins.  A more complex arm replaces it only with at
least `0.002` extra equal-stratum PDM and a positive paired-log lower bound.

## Machine decisions and mandatory next action

| Gate decision | Interpretation | Only authorized follow-up |
| --- | --- | --- |
| `AUTHORIZE_TOKEN_TOP5_EVALUATOR` | V3 already encodes the missing signal | train/freeze the H0-style selective evaluator |
| `AUTHORIZE_RELATIONAL_TOP5_EVALUATOR` | aligned candidate relations add real signal | train H1; do not add tracks |
| `AUTHORIZE_INTERACTION_TOP5_EVALUATOR` | tracks survive both simpler and shuffled controls | train H2 |
| `AUTHORIZE_LOCKED_TOP1_OBJECTIVE` | pair ordering is learnable but the graph/override cannot convert it | redesign only the incumbent-vs-challenger objective/calibration |
| `AUTHORIZE_HISTORY_AUDIT` | current-frame top-5 information is insufficient | audit temporal/history information before any new selector |

None of these decisions authorizes development or certification evaluation.
The selected architecture must first be materialized under a separate fixed
training contract; only then is a single development gate designed.  Closed-
loop Success Rate is unavailable in the train cache and is deliberately not
imputed here; it remains a mandatory floor in that later closed-loop gate.

## Run

From the worktree root on any instance with at least one H100/H200:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
HARD_PAIR_GPU=0 ./run_diffusiondrive_selector_hard_pair_audit_1gpu.sh
```

The command isolates one GPU even on an eight-GPU instance.  Its only decision
artifact is:

```text
experiments/diffusiondrive/hard_pair_audit_v1/audit/audit_<UTC>/hard_pair_gate.json
```
