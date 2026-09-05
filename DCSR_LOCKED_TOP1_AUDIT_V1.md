# DCSR Locked-Top1 Audit V1

## Why this is the final current-frame audit

The preceding train-only hard-pair audit established three facts under the
rare-tuned V3 incumbent:

- V3 top-5 retains `71.2%` of the available 20-candidate oracle headroom.
- Aligned trajectory relations add statistically significant pair information
  over both frozen V3 tokens and a matched shuffled-relation control.
- Candidate-agent interaction summaries add no incremental information, while
  all-pair/Borda reranking converts less than `0.5%` of the top-5 headroom.

The remaining hypothesis is a training/deployment mismatch: all-pair ordering
is not the deployed decision, which is whether one challenger should replace
V3 rank-1.  This audit tests that hypothesis once.  Failure permanently stops
current-frame architecture, pairwise, arbitration, and threshold variants.

## Difference from IVPS and PCRA

IVPS/PCRA used a separately trained residual proposal and then verified the
single candidate exposed by that proposal against the original reference
selector.  Their common failure was already entangled with proposal quality
and the old real/synthetic rollout mixture.

DCSR has no independent proposal.  Rare-tuned V3 first locks five candidates;
V3 rank-1 is the immutable incumbent.  One evaluator scores each of the other
four candidates against it, and the same evidence both nominates the
challenger and accepts or rejects the override.  The audit uses only the
current three train caches and treats pairwise verification as a non-promotable
historical control.

## Frozen arms and objective

All arms share one antisymmetric `257 + 257 + 41` comparator, hidden width 128,
512 AdamW updates, and approximately 1024 candidate edges per update.

| Arm | Input | Objective | Promotable |
| --- | --- | --- | --- |
| `P0_pair_token` | V3 token/logit | four regret-weighted BCE pairs | no |
| `P1_pair_relation` | token + aligned relation | four regret-weighted BCE pairs | no |
| `T0_structured_token` | V3 token/logit | structured top-1 | yes |
| `T1_structured_relation` | token + aligned relation | structured top-1 | yes |
| `C1_structured_relation_shuffle` | token + shuffled relation | structured top-1 | no |

For top-5 scores `s0=0, s1...s4`, reward-best index `y`, and official PDM
`r`, the structured loss is:

```text
max_i [s_i + (r_y - r_i)] - s_y
```

Thus training and inference share the same top-1 decision.  Reward ties across
the complete set are inactive; partial ties prefer the earlier V3 rank.  PDM
and components never enter model inputs.

## Log-disjoint calibration and gate

Outer fold `f` is test-only, fold `(f+1) mod 5` is calibration-only, and the
other three folds train the evaluator.  All noise views of one token remain in
the same audited log fold.

The calibration fold searches 101 non-negative evidence quantiles plus
keep-all.  It maximizes equal common/rare PDM subject to tighter calibration
floors: common PDM `-0.002`, rare PDM no drop, both hard-safety strata
`-0.002`, and common degraded fraction at most `0.05`.  Ties select the least
override coverage.  The test fold is touched only after the threshold freezes.

Final eligibility retains the preceding gate: equal PDM `+0.005`, common PDM
at worst `-0.005`, no rare drop, hard-safety at worst `-0.002`, common degraded
fraction at most `0.10`, positive paired-log bootstrap lower bound, and
noise-view agreement at worst `-0.02`.

The only decisions are:

- `AUTHORIZE_DCSR_RELATIONAL_V4`: T1 is eligible, beats P1/T0/shuffle by at
  least `0.002` equal PDM with positive paired-log lower bounds, and adds at
  least `0.01` boundary AUC over token and shuffle controls.
- `AUTHORIZE_DCSR_TOKEN_V4`: T0 is eligible and beats P0 by at least `0.002`
  with a positive paired-log lower bound.
- `STOP_CURRENT_FRAME_SELECTOR_AND_AUDIT_HISTORY`: every other outcome,
  including a gain confined to a pairwise historical control.

No decision in this audit consumes or authorizes development/certification.
A positive decision first authorizes three-seed materialization with a fixed
train/calibration split.  A stop decision authorizes only a temporal/history
information audit; it does not authorize another current-frame loss.

## Run

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
DCSR_GPU=0 ./run_diffusiondrive_selector_dcsr_audit_1gpu.sh
```

The runner isolates one H100/H200 and reuses all existing assets.  Its decision
artifact is:

```text
experiments/diffusiondrive/selector_dcsr_audit_v1/audit/audit_<UTC>/locked_top1_gate.json
```
