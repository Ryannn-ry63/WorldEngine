# DiffusionDrive Selector V4 Implementation Runbook

Status: **IMPLEMENTED; SOURCE PASSED; BASELINE A/B PENDING**

This runbook is operational.  The scientific motivation and historical
evidence are recorded in the V3 diagnostic synthesis; the frozen collection
contract is in `DIFFUSIONDRIVE_SELECTOR_V4_CAUSAL_CACHE_PROTOCOL_20260904.md`.

## What this implementation is for

V4 first measures the missing supervision rather than immediately training
another selector.  For each frozen DiffusionDrive proposal at a selected
decision, it executes that proposal once and then returns control to frozen
V3.  The resulting final closed-loop score is the policy-relative causal
value `Q^V3(s,a)`.  This separates three questions that earlier experiments
mixed together:

1. Is the best local-PDM proposal also the best downstream proposal?
2. Does ordinary group-GRPO fail to recover a useful proposal after V3 has
   suppressed its probability?
3. After fixing labels and update geometry, is an explicitly diffusion-set
   (causal-basin) representation still necessary?

No new data family is introduced.  The source is a frozen subset of the
existing original rare/common train split, plus a disjoint sealed validation
subset.  Candidate generation, perception, the original 20 proposals and V3
remain frozen.

## One-machine execution

Run from the selector worktree on one instance with exactly eight visible
Hopper GPUs:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_selector_v4_causal_cache_8hopper.sh all \
  v4_causal_cache_20260904
```

The command is resume-safe under the same run ID.  It re-audits a complete
stage and skips it; it never accepts a partial collection or silently mixes
artifacts from a different code/provenance hash.

Use `all` only for an allocation comfortably longer than one day.  The prior
CCV run measured roughly 6.5--7 minutes for one 33-scene arm on eight Hoppers.
Linear scaling gives an intentionally rough serial budget of 8--10 hours for
the four A/B baselines, 4--5 hours for pilot64, 12--14 hours for expand192 and
4--5 hours for dev64, plus merge/audit overhead.  A 25--35 hour total is more
realistic than treating this as another quick offline probe.  Re-estimate from
the first completed V4 collection before scheduling the remaining arms.
The same CCV footprint implies roughly 300--400 GB for all V4 rollout
artifacts.  Check free space before `expand192`; on 2026-09-04 the relevant
filesystem had about 1.3 TB free, but that observation is not a quota promise.

The immediate recommended command is therefore the bounded baseline stage:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_selector_v4_causal_cache_8hopper.sh baseline \
  v4_causal_cache_20260904
```

For diagnosis or scheduled execution, stages can be called separately:

```text
source -> baseline -> freeze -> sentinel -> pilot64 -> pilot_audit
       -> expand192 -> dev64
```

- `source` is CPU/I/O work and can run without GPUs.
- `baseline` runs two observe-only repetitions before targets are chosen.
- `freeze` chooses targets without looking at candidate reward or causal Q.
- `sentinel` verifies that intervening with V3's own action reproduces V3.
- `pilot64` collects all 20 counterfactual actions for 64 train targets.
- `pilot_audit` is the first scientific decision gate.
- `expand192` is authorized only by a passing pilot and completes train256.
- `dev64` collects and seals validation labels, but does not evaluate a model.

One individual treatment can be resumed with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_selector_v4_causal_cache_8hopper.sh arm \
  v4_causal_cache_20260904 pilot64 0
```

Valid treatment stages are `pilot64`, `expand192` and `dev64`; candidate
indices are 0 through 19.

## Scientific stop/go decisions

### Gate C0: reproducible intervention machinery

The sentinel must reproduce the observe-only V3 trajectory and outcome.  A
failure is an implementation/provenance failure, not evidence against V4.  Do
not inspect method scores until C0 passes.

### Gate C1: causal surface contains learnable decision signal

The exact 64-by-20 pilot must contain at least 32 states with causal span
above 0.02 and at least 16 where V3 leaves more than 0.02 causal value on the
table.  Failure means that one-step `Q^V3` is not a useful estimand at this
resolution.  Stop this route and revisit the intervention horizon or target
state definition; do not compensate by sweeping losses.

### Gate M0: method review

Passing C1 authorizes cache expansion, not selector training.  Freeze the V4
method protocol after the listed Astra/novelty review.  This preserves the
paper attribution: the causal cache diagnoses the problem; A1--A5 test the
label, update and basin hypotheses with matched data and optimization.

### Gate M1/M2: train-only method selection

Use four-fold pilot and five-fold full origin-log cross-validation exactly as
specified in the method protocol.  Select a single eligible arm using the
predeclared simpler-model tie break.  Only that arm may consume sealed dev64.

### Gate D/F: deployment and formal evidence

Sealed causal-cache success is necessary but not sufficient.  The winner must
then pass one closed-loop development gate.  Only after that result is the
method frozen for three-seed formal evaluation and one-time certification/test
consumption.

## Interpretation matrix

| Observation | Conclusion | Next action |
|---|---|---|
| C1 fails | Current one-step causal label is weak or targets are temporally uninformative | Revisit estimand/target timing; stop loss search |
| A3 beats A1, A4 does not beat A3 | Causal labels matter; GRPO update is adequate | Retain A3; drop IPCT claim |
| A4 beats A3 and A2 does not beat A1 | Non-vanishing update helps specifically with causal labels | Retain A4 core claim |
| A2 and A4 both improve their GRPO controls | Update geometry is general; causal label contribution needs A3/A1 evidence | Narrow attribution accordingly |
| A5 fails to beat A4 | Basin block is unnecessary | Publish the simpler A4 route |
| A5 beats A4 under its gate | Correlated diffusion-set structure adds value | Retain full causal-basin V4 |
| Offline causal Q improves but closed loop does not | Cache-to-deployment distribution shift | Stop; audit observability/coverage, not another objective sweep |

## Implemented artifacts and checks

The implementation includes source freezing, target construction, treatment
manifests, the isolated simulator manager, collection audits, sentinel and
pilot gates, cache assembly/combination, the IPCT reference objective, a
decision ledger updater and the resume-safe runner.  Unit tests cover target
selection invariants, objective behavior and provenance helpers.  Before any
expensive collection, the runner checks the exact checkpoint, source split,
Hopper/CUDA toolchains, implementation hash and code hash.

The run-local source of truth is:

```text
experiments/diffusiondrive/selector_v4_causal_cache_v1/runs/
  v4_causal_cache_20260904/decision_ledger.json
```

Treat a terminal `PASS` line as valid only when its referenced JSON audit also
has `status: PASS`.  A traceback is fail-closed and must never be recorded as
a negative scientific result.

## Current frozen run state

Run `v4_causal_cache_20260904` completed its real source build and audit on
2026-09-04 UTC:

- train: 1024 scenes, 512 rare_union + 512 matched_common, 250 origin logs;
- sealed validation source: 256 scenes, 128 + 128, 63 origin logs;
- train/validation origin-log overlap: zero;
- excluded data: test, augmented, broad-common, BWM and prior CCV targets;
- `development_consumed=false`, `test_consumed=false`.

The next authorized stage in the run ledger is `baseline_a_b`.  Reusing the
same run ID with `all` or `baseline` verifies and skips the 4.8 GB source
artifacts rather than rebuilding them.

The first `baseline_train_a` launch on 2026-09-04 stopped before any scenario
completed because a temporary `re.Match` object from the V4 config was exposed
to MMCV `pretty_text`.  This is an engineering startup failure, not a baseline
outcome.  The config now performs the regex check without retaining the match
object, and the runner evaluates `cfg.pretty_text` before starting WorldEngine.
The exact failing serialization path, followed by Python compilation of the
rendered config, passed after the fix.  The same run ID remains valid because
there is no collection audit, completion flag, completed-scenario ledger or
scientific result from the failed launch.
