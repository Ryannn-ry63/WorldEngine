# DiffusionDrive Selector Causal-Branch Pilot Protocol v2 (2026-09-03)

Status: **FROZEN AFTER BASELINE A/B AND BEFORE ANY INTERVENTION OUTCOME**

## Purpose and evidence boundary

This pilot asks whether correcting one reproducible local ranking error made by the
frozen scalar V3 selector causally improves closed-loop behavior. It is a
diagnostic experiment, not a new selector, an AutoVLA reproduction, a training
result, or a deployable oracle.

Baseline A/B on the same 147 train-only scenarios is already complete. Both runs
contain 1176 pre-action records and have an identical closed-loop metrics SHA256.
Candidates, rewards, logits, selected indices, and oracle indices reproduce. This
establishes stable candidate-set selection headroom; it does not by itself show
that the locally higher-reward action improves the later closed-loop outcome.

## Why v2 supersedes the original matched-control definition

The original protocol selected the non-improving candidate whose policy-relative
ADE was closest to the oracle perturbation, but did not cap the remaining match
error. Baseline-only inspection found that this could label a poor match as
"geometry matched." No oracle or matched intervention outcome had been observed.
Therefore v2 freezes a treatment-outcome-blind amendment before target creation:

- the primary comparison is oracle intervention versus the exactly reproducible
  no-intervention V3 baseline;
- the specificity comparison is oracle intervention versus a strictly
  magnitude-matched non-improving perturbation;
- the control is not claimed to match trajectory direction or full geometry.

## Fixed scope

- Policy: frozen scalar V3 selector, train seed 0.
- Generator, perception, base selector, and the 20-candidate set are frozen.
- Worlds: existing 147 BWM train-only scenarios; no navtest overlap or new data.
- All arms use the same candidate-noise namespace and rollout implementation.
- Official scalar pairwise PDM is the treatment reward; components are diagnostic.
- Each intervention replaces exactly one action at one frozen pre-action frame.
- Statistical resampling clusters by original log.

## Estimands and target contract

Primary estimand:

`one_step_local_oracle - no_intervention_V3_baseline`

Specificity estimand:

`one_step_local_oracle - magnitude_matched_nonimproving_control`

A target must satisfy all of the following in baseline A and B:

1. Candidate trajectories and logits reproduce within `1e-5`; policy, reward
   vector, and stable oracle index reproduce.
2. Local-oracle reward headroom is strictly greater than `0.02`.
3. A failed-scene target occurs strictly before the first violation.
4. The matched candidate is neither policy nor oracle and has reward at most
   `policy + 0.005` in both baselines.
5. Let `d_o` and `d_m` be mean 8-step XY displacement from the V3 trajectory for
   oracle and matched candidates. The match must satisfy both
   `abs(d_m-d_o) <= 0.5m` and
   `abs(d_m-d_o)/max(abs(d_o),1e-6m) <= 0.5`.
6. The earliest frame satisfying every rule is frozen for each scene.

Full XY-displacement and wrapped-yaw magnitude summaries are reported as
additional diagnostics only. Targets are capped at 24 per failed/solved stratum
and two per origin log, with deterministic source/pairing stratification. The
manifest uses schema v2, hashes the protocol, builder, helpers, baselines, and
scenario files, and is immutable after creation.

## Sequence and coverage

1. Reuse and SHA-audit the completed baseline A/B; do not rerun them.
2. Build the outcome-blind v2 target manifest and match-quality audit.
3. Require at least 8 targets and 6 origin logs in each of failed and solved.
4. Run oracle and matched arms on a smoke subset of 4 failed + 4 solved.
5. Run both arms on the complete frozen target set.
6. Require identical smoke/formal outcomes on their overlapping scenes.
7. Analyze the two predeclared estimands and issue exactly one decision.

Coverage failure is `INSUFFICIENT_CAUSAL_BRANCH_COVERAGE`; thresholds must not be
relaxed after seeing intervention outcomes.

## Gates and decisions

On failed targets, the primary point gate requires oracle-minus-baseline rescue
rate at least `0.15` and oracle rescues in at least four origin logs. The
specificity point gate requires oracle-minus-matched rescue rate at least `0.15`
and positive differential rescues in at least four origin logs. Each estimand
also requires an origin-log cluster-bootstrap 95% lower bound above zero for full
authorization.

- `AUTHORIZE_DIFFUSION_SET_SELECTOR_METHOD`: coverage, reproducibility, both point
  gates, and both bootstrap gates pass. Proceed to a fixed-20/frozen-generator
  diffusion-set ranking method.
- `EXPAND_CAUSAL_BRANCH_SEED1`: both point gates pass but at least one bootstrap
  lower bound is not positive. Repeat this causal pilot with V3 train seed 1.
- `REDIRECT_TO_REWARD_HORIZON_OR_ACTION_SENSITIVITY`: oracle beats baseline at the
  point gate but does not beat the magnitude-matched control. Do not claim that
  local reward ranking is the unique cause.
- `STOP_LOCAL_HEADROOM_CAUSAL_ROUTE`: oracle does not beat no-intervention baseline
  at the point gate. Treat local reward headroom as an offline diagnostic only.
- `INVALID_CAUSAL_BRANCH_REPRODUCIBILITY`: smoke/formal outcomes disagree. Repair
  infrastructure without drawing a scientific conclusion.
- `INSUFFICIENT_CAUSAL_BRANCH_COVERAGE`: either stratum has fewer than 8 targets or
  6 origin logs. Do not relax the frozen match contract post hoc.

Solved-target harm, score deltas, trajectory-shape diagnostics, and correlations
are reported but are not silently optimized after observing results.

## Resume-safe commands

Use the completed immutable run ID:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1

./run_diffusiondrive_selector_causal_branch_pilot_8hopper.sh targets causal_prefixfix_20260903
./run_diffusiondrive_selector_causal_branch_pilot_8hopper.sh smoke causal_prefixfix_20260903
./run_diffusiondrive_selector_causal_branch_pilot_8hopper.sh formal causal_prefixfix_20260903
./run_diffusiondrive_selector_causal_branch_pilot_8hopper.sh analyze causal_prefixfix_20260903
```

The runner verifies and skips completed immutable stages. Baseline A/B are reused
only after their scenario, checkpoint, noise, rollout implementation, and metrics
identity checks pass.
