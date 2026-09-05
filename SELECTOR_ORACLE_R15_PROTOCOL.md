# DiffusionDrive Selector Oracle R1.5 Protocol

## Purpose

R1 established correlation: failed reactive rollouts often contain a scalar-PDM-better candidate whose V3 probability is nearly zero. It did **not** establish that choosing that candidate before execution improves the closed loop, because R1 scored candidates after the action was executed.

R1.5 is a causal diagnostic, not a proposed deployable method. It changes exactly one variable: the candidate index deployed from the frozen V3-generated set. Generator, perception, V3 selector logits, candidate count, candidate noise, scenario, checkpoint, and scalar reward remain frozen.

The question is:

> If the scalar-PDM oracle candidate is substituted before the first recorded safety violation, does the failed scenario recover?

Only after this question is answered may a new selector objective be designed.

## Arms

- `observe_only`: pre-action score the exact 20 candidates, deploy the V3 argmax unchanged.
- `one_shot_oracle`: at one frozen target frame, deploy the stable scalar-PDM oracle once; all other frames use V3.
- `persistent_oracle`: at the target frame deploy the frozen baseline treatment; after that frame, deploy the current per-frame stable scalar-PDM oracle for the rest of decisions 4–11.

A stable oracle preserves the V3 action if it is tied for the maximum reward within `1e-8`. Components of PDM are diagnostics only; the scalar official pairwise PDM score alone defines the oracle.

## Frozen target rule

For scalar V3 train seeds 0 and 1, run a fresh `observe_only` baseline over the fixed 58-scene development split. For each closed-loop failure, select the earliest decision satisfying both:

1. `decision_step <= first_violation_step`;
2. `oracle_reward - policy_reward > 0.02`.

Candidate trajectories, rewards, logits, indices, source record hashes, target step, and baseline outcome are written into an immutable target manifest. Intervention rollouts must reproduce the treatment-defining candidate trajectories and logits within `1e-5` or fail. The baseline treatment index is never reselected from intervention-run rewards.

Each intervention collection has at least eight scenarios so all eight scenario workers contribute. Non-target scenes are deterministic fillers and must receive no intervention.

## Pre-intervention protocol amendment: reward repeatability

The first intervention smoke run failed before any action was replaced. Its target had exact candidates and logits but a `0.396359` scalar-reward difference. A paired repeatability audit was therefore performed on decision 4, the shared initial decision before seed-specific selector actions can alter the state. Across all 58 scenes:

- candidate trajectories were exact in 58/58 scenes (global maximum error `0`);
- scalar reward differed by more than `1e-5` in 24/58 scenes and by more than `0.02` in 24/58;
- the raw reward argmax disagreed in 14/58 scenes;
- among six reward components, only normalized `ego_progress` drifted (maximum error `0.951262`); all five safety/comfort/direction components were exact.

This identifies rerun-sensitive pairwise progress normalization, not candidate generation or selector nondeterminism. Accordingly, before observing any intervention outcome, the protocol was amended as follows:

1. The treatment at the frozen target is the baseline-manifest candidate index. It is not reselected using the intervention-run reward.
2. Intervention-run rewards are retained as a stability sensitivity, not used to define treatment identity.
3. The primary causal gate includes only target units for which that frozen treatment still exceeds the policy action by `>0.02` under the intervention-run score. All target outcomes remain reported.

This is stricter than silently accepting reward drift: it preserves preregistration while preventing a rerun-unstable reward label from supporting the primary causal claim. The immutable diagnostic is written to `targets/reward_repeatability_diagnostic.json`.

## Audit invariants

Every rollout must contain exactly eight pre-action records per scene for decisions 4–11. The audit rejects:

- post-action timing or state/decision misalignment;
- any candidate/logit/action mismatch against the prior R1 baseline in `observe_only`;
- policy action conversion error or replacement error above `1e-4`;
- an intervention outside its frozen schedule;
- target candidate/logit context, checkpoint, manifest, namespace, config, code, worker, report, ledger, or scenario drift;
- treatment-index reselection at the frozen target, missing reward-repeatability diagnostics, or incorrect current-score sensitivity values;
- missing or duplicate records;
- outcome coverage other than exact NC/DAC-scored scenario coverage.

The old R1 reward arrays are not required to match: R1 scored after execution, while R1.5 intentionally scores the same candidates before execution.

## Pre-registered decision

Aggregation is by distinct scene. If a scene appears for both train seeds, its score deltas are averaged before the mean-delta gate, preventing duplicated seeds from being treated as independent scenes. Bootstrap intervals (10,000 repetitions) are descriptive, not a hidden gate.

The primary gate is evaluated only on preregistered target units whose frozen treatment retains `>0.02` current-score headroom. All unstable units are still shown as sensitivity results but cannot authorize a method.

1. If `one_shot_oracle` rescues at least two reward-stable distinct scenes and its mean distinct-scene score delta is positive: `AUTHORIZE_CURRENT_SET_CAUSAL_METHOD`.
2. Otherwise, if `persistent_oracle` meets the same rule: `AUTHORIZE_TEMPORAL_OVERRIDE_METHOD`.
3. Otherwise, if either arm rescues exactly one distinct scene: `R15_SINGLE_CASE_ONLY`.
4. Otherwise: `STOP_R15_LOCAL_PDM_CAUSAL_ROUTE`.

Interpretation:

- Current-set authorization supports designing a diffusion-set-specific selector objective that corrects probability suppression within the current candidate set.
- Temporal authorization says a one-frame selector loss is insufficient; the next method must model sequential effects or state distribution shift.
- Single-case is qualitative evidence only and does not authorize a method claim.
- Stop means local scalar-PDM oracle choice is not the causal bottleneck; revisit reward validity, horizon, or candidate generation rather than tuning another selector loss.

No R1.5 result is a certification result, and the diagnostic oracle is unavailable at deployment.

## Execution

From the research worktree root:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_diffusiondrive_selector_oracle_r15_8hopper.sh preflight r15_main
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_diffusiondrive_selector_oracle_r15_8hopper.sh all r15_main
```

The same command and run ID are resume-safe after interruption. For explicit stage boundaries:

```bash
./run_diffusiondrive_selector_oracle_r15_8hopper.sh baseline r15_main
./run_diffusiondrive_selector_oracle_r15_8hopper.sh interventions r15_main
./run_diffusiondrive_selector_oracle_r15_8hopper.sh analyze r15_main
```

Final decision:

`experiments/diffusiondrive/selector_oracle_r15/runs/r15_main/causal_gate.json`

Target manifest:

`experiments/diffusiondrive/selector_oracle_r15/runs/r15_main/targets/target_manifest.json`

Reward repeatability diagnostic:

`experiments/diffusiondrive/selector_oracle_r15/runs/r15_main/targets/reward_repeatability_diagnostic.json`
