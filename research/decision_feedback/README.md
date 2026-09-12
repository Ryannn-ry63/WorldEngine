# Decision-feedback v1: diagnostic-first four-arm selector study

Implementation branch: research/selector-decision-feedback-v1. Source implementation
snapshot: aab4752 (feedback-v2). Historical runs/checkpoints are read-only; their
STOP_NO_MECHANISM_SIGNAL decision remains unchanged. Internal method name is not
a novelty claim. No formal GPU experiment is launched by implementing this code.

## Fixed protocol

Keep perception, generator, selector architecture, original20 candidates, noise0,
LogPlayController, 0.5s decisions4..11 and unexecuted publication12 unchanged.
NR and R are co-primary. The 58-scene development and 64-scene common benchmarks
are historically exposed, not blind tests. Confirmation230 is not authorized.

Audit freezes the previous 192 shared + 64 T rows and their state/candidate/
continuation hashes. On T's 64 second-round states it identifies final argmax
outside the queried pair (historical expectation NR9/R8) and selects eight
informative continuation comparisons per mode by deterministic hash.

Diagnosis queries these unsupported actions under the original shared continuation.
It also reruns the selected16 pairs under T continuation. The latter uses an
unmodified visiting action followed by T as its matching hybrid reference; it is
NOT compared against shared's full natural trajectory. Every forced action has
prefix/context and actual-controller-action audit. Standard and hybrid sentinels
must pass before treatments. Diagnostics never enter training automatically.

Pilot requires dominated actual choices on >=3 independent logs including >=2 R
decisions, and <=4 strict preference reversals among all16 continuation pairs.
Dominance is PDM gap>=.01 with NC/DAC/strict success nondecreasing. A failed
diagnostic stops, without changing thresholds or auto-selecting another method.

Three optimization seeds share seed0-acquired contexts. Each seed has one shared
round1 parent: seed0 uses the old verified optimizer/RNG checkpoint, seeds1/2
retrain500 steps on the same192 rows. All four round2 arms use that seed's parent
and same row sampling RNG, for500 updates:

| Arm | Objective/data |
|---|---|
| P0 | Exact legacy T loss and fixed data |
| P1 | P0 plus gap-weighted winner negative log-probability, coefficient1 |
| P2 | P1 plus random unqueried-action queries matched to P3 states/counts |
| P3 | P1 plus queries for current global argmax outside its evaluated subset |

All inherited constants unchanged: AdamW1e-4, decay1e-4, clip10, T1; dense16
(8common/8hard), feedback16 (8NR/8R); GRPO + pair loss + .01 KL(V3||current).
Additional NLL uses full20 softmax. At steps100/300 both P2/P3 schedules are frozen
before either sees new returns; a state receives at most one added candidate each
round. Candidate banks/contexts/continuation remain fixed to isolate action querying.
Every pair satisfying the original Pareto condition contributes. Divide each
state's sum by ALL possible pairs, then average over all sampled states.
No reward or negative label is invented for unqueried candidates. Tie/conflict
states still receive retention, but no fabricated ranking.

Queries use actual-visiting-action sentinels (up to8 per mode/arm/round). Each
arm owns its own cache. P2 can read P3's request schedule but never its labels.
Diagnostic labels are quarantined; this implementation deliberately recollects
requested pilot branches rather than introducing cross-purpose cache reuse.
Logical query counts are therefore charged even if an action coincides with a
diagnostic. P0/P1 cannot consume a generation>0 cache.

Step500 is the only evaluation model. P0 seed0 must reproduce old T parameters,
scores and argmax within inherited numerical tolerances; otherwise stop as an
engineering failure. Other seeds do not bypass the old gate: they belong solely
to this newly frozen diagnostic protocol.

## Reports and stopping

Evaluate all four arms, all3 seeds, NR/R, rare58/common64, plus scalar/Gate references
(56 conditions). No early seed dropping or best-seed selection. Report PDM, strict
NC&DAC success, NC, DAC, TTC, EP, rescue/broken counts, V3-failed/V3-safe-slow
strata, query counts, pair correctness, decisive winner selection and unsupported
argmax. Pair correctness in variable subsets is not confused with globally known
optimality. Compute seed-average-first log-cluster95% intervals and equal-log
sensitivity. These are exploratory, conditional on existing acquisition/noise.

P3 must satisfy IN BOTH modes: meanPDM>=V3+.01, >=Gate-.02, >=eachP0/P1/P2+.005;
rare success/NC/DAC>=V3; >=2/3 positive seeds vsV3; common PDM/success/NC/DAC>=V3-.01.
Point pass without supported superiority/noninferiority yields INSUFFICIENT_EVIDENCE.
Even a supported pass only yields REQUIRES_SEPARATE_CONFIRMATION_PLAN.
Neither a safety guarantee nor a paper acceptance claim is made.

## Four-card usage

Before reserving GPUs, run CPU audit from THIS worktree:

```bash
bash run_diffusiondrive_selector_decision_feedback.sh audit decision_feedback_20260912_v1 --gpus 4 --gpu-hours 128
```

Inspect audit_report.json and budget_forecast.json under
experiments/diffusiondrive/selector_decision_feedback_v1/runs/decision_feedback_20260912_v1.
The forecast includes the exact diagnostic and the adaptive-query upper estimate;
128 GPUh is a ceiling (32 reserved-four-card hours), not a completion promise.
The initial real-cache software audit estimated diagnosis7.55, adaptive queries
up to39.15, training allowance8, evaluation81.63 GPUh (136.33 total, about34.08
four-card hours). These are estimates, not measured runtime of this new protocol.
If a higher explicit cap is required, use a NEW run ID and that same cap at every
stage. Never edit an existing contract or silently reduce arms/seeds/scenes.

For an unattended full run, an explicit192 GPUh cap is recommended, using a separate
ID throughout (the executable default remains128):

```bash
bash run_diffusiondrive_selector_decision_feedback.sh audit decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192
bash run_diffusiondrive_selector_decision_feedback.sh preflight decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192
nohup bash run_diffusiondrive_selector_decision_feedback.sh all decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192 >> decision_feedback_20260912_192h_v1.log 2>&1 &
```

On the allocated four H100 instance:

```bash
bash run_diffusiondrive_selector_decision_feedback.sh preflight decision_feedback_20260912_v1 --gpus 4 --gpu-hours 128
nohup bash run_diffusiondrive_selector_decision_feedback.sh all decision_feedback_20260912_v1 --gpus 4 --gpu-hours 128 >> decision_feedback_20260912_v1.log 2>&1 &
tail -n 60 -f decision_feedback_20260912_v1.log
```

all runs diagnose then pilot ONLY if diagnosis passes. Standalone diagnose/pilot
preserve the same boundary. report can report missing conditions after any stop.
Failed diagnostics cannot be bypassed with pilot. No confirm command exists.
Reserve four distinct scheduler-provided CUDA IDs; do not clear that allocation.
Small conditions use min(4,scenes) workers but ALL four reserved GPUs are charged.
Resume verifies hashes, optimizer/RNG, cache ownership and cumulative ledger.
Incomplete simulation attempts are archived recoverably, not deleted. Completed
historical runs are never re-audited with a writing tool.

CPU software tests (synthetic only):

```bash
python -m unittest discover -s research/decision_feedback/tests -v
python -m unittest discover -s research/feedback/tests -q
python -m unittest discover -s research/cfpi/tests -p test_continuous_deployment.py -q
```

Software-audit run IDs are NOT formal experiment IDs; any code change invalidates
their contracts. Freeze final code before launching a new formal run.
