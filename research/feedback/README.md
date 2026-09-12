# Selector feedback repair v2

Implementation branch: research/frozen-selector-feedback-v2, based on rare-v1
0c237e2. Old checkpoints, runs, negative results and worktrees are read-only.
Internal name is not a novelty claim. CRAFT/RAD-2 are relevant strong precedents.

## Frozen purpose and boundaries

Test whether targeted feedback after a policy change improves hard-scene repair
and retention over STATIC (S) and UNIFORM-REFRESH (U). T is the only candidate
eligible for additional seeds; controls can be scientifically positive without
being silently promoted. No new architecture, generator, inference router,
safety rule, candidate count, controller, or third feedback iteration.

NR and R are co-primary and run separately with trajectory and IDM surrounding
agents, respectively. Both use the original LogPlayController, eight executed
decisions 4..11 at 0.5s, original20 candidates, and token-keyed
selector_cfpi_v1_noise0. Publication12 is recorded but never counted as executed.
Existing 58-scene/22-log rare development and 64 common scenes are fixed.
They are legacy-exposed, not independent unseen/blind evaluation.
Confirmation230 is not part of this run.

## Data and exact defaults

CPU audit chooses 128 train scenes in baseline-R strata: 16 failures,
32 successful EP<0.5, 48 other rare, 32 successful common EP>=0.5.
Global max two scenes/origin-log. Fixed hash ordering; no quota relaxation.
All known evaluation, legacy development/certification, historical B and CCV
logs are excluded. Pilot training logs may be reused and are not called unseen.
Dense replay is the old common/hard mixture filtered by the same exclusions
and train membership; paired common token origins are independently checked.

Round1: 96 scenes/mode = failure16+slow32+other16+common32, two branches/state.
V3 visits and continues. Proposals are V3 action vs Gate action, falling back
to Gate's highest-ranked different original candidate. Round1 has one shared
seed0 preliminary policy for feedback acquisition, never selected on dev.

Round2: every arm has 24 rare+8 common targets per mode and two branches/state.
S visits/continues V3; U/T share the preliminary policy's naturally visited
trajectories and continuation. Proposals are V3 vs preliminary at THAT state;
same-action fallback uses Gate. S/U use the same scene/timing hash namespace.
T rare quotas: unresolved12, newly-regressed8, uniform4; common: regressed4,
uniform4. Regression is solved->failed OR PDM drop >=0.05. It takes precedence
over unresolved (failed OR EP<0.5). Missing slots use within-family uniform
sampling and are explicitly reported. No Q-guided target/candidate selection.
Event targets use the last decision STRICTLY before first_violation_step.
Other targets use uniform hash among decisions4..11, restricted to strictly
pre-violation decisions if the visiting rollout failed (also for S and U).
Round1 uses the last pre-violation decision for every actual mode-specific
failure, not just scenes labelled failed by the source R stratification.
No legal pre-event state
is a protocol stop, not permission to drop that scene.

Before treatment in each mode/round/arm, eight hash-selected actual-visiting-
action sentinels must reproduce the full natural rollout, components, success,
and first violation timing. Every treatment must reproduce the original prefix
and target context, force exactly one original candidate, then use its frozen
continuation policy. The policy logits are NEVER forged to justify a forced
action. Pair slot0 is V3, not necessarily the visiting action; sentinel is
therefore a separate condition after refresh.

## Training and gates

All arms use the identical V3 scene selector and initialization. Each seed has
500 round1 + 500 round2 AdamW steps, lr1e-4, wd1e-4, T1, grad clip10.
Round2 continues its seed's round1 model AND optimizer; S/U/T seed0 share the
same round1 parent. Extra T seeds use the SAME seed0-acquired feedback, so this
is optimization-seed variation, not repeated end-to-end acquisition.

Each batch has dense16 (8 common/8 hard) and feedback16 (8 NR/8 R).
Round2 trains on round1+that arm's round2 feedback; uniform state sampling
within each mode. A dense hard item is uniform over the filtered logical
hard pool; its real-cache noise variant is uniform0/1/2.
Loss = dense scalar exact-group GRPO + paired logistic loss
+0.01 KL(V3 || current), the latter over all32 rows.
Pair preference exists only for PDM gap>=0.01 with NC,DAC,success all
nondecreasing. Raw gap weights softplus(z_loser-z_winner); divide by all16
feedback rows, not only active pairs. Conflict/ties receive no invented
ranking (but still receive retention). This known loss is an experimental
constant, NOT the proposed contribution.

Screen: S/U/T seed0 and scalar/Gate controls on all58 rare scenes in BOTH modes.
T must gain>=0.01 PDM vs scalar, stay within0.02 of Gate, retain scalar success,
and exceed S/U each by>=0.005 in BOTH modes. Otherwise STOP_NO_MECHANISM_SIGNAL.
Passing T enables seeds1/2 (both rounds), rare58 and common64 for all T seeds,
plus common scalar controls. No control is substituted as winner.

Final per-mode gates: three-seed mean PDM >=scalar+0.02 and >=Gate-0.01;
success>=scalar and >=Gate+0.01; rare NC/DAC/EP>=scalar;
at least2/3 seeds improve PDM vs scalar; common PDM/success/NC/DAC>=scalar-0.01.
Report all seeds, rescue/broken counts and scene-weighted log-cluster bootstrap
(seed-average first), plus equal-log sensitivity. Point gates are not inference:
unsupported scalar superiority, Gate PDM noninferiority or success comparisons
produce INSUFFICIENT_EVIDENCE. The bootstrap is conditional on this one
feedback acquisition and fixed benchmark, not a formal safety guarantee.

## Execution and budget

Default 64 GPUh cumulative ceiling: feedback24, training4, evaluation30, shared buffer6.
Following the user's budget expansion authorization on 2026-09-07, the runner
accepts an explicit positive finite --gpu-hours value. Phase allowances scale
proportionally; the recommended 128 GPUh cap gives feedback48, training8,
evaluation60, shared buffer12. This changes resources ONLY: no extra scenes,
steps, rounds, seeds or relaxed effect gates. Budget remains immutable within
a run; pass the SAME value for audit, preflight, all and report.
Four or eight H100 GPUs are supported through --gpus; freeze this count at audit
and keep it identical on resume. All RESERVED GPUs are charged during launched stages, including CPU
preparation/reporting. CPU audit must be run before reserving the GPU instance.
Forecast accounts for historical startup/service times; NR uses a conservative
1.2 multiplier until measured. Live runtime samples record their GPU count and
are normalized to the historical eight-GPU GPUh basis, so twice the elapsed
time on four GPUs is not mistaken for twice the GPU cost. The audit also reports
wall hours at the selected count. Four-GPU scaling is an estimate, not measured.
The full design may not fit: stop before a
condition that cannot fit; do not reduce scenes/seeds or reset the run ID.
Only owned process groups are terminated. Incomplete rollout attempts are
recoverably archived; complete conditions are hash-verified and reused.

The four-GPU full design retains all scenes, seeds, steps, eight decision times
and eight sentinels; only parallelism changes. The collection watchdog is
12 hours on four GPUs (6 hours on eight), still subject to cumulative GPUh and
phase caps. At 128 GPUh, four-card cumulative reserved runtime is capped at
32 hours; the prior 78.4 GPUh full-design forecast corresponds to about 19.6 hours.

CUDA_VISIBLE_DEVICES is respected and must contain exactly the requested number
of distinct IDs. When unset, the selected IDs are 0..N-1. A mismatched inherited
map fails early; do not blindly unset a scheduler-provided allocation map.
Ray physical IDs / UUIDs are mapped to logical split_0..N-1 for the feedback
protocol only. The simulator sees N GPUs; each planner sees only its paired GPU.
Changing GPU count requires a NEW run ID, never relabel old eight-GPU outputs.

Run from this worktree; no Bash arrays are needed. The neutral-name launcher
below supports either count; the old _8h100 filename remains compatible:

```bash
bash run_diffusiondrive_selector_feedback.sh audit rare_feedback_20260907_4gpu_v1 --gpus 4 --gpu-hours 128
bash run_diffusiondrive_selector_feedback.sh preflight rare_feedback_20260907_4gpu_v1 --gpus 4 --gpu-hours 128
nohup bash run_diffusiondrive_selector_feedback.sh all rare_feedback_20260907_4gpu_v1 --gpus 4 --gpu-hours 128 >> rare_feedback_20260907_4gpu_v1.log 2>&1 &
tail -n 60 -f rare_feedback_20260907_4gpu_v1.log
```

Standalone boundaries: feedback, screen, final. CPU-only report remains
available after a budget/effect stop. Do not edit code after freezing a formal
run contract. Software audit IDs are developmental and must not be used as
formal experiment IDs after code changes.

Wait for audit COMPLETE, then preflight COMPLETE, before launching all.
all automatically runs feedback -> screen -> final, but only when the T effect
gate passes. STOPPED_AT_EFFECT_GATE is a scientific stop, not a software crash.
Ctrl+C while running tail stops log viewing, not the nohup job. Do not launch
two all jobs for the same run; use the same all command to resume only after
confirming the previous runner has exited.

```bash
bash run_diffusiondrive_selector_feedback.sh report rare_feedback_20260907_4gpu_v1 --gpus 4 --gpu-hours 128
```

## Verification

```bash
python -m unittest discover -s research/feedback/tests -v
python -m unittest discover -s research/rare/tests -v
python -m unittest discover -s research/cfpi/tests -p test_continuous_deployment.py -v
```

Software checks cover fixed counts, source separation, deterministic selection,
fallback accounting, Pareto conflicts/ties, suppressed gradients, true-vs-forced
action semantics, continuation identity, NR/R exposure metadata, co-primary
gates, incomplete reporting, budgets and interrupted-run resume. Real-cache
preflight checks V3 identity/bank parity and the full frozen baseline tensors.
None of these checks claims real closed-loop execution has passed.
