# DiffusionDrive V3 rare-rollout next-stage runbook

## 1. Frozen incumbent and experimental scope

The accepted incumbent is V3 rare-rollout v1 on branch
`diffusiondrive-selector-grpo-v3-rare-rollout-v1`, commit `8dc8bbf`, with
annotated tag `diffusiondrive-selector-grpo-v3-rare-rollout-v1-formal-20260822`.
Its tracked report and artifact checksums are:

- `WORLDENGINE_DIFFUSIONDRIVE_V3_RARE_ROLLOUT_V1_FORMAL_20260822.md`
- `WORLDENGINE_DIFFUSIONDRIVE_V3_RARE_ROLLOUT_V1_ARTIFACT_MANIFEST_20260822.json`

The causal result against compute-matched rare-original is Reactive CL-PDMS
`+1.86` points and valid rate `+3.35` points. The result is kept as the
incumbent; neither experiment below may overwrite its data, models or reports.

Two independent follow-up lines are implemented. They must first be screened
with seed 0 and must not be combined unless both independently pass their
predeclared promotion gates.

## 2. Line A: gate-conditioned GRPO

Branch: `diffusiondrive-selector-grpo-v3-gate-conditioned-v1`

This line reuses the frozen rare-rollout-v1 data exactly. It does not collect
new synthetic data and does not modify the selector architecture or official
PDM evaluation. The loss retains official PDM advantage and adds a conditional
quality advantage only among candidates whose NC and DAC gates both pass.
The quality term is `(5 EP + 5 TTC + 2 Comfort) / 12`; unsafe candidates never
enter that conditional softmax. This directly targets the observed EP loss
without rewarding progress through collisions or off-road behavior.

One-H100 smoke:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
git switch diffusiondrive-selector-grpo-v3-gate-conditioned-v1
./run_diffusiondrive_grpo_selector_v3_gate_conditioned_1h100_smoke.sh
```

Seed-0 screening on one 8-H100 allocation:

```bash
./run_diffusiondrive_grpo_selector_v3_gate_conditioned_8h100.sh seed 0
```

Only after seed 0 passes, run seeds 1 and 2 independently, or all three in one
sequential allocation:

```bash
./run_diffusiondrive_grpo_selector_v3_gate_conditioned_8h100.sh seed 1
./run_diffusiondrive_grpo_selector_v3_gate_conditioned_8h100.sh seed 2
# Alternative single allocation:
./run_diffusiondrive_grpo_selector_v3_gate_conditioned_8h100.sh all
```

The predeclared mean-delta gate is: Reactive EP at least `+1.0` point;
Reactive CL-PDMS, success rate, NC and DAC each no worse than `-0.5` point.

## 3. Line B: BWM generalization rollout

Branch: `diffusiondrive-selector-grpo-v3-rare-rollout-bwm-v1`

This follows the senior experiment's generalization part in:

- senior config
  `projects/AlgEngine/configs/WE/e2e_vadv2_50pct_rlft_rare_rollout_bwm.py`
- senior dataset implementation
  `mmdet3d_plugin/datasets/navsim_openscene_synthetic.py`, especially
  `NavSimOpenSceneE2EFineTuneSynthetic` and `customized_filter="v1"`

The senior config combines reactive synthetic data with BWM-augmented
collision, low-EP and off-road worlds. We reuse those published worlds, but do
not reuse the senior policy outputs: the immutable epoch-100 DiffusionDrive is
deployed again in SimEngine and our own candidate contexts/rewards are
collected. This preserves model attribution.

The local preparation audit currently fixes 147 BWM worlds from 22 origins:
14 collision, 80 low-EP and 53 off-road. Of these, 76 use an existing direct
DiffusionDrive rare/common pair and 71 use a deterministic common token from
the same log. navtest token/log overlap is zero. The 147 worlds are divided
into immutable lanes of 47, 44 and 56 scenarios, with 9 reward records expected
per scenario.

Prepare/check the scenario contract locally (CPU only):

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
git switch diffusiondrive-selector-grpo-v3-rare-rollout-bwm-v1
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_prepare_local.sh
```

One-H100 end-to-end smoke (8 BWM scenarios, then tiny train/materialization):

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_1h100_smoke.sh
```

Formal collection can use three parallel 8-H100 allocations:

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh collect-lane 0
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh collect-lane 1
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh collect-lane 2
```

If queue cost matters more than wall time, one allocation can run all lanes
sequentially and stops immediately if any lane fails:

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh collect-all
```

After all three collection audits pass, build the cache locally (CPU only),
then screen seed 0:

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_build_local.sh
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh finish-seed 0
```

Only after seed 0 passes, run seeds 1 and 2, either independently or in one
sequential allocation:

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh finish-seed 1
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh finish-seed 2
# Alternative single allocation:
./run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_8h100.sh finish-all
```

The built training pool retains the frozen rare-rollout-v1 hard rows and
appends only BWM rows that pass the same senior-v1 filter. Training remains
globally 50% paired common and 50% uniformly sampled hard rows, with the same
304,272-example/4,800-step budget. The selector is fresh zero-initialized and
the epoch-100 backbone remains unchanged.

The predeclared mean-delta gate is: Reactive CL-PDMS strictly better than the
seed-matched rare-rollout-v1 incumbent, Reactive success rate non-decreasing,
and OpenLoop-navtest PDM regression no larger than 1.0 point.

## 4. Machine-readable promotion audit

For a seed-0 BWM screen:

```bash
python projects/AlgEngine/scripts/diffusiondrive/audit_grpo_selector_v3_candidate_promotion.py \
  --mode bwm_generalization \
  --baseline 0=experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s0/summary.json \
  --candidate 0=experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_v1_s0/summary.json \
  --output experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1/formal/seed0_promotion_audit.json
```

Use `--mode gate_conditioned` for Line A. For a three-seed decision, append
matching `--baseline 1=... --candidate 1=...` and seed-2 arguments. The audit
always completes successfully when inputs are valid; its decision is explicitly
`PROMOTE` or `HOLD_INCUMBENT`, so a scientifically valid negative experiment is
not confused with a pipeline failure.

## 5. Isolation rules

- Do not overwrite `experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1`.
- Keep Line A and Line B on separate branches and separate experiment roots.
- Do not choose hyperparameters from navtest or closed-loop test results.
- Do not hide EP or open-loop regressions when safety/validity improves.
- Keep V2 rare-log/rare-rollout work in its existing separate worktree/branch.
- Combine gate-conditioned loss with BWM data only after both ablations pass
  independently; otherwise keep the frozen rare-rollout-v1 incumbent.
