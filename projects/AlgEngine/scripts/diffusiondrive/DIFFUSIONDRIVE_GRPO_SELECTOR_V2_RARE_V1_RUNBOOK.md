# DiffusionDrive Selector V2 Rare V1 Runbook

## Purpose

This branch supplies the corrected-selector-V2 controls for the existing V3
rare-log and rare-rollout experiments. It does not change the DiffusionDrive
generator or collect a second synthetic dataset.

- V2 trains the original `plan_cls_branch` MLP (exactly 10 tensors).
- V3 trains the 54-tensor scene-conditioned residual selector.
- Both consume the same audited real rare/common caches.
- Both consume the same immutable epoch-100 online rollout records.
- Fixed results use 304,272 examples and 4,800 optimizer steps.
- Tuned results use the same predeclared 8-trial/56-candidate search budget,
  development-only selection, and one-shot certification.

The shared cache stores candidate features in FP16 but reference logits in
FP32. V2 therefore evaluates cached policy logits as

```text
reference_logits + (trainable_MLP(fp16_features) - frozen_MLP(fp16_features))
```

This makes initialization exactly equal to the epoch-100 behavior policy while
preserving gradients of the original MLP. Materialized checkpoints still write
only the 10 original MLP tensors. The checkpoint audit rejects every other
baseline change.

## Isolated worktree

Keep V3 collection jobs on the main worktree. Use the persistent V2 worktree:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-v2-rare
git status --short --branch
```

`data`, `experiments`, and `artifacts` are links to the canonical WorldEngine
workspace; code comes from the V2 branch.

## Rare-log order

One-H100 smoke:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-v2-rare
./run_diffusiondrive_grpo_selector_v2_rare_log_1h100_smoke.sh
```

Submit only these three 8-H100 allocations. Submit seed 0 first, then seed 1 and
seed 2 immediately; a separate tuning allocation is no longer required:

```bash
./run_diffusiondrive_grpo_selector_v2_rare_log_8h100.sh formal-seed 0
./run_diffusiondrive_grpo_selector_v2_rare_log_8h100.sh formal-seed 1
./run_diffusiondrive_grpo_selector_v2_rare_log_8h100.sh formal-seed 2
```

Seed 0 owns the eight-lane development sweep and one-shot certification, then
runs its fixed and tuned formal models. Seeds 1 and 2 train and formally evaluate
the fixed model first, wait for the verified seed-0 tuning result, and then run
the tuned model. They fail instead of continuing if seed 0 reports failure or if
the wait exceeds 24 hours. An already verified tuning result is reused. The
standalone `tune` mode remains available only for recovery/debugging.

After all three pass, run locally on the one-GPU instance (no 8-H100 queue):

```bash
./run_diffusiondrive_grpo_selector_v2_rare_log_8h100.sh summarize
```

## Rare-rollout order

Wait until the V3 collection/finalization pipeline has produced all three:

```text
experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/manifest.json
experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/hard_pool.jsonl
experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/synthetic_cache.pt
```

Do not recollect for V2. First run the one-H100 smoke:

```bash
./run_diffusiondrive_grpo_selector_v2_rare_rollout_1h100_smoke.sh
```

Then submit only these three 8-H100 allocations, with seed 0 submitted first:

```bash
./run_diffusiondrive_grpo_selector_v2_rare_rollout_8h100.sh formal-seed 0
./run_diffusiondrive_grpo_selector_v2_rare_rollout_8h100.sh formal-seed 1
./run_diffusiondrive_grpo_selector_v2_rare_rollout_8h100.sh formal-seed 2
```

The same coordinator contract applies: seed 0 performs tuning/certification,
while seeds 1 and 2 do fixed work before waiting for the tuned parameters. The
standalone `tune` mode is optional recovery/debugging only.

Then aggregate locally on the one-GPU instance:

```bash
./run_diffusiondrive_grpo_selector_v2_rare_rollout_8h100.sh summarize
```

The fixed rows are the architecture/data attribution results. Tuned rows are
secondary best-tuned results and must not be described as compute matched when
their selected epoch differs from 16.
