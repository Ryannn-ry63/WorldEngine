# Training and export recipes

## Identity before running

The historical immutable epoch-100 generator SHA256 is
`1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`.
Materializers deliberately check this identity. A different generator requires a new documented experiment,
not removal of the check. Record the upstream pretraining provenance separately from selector post-training.

Common, rarelog and rarerollout describe **training data**, not different columns of the evaluation table.
The source builder, split audit and cache manifest are part of a recipe. Do not infer them from a model nickname.
The cached candidate generator remains frozen; regenerate/validate caches if the reward implementation changes.

| Model | Source | Training tool | Materializer |
| --- | --- | --- | --- |
| V2 | corrected common | train_grpo_selector_v2_cached.py | materialize_grpo_selector_v2_replica.py |
| V2 | rarelog | train_grpo_selector_v2_cached_rare_original.py | materialize_grpo_selector_v2_rare.py |
| V2 | rarerollout | train_grpo_selector_v2_cached_rare_rollout.py | materialize_grpo_selector_v2_rare.py |
| V3 | common | train_grpo_selector_v3_cached.py | materialize_grpo_selector_v3.py |
| V3 | rarelog | train_grpo_selector_v3_cached_rare_original.py | materialize_grpo_selector_v3.py |
| V3 | rarerollout | train_grpo_selector_v3_cached_rare_rollout.py | materialize_grpo_selector_v3.py |

All tools are under `projects/AlgEngine/scripts/diffusiondrive`; access their required manifest/configuration arguments
with `bash run_selector.sh --settings SETTINGS --devices 0 tool TOOL --help`.
Public recipes expose inputs rather than silently choosing a historical best seed or server-local cache.

## Example: standard V3 common, one optimization seed

```bash
bash run_selector.sh --settings /absolute/selector.local.json --devices 0 tool train_grpo_selector_v3_cached \
  --train-cache /absolute/cache/train_seed0/cache.pt \
  --train-cache /absolute/cache/train_seed1/cache.pt \
  --train-cache /absolute/cache/train_seed2/cache.pt \
  --development-cache /absolute/cache/development_seed3/cache.pt \
  --development-cache /absolute/cache/development_seed4/cache.pt \
  --development-cache /absolute/cache/development_seed5/cache.pt \
  --output-dir /absolute/runs/v3_common_seed0 \
  --temperature 1 --learning-rate 0.0001 --kl-weight 0.001 \
  --seed 0 --epochs 64 --checkpoint-epochs 16,64 --batch-size 64 --device cuda
```

This is an explicit training recipe, not a claim that it reproduces every historical V3 row. Use each archived
selection/manifest's exact hyperparameters and budget for that row. Repeat optimization seeds 0/1/2 into separate
outputs. Do not confuse candidate-noise seeds with independent training seeds.

Use `prepare_grpo_selector_v3_rare_original_data.py` and `split_grpo_selector_v3_rare_original.py` for audited
real rare / paired-common construction. Use `prepare_grpo_selector_v3_rare_rollout_scenarios.py`, the base rollout
configs, exported exact-candidate contexts, `audit_grpo_selector_v3_rare_rollout_collection.py` and
`build_grpo_selector_v3_rare_rollout_data.py` for the recorded base-policy rollout pipeline.
These retain their explicit CLI contracts; `--help` documents required input artifacts. They do not download missing data.

Rarelog historically samples 50% real rare + 50% paired common. Rarerollout samples 50% common + 50% hard,
where hard includes real rare and filtered base-policy rollout records. Preserve the actual pool manifest and filtering.
An offline rollout cache is not online feedback learning.

## Structural ablations

For inference, `e2e_diffusiondrive_grpo_selector_v3_ablation.py` reads `SELECTOR_ABLATION` for full,
feature_only, no_geometry, no_route_bev, no_scene or no_set. Choose the matching checkpoint architecture.
`selector_ablation_models.py` preserves the original-scorer and self-only-attention controls' state layout.
It is a model adapter, not a replacement for a frozen experiment's trainer, export contract or queue runner.
For continuing an existing structural study, keep its original immutable manifest and launcher.
For new controls, freeze training data/budget/seeds and audit native export before full evaluation.

Fixed-budget and tuned models are different comparisons. Report all three seeds; never select the best test seed
as the primary mean. Keep earlier progress-normalization-bug results as history only.
