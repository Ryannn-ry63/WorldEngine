# DiffusionDrive selector V2 / V3

This contribution studies selection among a frozen DiffusionDrive generator's 20 candidate trajectories.
V2 optimizes the original candidate scorer with complete-action GRPO. Standard V3 adds a zero-initialized
residual selector using trajectory geometry, sampled route BEV features, ego/agent context and candidate-set interaction.
Perception and trajectory generation remain frozen. PDM rewards are training targets, not inference inputs.
The V3 here precedes GateV3 and does not include feedback-pair post-training or later methods.

## Code map

Paths below are relative to `projects/AlgEngine/`:

- `mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py`: frozen-reference selection, exact-group objective, inference integration.
- `mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py`: standard V3 architecture.
- `mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py`: PDM reward, corrected progress normalization.
- `scripts/diffusiondrive/train_grpo_selector_v2_cached.py`: corrected V2 common training.
- `scripts/diffusiondrive/train_grpo_selector_v3_cached.py`: standard V3 cached training.
- `scripts/diffusiondrive/*cached_rare_original.py`, `*cached_rare_rollout.py`: source-specific V2/V3 training.
- `scripts/diffusiondrive/selector_ablation_models.py`: original-scorer and parameter-matched unary controls, extracted from the structural study without feedback objectives.

`online` in historical module names means candidate PDM scoring during a forward pass. It does not establish
step-by-step online closed-loop learning. Historical rollout-source training collects data first, then trains the selector.

## Start here

Copy `selector.example.json` to an external personal configuration and set absolute paths.
No weights, datasets, conda environments or experiment outputs are bundled. Use the upstream environment setup
plus the same NAVSIM v1 scorer and compatible mmcv/gsplat extensions as the checkpoint's provenance requires.

```bash
bash run_selector.sh --settings /absolute/selector.local.json --devices 0 check
bash run_selector.sh --settings /absolute/selector.local.json --devices 0 tool train_grpo_selector_v3_cached --help
```

The first command is **path checking only**. It is not a data-coverage, CUDA, throughput or formal-evaluation test.
The launcher uses the configured interpreters and resets PYTHONPATH; it never sources another user's setup script.
See [reproduction recipes](REPRODUCTION.md) and [full evaluation protocol](EVALUATION.md).

Existing upstream code and licenses remain in place. Only the selector contribution and narrow integration changes
should enter a PR. Machine-specific handoff material is deliberately outside the Git root.

Validation: portable launcher and strict table tests live in `tests/selector_v2_v3`.
Historical tests inspecting removed machine-specific shell scripts are not shipped; scientific model, loss and cache tests remain.
Use `method_env` in the runtime JSON for explicit temperature, KL weight, V3 width or `SELECTOR_ABLATION`; inherited research environment variables are cleared.
