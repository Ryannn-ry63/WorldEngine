# Base Model Rollout Summary

## Goal

Run the trained DiffusionDrive base model in the existing SimEngine closed-loop pipeline and save rollout data for later score-head / GRPO fine-tuning.

## Key Changes

- Added `score_mode=rollout` for closed-loop rollout.
  - It skips online NAVSIM PDM recompute because synthetic rollout tokens do not always have metric cache entries.
  - It still returns model trajectory, selected mode, all candidate trajectories, and logits.
- Added rollout record saving in `closed_loop/sim_test.py`.
  - Records are written to `rollout_records/{prefix}_{step}.pkl`.
  - An index is appended to `rollout_records/rollout_index.csv`.
- Updated SimEngine/AlgEngine rollout scripts to pass `sim.rollout_record_path`.
- Kept the existing `plan_traj/*.npy` interface unchanged, so SimEngine still consumes trajectories as before.

## Saved Data

Each rollout record contains:

- `token`, `prefix`, `step`
- current frame pkl, history queue, merged annotation pkl
- final planned trajectory
- `trajectory_8`, `all_trajectories_8`
- expanded 40-step `trajectory`, `all_trajectories`
- `poses_cls`, selected mode index
- metric fields if available; `rollout` mode stores them as `NaN`

## Current Run

Checkpoint:

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/Worldengine-Diffusion/experiments/navformer/e2e_diffusiondrive/epoch_8.pth`

Output:

`experiments/closed_loop_exps/e2e_diffusiondrive_base_rollout/navtest_failures_NR/`

The current smoke rollout is running on `navtest_failures` in non-reactive mode with one GPU.
