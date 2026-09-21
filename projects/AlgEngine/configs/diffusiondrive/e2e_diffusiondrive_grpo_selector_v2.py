"""DiffusionDrive selector GRPO V2: exact complete-action objective.

V2 changes only the policy objective. The 20 trajectories, online NAVSIM PDM
reward, frozen generator, frozen reference selector, and deployed argmax are
identical to V1. Offline hyper-parameter selection is performed by the
dedicated cached-feature sweep; this config remains the real-candidate parity
and optional online-training definition.
"""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector.py"]

model = dict(
    planning_head=dict(
        policy_objective="exact_group_grpo",
        policy_temperature=float(
            os.getenv("DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE", "1.0")
        ),
        kl_weight=float(os.getenv("DIFFUSIONDRIVE_GRPO_KL_WEIGHT", "0.001")),
    )
)

selector_reward_contract = dict(
    policy_objective="exact_complete_action_expected_advantage",
    policy_temperature="environment_selected",
    fixed_ratio_clip=False,
    generator_frozen=True,
    imitation_loss=False,
    model_selection_split="navtrain_scene_disjoint_calibration",
)
