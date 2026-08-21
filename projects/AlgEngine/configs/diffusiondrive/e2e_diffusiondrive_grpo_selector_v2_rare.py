"""Corrected selector V2 evaluation config for rare-log/rare-rollout models."""

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
    reward_contract="navsim_pairwise_raw_progress_then_candidate_gate_v1",
    fixed_ratio_clip=False,
    generator_frozen=True,
    imitation_loss=False,
    trainable_tensors="plan_cls_branch_exact_10_tensors",
)
