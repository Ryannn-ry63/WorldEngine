"""DiffusionDrive selector GRPO V3: context-conditioned group competition.

The frozen DiffusionDrive generator still produces the complete 20-trajectory
action set and NAVSIM PDM remains the only reward.  The trainable selector
receives final candidate features, trajectory geometry,
frozen BEV features sampled along each final trajectory, and frozen
status/ego/agent context.  It predicts residual logits on top of the immutable
epoch-100 selector, so initialization is exactly deployment-equivalent.
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
        scene_selector=dict(
            feature_dim=256,
            model_dim=int(os.getenv("DIFFUSIONDRIVE_GRPO_V3_MODEL_DIM", "256")),
            route_bev_dim=256,
            context_dim=256,
            geometry_hidden_dim=128,
            num_heads=4,
            feedforward_dim=512,
            num_route_steps=8,
            num_set_layers=1,
            use_set_attention=True,
            use_trajectory_geometry=True,
            use_route_bev=True,
            use_scene_context=True,
        ),
    )
)

selector_reward_contract = dict(
    policy_objective="exact_complete_action_expected_advantage",
    architecture="scene_conditioned_residual_set_selector_v3",
    policy_temperature="environment_selected",
    fixed_ratio_clip=False,
    residual_on_frozen_reference=True,
    generator_frozen=True,
    trainable_selector_tensors="architecture_defined",
    imitation_loss=False,
    ranking_loss=False,
    gate=False,
    reward_components_as_inputs=False,
    model_selection_split="navtrain_scene_disjoint_development",
    certification_split="navtrain_scene_disjoint_certification",
)
