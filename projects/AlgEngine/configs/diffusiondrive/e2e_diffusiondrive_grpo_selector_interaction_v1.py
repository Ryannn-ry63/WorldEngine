"""Interaction-aware V3 cache extraction and closed-loop materialization.

The DiffusionDrive generator and epoch-100 tracker are frozen.  The tracker is
run only in inference mode; no GT association, future motion label, reward
component, or additional dataset is exposed to the selector.
"""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]

INTERACTION_ARCHITECTURE = os.getenv(
    "DIFFUSIONDRIVE_INTERACTION_ARCHITECTURE", "interaction_generic"
)
if INTERACTION_ARCHITECTURE not in {
    "interaction_generic",
    "interaction_relation",
}:
    raise ValueError(
        f"unsupported interaction architecture: {INTERACTION_ARCHITECTURE}"
    )

model = dict(
    # Keep the original V3 BEV/generator path. NAVFormer activates a separate
    # frozen inference-only tracking branch only because the selector requests
    # explicit track states.
    process_perception=False,
    planning_head=dict(
        scene_selector=dict(
            architecture=INTERACTION_ARCHITECTURE,
            feature_dim=256,
            model_dim=256,
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
            interaction_hidden_dim=128,
            num_agent_classes=10,
            max_agents=30,
            seconds_per_step=0.5,
            num_interaction_heads=4,
        )
    ),
)

selector_reward_contract = dict(
    policy_objective="exact_complete_action_expected_advantage",
    architecture=INTERACTION_ARCHITECTURE,
    residual_on_frozen_reference=True,
    generator_frozen=True,
    tracker_frozen=True,
    tracker_execution="inference_only_without_gt_matching",
    track_score_threshold=0.35,
    maximum_tracks=30,
    motion_assumption="deterministic_constant_velocity_0p5s_x_8",
    new_annotations=False,
    reward_components_as_inputs=False,
    fixed_candidate_count=20,
)
