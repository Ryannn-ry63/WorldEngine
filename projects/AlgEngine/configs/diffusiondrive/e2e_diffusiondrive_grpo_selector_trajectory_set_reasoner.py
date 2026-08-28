"""Ungated exact-GRPO selector with explicit trajectory-set reasoning.

This config keeps the epoch-100 generator and reference selector frozen.  The
new selector sees only deployment-available candidate/scene features and adds
a zero-initialized residual to the immutable reference logits.
"""

import os


reasoner_ablation = os.getenv(
    "DIFFUSIONDRIVE_GRPO_REASONER_ABLATION", "full"
)
if reasoner_ablation not in ("full", "temporal_only", "relational_only", "no_scene_context"):
    raise ValueError(f"unsupported reasoner ablation: {reasoner_ablation}")

_base_ = ["./e2e_diffusiondrive_grpo_selector.py"]

model = dict(
    planning_head=dict(
        policy_objective="exact_group_grpo",
        policy_temperature=float(
            os.getenv("DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE", "1.0")
        ),
        kl_weight=float(os.getenv("DIFFUSIONDRIVE_GRPO_KL_WEIGHT", "0.001")),
        scene_selector=dict(
            architecture="trajectory_set_reasoner",
            feature_dim=256,
            model_dim=int(os.getenv("DIFFUSIONDRIVE_GRPO_REASONER_MODEL_DIM", "256")),
            route_bev_dim=256,
            context_dim=256,
            step_geometry_dim=11,
            relation_dim=41,
            geometry_hidden_dim=128,
            relation_hidden_dim=128,
            num_heads=4,
            feedforward_dim=512,
            num_route_steps=8,
            num_temporal_layers=2,
            num_relation_layers=2,
            use_temporal_reasoning=reasoner_ablation != "relational_only",
            use_relational_reasoning=reasoner_ablation != "temporal_only",
            use_route_bev=True,
            use_scene_context=reasoner_ablation != "no_scene_context",
        ),
    )
)

selector_reward_contract = dict(
    policy_objective="exact_complete_action_expected_advantage",
    architecture="trajectory_set_reasoning_residual_selector_v1",
    ablation=reasoner_ablation,
    policy_temperature="environment_selected",
    fixed_ratio_clip=False,
    residual_on_frozen_reference=True,
    generator_frozen=True,
    trainable_selector_tensors="architecture_defined",
    imitation_loss=False,
    ranking_loss=False,
    gate=False,
    reward_components_as_inputs=False,
    future_labels_as_inputs=False,
    model_selection_split="navtrain_scene_disjoint_development",
    certification_split="navtrain_scene_disjoint_certification",
)
