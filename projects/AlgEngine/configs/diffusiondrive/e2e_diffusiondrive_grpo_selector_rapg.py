"""Scalar-only post-trained selectors over the frozen DiffusionDrive set.

The epoch-100 perception stack, generator, candidate set and original selector
remain frozen.  The new module consumes only deployment-available scene,
trajectory and frozen-reference information and returns a residual logit.
"""

import os


selector_architecture = os.getenv(
    "DIFFUSIONDRIVE_RAPG_ARCHITECTURE",
    "reference_anchored_preference_graph",
)
if selector_architecture not in (
    "reference_anchored_preference_graph",
    "trajectory_set_reasoner",
    "capacity_matched_unary",
    "incumbent_verified_preference",
):
    raise ValueError(f"unsupported RAPG architecture: {selector_architecture}")

reasoner_ablation = os.getenv("DIFFUSIONDRIVE_RAPG_ABLATION", "relational_only")
if reasoner_ablation not in (
    "full",
    "temporal_only",
    "relational_only",
    "no_scene_context",
):
    raise ValueError(f"unsupported RAPG ablation: {reasoner_ablation}")

_base_ = ["./e2e_diffusiondrive_grpo_selector.py"]

scene_selector = dict(
    architecture=selector_architecture,
    feature_dim=256,
    model_dim=256,
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
)
if selector_architecture == "reference_anchored_preference_graph":
    scene_selector["pairwise_hidden_dim"] = int(
        os.getenv("DIFFUSIONDRIVE_RAPG_PAIR_HIDDEN_DIM", "128")
    )
elif selector_architecture == "capacity_matched_unary":
    scene_selector["capacity_hidden_dim"] = int(
        os.getenv("DIFFUSIONDRIVE_RAPG_CONTROL_HIDDEN_DIM", "277")
    )
elif selector_architecture == "incumbent_verified_preference":
    scene_selector["verifier_hidden_dim"] = int(
        os.getenv("DIFFUSIONDRIVE_IVPS_VERIFIER_HIDDEN_DIM", "128")
    )
    scene_selector["override_threshold"] = float(
        os.getenv("DIFFUSIONDRIVE_IVPS_OVERRIDE_THRESHOLD", "0.0")
    )

model = dict(
    planning_head=dict(
        policy_objective="exact_group_grpo",
        policy_temperature=float(
            os.getenv("DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE", "1.0")
        ),
        kl_weight=float(os.getenv("DIFFUSIONDRIVE_GRPO_KL_WEIGHT", "0.001")),
        scene_selector=scene_selector,
    )
)

selector_reward_contract = dict(
    method=(
        "incumbent_verified_preference_selection_v1"
        if selector_architecture == "incumbent_verified_preference"
        else "reference_anchored_preference_graph_v1"
    ),
    policy_objective="official_scalar_pdm_only",
    inference=(
        "proposal_argmax_then_learned_incumbent_verification"
        if selector_architecture == "incumbent_verified_preference"
        else "plain_argmax_over_same_20_candidates"
    ),
    residual_on_frozen_reference=True,
    reference_logits_are_deployment_available=True,
    generator_frozen=True,
    reward_components_as_inputs=False,
    future_labels_as_inputs=False,
    component_conditioned_loss=False,
    development_split="log_disjoint_navtrain_train_only",
    certification_split="log_disjoint_navtrain_held_out_once",
)
