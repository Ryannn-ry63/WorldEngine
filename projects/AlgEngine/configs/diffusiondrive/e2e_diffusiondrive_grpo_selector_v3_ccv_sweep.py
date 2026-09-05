"""Frozen scalar-V3 rollout contract for candidate causal-value sweep v1."""

import os
import re


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]


def required_environment(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"CCV sweep requires {name}")
    return value.strip()


train_seed = int(required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED"))
if train_seed != 0:
    raise RuntimeError("CCV sweep v1 freezes scalar V3 train seed 0")
treatment_id = required_environment("DIFFUSIONDRIVE_CCV_TREATMENT_ID")
if re.fullmatch(r"(?:sentinel_(?:policy|oracle|matched)|arm_(?:0[0-9]|1[0-9]))", treatment_id) is None:
    raise RuntimeError(f"unsupported CCV treatment id: {treatment_id}")
treatment_sha256 = required_environment("DIFFUSIONDRIVE_CCV_TREATMENT_MANIFEST_SHA256")
checkpoint_sha256 = required_environment("DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256")
checkpoint_manifest_sha256 = required_environment(
    "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256"
)
implementation_sha256 = required_environment(
    "DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"
)
noise_namespace = required_environment("DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE")

model = dict(
    planning_head=dict(
        online_reward=None,
        export_rollout_context=True,
        kl_weight=0.0,
        candidate_noise_namespace=noise_namespace,
    )
)

selector_rollout_contract = dict(
    schema_version=8,
    experiment="diffusiondrive_selector_ccv_sweep_v1",
    source_policy="trained_v3_selector",
    behavior_policy_family="scalar_v3",
    behavior_policy_train_seed=train_seed,
    expected_checkpoint_sha256=checkpoint_sha256,
    behavior_checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    rollout_implementation_sha256=implementation_sha256,
    trained_v3_checkpoint_loaded=True,
    deployed_action_parity_required=False,
    num_dynamic_candidates=20,
    candidate_context_export=True,
    reward_owner="simengine_preaction_candidate_reward",
    reward_scalar="official_pairwise_pdm",
    reward_components_role="diagnostics_only",
    generator_frozen=True,
    perception_frozen=True,
    base_selector_frozen=True,
    residual_selector_frozen=True,
    diagnostic_split=treatment_id,
    react_type="R",
    training_data_consumed=False,
    train_only=True,
    development_only=False,
    diagnostic_only=True,
    action_score_timing="pre_action",
    intervention_mode="one_shot_manifest",
    treatment_id=treatment_id,
    treatment_manifest_sha256=treatment_sha256,
    treatment_index_source="frozen_manifest",
    treatment_future_outcome_used=False,
    target_selection_uses_prior_baseline_outcome=True,
    causal_value_definition="one_candidate_action_then_frozen_v3",
    causal_value_authorized_for_training=False,
)
