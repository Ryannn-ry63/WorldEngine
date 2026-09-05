"""Frozen scalar-V3 rollout contract for the method-neutral CFPI causal cache."""

import os
import re


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]


def required_environment(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"CFPI causal cache requires {name}")
    return value.strip()


train_seed = int(required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED"))
if train_seed != 0:
    raise RuntimeError("CFPI causal cache freezes scalar V3 train seed 0")
collection_id = required_environment("DIFFUSIONDRIVE_CFPI_COLLECTION_ID")
if re.fullmatch(
    r"(?:baseline_train_[ab]|sentinel_policy|(?:pilot64|repeat8)_arm_(?:0[0-9]|1[0-9]))",
    collection_id,
) is None:
    raise RuntimeError(f"unsupported CFPI collection id: {collection_id}")
source_split = required_environment("DIFFUSIONDRIVE_CFPI_SOURCE_SPLIT")
if source_split != "train":
    raise RuntimeError(f"unsupported CFPI source split: {source_split}")
mode = required_environment("DIFFUSIONDRIVE_CFPI_INTERVENTION_MODE")
if mode not in {"observe_only", "one_shot_manifest"}:
    raise RuntimeError(f"unsupported CFPI intervention mode: {mode}")
treatment_sha256 = required_environment("DIFFUSIONDRIVE_CFPI_TREATMENT_MANIFEST_SHA256")
if (mode == "observe_only") != (treatment_sha256 == "none"):
    raise RuntimeError("observe-only must use treatment SHA 'none' and interventions must not")
checkpoint_sha256 = required_environment("DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256")
checkpoint_manifest_sha256 = required_environment(
    "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256"
)
implementation_sha256 = required_environment("DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256")
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
    schema_version=9,
    experiment="diffusiondrive_selector_cfpi_causal_cache_v1",
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
    diagnostic_split=collection_id,
    source_data_split=source_split,
    react_type="R",
    training_data_consumed=False,
    development_consumed=False,
    test_consumed=False,
    diagnostic_only=True,
    action_score_timing="pre_action",
    intervention_mode=mode,
    collection_id=collection_id,
    treatment_manifest_sha256=treatment_sha256,
    treatment_future_outcome_used=False,
    causal_value_definition="one_candidate_action_then_frozen_v3",
    method_training_performed=False,
    inference_uses_reward_or_q=False,
)
