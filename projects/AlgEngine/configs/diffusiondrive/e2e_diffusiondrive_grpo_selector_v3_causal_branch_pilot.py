"""Frozen scalar-V3 rollout contract for train-only causal-branch pilot v2."""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]


def required_environment(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"causal-branch pilot requires {name}")
    return value.strip()


train_seed = int(required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED"))
if train_seed != 0:
    raise RuntimeError("causal-branch pilot v2 freezes scalar V3 train seed 0")
phase = required_environment("DIFFUSIONDRIVE_CAUSAL_BRANCH_PHASE")
if phase not in {
    "pipeline_smoke8",
    "baseline_a",
    "baseline_b",
    "intervention_smoke8",
    "intervention_target",
}:
    raise RuntimeError(f"unsupported causal-branch phase: {phase}")
mode = required_environment("DIFFUSIONDRIVE_CAUSAL_BRANCH_MODE")
if mode not in {"observe_only", "one_shot_oracle", "one_shot_matched"}:
    raise RuntimeError(f"unsupported causal-branch mode: {mode}")
target_sha256 = required_environment(
    "DIFFUSIONDRIVE_CAUSAL_BRANCH_TARGET_MANIFEST_SHA256"
)
if (mode == "observe_only") != (target_sha256 == "none"):
    raise RuntimeError("observe-only must use target SHA 'none' and interventions must not")

checkpoint_sha256 = required_environment(
    "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256"
)
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
    schema_version=7,
    experiment="diffusiondrive_selector_causal_branch_pilot",
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
    diagnostic_split=phase,
    react_type="R",
    training_data_consumed=False,
    train_only=True,
    development_only=False,
    diagnostic_only=True,
    action_score_timing="pre_action",
    oracle_intervention_mode=mode,
    oracle_target_manifest_sha256=target_sha256,
    oracle_future_information_used=True,
    oracle_intervention_treatment="frozen_baseline_index_one_step_or_magnitude_matched_control",
    target_reward_repeatability_role="eligibility_rechecked_in_each_intervention_run",
    causal_estimand="one_step_local_oracle_with_noop_primary_and_magnitude_matched_specificity",
    primary_causal_estimand="one_step_local_oracle_minus_no_intervention_v3_baseline",
    specificity_causal_estimand="one_step_local_oracle_minus_magnitude_matched_nonimproving_control",
    causal_gate_requires_current_oracle_headroom_gt_0p02=True,
    matched_control_requires_current_headroom_le_0p005=True,
    matched_control_match_metric="policy_relative_xy_ade_magnitude",
    matched_control_max_absolute_ade_error_m=0.5,
    matched_control_max_relative_ade_error=0.5,
    matched_control_trajectory_shape_and_yaw_role="diagnostics_only",
)

