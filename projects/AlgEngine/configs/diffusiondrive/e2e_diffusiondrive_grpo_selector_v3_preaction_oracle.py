"""Frozen V3 selector for the R1.5 pre-action oracle causal intervention."""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]


def required_environment(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"R1.5 rollout requires {name}")
    return value.strip()


train_seed = int(required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED"))
if train_seed not in {0, 1}:
    raise RuntimeError(f"unsupported R1.5 train seed: {train_seed}")
phase = required_environment("DIFFUSIONDRIVE_ORACLE_R15_PHASE")
if phase not in {
    "baseline_smoke8",
    "baseline_cl_dev58",
    "intervention_smoke8",
    "intervention_target8",
}:
    raise RuntimeError(f"unsupported R1.5 phase: {phase}")
mode = required_environment("DIFFUSIONDRIVE_ORACLE_R15_MODE")
if mode not in {"observe_only", "one_shot_oracle", "persistent_oracle"}:
    raise RuntimeError(f"unsupported R1.5 mode: {mode}")
target_sha256 = required_environment(
    "DIFFUSIONDRIVE_ORACLE_TARGET_MANIFEST_SHA256"
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
    schema_version=5,
    experiment="diffusiondrive_selector_oracle_r15",
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
    development_only=True,
    action_score_timing="pre_action",
    oracle_intervention_mode=mode,
    oracle_target_manifest_sha256=target_sha256,
    oracle_future_information_used=True,
    oracle_intervention_treatment="frozen_baseline_index_at_target_then_current_oracle",
    target_reward_repeatability_role="stability_sensitivity_not_treatment_identity",
    causal_gate_requires_current_headroom_gt_0p02=True,
)
