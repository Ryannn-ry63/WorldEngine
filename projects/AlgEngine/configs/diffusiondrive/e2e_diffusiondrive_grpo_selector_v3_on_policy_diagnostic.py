"""Frozen-generator, trained-selector rollout for the R1 root-cause audit."""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]


def required_environment(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"R1 on-policy rollout requires {name}")
    return value.strip()


policy_family = required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_FAMILY")
if policy_family not in {"scalar_v3", "gate_conditioned"}:
    raise RuntimeError(f"unsupported R1 behavior policy: {policy_family}")
train_seed = int(required_environment("DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED"))
if train_seed not in {0, 1, 2}:
    raise RuntimeError(f"unsupported R1 train seed: {train_seed}")
diagnostic_split = required_environment("DIFFUSIONDRIVE_DIAGNOSTIC_SPLIT")
if diagnostic_split not in {"smoke8", "cl_dev58"}:
    raise RuntimeError(f"unsupported R1 diagnostic split: {diagnostic_split}")
checkpoint_sha256 = required_environment(
    "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256"
)
manifest_sha256 = required_environment(
    "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256"
)
implementation_sha256 = required_environment(
    "DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"
)

model = dict(
    planning_head=dict(
        online_reward=None,
        export_rollout_context=True,
        kl_weight=0.0,
        candidate_noise_namespace=required_environment(
            "DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE"
        ),
    )
)

selector_rollout_contract = dict(
    schema_version=3,
    experiment="diffusiondrive_selector_root_cause_r1",
    source_policy="trained_v3_selector",
    behavior_policy_family=policy_family,
    behavior_policy_train_seed=train_seed,
    expected_checkpoint_sha256=checkpoint_sha256,
    behavior_checkpoint_manifest_sha256=manifest_sha256,
    rollout_implementation_sha256=implementation_sha256,
    trained_v3_checkpoint_loaded=True,
    deployed_action_parity_required=False,
    num_dynamic_candidates=20,
    candidate_context_export=True,
    reward_owner="simengine_dynamic_candidate_reward",
    reward_scalar="official_pairwise_pdm",
    reward_components_role="diagnostics_only",
    generator_frozen=True,
    perception_frozen=True,
    base_selector_frozen=True,
    diagnostic_split=diagnostic_split,
    react_type="R",
    training_data_consumed=False,
    development_only=True,
)

