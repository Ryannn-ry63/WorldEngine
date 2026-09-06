"""Opt-in deployment only. The shared full checkpoint remains scalar V3."""
import os

_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]

def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"CFPI deployment requires {name}")
    return value

model = dict(planning_head=dict(
    online_reward=None, export_rollout_context=True, kl_weight=0.0,
    scene_selector_score_mode="residual",
    candidate_noise_namespace=required("DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE")))

selector_rollout_contract = dict(
    schema_version=10, experiment="selector_cfpi_continuous_deployment_v1",
    expected_checkpoint_sha256=required("DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256"),
    behavior_checkpoint_manifest_sha256=required("DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256"),
    rollout_implementation_sha256=required("DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"),
    deployment_routing_sha256=required("DIFFUSIONDRIVE_CFPI_DEPLOYMENT_ROUTING_SHA256"),
    behavior_policy_family="scalar_v3_shared_forward_selector_bank", behavior_policy_train_seed=0,
    generator_frozen=True, perception_frozen=True, base_selector_frozen=True,
    residual_selector_frozen=True, deployed_action_parity_required=False,
    inference_uses_reward_or_q=False, candidate_context_export=True, num_dynamic_candidates=20,
    action_score_timing="pre_action", source_data_split="train", react_type="R",
    diagnostic_split=required("DIFFUSIONDRIVE_CFPI_DEPLOYMENT_COLLECTION_ID"),
    development_consumed=False, test_consumed=False,
    reward_owner="closed_loop_metric_manager_only", online_candidate_reward_computed=False)

# Top-level flags avoid replacing the inherited/default sim configuration.
cfpi_deployment_routing = required("DIFFUSIONDRIVE_CFPI_DEPLOYMENT_ROUTING")
cfpi_deployment_routing_sha256 = required("DIFFUSIONDRIVE_CFPI_DEPLOYMENT_ROUTING_SHA256")
