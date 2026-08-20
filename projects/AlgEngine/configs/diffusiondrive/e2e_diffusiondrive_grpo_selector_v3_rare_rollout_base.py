"""Epoch-100 DiffusionDrive behavior policy for rare-scene online rollout.

The V3 residual selector is initialized to exact zero and no trained selector
weights are loaded.  Therefore the deployed action is exactly the immutable
epoch-100 DiffusionDrive action, while the forward pass exports the complete
dynamic 20-candidate set and detached scene context for SimEngine scoring.
"""

import os


_base_ = ["./e2e_diffusiondrive_grpo_selector_v3.py"]

model = dict(
    planning_head=dict(
        online_reward=None,
        export_rollout_context=True,
        kl_weight=0.0,
        candidate_noise_namespace=os.getenv(
            "DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE",
            "diffusiondrive_v3_rare_rollout_v1",
        ),
    )
)

selector_rollout_contract = dict(
    schema_version=2,
    experiment="diffusiondrive_v3_rare_rollout_v1",
    source_policy="immutable_epoch100_diffusiondrive",
    source_checkpoint_sha256=(
        "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
    ),
    residual_selector_initialization="exact_zero",
    trained_v3_checkpoint_loaded=False,
    deployed_action_parity_required=True,
    num_dynamic_candidates=20,
    candidate_context_export=True,
    reward_owner="simengine_dynamic_candidate_reward",
    pdm_progress_contract="pairwise_raw_progress_then_candidate_gate_v1",
    fixed_vocabulary_size=None,
    imitation_loss=False,
)
