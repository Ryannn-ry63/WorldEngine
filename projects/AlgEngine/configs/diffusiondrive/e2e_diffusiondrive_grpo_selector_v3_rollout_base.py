"""Frozen epoch-100 DiffusionDrive rollout source for selector GRPO R1.

The residual V3 selector starts at exact zero and no post-trained selector is
loaded.  Thus the deployed top-1 action is identical to the immutable
epoch-100 DiffusionDrive baseline, while the same forward pass exports all 20
candidates and their frozen context.  SimEngine, not AlgEngine, scores the
complete action set at the actual closed-loop state.
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
            "diffusiondrive_rollout_v1_seed0",
        ),
    )
)

selector_rollout_contract = dict(
    schema_version=1,
    source_policy="immutable_epoch100_diffusiondrive",
    residual_selector_initialization="exact_zero",
    deployed_action_parity_required=True,
    num_dynamic_candidates=20,
    candidate_context_export=True,
    reward_owner="simengine_dynamic_candidate_reward",
    fixed_vocabulary_size=None,
    imitation_loss=False,
)
