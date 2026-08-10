import numpy as np
import pytest
import torch

from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import (
    DiffusionGRPOOnlineSelectorPlanningHead,
)
from mmdet3d_plugin.navformer.dense_heads.diffusion_planning_head import (
    DiffusionPlanningHead,
)


def head_kwargs(tmp_path):
    anchor_path = tmp_path / "formal_anchors.npy"
    np.save(anchor_path, np.zeros((4, 8, 2), dtype=np.float32))
    return dict(
        num_poses=8,
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_bounding_boxes=2,
        num_query_decoder_layers=1,
        query_keyval_size=2,
        num_anchors=4,
        num_diff_decoder_layers=2,
        plan_anchor_path=str(anchor_path),
        bev_h=4,
        bev_w=4,
        num_train_timesteps=1000,
        train_timestep_max=5,
        inference_steps=2,
        trunc_timesteps=2,
        use_nerf=False,
    )


def identical_base_and_formal_heads(tmp_path):
    torch.manual_seed(101)
    base = DiffusionPlanningHead(**head_kwargs(tmp_path))
    torch.manual_seed(202)
    formal = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path), online_reward=None
    )
    current_state = {
        key: value
        for key, value in formal.state_dict().items()
        if not key.startswith("reference_selector.")
    }
    missing, unexpected = base.load_state_dict(current_state, strict=False)
    assert not missing
    assert not unexpected
    formal.initialize_reference_selector()
    return base, formal


def test_baseline_inference_is_exactly_the_reviewed_diffusiondrive_head(tmp_path):
    base, formal = identical_base_and_formal_heads(tmp_path)
    base.eval()
    formal.eval()
    bev = torch.randn(1, 32, 4, 4)
    ego = torch.randn(1, 1, 32)
    agents = torch.randn(1, 2, 32)
    status = torch.randn(1, 1, 32)

    torch.manual_seed(37)
    base_result = base._forward_test(bev, ego, agents, status)
    torch.manual_seed(37)
    formal_result = formal._forward_test(bev, ego, agents, status)

    assert torch.equal(
        base_result["selected_indices"], formal_result["selected_indices"]
    )
    assert torch.allclose(
        base_result["trajectory_8"],
        formal_result["trajectory_8"],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        base_result["trajectory"],
        formal_result["trajectory"],
        atol=1e-6,
        rtol=0.0,
    )


def test_selector_perturbation_cannot_change_generated_candidates(tmp_path):
    _, formal = identical_base_and_formal_heads(tmp_path)
    formal.eval()
    formal.set_candidate_noise_namespace("candidate-invariance")
    bev = torch.randn(1, 32, 4, 4)
    ego = torch.randn(1, 1, 32)
    agents = torch.randn(1, 2, 32)
    status = torch.randn(1, 1, 32)

    formal._active_sample_tokens = ["token-a"]
    before, before_feature = formal._generate_frozen_candidates(
        bev, ego, agents, status
    )
    with torch.no_grad():
        for parameter in formal._current_selector().parameters():
            parameter.add_(torch.randn_like(parameter))
    formal._active_sample_tokens = ["token-a"]
    after, after_feature = formal._generate_frozen_candidates(
        bev, ego, agents, status
    )
    formal._active_sample_tokens = None

    assert torch.equal(before, after)
    assert torch.equal(before_feature, after_feature)


def test_candidate_noise_is_token_paired_and_global_rng_independent(tmp_path):
    _, formal = identical_base_and_formal_heads(tmp_path)
    image = torch.zeros((1, 4, 8, 2))
    formal.set_candidate_noise_namespace("calibration-seed-0")
    formal._active_sample_tokens = ["token-a"]
    first = formal._candidate_noise(image)
    torch.manual_seed(999)
    second = formal._candidate_noise(image)
    formal._active_sample_tokens = ["token-b"]
    other = formal._candidate_noise(image)
    formal._active_sample_tokens = None

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_per_sample_diagnostics_use_deployed_argmax_not_expected_reward(tmp_path):
    formal = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path), online_reward=None
    )
    rewards = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    components = rewards[:, :, None].repeat(1, 1, 6)
    result = {
        "selector_logits": torch.tensor([[0.0, 0.0, 5.0, 0.0]]),
        "reference_selector_logits": torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        "candidate_rewards": rewards,
        "candidate_reward_components": components,
        "candidate_reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "candidate_trajectories_8": torch.zeros((1, 4, 8, 3)),
    }
    diagnostics = formal.selector_diagnostics_per_sample(result)
    assert diagnostics["current_index"].item() == 2
    assert diagnostics["reference_index"].item() == 0
    assert diagnostics["oracle_index"].item() == 3
    assert diagnostics["current_reward"].item() == pytest.approx(0.3)
    assert diagnostics["reference_reward"].item() == pytest.approx(0.1)
    assert diagnostics["top1_reward_gain"].item() == pytest.approx(0.2)


def test_clipped_objective_has_correct_sign_for_negative_advantage(tmp_path):
    formal = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path),
        online_reward=None,
        clip_epsilon=0.2,
        kl_weight=0.0,
    )
    reference_logits = torch.zeros((1, 4))
    current_logits = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    losses = formal.loss(
        result={
            "selector_logits": current_logits,
            "reference_selector_logits": reference_logits,
            "candidate_rewards": rewards,
            "candidate_reward_valid_mask": torch.ones_like(
                rewards, dtype=torch.bool
            ),
        }
    )
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    advantages = centered / centered.square().mean(dim=-1, keepdim=True).sqrt()
    current_log_prob = current_logits.log_softmax(dim=-1)
    reference_log_prob = reference_logits.log_softmax(dim=-1)
    ratio = (current_log_prob - reference_log_prob).exp()
    clipped = ratio.clamp(0.8, 1.2)
    expected = -(
        reference_logits.softmax(dim=-1)
        * torch.minimum(ratio * advantages, clipped * advantages)
    ).sum(dim=-1).mean()
    assert losses["loss.grpo_selector_policy"].item() == pytest.approx(
        expected.item()
    )


def test_none_online_reward_fails_closed_with_train_mode_hint(tmp_path):
    formal = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path), online_reward=None
    )
    with pytest.raises(KeyError, match="model must be in train mode"):
        formal.loss(
            result={
                "selector_logits": torch.zeros((1, 4)),
                "reference_selector_logits": torch.zeros((1, 4)),
                "candidate_rewards": None,
            },
            gt_pdm_score={"score": None},
        )
