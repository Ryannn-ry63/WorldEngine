import numpy as np
import pytest
import torch

from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import (
    DiffusionGRPOOnlineSelectorPlanningHead,
)


def head_kwargs(tmp_path):
    anchor_path = tmp_path / "v2_anchors.npy"
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


def test_exact_group_objective_matches_complete_action_expectation(tmp_path):
    head = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path),
        online_reward=None,
        policy_objective="exact_group_grpo",
        policy_temperature=4.0,
        kl_weight=0.0,
    )
    logits = torch.tensor([[2.0, -1.0, 0.5, 7.0]], requires_grad=True)
    reference = torch.tensor([[1.0, 0.0, -0.5, 9.0]])
    rewards = torch.tensor([[0.1, 0.8, 0.4, float("nan")]])
    valid = torch.tensor([[True, True, True, False]])
    losses = head.loss(
        result={
            "selector_logits": logits,
            "reference_selector_logits": reference,
            "candidate_rewards": rewards,
            "candidate_reward_valid_mask": valid,
        }
    )

    safe_rewards = rewards[:, :3]
    centered = safe_rewards - safe_rewards.mean(dim=-1, keepdim=True)
    advantage = centered / centered.square().mean(
        dim=-1, keepdim=True
    ).sqrt()
    probability = (logits[:, :3] / 4.0).softmax(dim=-1)
    expected = -(probability * advantage).sum(dim=-1).mean()
    assert losses["loss.grpo_selector_policy"].item() == pytest.approx(
        expected.item()
    )
    assert losses["grpo.selector.fixed_clip_applied"].item() == 0.0
    assert losses["grpo.selector.policy_temperature"].item() == 4.0


def test_exact_group_objective_crosses_v1_fixed_clip_barrier(tmp_path):
    rewards = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    reference = torch.tensor([[torch.log(torch.tensor(40.0)), 0.0]])
    current_probability = torch.tensor([[0.78, 0.22]])
    current = current_probability.log().detach().requires_grad_(True)

    clipped = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path),
        online_reward=None,
        policy_objective="clipped_reference_grpo",
        clip_epsilon=0.2,
        kl_weight=0.0,
    )
    clipped_loss = clipped.loss(
        result={
            "selector_logits": current,
            "reference_selector_logits": reference,
            "candidate_rewards": rewards,
            "candidate_reward_valid_mask": valid,
        }
    )["loss.grpo_selector_policy"]
    clipped_loss.backward()
    assert torch.allclose(current.grad, torch.zeros_like(current.grad), atol=1e-7)

    exact_logits = current.detach().clone().requires_grad_(True)
    exact = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path),
        online_reward=None,
        policy_objective="exact_group_grpo",
        kl_weight=0.0,
    )
    exact_loss = exact.loss(
        result={
            "selector_logits": exact_logits,
            "reference_selector_logits": reference,
            "candidate_rewards": rewards,
            "candidate_reward_valid_mask": valid,
        }
    )["loss.grpo_selector_policy"]
    exact_loss.backward()
    assert exact_logits.grad[0, 1] < 0.0
    assert exact_logits.grad[0, 0] > 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"policy_objective": "unknown"}, "unsupported policy_objective"),
        ({"policy_temperature": 0.0}, "policy_temperature must be positive"),
    ],
)
def test_v2_objective_configuration_fails_closed(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        DiffusionGRPOOnlineSelectorPlanningHead(
            **head_kwargs(tmp_path), online_reward=None, **kwargs
        )
