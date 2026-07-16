"""Focused tests for the standalone generation-GRPO math."""

import importlib.util
from pathlib import Path

import torch


UTILS_PATH = (
    Path(__file__).parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_utils.py"
)
SPEC = importlib.util.spec_from_file_location("diffusion_grpo_utils", UTILS_PATH)
grpo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grpo)


def test_diagonal_gaussian_log_prob_shape_and_peak():
    mean = torch.zeros(2, 3, 8, 2)
    at_mean = grpo.diagonal_gaussian_log_prob(mean, mean, 0.5)
    displaced = grpo.diagonal_gaussian_log_prob(
        torch.ones_like(mean), mean, 0.5
    )
    assert at_mean.shape == (2, 3)
    assert torch.all(at_mean > displaced)


def test_generation_objective_is_finite_and_differentiable():
    current = torch.zeros(2, 4, 2, requires_grad=True)
    old = torch.zeros_like(current)
    kl = torch.zeros_like(current)
    rewards = torch.tensor([
        [0.1, 0.4, 0.3, 0.2],
        [0.8, 0.1, 0.5, 0.4],
    ])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    result = grpo.generation_grpo_objective(
        current, old, kl, rewards, valid, selected_indices=torch.tensor([1, 0])
    )
    loss = result["policy_loss"] + 0.1 * result["kl_loss"]
    assert torch.isfinite(loss)
    assert result["valid_group_fraction"].item() == 1.0
    assert result["selection_regret"].item() == 0.0
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_constant_reward_group_is_skipped():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    rewards = torch.ones(1, 3)
    result = grpo.generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        torch.ones_like(rewards, dtype=torch.bool),
    )
    assert result["policy_loss"].item() == 0.0
    assert result["valid_group_fraction"].item() == 0.0


def test_all_invalid_group_has_finite_zero_metrics():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    rewards = torch.full((1, 3), torch.nan)
    result = grpo.generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        torch.zeros_like(rewards, dtype=torch.bool),
    )
    for key in (
        "policy_loss", "kl_loss", "selected_reward", "oracle_reward",
        "selection_regret", "ratio_mean", "clip_fraction",
    ):
        assert torch.isfinite(result[key])
        assert result[key].item() == 0.0


def test_interpolation_preserves_each_two_hz_endpoint_and_wraps_heading():
    trajectory = torch.zeros(1, 8, 3)
    trajectory[0, :, 0] = torch.arange(1, 9)
    trajectory[0, :, 2] = torch.tensor([
        3.0, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8
    ])
    dense = grpo.interpolate_trajectory_8_to_40(trajectory)
    assert dense.shape == (1, 40, 3)
    assert torch.allclose(dense[:, 4::5, :2], trajectory[:, :, :2])
    heading_error = torch.atan2(
        torch.sin(dense[:, 4::5, 2] - trajectory[:, :, 2]),
        torch.cos(dense[:, 4::5, 2] - trajectory[:, :, 2]),
    )
    assert torch.allclose(heading_error, torch.zeros_like(heading_error), atol=1e-6)
