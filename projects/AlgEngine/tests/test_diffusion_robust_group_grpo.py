from pathlib import Path
import importlib.util

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/diffusion_robust_group_grpo.py"
)
SPEC = importlib.util.spec_from_file_location("diffusion_robust_group_grpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_tensors():
    anchor = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, 0.0, -0.5], [0.2, 0.1, 0.0]]]
    )
    rewards = torch.tensor(
        [[[0.3, 0.8, 0.1], [0.9, 0.2, 0.1], [0.2, 0.4, 0.7]]]
    )
    valid = torch.ones_like(rewards, dtype=torch.bool)
    return anchor, rewards, valid


@pytest.mark.parametrize("selection_mode", ["soft", "straight_through_top1"])
@pytest.mark.parametrize(
    "aggregation", ["mean", "softmin", "bounded_mean_risk"]
)
def test_anchor_initialization_has_exactly_zero_gain(selection_mode, aggregation):
    anchor, rewards, valid = fixture_tensors()
    logits = anchor.clone().requires_grad_(True)
    loss, policy, kl, diagnostics = MODULE.diffusion_robust_group_loss(
        logits,
        anchor,
        rewards,
        valid,
        selection_mode=selection_mode,
        aggregation=aggregation,
        risk_temperature=0.02,
    )
    assert torch.equal(diagnostics["per_draw_gain"], torch.zeros(1, 3))
    assert torch.equal(diagnostics["aggregate_gain"], torch.zeros(1))
    assert policy.item() == 0.0
    assert kl.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_softmin_focuses_on_worst_draw_and_is_below_mean():
    gains = torch.tensor([[0.10, -0.05, 0.02]])
    robust, weights = MODULE.aggregate_draw_gains(
        gains, aggregation="softmin", risk_temperature=0.02
    )
    assert robust.item() < gains.mean().item()
    assert robust.item() > gains.min().item()
    assert weights.argmax(dim=-1).item() == gains.argmin(dim=-1).item()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


def test_bounded_mean_risk_keeps_every_draw_active():
    gains = torch.tensor([[10.0, 0.0, -10.0]])
    aggregate, weights = MODULE.aggregate_draw_gains(
        gains,
        aggregation="bounded_mean_risk",
        risk_temperature=0.05,
        risk_mix=0.5,
    )
    assert torch.isfinite(aggregate).all()
    assert weights.min().item() >= 1.0 / 6.0 - 1e-6
    assert weights.max().item() <= 2.0 / 3.0 + 1e-6
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


def test_straight_through_forward_is_deployed_top1_but_has_gradient():
    anchor, rewards, valid = fixture_tensors()
    logits = anchor.clone()
    logits[0, 0, 1] = 2.0
    logits.requires_grad_(True)
    gains, _, diagnostics = MODULE.per_draw_reference_gain(
        logits,
        anchor,
        rewards,
        valid,
        selection_mode="straight_through_top1",
    )
    expected = rewards[0, 0, 1] - rewards[0, 0, 0]
    assert gains[0, 0].item() == pytest.approx(expected.item())
    diagnostics["current_value"].sum().backward()
    assert logits.grad.abs().sum() > 0.0


def test_raw_reward_keeps_regret_magnitude_information():
    anchor = torch.zeros(2, 2, 2)
    logits = anchor.clone().requires_grad_(True)
    rewards = torch.tensor(
        [[[0.0, 0.01], [0.0, 0.01]], [[0.0, 1.0], [0.0, 1.0]]]
    )
    valid = torch.ones_like(rewards, dtype=torch.bool)
    gains, _, _ = MODULE.per_draw_reference_gain(
        logits, anchor, rewards, valid, selection_mode="soft"
    )
    gradients = torch.autograd.grad(gains.sum(), logits)[0]
    small = gradients[0].abs().sum()
    large = gradients[1].abs().sum()
    assert large.item() == pytest.approx(100.0 * small.item(), rel=1e-5)


def test_invalid_candidate_never_receives_probability():
    anchor, rewards, valid = fixture_tensors()
    valid[..., 2] = False
    rewards[..., 2] = float("nan")
    _, _, diagnostics = MODULE.per_draw_reference_gain(
        anchor, anchor, rewards, valid, selection_mode="soft"
    )
    assert torch.equal(
        diagnostics["probability"][..., 2],
        torch.zeros_like(diagnostics["probability"][..., 2]),
    )


def test_draw_permutation_does_not_change_objective():
    anchor, rewards, valid = fixture_tensors()
    logits = (anchor + torch.tensor([[[0.0, 0.3, 0.0]] * 3])).requires_grad_(True)
    first = MODULE.diffusion_robust_group_loss(
        logits, anchor, rewards, valid, aggregation="softmin"
    )[0]
    permutation = torch.tensor([2, 0, 1])
    second = MODULE.diffusion_robust_group_loss(
        logits[:, permutation],
        anchor[:, permutation],
        rewards[:, permutation],
        valid[:, permutation],
        aggregation="softmin",
    )[0]
    assert torch.allclose(first, second)


def test_independent_candidate_permutation_per_draw_is_invariant():
    anchor, rewards, valid = fixture_tensors()
    logits = (anchor + 0.2 * rewards).requires_grad_(True)
    first = MODULE.diffusion_robust_group_loss(
        logits,
        anchor,
        rewards,
        valid,
        aggregation="bounded_mean_risk",
        risk_temperature=0.05,
        risk_mix=0.5,
    )[0]
    permutations = (
        torch.tensor([2, 0, 1]),
        torch.tensor([1, 2, 0]),
        torch.tensor([0, 2, 1]),
    )
    permuted_logits = torch.stack(
        [logits[:, draw, permutations[draw]] for draw in range(3)], dim=1
    )
    permuted_anchor = torch.stack(
        [anchor[:, draw, permutations[draw]] for draw in range(3)], dim=1
    )
    permuted_rewards = torch.stack(
        [rewards[:, draw, permutations[draw]] for draw in range(3)], dim=1
    )
    permuted_valid = torch.stack(
        [valid[:, draw, permutations[draw]] for draw in range(3)], dim=1
    )
    second = MODULE.diffusion_robust_group_loss(
        permuted_logits,
        permuted_anchor,
        permuted_rewards,
        permuted_valid,
        aggregation="bounded_mean_risk",
        risk_temperature=0.05,
        risk_mix=0.5,
    )[0]
    assert torch.allclose(first, second)


def test_absolute_value_changes_only_outer_training_signal():
    anchor, rewards, valid = fixture_tensors()
    logits = anchor.clone().requires_grad_(True)
    _, relative_policy, _, relative = MODULE.diffusion_robust_group_loss(
        logits,
        anchor,
        rewards,
        valid,
        aggregation="bounded_mean_risk",
        objective_mode="relative_gain",
    )
    _, absolute_policy, _, absolute = MODULE.diffusion_robust_group_loss(
        logits,
        anchor,
        rewards,
        valid,
        aggregation="bounded_mean_risk",
        objective_mode="absolute_value",
    )
    assert torch.equal(relative["per_draw_gain"], absolute["per_draw_gain"])
    assert relative_policy.item() == 0.0
    assert absolute_policy.item() != 0.0
