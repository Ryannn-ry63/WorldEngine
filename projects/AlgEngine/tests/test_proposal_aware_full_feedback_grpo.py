from pathlib import Path
import importlib.util

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/proposal_aware_full_feedback_grpo.py"
)
SPEC = importlib.util.spec_from_file_location("proposal_aware_full_feedback_grpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_low_probability_high_reward_direct_gradient_vanishes_but_paf_is_finite():
    anchor = torch.tensor([[20.0, -20.0]])
    rewards = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    direct_logits = anchor.clone().requires_grad_(True)
    direct_loss = MODULE.proposal_aware_loss(
        direct_logits, anchor, rewards, valid, arm="direct_grpo", kl_weight=0.0
    )[0]
    direct_gradient = torch.autograd.grad(direct_loss, direct_logits)[0]

    paf_logits = anchor.clone().requires_grad_(True)
    paf_loss = MODULE.proposal_aware_loss(
        paf_logits, anchor, rewards, valid, arm="paf_grpo", kl_weight=0.0
    )[0]
    paf_gradient = torch.autograd.grad(paf_loss, paf_logits)[0]

    assert direct_gradient[0, 1].abs().item() < 1e-12
    assert paf_gradient[0, 1].item() < -0.1
    assert paf_gradient[0, 1].abs() > 1e10 * direct_gradient[0, 1].abs()


def test_no_recoverable_headroom_preserves_frozen_v3_target():
    anchor = torch.tensor([[3.0, 1.0, -1.0]])
    rewards = torch.tensor([[0.9, 0.1, 0.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    _, _, _, diagnostics = MODULE.proposal_aware_loss(
        anchor.clone().requires_grad_(True), anchor, rewards, valid, arm="paf_grpo"
    )
    assert diagnostics["headroom"].item() == pytest.approx(0.0)
    assert diagnostics["opportunity_weight"].item() == pytest.approx(0.0)
    assert torch.equal(diagnostics["target"], diagnostics["anchor_probability"])


def test_opportunity_weight_has_locked_boundaries():
    anchor_probability = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    rewards = torch.tensor([[0.0, 0.005], [0.0, 0.0525], [0.0, 0.1]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    weight, headroom, _, _ = MODULE.opportunity(rewards, valid, anchor_probability)
    assert torch.allclose(headroom, torch.tensor([0.005, 0.0525, 0.1]))
    assert torch.allclose(weight, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


@pytest.mark.parametrize("arm", MODULE.ARMS)
def test_ties_are_finite_and_carry_no_update(arm):
    anchor = torch.tensor([[1.0, 0.0, -2.0]])
    logits = anchor.clone().requires_grad_(True)
    rewards = torch.tensor([[0.4, 0.4, float("nan")]])
    valid = torch.tensor([[True, True, False]])
    loss, policy, kl, diagnostics = MODULE.proposal_aware_loss(
        logits, anchor, rewards, valid, arm=arm
    )
    assert loss.item() == pytest.approx(0.0)
    assert policy.item() == pytest.approx(0.0)
    assert kl.item() == pytest.approx(0.0)
    assert not diagnostics["active"].item()
    assert diagnostics["probability"][0, 2].item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_candidate_permutation_preserves_every_arm_loss():
    anchor = torch.tensor([[[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]]])
    logits = anchor + torch.tensor([[[0.1, 0.2, 0.0], [0.0, 0.3, -0.1]]])
    rewards = torch.tensor([[[0.2, 0.8, 0.1], [0.9, 0.2, 0.5]]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    for arm in MODULE.ARMS:
        first = MODULE.proposal_aware_loss(logits, anchor, rewards, valid, arm=arm)[0]
        second = MODULE.proposal_aware_loss(
            logits[..., permutation], anchor[..., permutation],
            rewards[..., permutation], valid[..., permutation], arm=arm,
        )[0]
        assert torch.allclose(first, second, atol=1e-6)


def test_direct_arm_matches_analytic_expected_advantage():
    logits = torch.tensor([[0.4, -0.2, 0.1]], requires_grad=True)
    anchor = torch.zeros_like(logits)
    rewards = torch.tensor([[0.2, 0.9, 0.1]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    _, probability = MODULE.masked_log_policy(logits, valid)
    advantage, _ = MODULE.normalized_advantage(rewards, valid)
    expected = -(probability * advantage).sum(dim=-1).mean()
    actual = MODULE.proposal_aware_loss(
        logits, anchor, rewards, valid, arm="direct_grpo", kl_weight=0.0
    )[1]
    assert torch.allclose(actual, expected)
