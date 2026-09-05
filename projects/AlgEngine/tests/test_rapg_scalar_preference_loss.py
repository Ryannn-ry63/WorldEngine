import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/grpo_selector_v3_cached_common.py"
)
SPEC = importlib.util.spec_from_file_location("rapg_common", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_equal_scalar_rewards_produce_neutral_pair_targets():
    rewards = torch.tensor([[0.2, 0.2, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    targets, weights, pair_mask, _, active = MODULE.scalar_preference_targets(
        rewards, valid
    )
    assert active.tolist() == [True]
    assert targets[0, 0, 1] == 0.5
    assert targets[0, 1, 0] == 0.5
    assert weights[0, 0, 1] > 0.0
    assert pair_mask[0, 0, 0] == 0


def test_invalid_candidates_never_enter_pair_loss():
    rewards = torch.tensor([[0.1, 0.4, 0.9]])
    valid = torch.tensor([[True, True, False]])
    _, weights, pair_mask, _, _ = MODULE.scalar_preference_targets(rewards, valid)
    assert torch.equal(pair_mask[:, 2], torch.zeros_like(pair_mask[:, 2]))
    assert torch.equal(pair_mask[:, :, 2], torch.zeros_like(pair_mask[:, :, 2]))
    assert torch.equal(weights[:, 2], torch.zeros_like(weights[:, 2]))


def test_preference_gradient_pushes_higher_reward_logit_up():
    logits = torch.zeros(1, 3, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, _ = MODULE.scalar_preference_loss(logits, rewards, valid)
    loss.backward()
    assert logits.grad[0, 2] < 0.0
    assert logits.grad[0, 0] > 0.0


def test_zero_preference_weight_is_exact_group_loss():
    torch.manual_seed(5)
    logits = torch.randn(2, 5)
    reference = torch.randn(2, 5)
    rewards = torch.rand(2, 5)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    exact, policy, kl = MODULE.exact_group_loss(
        logits, reference, rewards, valid, 1.0, 1e-3
    )
    combined, combined_policy, _, combined_kl, _ = (
        MODULE.exact_group_preference_loss(
            logits, reference, rewards, valid, 1.0, 1e-3, 0.0
        )
    )
    assert torch.allclose(combined, exact)
    assert torch.allclose(combined_policy, policy)
    assert torch.allclose(combined_kl, kl)


def test_all_equal_group_is_inactive_and_finite():
    logits = torch.randn(2, 4, requires_grad=True)
    rewards = torch.ones(2, 4)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, diagnostics = MODULE.scalar_preference_loss(logits, rewards, valid)
    assert torch.isfinite(loss)
    assert loss == 0.0
    assert not diagnostics["active"].any()


def test_incumbent_verification_targets_are_tie_aware_and_margin_shifted():
    rewards = torch.tensor([[0.7, 0.7, 0.9]])
    reference = torch.tensor([[3.0, 1.0, 0.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    neutral, mask, difference, incumbent, _ = MODULE.incumbent_verification_targets(
        rewards, reference, valid, reward_margin=0.0, reward_temperature=0.05
    )
    conservative, _, _, _, _ = MODULE.incumbent_verification_targets(
        rewards, reference, valid, reward_margin=0.03, reward_temperature=0.05
    )
    assert incumbent.tolist() == [0]
    assert difference[0, 1] == 0.0
    assert neutral[0, 1] == 0.5
    assert conservative[0, 1] < 0.5
    assert not mask[0, 0]


def test_incumbent_verification_gradient_accepts_better_and_rejects_worse():
    verification = torch.zeros(1, 3, requires_grad=True)
    reference = torch.tensor([[3.0, 1.0, 0.0]])
    proposal = torch.tensor([[0.0, 2.0, 4.0]])
    rewards = torch.tensor([[0.5, 0.2, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, diagnostics = MODULE.incumbent_verification_loss(
        verification, reference, proposal, rewards, valid
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["pair_mask"].sum() == 2
    assert verification.grad[0, 1] > 0.0
    assert verification.grad[0, 2] < 0.0
