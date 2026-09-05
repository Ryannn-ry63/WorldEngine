from pathlib import Path
import importlib.util
import sys

import pytest
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "lineage_consistent_proximal_grpo",
    SCRIPT_ROOT / "lineage_consistent_proximal_grpo.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LEGACY = __import__("proposal_aware_full_feedback_grpo")


def tensors():
    logits = torch.tensor(
        [[[0.4, -0.2, 0.1], [0.3, 0.0, -0.4], [-0.1, 0.5, 0.2]]],
        requires_grad=True,
    )
    anchor = torch.tensor(
        [[[0.2, -0.1, 0.0], [0.1, 0.1, -0.2], [0.0, 0.3, 0.1]]]
    )
    rewards = torch.tensor(
        [[[0.2, 0.9, 0.1], [0.4, 0.7, 0.2], [0.3, 0.8, 0.0]]]
    )
    valid = torch.ones_like(rewards, dtype=torch.bool)
    return logits, anchor, rewards, valid


def test_a0_is_exact_legacy_direct_loss_and_gradient():
    logits, anchor, rewards, valid = tensors()
    legacy_logits = logits.detach().clone().requires_grad_(True)
    legacy_result = LEGACY.proposal_aware_loss(
        legacy_logits,
        anchor,
        rewards,
        valid,
        arm="direct_grpo",
        temperature=1.0,
        kl_weight=1e-3,
    )
    result = MODULE.lineage_consistent_loss(
        logits,
        anchor,
        rewards,
        valid,
        arm="direct",
        temperature=1.0,
        kl_weight=1e-3,
    )
    for actual, expected in zip(result[:3], legacy_result[:3]):
        assert torch.equal(actual, expected)
    gradient = torch.autograd.grad(result[0], logits)[0]
    legacy_gradient = torch.autograd.grad(legacy_result[0], legacy_logits)[0]
    assert torch.equal(gradient, legacy_gradient)


def test_leave_one_draw_lineage_utility_excludes_heldout_reward():
    rewards = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    utility, utility_valid, complete = MODULE.leave_one_draw_lineage_utility(
        rewards, valid
    )
    expected = torch.tensor([[[4.0, 8.0], [3.0, 6.0], [2.0, 4.0]]])
    assert torch.equal(utility, expected)
    assert utility_valid.all()
    assert complete.all()
    changed = rewards.clone()
    changed[:, 0] = 1000.0
    changed_utility = MODULE.leave_one_draw_lineage_utility(changed, valid)[0]
    assert torch.equal(changed_utility[:, 0], utility[:, 0])


@pytest.mark.parametrize("arm", MODULE.ARMS)
def test_common_candidate_permutation_is_equivariant(arm):
    logits, anchor, rewards, valid = tensors()
    permutation = torch.tensor([2, 0, 1])
    kwargs = {"target_kl": 0.03} if arm in MODULE.PROXIMAL_ARMS else {}
    original = MODULE.lineage_consistent_loss(
        logits, anchor, rewards, valid, arm=arm, **kwargs
    )[0]
    permuted = MODULE.lineage_consistent_loss(
        logits[..., permutation],
        anchor[..., permutation],
        rewards[..., permutation],
        valid[..., permutation],
        arm=arm,
        **kwargs,
    )[0]
    assert torch.allclose(original, permuted, atol=1e-6)


@pytest.mark.parametrize("target_kl", (0.01, 0.03, 0.10))
def test_bisection_hits_requested_kl(target_kl):
    anchor = torch.tensor([[[3.0, 1.0, -1.0]]], dtype=torch.float64)
    advantage = torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=torch.float64)
    valid = torch.ones_like(anchor, dtype=torch.bool)
    active = torch.ones((1, 1), dtype=torch.bool)
    target, eta, achieved, reached = MODULE.kl_budgeted_target(
        anchor,
        advantage,
        valid,
        active,
        target_kl=target_kl,
    )
    assert eta.item() > 0.0
    assert achieved.item() == pytest.approx(target_kl, abs=2e-7)
    assert reached.item()
    assert target.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_solved_tied_and_incomplete_sets_retain_v3_target():
    anchor = torch.tensor(
        [[[4.0, 1.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]
    )
    logits = anchor.clone().requires_grad_(True)
    # Candidate zero is best in every draw, so V3 is solved.
    rewards = torch.tensor(
        [[[0.9, 0.2, 0.1], [0.8, 0.3, 0.1], [0.95, 0.4, 0.1]]]
    )
    valid = torch.ones_like(rewards, dtype=torch.bool)
    _, _, _, diagnostics = MODULE.lineage_consistent_loss(
        logits,
        anchor,
        rewards,
        valid,
        arm="lineage_proximal",
        target_kl=0.03,
    )
    assert diagnostics["retained"].all()
    assert torch.equal(diagnostics["target"], diagnostics["anchor_probability"])

    tied = torch.full_like(rewards, 0.5)
    tied_diagnostics = MODULE.lineage_consistent_loss(
        logits,
        anchor,
        tied,
        valid,
        arm="lineage_proximal",
        target_kl=0.03,
    )[3]
    assert tied_diagnostics["retained"].all()

    incomplete_valid = valid.clone()
    incomplete_valid[:, 1:, 2] = False
    incomplete = MODULE.lineage_consistent_loss(
        logits,
        anchor,
        rewards,
        incomplete_valid,
        arm="lineage_proximal",
        target_kl=0.03,
    )[3]
    assert incomplete["retained"][0, 0]


def test_independent_shuffle_preserves_each_draw_multiset_and_breaks_alignment():
    rewards = torch.arange(24.0).reshape(2, 3, 4)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    shuffled, shuffled_valid, permutations = MODULE.independently_shuffle_lineages(
        rewards, valid, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(rewards.sort(dim=-1).values, shuffled.sort(dim=-1).values)
    assert shuffled_valid.all()
    assert not torch.equal(permutations[:, 0], permutations[:, 1])


def test_objective_source_never_mentions_decomposed_reward_field():
    source = (SCRIPT_ROOT / "lineage_consistent_proximal_grpo.py").read_text()
    forbidden = "candidate_reward_" + "components"
    assert forbidden not in source
