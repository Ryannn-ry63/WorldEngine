from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
common = importlib.import_module("grpo_selector_v3_cached_common")


def components(batch=2, candidates=4):
    values = torch.ones(batch, candidates, 6)
    values[..., 5] = 0.0
    return values


def test_gate_conditioned_loss_matches_official_when_quality_is_inactive():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    reference = torch.zeros_like(logits)
    rewards = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    values = components(batch=1)
    values[:, 1:, 0] = 0.0

    expected, expected_policy, expected_kl = common.exact_group_loss(
        logits, reference, rewards, valid, 1.0, 1e-3
    )
    actual, policy, quality, kl, diagnostics = (
        common.gate_conditioned_exact_group_loss(
            logits, reference, rewards, values, valid, 1.0, 1e-3
        )
    )
    assert torch.equal(actual, expected)
    assert torch.equal(policy, expected_policy)
    assert torch.equal(kl, expected_kl)
    assert quality.item() == 0.0
    assert diagnostics["quality_active"].tolist() == [False]


def test_quality_term_ranks_progress_only_inside_safe_subset():
    logits = torch.zeros(1, 4, requires_grad=True)
    reference = torch.zeros_like(logits)
    rewards = torch.tensor([[0.0, 0.0, 0.4, 0.8]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    values = components(batch=1)
    values[0, :2, 0] = 0.0
    values[0, 2:, 2] = torch.tensor([0.2, 0.8])

    loss, _, quality, _, diagnostics = common.gate_conditioned_exact_group_loss(
        logits, reference, rewards, values, valid, 1.0, 0.0
    )
    assert diagnostics["safe"].tolist() == [[False, False, True, True]]
    assert diagnostics["quality_active"].tolist() == [True]
    # The expectation is zero at uniform logits because the normalized
    # advantage is zero-mean, while its gradient still favors higher quality.
    assert quality.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] < logits.grad[0, 2]


def test_quality_term_ignores_ddc_and_requires_nc_dac():
    logits = torch.zeros(1, 4)
    reference = torch.zeros_like(logits)
    rewards = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    values = components(batch=1)
    values[0, 0, 0] = 0.0
    values[0, 1, 1] = 0.0
    values[0, 2, 5] = 0.0
    values[0, 3, 5] = 1.0

    _, _, _, _, diagnostics = common.gate_conditioned_exact_group_loss(
        logits, reference, rewards, values, valid, 1.0, 0.0
    )
    assert diagnostics["safe"].tolist() == [[False, False, True, True]]


def test_shape_mismatch_fails_closed():
    logits = torch.zeros(1, 4)
    rewards = torch.zeros_like(logits)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    try:
        common.gate_conditioned_exact_group_loss(
            logits, logits, rewards, torch.zeros(1, 4, 5), valid, 1.0, 0.0
        )
    except ValueError as error:
        assert "shape [B, K, 6]" in str(error)
    else:
        raise AssertionError("invalid component shape was accepted")
