import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "probe_dcsr_top1_objective", SCRIPT_DIR / "probe_dcsr_top1_objective.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_comparator_is_antisymmetric():
    torch.manual_seed(2)
    model = MODULE.Top1Comparator(hidden_dim=16)
    challenger = torch.randn(6, 257)
    incumbent = torch.randn(6, 257)
    relation_ci = torch.randn(6, 41)
    relation_ic = torch.randn(6, 41)
    forward = model(challenger, incumbent, relation_ci, relation_ic)
    reverse = model(incumbent, challenger, relation_ic, relation_ci)
    torch.testing.assert_close(forward, -reverse)


def test_structured_loss_is_zero_when_top1_margin_covers_all_regret_costs():
    rewards = torch.tensor([[0.5, 0.8, 0.4, 0.3, 0.2]])
    evidence = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss, active = MODULE.structured_top1_loss(evidence, rewards)
    assert active.tolist() == [True]
    torch.testing.assert_close(loss, torch.tensor(0.0))

    wrong = torch.zeros_like(evidence)
    wrong_loss, _ = MODULE.structured_top1_loss(wrong, rewards)
    torch.testing.assert_close(wrong_loss, torch.tensor(0.6))


def test_structured_loss_skips_all_reward_ties():
    evidence = torch.randn(3, 4, requires_grad=True)
    rewards = torch.full((3, 5), 0.7)
    loss, active = MODULE.structured_top1_loss(evidence, rewards)
    assert not bool(active.any())
    assert float(loss) == 0.0
    loss.backward()
    assert evidence.grad is not None


def minimal_group(stratum, token, rewards):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    components = torch.ones(5, 6)
    return {
        "stratum": stratum,
        "token": token,
        "log_name": token,
        "top_rewards": rewards,
        "top_components": components,
    }


def test_calibration_falls_back_to_keep_all_when_overrides_are_infeasible():
    groups = [
        minimal_group("common", "common", [0.9, 0.7, 0.6, 0.5, 0.4]),
        minimal_group("rare", "rare", [0.8, 0.6, 0.5, 0.4, 0.3]),
    ]
    scores = {
        0: torch.tensor([0.0, 2.0, 1.0, 0.5, 0.2]),
        1: torch.tensor([0.0, 2.0, 1.0, 0.5, 0.2]),
    }
    threshold, report = MODULE.calibrate_threshold(scores, groups, [0, 1])
    assert threshold == float("inf")
    assert report["keep_all"] is True
    assert report["threshold"] is None


def arm_report(eligible):
    return {"gates": {"eligible": eligible}}


def comparison(passed):
    return {"passed": passed}


def test_only_structured_arms_can_be_authorized():
    reports = {
        "P0_pair_token": arm_report(True),
        "P1_pair_relation": arm_report(True),
        "T0_structured_token": arm_report(False),
        "T1_structured_relation": arm_report(False),
    }
    comparisons = {
        "T0_over_P0": comparison(False),
        "T1_over_P1": comparison(False),
        "T1_over_T0": comparison(False),
        "T1_over_C1": comparison(False),
    }
    decision, arm = MODULE.choose_decision(
        reports, {"passed": True}, comparisons
    )
    assert decision == "STOP_CURRENT_FRAME_SELECTOR_AND_AUDIT_HISTORY"
    assert arm is None


def test_relation_v4_requires_every_objective_and_control_comparison():
    reports = {
        "P0_pair_token": arm_report(False),
        "P1_pair_relation": arm_report(False),
        "T0_structured_token": arm_report(True),
        "T1_structured_relation": arm_report(True),
    }
    comparisons = {
        "T0_over_P0": comparison(True),
        "T1_over_P1": comparison(True),
        "T1_over_T0": comparison(True),
        "T1_over_C1": comparison(True),
    }
    decision, arm = MODULE.choose_decision(
        reports, {"passed": True}, comparisons
    )
    assert (decision, arm) == (
        "AUTHORIZE_DCSR_RELATIONAL_V4",
        "T1_structured_relation",
    )
    comparisons["T1_over_C1"] = comparison(False)
    decision, arm = MODULE.choose_decision(
        reports, {"passed": True}, comparisons
    )
    assert (decision, arm) == (
        "AUTHORIZE_DCSR_TOKEN_V4",
        "T0_structured_token",
    )

