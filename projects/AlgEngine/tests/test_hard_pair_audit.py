import importlib.util
import inspect
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "probe_hard_pair_information", SCRIPT_DIR / "probe_hard_pair_information.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_locked_top5_pair_constructor_cannot_receive_rewards():
    assert tuple(inspect.signature(MODULE.topk_pair_ranks).parameters) == ("k",)
    left, right = MODULE.topk_pair_ranks(5)
    assert len(left) == 10
    assert bool((left < right).all())


def test_comparator_is_antisymmetric_by_construction():
    torch.manual_seed(4)
    model = MODULE.AntisymmetricComparator(hidden=16)
    left = torch.randn(7, 257)
    right = torch.randn(7, 257)
    relation_lr = torch.randn(7, 41)
    relation_rl = torch.randn(7, 41)
    left_interaction = torch.randn(7, 48)
    right_interaction = torch.randn(7, 48)
    forward = model(
        left, right, relation_lr, relation_rl, left_interaction, right_interaction
    )
    reverse = model(
        right, left, relation_rl, relation_lr, right_interaction, left_interaction
    )
    torch.testing.assert_close(forward, -reverse)


def test_graph_retains_incumbent_on_an_exact_tie():
    assert MODULE.graph_challenger(torch.zeros(5, 5), incumbent=0) == 0
    evidence = torch.zeros(5, 5)
    evidence[3, :] = 4.0
    evidence[:, 3] = -4.0
    evidence[3, 3] = 0.0
    assert MODULE.graph_challenger(evidence, incumbent=0) == 3


def test_batched_graph_inference_matches_single_group_reference():
    torch.manual_seed(8)
    model = MODULE.AntisymmetricComparator(hidden=16).eval()
    groups = []
    for _ in range(3):
        relation = torch.randn(5, 5, 41)
        interaction = torch.randn(5, 48)
        groups.append(
            {
                "base": torch.randn(5, 257),
                "relation": relation,
                "interaction": interaction,
                "shuffled_relation": relation.flip(0),
                "shuffled_interaction": interaction.flip(0),
            }
        )
    batched = MODULE.predict_group_evidence(
        model, groups, "H2_interaction", [0, 1, 2], torch.device("cpu"), 2
    )
    for index, group in enumerate(groups):
        reference = MODULE.group_evidence(
            model, group, "H2_interaction", torch.device("cpu")
        )
        torch.testing.assert_close(batched[index], reference)


def passing_selection(gain=0.006):
    return {
        "gain_equal_stratum_pdm": gain,
        "common_pdm": 0.92,
        "rare_pdm": 0.85,
        "v3_common_pdm": 0.92,
        "v3_rare_pdm": 0.84,
        "common_hard_safety": 0.98,
        "rare_hard_safety": 0.96,
        "v3_common_hard_safety": 0.98,
        "v3_rare_hard_safety": 0.96,
        "common_degraded_fraction": 0.05,
        "selected_noise_view_agreement": 0.90,
        "v3_noise_view_agreement": 0.90,
        "log_bootstrap_gain": {"lower_95": 0.001},
    }


def test_eligibility_gate_locks_pdm_safety_and_stability_floors():
    assert all(MODULE.eligibility_gates(passing_selection()).values())
    degraded = passing_selection()
    degraded["common_degraded_fraction"] = 0.11
    assert not MODULE.eligibility_gates(degraded)[
        "common_degraded_fraction_at_most_010"
    ]
    unstable = passing_selection()
    unstable["selected_noise_view_agreement"] = 0.879
    assert not MODULE.eligibility_gates(unstable)["noise_agreement_floor"]


def report(eligible, common_auc=0.80, rare_auc=0.80):
    return {
        "gates": {"eligible": eligible},
        "pair_auc": {"common": common_auc, "rare": rare_auc},
    }


def test_decision_is_simple_first_and_needs_matched_control_signal():
    reports = {
        "H0_token": report(True),
        "H1_relation": report(True),
        "H2_interaction": report(True),
    }
    decision, arm = MODULE.choose_decision(
        reports,
        {"H1_relation": True, "H2_interaction": True},
        {"H1_over_H0": False, "H2_over_H1": True},
    )
    assert (decision, arm) == ("AUTHORIZE_TOKEN_TOP5_EVALUATOR", "H0_token")

    decision, arm = MODULE.choose_decision(
        reports,
        {"H1_relation": True, "H2_interaction": False},
        {"H1_over_H0": True, "H2_over_H1": True},
    )
    assert (decision, arm) == (
        "AUTHORIZE_RELATIONAL_TOP5_EVALUATOR",
        "H1_relation",
    )


def test_failed_selection_with_pair_signal_changes_objective_not_representation():
    reports = {
        "H0_token": report(False, 0.78, 0.77),
        "H1_relation": report(False),
        "H2_interaction": report(False),
    }
    decision, arm = MODULE.choose_decision(
        reports,
        {"H1_relation": False, "H2_interaction": False},
        {"H1_over_H0": False, "H2_over_H1": False},
    )
    assert decision == "AUTHORIZE_LOCKED_TOP1_OBJECTIVE"
    assert arm is None
