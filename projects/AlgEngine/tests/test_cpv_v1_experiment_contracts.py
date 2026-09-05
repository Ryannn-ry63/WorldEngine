import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/diffusiondrive"
COMMON_PATH = SCRIPTS / "grpo_selector_v3_cached_common.py"
TRAINER_PATH = SCRIPTS / "train_grpo_selector_v3_cached_rare_rollout.py"
GATE_PATH = SCRIPTS / "select_cpv_offline_development.py"
PROPOSAL_SHA = "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load("cpv_common", COMMON_PATH)


def arbitration_tensors(gains):
    gains = torch.tensor(gains, dtype=torch.float32)
    rewards = torch.stack((torch.full_like(gains, 0.6), 0.6 + gains), dim=1)
    return (
        torch.zeros(len(gains), requires_grad=True),
        torch.zeros(len(gains), dtype=torch.long),
        torch.ones(len(gains), dtype=torch.long),
        rewards,
        torch.ones_like(rewards, dtype=torch.bool),
    )


def test_source_sign_regret_matches_six_hand_computed_cell_means():
    tensors = arbitration_tensors([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    labels = [
        "common", "common", "hard_real_rare", "hard_real_rare",
        "hard_synthetic", "hard_synthetic",
    ]
    loss, diagnostics = COMMON.proposal_conditioned_arbitration_loss(
        *tensors,
        loss_kind="regret",
        risk_aggregation="source_sign",
        source_labels=labels,
    )
    assert float(loss) == pytest.approx(0.35 * math.log(2.0), rel=1e-6)
    assert diagnostics["missing_groups"] == []
    assert set(diagnostics["group_active_counts"].values()) == {1}


def test_duplicating_one_source_sign_cell_does_not_change_a3_risk():
    base_gains = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    base_labels = [
        "common", "common", "hard_real_rare", "hard_real_rare",
        "hard_synthetic", "hard_synthetic",
    ]
    base, _ = COMMON.proposal_conditioned_arbitration_loss(
        *arbitration_tensors(base_gains),
        loss_kind="regret",
        risk_aggregation="source_sign",
        source_labels=base_labels,
    )
    duplicated, _ = COMMON.proposal_conditioned_arbitration_loss(
        *arbitration_tensors(base_gains + [0.1, 0.1, 0.1]),
        loss_kind="regret",
        risk_aggregation="source_sign",
        source_labels=base_labels + ["common"] * 3,
    )
    assert float(duplicated) == pytest.approx(float(base), rel=1e-7)


def test_larger_harm_increases_regret_cell_loss():
    small, _ = COMMON.proposal_conditioned_arbitration_loss(
        *arbitration_tensors([-0.1]),
        loss_kind="regret",
        risk_aggregation="source_sign",
        source_labels=["common"],
    )
    large, _ = COMMON.proposal_conditioned_arbitration_loss(
        *arbitration_tensors([-0.4]),
        loss_kind="regret",
        risk_aggregation="source_sign",
        source_labels=["common"],
    )
    assert float(large) == pytest.approx(4.0 * float(small), rel=1e-6)


def load_trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load("cpv_trainer_contract", TRAINER_PATH)


def test_fixed_budget_balances_every_declared_risk_group(monkeypatch):
    trainer = load_trainer(monkeypatch)
    totals = {
        risk: {group: 0 for group in trainer.decision_group_names(risk)}
        for risk in ("source", "sign", "source_sign")
    }
    global_block = 0
    for _epoch in range(16):
        for _cache in range(3):
            for risk, risk_totals in totals.items():
                counts = trainer.balanced_group_counts(
                    6339, tuple(risk_totals), global_block
                )
                for group, count in counts.items():
                    risk_totals[group] += count
            global_block += 1
    assert set(totals["source"].values()) == {101424}
    assert set(totals["sign"].values()) == {152136}
    assert set(totals["source_sign"].values()) == {50712}
    ranges = trainer.balanced_batch_ranges(6339, 64)
    assert len(ranges) == 100
    assert sum(end - start for start, end in ranges) == 6339
    assert {end - start for start, end in ranges} == {63, 64}


def checkpoint(risk, equal, common, rare, synthetic, common_auc, scores, gains):
    method = (
        "pcra_v1_pair_regret_proposal_conditioned_regret"
        if risk == "global"
        else f"cpv_v1_{risk}_pair_regret"
    )
    return {
        "method": method,
        "selector_architecture": "proposal_conditioned_regret_arbitration",
        "objective": "official_pdm_proposal_regret_arbitration",
        "arbiter_loss": "regret",
        "arbiter_risk": risk,
        "use_decision_context": False,
        "arbiter_target": "actual_top1_proposal_vs_reference_incumbent",
        "proposal_checkpoint_sha256": PROPOSAL_SHA,
        "reward_components_consumed": False,
        "training_data_split": "train",
        "training_epochs": 16,
        "training_examples_per_cache_epoch": 6339,
        "training_batch_size": 64,
        "formal_contract": True,
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "sampling_mode": "common_hard_balanced" if risk == "global" else f"cpv_decision_{risk}",
        "override_threshold": 0.0,
        "train_seed": 0,
        "epoch": 16,
        "equal_weight_selected_reward": equal,
        "strata": {
            "common": {
                "current_reward": common,
                "reference_reward": 0.927,
                "degraded_fraction": 0.05 if common >= 0.922 else 0.20,
                "proposal_benefit_roc_auc": common_auc,
                "proposal_ranking_records": {"evidence": scores, "gain": gains},
            },
            "real_rare": {
                "current_reward": rare,
                "proposal_benefit_roc_auc": 0.79,
            },
            "synthetic": {
                "current_reward": synthetic,
                "proposal_benefit_roc_auc": 0.79,
            },
        },
    }


def proposal(equal, common, rare, synthetic):
    return {
        "equal_weight_selected_reward": equal,
        "strata": {
            "common": {"current_reward": common},
            "real_rare": {"current_reward": rare},
            "synthetic": {"current_reward": synthetic},
        },
    }


def write_evaluation(path, row):
    path.write_text(json.dumps({
        "status": "PASS",
        "split": "development",
        "certification_consumed": False,
        "checkpoints": [row],
    }))


def run_gate(tmp_path, monkeypatch, mode):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    gate = load(f"cpv_gate_{mode}", GATE_PATH)
    gains = [-0.2] * 20 + [0.1] * 20
    a0_scores = [0.0] * 40
    if mode in ("S", "C"):
        a3_scores = [-1.0] * 20 + [1.0] * 20
        common_auc = 1.0
    else:
        a3_scores = [0.0] * 40
        common_auc = 0.5
    common = 0.923 if mode == "S" else 0.900
    equal = 0.814 if mode == "S" else 0.808
    paths = {name: tmp_path / f"{name}.json" for name in (
        "proposal32", "proposal48", "a0", "a1", "a2", "a3"
    )}
    write_evaluation(paths["proposal32"], proposal(0.807, 0.891, 0.728, 0.804))
    write_evaluation(paths["proposal48"], proposal(0.806, 0.885, 0.739, 0.797))
    write_evaluation(paths["a0"], checkpoint(
        "global", 0.807, 0.897, 0.723, 0.802, 0.5, a0_scores, gains
    ))
    write_evaluation(paths["a1"], checkpoint(
        "source", 0.807, 0.897, 0.723, 0.802, 0.5, a0_scores, gains
    ))
    write_evaluation(paths["a2"], checkpoint(
        "sign", 0.807, 0.897, 0.723, 0.802, 0.5, a0_scores, gains
    ))
    write_evaluation(paths["a3"], checkpoint(
        "source_sign", equal, common, 0.730, 0.800,
        common_auc, a3_scores, gains,
    ))
    output = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", [
        str(GATE_PATH),
        "--proposal-32", str(paths["proposal32"]),
        "--proposal-48", str(paths["proposal48"]),
        "--a0", str(paths["a0"]), "--a1", str(paths["a1"]),
        "--a2", str(paths["a2"]), "--a3", str(paths["a3"]),
        "--bootstrap-repetitions", "200", "--output", str(output),
    ])
    gate.main()
    return json.loads(output.read_text())


@pytest.mark.parametrize(
    ("mode", "followup", "passed"),
    [
        ("S", "closed_loop_development", True),
        ("C", "E1b_train_only_calibration", False),
        ("D", "E2_common_coverage", False),
    ],
)
def test_gate_follows_predeclared_s_c_d_branches(
    tmp_path, monkeypatch, mode, followup, passed
):
    report = run_gate(tmp_path, monkeypatch, mode)
    assert report["stage_outcome"] == mode
    assert report["authorized_followup_stage"] == followup
    assert report["development_gate_passed"] is passed
