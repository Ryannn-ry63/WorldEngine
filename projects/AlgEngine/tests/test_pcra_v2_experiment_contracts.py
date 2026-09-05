import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diffusiondrive/select_pcra_v2_offline_development.py"
SCRIPTS = SCRIPT.parent


def load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("pcra_v2_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(equal, common, rare, synthetic, reference=0.927, degraded=0.05):
    return {
        "equal_weight_selected_reward": equal,
        "selector_architecture": "proposal_conditioned_counterfactual_evaluator",
        "objective": "official_pdm_counterfactual_evaluation",
        "counterfactual_loss": "opportunity_risk",
        "train_evaluator_encoder": True,
        "source_risk": "original_mixture",
        "arbiter_target": "actual_top1_proposal_vs_reference_incumbent",
        "proposal_checkpoint_sha256": (
            "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
        ),
        "reward_components_consumed": False,
        "override_threshold": 0.0,
        "evaluator_initialization_max_abs_delta": 0.0,
        "strata": {
            "common": {
                "current_reward": common,
                "reference_reward": reference,
                "degraded_fraction": degraded,
            },
            "real_rare": {"current_reward": rare},
            "synthetic": {"current_reward": synthetic},
        },
    }


def write_eval(path, row):
    path.write_text(json.dumps({
        "status": "PASS",
        "split": "development",
        "certification_consumed": False,
        "checkpoints": [row],
    }))


def proposal(equal, common, rare, synthetic):
    row = checkpoint(equal, common, rare, synthetic)
    row.update(
        checkpoint_sha256=(
            "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
        ),
        selector_architecture="trajectory_set_reasoner",
        objective="official_pdm_plus_scalar_preference",
        counterfactual_loss=None,
        train_evaluator_encoder=None,
        source_risk=None,
        arbiter_target=None,
        reward_components_consumed=False,
        override_threshold=None,
        evaluator_initialization_max_abs_delta=None,
    )
    return row


def run_gate(tmp_path, monkeypatch, candidate):
    module = load_module(monkeypatch)
    p32, p48, control, main = (
        tmp_path / name for name in ("p32.json", "p48.json", "control.json", "main.json")
    )
    write_eval(p32, proposal(0.807, 0.891, 0.728, 0.804))
    write_eval(p48, proposal(0.806, 0.885, 0.739, 0.797))
    write_eval(control, candidate)
    write_eval(main, candidate)
    output = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--proposal-32", str(p32), "--proposal-48", str(p48),
        "--control", f"control={control}",
        "--candidate", f"independent_opportunity_risk={main}",
        "--output", str(output),
    ])
    module.main()
    return json.loads(output.read_text())


def test_pcra_v2_gate_promotes_only_the_predeclared_main(monkeypatch, tmp_path):
    candidate = checkpoint(0.820, 0.925, 0.740, 0.800, degraded=0.09)
    gate = run_gate(tmp_path, monkeypatch, candidate)
    assert gate["development_gate_passed"] is True
    assert gate["selected"]["label"] == "independent_opportunity_risk"
    assert all(gate["selected"]["gates"].values())


def test_pcra_v2_gate_rejects_common_degradation(monkeypatch, tmp_path):
    candidate = checkpoint(0.820, 0.900, 0.740, 0.800, degraded=0.20)
    gate = run_gate(tmp_path, monkeypatch, candidate)
    assert gate["development_gate_passed"] is False
    assert gate["selected"] is None
    gates = gate["candidates"][0]["gates"]
    assert gates["common_preserves_reference_within_005"] is False
    assert gates["common_degraded_fraction_at_most_010"] is False
