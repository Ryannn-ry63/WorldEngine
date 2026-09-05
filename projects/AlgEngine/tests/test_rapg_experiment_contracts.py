import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"


def load(name):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_payload(nav, rare, cl_nr, cl_r, success):
    return {
        "seeds": {
            str(seed): {
                "navtest_pdm": nav,
                "rare_pdm": rare,
                "cl_nonreactive_pdm": cl_nr + 0.001 * seed,
                "cl_reactive_pdm": cl_r + 0.001 * seed,
                "success_rate": success,
            }
            for seed in range(3)
        }
    }


def offline_checkpoint(tmp_path, arm, equal_reward, rewards, gains, common_degraded):
    return {
        "checkpoint": str(tmp_path / arm / "epoch_16_scene_selector.pt"),
        "equal_weight_selected_reward": equal_reward,
        "strata": {
            name: {
                "current_reward": rewards[name],
                "top1_reward_gain": gains[name],
                "degraded_fraction": common_degraded if name == "common" else 0.0,
            }
            for name in ("common", "real_rare", "synthetic")
        },
    }


def write_offline_evaluation(path, checkpoints):
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "split": "development",
                "certification_consumed": False,
                "checkpoints": checkpoints,
            }
        )
    )


def test_offline_shortlist_expands_multi_checkpoint_evaluation(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = load("select_rapg_offline_development")
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    baseline = tmp_path / "b00.json"
    batch = tmp_path / "remaining.json"
    output = tmp_path / "shortlist.json"
    write_offline_evaluation(
        baseline,
        [
            offline_checkpoint(
                tmp_path,
                "b00",
                0.60,
                {"common": 0.70, "real_rare": 0.50, "synthetic": 0.60},
                {"common": -0.05, "real_rare": 0.20, "synthetic": 0.40},
                0.20,
            )
        ],
    )
    write_offline_evaluation(
        batch,
        [
            offline_checkpoint(
                tmp_path,
                "b10",
                0.70,
                {"common": 0.69, "real_rare": 0.70, "synthetic": 0.71},
                {"common": -0.01, "real_rare": 0.30, "synthetic": 0.50},
                0.20,
            ),
            offline_checkpoint(
                tmp_path,
                "b01_l100",
                0.80,
                {"common": 0.72, "real_rare": 0.82, "synthetic": 0.86},
                {"common": 0.01, "real_rare": 0.42, "synthetic": 0.66},
                0.19,
            ),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_rapg_offline_development.py",
            "--baseline-b00",
            str(baseline),
            "--candidate",
            f"remaining={batch}",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert [row["label"] for row in report["candidates"]] == [
        "remaining:b10",
        "remaining:b01_l100",
    ]
    assert report["shortlist"][0]["label"] == "remaining:b01_l100"
    assert report["development_gate_passed"] is True


def test_formal_gate_passes_only_the_frozen_080_contract(tmp_path, monkeypatch):
    module = load("audit_rapg_formal_promotion")
    baseline = tmp_path / "cqr.json"
    candidate = tmp_path / "rapg.json"
    output = tmp_path / "audit.json"
    baseline.write_text(json.dumps(metric_payload(0.856, 0.627, 0.793, 0.790, 0.882)))
    candidate.write_text(json.dumps(metric_payload(0.850, 0.620, 0.806, 0.804, 0.870)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_rapg_formal_promotion.py",
            "--cqr-baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["promoted"] is True
    assert all(report["gates"].values())
    assert report["cqr_position"] == "diagnostic_pareto_baseline"


def test_ivps_gate_requires_reference_preservation_and_proposal_gain(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = load("select_ivps_offline_development")
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    baseline = tmp_path / "proposal.json"
    candidate = tmp_path / "ivps.json"
    output = tmp_path / "gate.json"
    baseline_checkpoint = offline_checkpoint(
        tmp_path,
        "proposal",
        0.80,
        {"common": 0.87, "real_rare": 0.74, "synthetic": 0.80},
        {"common": -0.05, "real_rare": 0.45, "synthetic": 0.70},
        0.39,
    )
    write_offline_evaluation(baseline, [baseline_checkpoint])
    ivps_checkpoint = offline_checkpoint(
        tmp_path,
        "ivps_m030",
        0.82,
        {"common": 0.923, "real_rare": 0.73, "synthetic": 0.79},
        {"common": -0.004, "real_rare": 0.44, "synthetic": 0.69},
        0.05,
    )
    ivps_checkpoint["selector_architecture"] = "incumbent_verified_preference"
    ivps_checkpoint["strata"]["common"].update(
        reference_reward=0.927,
        override_coverage=0.10,
        override_precision=0.60,
    )
    write_offline_evaluation(candidate, [ivps_checkpoint])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_ivps_offline_development.py",
            "--proposal-baseline",
            str(baseline),
            "--candidate",
            f"m030={candidate}",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["development_gate_passed"] is True
    assert report["shortlist"][0]["label"] == "m030"


def test_formal_gate_rejects_high_cl_with_failed_success_guardrail(
    tmp_path, monkeypatch
):
    module = load("audit_rapg_formal_promotion")
    baseline = tmp_path / "cqr.json"
    candidate = tmp_path / "rapg.json"
    output = tmp_path / "audit.json"
    baseline.write_text(json.dumps(metric_payload(0.856, 0.627, 0.793, 0.790, 0.882)))
    candidate.write_text(json.dumps(metric_payload(0.850, 0.620, 0.820, 0.820, 0.850)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_rapg_formal_promotion.py",
            "--cqr-baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["promoted"] is False
    assert report["gates"]["success_guardrail"] is False


def test_closed_loop_development_prioritizes_cl_r_subject_to_success(
    tmp_path, monkeypatch
):
    module = load("select_rapg_closed_loop_development")
    baseline = tmp_path / "cqr.json"
    unsafe = tmp_path / "unsafe.json"
    selected = tmp_path / "selected.json"
    output = tmp_path / "selection.json"
    baseline.write_text(json.dumps(metric_payload(0.0, 0.0, 0.79, 0.79, 0.88)))
    unsafe.write_text(json.dumps(metric_payload(0.0, 0.0, 0.80, 0.84, 0.85)))
    selected.write_text(json.dumps(metric_payload(0.0, 0.0, 0.80, 0.82, 0.87)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_rapg_closed_loop_development.py",
            "--cqr-baseline",
            str(baseline),
            "--candidate",
            f"unsafe={unsafe}",
            "--candidate",
            f"selected={selected}",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["selected"]["label"] == "selected"
    assert report["certification_consumed"] is False


def test_formal_collector_maps_openloop_failures_to_rare_pdm(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = load("collect_rapg_formal_metrics")
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    summaries = []
    for seed in range(3):
        path = tmp_path / f"summary_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "eval_seed": seed,
                    "checkpoint_sha256": str(seed) * 64,
                    "metrics": {
                        "openloop_navtest": {"score": 0.85},
                        "openloop_failures": {"score": 0.62 + seed * 0.01},
                        "closedloop_nonreactive": {"score": 0.81},
                        "closedloop_reactive": {
                            "score": 0.80,
                            "no_at_fault_collisions": 0.90,
                            "drivable_area_compliance": 0.91,
                        },
                        "success_rate": 0.87,
                    },
                }
            )
        )
        summaries.append(path)
    output = tmp_path / "collected.json"
    argv = ["collect_rapg_formal_metrics.py"]
    for path in summaries:
        argv.extend(("--summary", str(path)))
    argv.extend(("--output", str(output)))
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    report = json.loads(output.read_text())
    assert report["seeds"]["2"]["rare_pdm"] == 0.64
    assert report["rare_definition"] == "openloop_navtest_failures_official_pdm"

def test_pcra_gate_uses_best_proposal_floors_and_prefers_simple_tie(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = load("select_pcra_offline_development")
    finally:
        sys.path.remove(str(SCRIPT_DIR))

    def proposal(name, equal, common, rare, synthetic):
        checkpoint = offline_checkpoint(
            tmp_path,
            name,
            equal,
            {"common": common, "real_rare": rare, "synthetic": synthetic},
            {"common": 0.0, "real_rare": 0.0, "synthetic": 0.0},
            0.0,
        )
        return checkpoint

    def candidate(name, equal, context):
        checkpoint = proposal(name, equal, 0.921, 0.741, 0.791)
        checkpoint.update(
            selector_architecture="proposal_conditioned_regret_arbitration",
            arbiter_loss="regret",
            use_decision_context=context,
            arbiter_target="actual_top1_proposal_vs_reference_incumbent",
            override_threshold=0.0,
            proposal_checkpoint_sha256=module.PROPOSAL32_SHA256,
        )
        checkpoint["strata"]["common"].update(
            reference_reward=0.925,
            degraded_fraction=0.05,
        )
        return checkpoint

    proposal32 = tmp_path / "proposal32.json"
    proposal48 = tmp_path / "proposal48.json"
    pair = tmp_path / "pair.json"
    context = tmp_path / "context.json"
    output = tmp_path / "pcra_gate.json"
    write_offline_evaluation(
        proposal32, [proposal("proposal32", 0.800, 0.90, 0.75, 0.78)]
    )
    write_offline_evaluation(
        proposal48, [proposal("proposal48", 0.810, 0.91, 0.74, 0.80)]
    )
    write_offline_evaluation(pair, [candidate("pair", 0.8195, False)])
    write_offline_evaluation(context, [candidate("context", 0.8200, True)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_pcra_offline_development.py",
            "--proposal-32",
            str(proposal32),
            "--proposal-48",
            str(proposal48),
            "--candidate",
            f"pair={pair}",
            "--candidate",
            f"context={context}",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["development_gate_passed"] is True
    assert report["selected"]["label"] == "pair"
    assert report["selection_rule"] == "prefer_no_context_within_001"
    assert all(report["selected"]["gates"].values())

def test_pcra_closed_loop_gate_requires_offline_locked_arm_and_three_seed_gain(
    tmp_path, monkeypatch
):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = load("select_pcra_closed_loop_development")
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    offline = tmp_path / "offline_gate.json"
    proposal = tmp_path / "proposal48_metrics.json"
    cqr = tmp_path / "cqr_metrics.json"
    candidate = tmp_path / "pair_metrics.json"
    output = tmp_path / "closed_loop_gate.json"
    offline.write_text(
        json.dumps(
            {
                "status": "PASS",
                "development_gate_passed": True,
                "certification_consumed": False,
                "selected": {"label": "pair_regret"},
            }
        )
    )
    proposal.write_text(json.dumps(metric_payload(0.0, 0.0, 0.790, 0.792, 0.880)))
    cqr.write_text(json.dumps(metric_payload(0.0, 0.0, 0.794, 0.796, 0.880)))
    candidate.write_text(
        json.dumps(metric_payload(0.0, 0.0, 0.803, 0.805, 0.870))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_pcra_closed_loop_development.py",
            "--offline-gate",
            str(offline),
            "--proposal48-baseline",
            str(proposal),
            "--cqr-baseline",
            str(cqr),
            "--candidate",
            f"pair_regret={candidate}",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads(output.read_text())
    assert report["closed_loop_development_passed"] is True
    assert report["certification_allowed"] is True
    assert all(report["gates"].values())
