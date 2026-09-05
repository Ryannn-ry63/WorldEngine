#!/usr/bin/env python3
"""Apply the frozen LC-PGRPO efficacy gate to the sole development winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-gate", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    control_path = args.control_gate.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    control = json.loads(control_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if control.get("decision") != "AUTHORIZE_SINGLE_WINNER_DEVELOPMENT":
        raise RuntimeError("negative-control gate did not authorize development")
    winner = control["winner"]
    expected = {
        "status": "PASS",
        "method": protocol.EVALUATION_METHOD,
        "split": "development",
        "heldout_fold": -1,
        "noise_seeds": list(protocol.DEVELOPMENT_NOISE_SEEDS),
        "arm": winner["arm"],
        "target_kl": winner["target_kl"],
        "epoch": winner["epoch"],
        "lineage_control": "real",
    }
    drift = {
        key: {"actual": evaluation.get(key), "expected": value}
        for key, value in expected.items()
        if evaluation.get(key) != value
    }
    if drift:
        raise RuntimeError(f"development evaluation contract drifted: {drift}")
    records_path = Path(evaluation["records"]).resolve()
    if common.sha256_file(records_path) != evaluation["records_sha256"]:
        raise RuntimeError("development records SHA256 drifted")
    gates = protocol.efficacy_gates(
        evaluation["summary"], require_folds=False
    )
    if all(gates.values()):
        decision = "AUTHORIZE_BLIND_CERTIFICATION"
        followup = "evaluate_once_on_certification_seeds_6_7_8"
    else:
        decision = "STOP_LCPGRPO_RETAIN_V3"
        followup = "retain_v3_and_analyze_temporal_information_audit"

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable development gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "lcpgrpo_single_winner_development_gate_v1",
        "decision": decision,
        "authorized_followup_stage": followup,
        "winner": winner,
        "efficacy_gates": gates,
        "summary": evaluation["summary"],
        "strata": evaluation["strata"],
        "pre_registered_thresholds": protocol.THRESHOLDS,
        "control_gate": str(control_path),
        "control_gate_sha256": common.sha256_file(control_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": common.sha256_file(evaluation_path),
        "scientific_contract": {
            "only_cv_selected_winner_evaluated": True,
            "development_used_once": True,
            "development_used_for_retuning": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(protocol.__file__).resolve()): common.sha256_file(
                Path(protocol.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "decision": decision, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
