#!/usr/bin/env python3
"""Freeze the one-shot blind certification result without further selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    gate_path = args.development_gate.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    gate = json.loads(gate_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if gate.get("decision") != "AUTHORIZE_BLIND_CERTIFICATION":
        raise RuntimeError("development gate did not authorize certification")
    winner = gate["winner"]
    expected = {
        "status": "PASS",
        "method": protocol.EVALUATION_METHOD,
        "split": "certification",
        "heldout_fold": -1,
        "noise_seeds": list(protocol.CERTIFICATION_NOISE_SEEDS),
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
        raise RuntimeError(f"certification evaluation contract drifted: {drift}")
    gates = protocol.efficacy_gates(evaluation["summary"], require_folds=False)
    conclusion = (
        "LCPGRPO_CERTIFIED" if all(gates.values()) else "LCPGRPO_NOT_CERTIFIED"
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite certification report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "lcpgrpo_blind_certification_summary_v1",
        "conclusion": conclusion,
        "winner": winner,
        "efficacy_gates": gates,
        "summary": evaluation["summary"],
        "strata": evaluation["strata"],
        "development_gate": str(gate_path),
        "development_gate_sha256": common.sha256_file(gate_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": common.sha256_file(evaluation_path),
        "scientific_contract": {
            "one_shot_blind_certification": True,
            "certification_used_for_retuning": False,
            "no_further_selection_authorized": True,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "conclusion": conclusion, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
