#!/usr/bin/env python3
"""Verify that a policy-index CFPI intervention exactly reproduces baseline V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cfpi_common as common
from prepare_selector_cfpi import load_outcomes


def verify(args) -> dict:
    baseline_path = args.baseline_audit.expanduser().resolve()
    sentinel_path = args.sentinel_audit.expanduser().resolve()
    baseline = json.loads(baseline_path.read_text())
    sentinel = json.loads(sentinel_path.read_text())
    if (
        baseline.get("status") != "PASS"
        or baseline.get("method") != common.COLLECTION_AUDIT_METHOD
        or baseline.get("collection_id") != "baseline_train_a"
        or baseline.get("intervention_mode") != "observe_only"
    ):
        raise RuntimeError("invalid CFPI baseline for sentinel comparison")
    if (
        sentinel.get("status") != "PASS"
        or sentinel.get("method") != common.COLLECTION_AUDIT_METHOD
        or sentinel.get("collection_id") != "sentinel_policy"
        or sentinel.get("intervention_mode") != "one_shot_manifest"
        or int(sentinel.get("intervention_count", -1)) != 8
    ):
        raise RuntimeError("invalid CFPI policy sentinel collection")
    treatment_path, treatment = common.load_treatment(args.treatment_manifest)
    if (
        treatment["collection_id"] != "sentinel_policy"
        or treatment["treatment_kind"] != "policy_sentinel"
    ):
        raise RuntimeError("invalid CFPI policy sentinel treatment")
    if sentinel["treatment_manifest_sha256"] != common.sha256_file(treatment_path):
        raise RuntimeError("CFPI sentinel treatment provenance drifted")
    keys = (
        "checkpoint_sha256",
        "selector_state_sha256",
        "candidate_noise_namespace",
        "rollout_implementation_sha256",
    )
    if any(baseline[key] != sentinel[key] for key in keys):
        raise RuntimeError("CFPI baseline/sentinel provenance drifted")

    baseline_outcomes = load_outcomes(
        Path(baseline["closed_loop_outcome"]["metrics_csv"])
    )
    sentinel_outcomes = load_outcomes(
        Path(sentinel["closed_loop_outcome"]["metrics_csv"])
    )
    if set(sentinel_outcomes) != {row["scene_id"] for row in treatment["targets"]}:
        raise RuntimeError("CFPI sentinel outcome coverage drifted")
    errors = {}
    for scene, observed in sentinel_outcomes.items():
        reference = baseline_outcomes[scene]
        errors[scene] = {
            key: abs(float(observed[key]) - float(reference[key]))
            for key in (
                "score",
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "ego_progress",
            )
        }
    maximum_error = max(
        value for row in errors.values() for value in row.values()
    )
    categorical_match = all(
        sentinel_outcomes[scene]["success"] == baseline_outcomes[scene]["success"]
        and sentinel_outcomes[scene]["first_violation_step"]
        == baseline_outcomes[scene]["first_violation_step"]
        for scene in sentinel_outcomes
    )
    treatment_match = all(
        int(row["treatment_index"]) == int(row["policy_index"])
        for row in treatment["targets"]
    )
    reproducible = bool(
        maximum_error <= common.OUTCOME_TOLERANCE
        and categorical_match
        and treatment_match
    )
    return {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS" if reproducible else "FAIL",
        "method": "diffusiondrive_selector_cfpi_policy_sentinel_gate_v1",
        "decision": (
            "AUTHORIZE_CFPI_PILOT64_CAUSAL_COLLECTION"
            if reproducible
            else "STOP_CFPI_SENTINEL_NOT_REPRODUCIBLE"
        ),
        "all_policy_treatments_match_incumbent": treatment_match,
        "categorical_outcomes_match": categorical_match,
        "maximum_outcome_absolute_error": maximum_error,
        "outcome_tolerance": common.OUTCOME_TOLERANCE,
        "per_scene_errors": errors,
        "baseline_audit": str(baseline_path),
        "baseline_audit_sha256": common.sha256_file(baseline_path),
        "sentinel_audit": str(sentinel_path),
        "sentinel_audit_sha256": common.sha256_file(sentinel_path),
        "treatment_manifest": str(treatment_path),
        "treatment_manifest_sha256": common.sha256_file(treatment_path),
        "method_training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--sentinel-audit", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args)
    common.atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(report["decision"])


if __name__ == "__main__":
    main()
