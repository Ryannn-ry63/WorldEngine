#!/usr/bin/env python3
"""Verify CCV policy/oracle/matched sentinels against prior immutable outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ccv_sweep_common as common
import selector_root_cause_metrics as metrics_common


SENTINELS = ("sentinel_policy", "sentinel_oracle", "sentinel_matched")
OUTCOME_FIELDS = (
    "score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress"
)


def load_audit(path, treatment_path, treatment_sha):
    path = Path(path).expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != common.COLLECTION_AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("treatment_manifest_sha256") != treatment_sha
        or Path(row.get("treatment_manifest", "")).resolve() != treatment_path
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid CCV sentinel collection audit: {path}")
    return path, row


def verify(args):
    master_path, master = common.load_master(args.master_manifest)
    run_root = args.run_root.expanduser().resolve()
    amendment_path = args.amendment.expanduser().resolve()
    amendment_text = amendment_path.read_text()
    if "FROZEN AFTER SENTINEL AND BEFORE FORMAL ARMS" not in amendment_text:
        raise RuntimeError("invalid CCV sentinel amendment")
    amendment_sha = common.sha256_file(amendment_path)

    summaries = {}
    all_rows = []
    audit_refs = {}
    implementation_shas = set()
    code_shas = set()
    checkpoint_shas = set()
    checkpoint_manifest_shas = set()
    noise_namespaces = set()
    for treatment_id in SENTINELS:
        reference = master["treatment_manifests"]["sentinels"][treatment_id]
        treatment_path, treatment = common.load_treatment(
            reference["path"], reference["sha256"]
        )
        expected_path = Path(treatment["sentinel_expected_audit"]).resolve()
        if common.sha256_file(expected_path) != treatment["sentinel_expected_audit_sha256"]:
            raise RuntimeError(f"prior sentinel expectation drifted: {treatment_id}")
        expected_audit = json.loads(expected_path.read_text())
        expected_metrics = metrics_common.load_outcomes(
            expected_audit["closed_loop_outcome"]["metrics_csv"]
        )
        audit_path, audit = load_audit(
            run_root / "collections" / treatment_id / "collection_audit.json",
            treatment_path,
            reference["sha256"],
        )
        implementation_shas.add(str(audit["rollout_implementation_sha256"]))
        code_shas.add(str(audit["code_sha"]))
        checkpoint_shas.add(str(audit["checkpoint_sha256"]))
        checkpoint_manifest_shas.add(str(audit["checkpoint_manifest_sha256"]))
        noise_namespaces.add(str(audit["candidate_noise_namespace"]))
        actual_metrics = metrics_common.load_outcomes(
            audit["closed_loop_outcome"]["metrics_csv"]
        )
        target_ids = {str(row["scene_id"]) for row in treatment["targets"]}
        if set(actual_metrics) != target_ids or not target_ids <= set(expected_metrics):
            raise RuntimeError(f"CCV sentinel outcome coverage drifted: {treatment_id}")
        maximum_error = 0.0
        exact_success = True
        for scene in sorted(target_ids):
            errors = {
                field: abs(float(actual_metrics[scene][field]) - float(expected_metrics[scene][field]))
                for field in OUTCOME_FIELDS
            }
            maximum_error = max(maximum_error, *errors.values())
            success_equal = actual_metrics[scene]["success"] == expected_metrics[scene]["success"]
            exact_success = exact_success and success_equal
            all_rows.append({
                "treatment_id": treatment_id,
                "scene_id": scene,
                "success_equal": success_equal,
                "field_absolute_errors": errors,
                "pass": bool(success_equal and max(errors.values()) <= common.OUTCOME_TOLERANCE),
            })
        passed = bool(
            maximum_error <= common.OUTCOME_TOLERANCE
            and exact_success
            and float(audit["maximum_action_parity_error"]) <= common.ACTION_TOLERANCE
        )
        summaries[treatment_id] = {
            "pass": passed,
            "reward_recomputation_exact": bool(float(audit["target_reward_max_abs_error"]) <= common.ARRAY_TOLERANCE),
            "reward_recomputation_role": "diagnostic_only_not_behavior_reproduction_gate",
            "scene_count": len(target_ids),
            "maximum_outcome_absolute_error": maximum_error,
            "all_success_labels_equal": exact_success,
            "maximum_action_parity_error": audit["maximum_action_parity_error"],
            "target_reward_max_abs_error": audit["target_reward_max_abs_error"],
            "expected_audit": str(expected_path),
            "expected_audit_sha256": treatment["sentinel_expected_audit_sha256"],
            "actual_audit": str(audit_path),
            "actual_audit_sha256": common.sha256_file(audit_path),
        }
        audit_refs[treatment_id] = str(audit_path)
    provenance_sets = {
        "rollout_implementation_sha256": implementation_shas,
        "code_sha": code_shas,
        "checkpoint_sha256": checkpoint_shas,
        "checkpoint_manifest_sha256": checkpoint_manifest_shas,
        "candidate_noise_namespace": noise_namespaces,
    }
    drifted = {
        key: sorted(values) for key, values in provenance_sets.items()
        if len(values) != 1
    }
    if drifted:
        raise RuntimeError(f"CCV sentinel collection provenance drifted: {drifted}")
    if next(iter(checkpoint_shas)) != master["checkpoint_sha256"]:
        raise RuntimeError("CCV sentinel checkpoint differs from frozen master")
    if next(iter(noise_namespaces)) != master["candidate_noise_namespace"]:
        raise RuntimeError("CCV sentinel noise namespace differs from frozen master")
    all_pass = all(row["pass"] for row in summaries.values()) and all(
        row["pass"] for row in all_rows
    )
    if not all_pass:
        decision = "STOP_CCV_SENTINEL_NOT_REPRODUCIBLE"
    else:
        decision = "AUTHORIZE_CCV_20_ARM_SWEEP"
    return {
        "sentinel_amendment": str(amendment_path),
        "sentinel_amendment_sha256": amendment_sha,
        "reward_recomputation_gate_required": False,
        "reward_axis_source": "frozen_causal_pilot_target_records",
        "schema_version": common.SCHEMA_VERSION,
        "status": "PASS",
        "method": common.SENTINEL_METHOD,
        "decision": decision,
        "all_sentinels_reproducible": all_pass,
        "master_manifest": str(master_path),
        "master_manifest_sha256": common.sha256_file(master_path),
        "outcome_tolerance": common.OUTCOME_TOLERANCE,
        "candidate_context_tolerance": common.ARRAY_TOLERANCE,
        "action_parity_tolerance": common.ACTION_TOLERANCE,
        "rollout_implementation_sha256": next(iter(implementation_shas)),
        "code_sha": next(iter(code_shas)),
        "checkpoint_sha256": next(iter(checkpoint_shas)),
        "checkpoint_manifest_sha256": next(iter(checkpoint_manifest_shas)),
        "candidate_noise_namespace": next(iter(noise_namespaces)),
        "summaries": summaries,
        "rows": all_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args)
    common.atomic_json(args.output, report)
    if report["decision"] != "AUTHORIZE_CCV_20_ARM_SWEEP":
        raise RuntimeError(report["decision"])
    print(json.dumps({"status": "PASS", "decision": report["decision"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
