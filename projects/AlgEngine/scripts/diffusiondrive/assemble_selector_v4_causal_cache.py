#!/usr/bin/env python3
"""Assemble one complete V4 causal-value surface from 20 audited arms."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import v4_causal_cache_common as common
from build_selector_v4_causal_targets import load_outcomes


STAGE_COUNTS = {
    "pilot64": common.PILOT_TARGETS,
    "expand192": common.TRAIN_TARGETS - common.PILOT_TARGETS,
    "dev64": common.DEV_TARGETS,
}
CONTEXT_SHAPES = {
    "candidate_features": (20, 256),
    "candidate_trajectories_8": (20, 8, 3),
    "route_bev_features": (20, 8, 256),
    "status_tokens": (1, 256),
    "ego_queries": (1, 256),
    "agents_queries": (30, 256),
    "reference_logits": (20,),
    "current_logits": (20,),
    "candidate_rewards": (20,),
    "candidate_reward_components": (20, 6),
}


def load_master(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != common.MASTER_METHOD
        or payload.get("design_version") != common.DESIGN_VERSION
        or payload.get("method_training_authorized") is not False
    ):
        raise RuntimeError(f"invalid V4 master manifest: {path}")
    target_path, _ = common.load_target_manifest(payload["target_manifest"])
    if common.sha256_file(target_path) != payload["target_manifest_sha256"]:
        raise RuntimeError("V4 master target provenance drifted")
    return path, payload


def load_arm_audit(
    run_root: Path, master: dict, stage: str, index: int
) -> tuple[Path, dict, dict[str, dict]]:
    collection_id = f"{stage}_arm_{index:02d}"
    collection = master["collections"].get(collection_id)
    if collection is None or int(collection["treatment_index"]) != index:
        raise RuntimeError(f"V4 master is missing {collection_id}")
    treatment_path, treatment = common.load_treatment(
        collection["path"], collection["sha256"]
    )
    audit_path = run_root / "collections" / collection_id / "collection_audit.json"
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("method") != common.COLLECTION_AUDIT_METHOD
        or audit.get("layout") != "merged"
        or audit.get("collection_id") != collection_id
        or audit.get("intervention_mode") != "one_shot_manifest"
        or audit.get("treatment_manifest_sha256") != collection["sha256"]
        or int(audit.get("intervention_count", -1)) != int(treatment["target_count"])
        or not audit.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid V4 arm audit: {audit_path}")
    outcomes = load_outcomes(Path(audit["closed_loop_outcome"]["metrics_csv"]))
    expected = {str(row["scene_id"]) for row in treatment["targets"]}
    if set(outcomes) != expected:
        raise RuntimeError(f"V4 arm outcome coverage drifted: {collection_id}")
    return audit_path, audit, outcomes


def load_context(target: dict) -> tuple[dict, dict]:
    path = Path(target["source_record_a"]).expanduser().resolve()
    if common.sha256_file(path) != target["source_record_a_sha256"]:
        raise RuntimeError(f"V4 target source record drifted: {path}")
    with path.open("rb") as stream:
        record = pickle.load(stream)
    if (
        str(record["rollout_scene_id"]) != target["scene_id"]
        or int(record["decision_step"]) != int(target["decision_step"])
        or int(record["policy_selected_index"]) != int(target["policy_index"])
    ):
        raise RuntimeError("V4 target/source record identity drifted")
    arrays = {
        key: common.checked_array(record[key], shape, key).astype(np.float32)
        for key, shape in CONTEXT_SHAPES.items()
    }
    if not np.allclose(
        arrays["candidate_trajectories_8"],
        np.asarray(target["candidate_trajectories_8"], dtype=np.float32),
        rtol=0.0,
        atol=common.ARRAY_TOLERANCE,
    ):
        raise RuntimeError("V4 cache target trajectory drifted")
    if not np.allclose(
        arrays["current_logits"],
        np.asarray(target["current_logits"], dtype=np.float32),
        rtol=0.0,
        atol=common.ARRAY_TOLERANCE,
    ):
        raise RuntimeError("V4 cache target logits drifted")
    return arrays, record


def assemble(args) -> tuple[Path, Path]:
    run_root = args.run_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    if audit_output.exists():
        if not output_path.exists():
            raise RuntimeError("V4 cache audit exists without its cache")
        audit = json.loads(audit_output.read_text())
        if (
            audit.get("status") != "PASS"
            or audit.get("method") != common.CACHE_METHOD
            or audit.get("stage") != args.stage
            or common.sha256_file(output_path) != audit["cache_file_sha256"]
        ):
            raise RuntimeError("existing V4 cache assembly drifted")
        return output_path, audit_output

    master_path, master = load_master(args.master_manifest)
    target_path, target_payload = common.load_target_manifest(
        master["target_manifest"]
    )
    targets = [
        row for row in target_payload["targets"] if row.get("stage") == args.stage
    ]
    if len(targets) != STAGE_COUNTS[args.stage]:
        raise RuntimeError(f"V4 {args.stage} target count drifted")
    target_ids = [str(row["scene_id"]) for row in targets]
    if len(set(target_ids)) != len(target_ids):
        raise RuntimeError(f"duplicate V4 {args.stage} targets")

    arm_audits = []
    arm_outcomes = []
    provenance = None
    for index in range(common.NUM_CANDIDATES):
        audit_path, audit, outcomes = load_arm_audit(
            run_root, master, args.stage, index
        )
        current = (
            audit["checkpoint_sha256"],
            audit["selector_state_sha256"],
            audit["candidate_noise_namespace"],
            audit["rollout_implementation_sha256"],
            audit["code_sha"],
        )
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise RuntimeError("V4 arm collection provenance drifted")
        arm_audits.append({
            "candidate_index": index,
            "path": str(audit_path),
            "sha256": common.sha256_file(audit_path),
            "metrics_csv": audit["closed_loop_outcome"]["metrics_csv"],
            "metrics_csv_sha256": audit["closed_loop_outcome"]["metrics_csv_sha256"],
        })
        arm_outcomes.append(outcomes)

    rows = []
    for target in targets:
        scene_id = str(target["scene_id"])
        context, source_record = load_context(target)
        q = np.asarray(
            [arm_outcomes[index][scene_id]["score"] for index in range(20)],
            dtype=np.float32,
        )
        outcome_components = np.asarray([
            [
                arm_outcomes[index][scene_id]["no_at_fault_collisions"],
                arm_outcomes[index][scene_id]["drivable_area_compliance"],
                arm_outcomes[index][scene_id]["ego_progress"],
            ]
            for index in range(20)
        ], dtype=np.float32)
        if q.shape != (20,) or outcome_components.shape != (20, 3):
            raise RuntimeError("V4 causal outcome shape drifted")
        row = {
            **context,
            "scene_id": scene_id,
            "origin_log": target["origin_log"],
            "origin_token": target["origin_token"],
            "scenario_family": target["scenario_family"],
            "split": target["split"],
            "stage": target["stage"],
            "target_decision_step": int(target["decision_step"]),
            "target_state_step": int(target["state_step"]),
            "policy_index": int(target["policy_index"]),
            "behavior_outcome_stratum": target["outcome_stratum"],
            "behavior_score": float(target["baseline_score"]),
            "behavior_success": bool(target["baseline_success"]),
            "behavior_ego_progress": float(target["baseline_ego_progress"]),
            "first_violation_step": target["first_violation_step"],
            "v3_logits": context["current_logits"].copy(),
            "local_official_pdm": context["candidate_rewards"].copy(),
            "local_reward_components": context[
                "candidate_reward_components"
            ].copy(),
            "causal_q_v3": q,
            "causal_outcome_components": outcome_components,
            "source_record_a": target["source_record_a"],
            "source_record_a_sha256": target["source_record_a_sha256"],
            "checkpoint_sha256": source_record["checkpoint_sha256"],
            "selector_state_sha256": target_payload.get("selector_state_sha256"),
            "candidate_noise_namespace": source_record[
                "candidate_noise_namespace"
            ],
            "rollout_implementation_sha256": target_payload[
                "baseline_rollout_implementation_sha256"
            ] if "baseline_rollout_implementation_sha256" in target_payload else provenance[3],
            "target_manifest_sha256": common.sha256_file(target_path),
        }
        rows.append(row)

    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.CACHE_METHOD,
        "scientific_role": "method_neutral_train_label_cache",
        "stage": args.stage,
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "training_data_consumed": False,
        "development_consumed": False,
        "test_consumed": False,
        "target_manifest": str(target_path),
        "target_manifest_sha256": common.sha256_file(target_path),
        "master_manifest": str(master_path),
        "master_manifest_sha256": common.sha256_file(master_path),
        "checkpoint_sha256": provenance[0],
        "selector_state_sha256": provenance[1],
        "candidate_noise_namespace": provenance[2],
        "rollout_implementation_sha256": provenance[3],
        "collection_code_sha": provenance[4],
        "num_candidates": common.NUM_CANDIDATES,
        "num_rows": len(rows),
        "rows": rows,
    }
    common.atomic_pickle(output_path, payload)
    spans = np.asarray([
        float(np.max(row["causal_q_v3"]) - np.min(row["causal_q_v3"]))
        for row in rows
    ])
    gaps = np.asarray([
        float(np.max(row["causal_q_v3"]) - row["causal_q_v3"][row["policy_index"]])
        for row in rows
    ])
    audit = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.CACHE_METHOD,
        "stage": args.stage,
        "cache_file": str(output_path),
        "cache_file_sha256": common.sha256_file(output_path),
        "num_rows": len(rows),
        "num_candidates_per_row": common.NUM_CANDIDATES,
        "complete_causal_surfaces": len(rows),
        "q_span": {
            "minimum": float(np.min(spans)),
            "median": float(np.median(spans)),
            "maximum": float(np.max(spans)),
            "above_0p02": int(np.sum(spans > common.Q_SPAN_THRESHOLD)),
        },
        "q_oracle_minus_policy": {
            "minimum": float(np.min(gaps)),
            "median": float(np.median(gaps)),
            "maximum": float(np.max(gaps)),
            "above_0p02": int(np.sum(gaps > common.Q_POLICY_GAP_THRESHOLD)),
        },
        "target_manifest": str(target_path),
        "target_manifest_sha256": common.sha256_file(target_path),
        "master_manifest": str(master_path),
        "master_manifest_sha256": common.sha256_file(master_path),
        "arm_collection_audits": arm_audits,
        "development_consumed": False,
        "method_training_performed": False,
    }
    common.atomic_json(audit_output, audit)
    return output_path, audit_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_COUNTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    cache, audit = assemble(args)
    print(json.dumps({
        "status": "PASS",
        "cache": str(cache),
        "audit": str(audit),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
