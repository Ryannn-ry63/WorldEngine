#!/usr/bin/env python3
"""Freeze pre-violation R1.5 oracle intervention targets from baseline rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

import oracle_r15_common as r15


AUDIT_METHOD = "diffusiondrive_selector_oracle_r15_collection_audit_v1"
TARGET_METHOD = "diffusiondrive_selector_oracle_r15_targets_v1"


def load_audit(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("diagnostic_split") != "baseline_cl_dev58"
        or row.get("oracle_intervention_mode") != "observe_only"
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"not a completed R1.5 baseline audit: {path}")
    return path, row


def load_outcomes(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            scene = str(row["token"])
            if scene == "overall_average":
                continue
            nc = float(row["no_at_fault_collisions"])
            dac = float(row["drivable_area_compliance"])
            violation = row.get("first_violation_step", "").strip()
            rows[scene] = {
                "score": float(row["score"]),
                "success": bool(nc >= 1.0 and dac >= 1.0),
                "no_at_fault_collisions": nc,
                "drivable_area_compliance": dac,
                "first_violation_step": None if not violation else int(float(violation)),
            }
    return rows


def record_paths(root: Path) -> list[Path]:
    paths = sorted(
        (root / "WE_output/openscene_format/diffusiondrive_preaction_records").glob(
            "*_preaction.pkl"
        )
    )
    if not paths:
        raise RuntimeError(f"no merged pre-action records under {root}")
    return paths


def target_for_scene(records: list[tuple[Path, dict]], outcome: dict, threshold: float):
    if outcome["success"] or outcome["first_violation_step"] is None:
        return None
    eligible = []
    for path, row in records:
        decision = int(row["decision_step"])
        rewards = r15.checked_rewards(row["candidate_rewards"])
        policy = int(row["policy_selected_index"])
        oracle = r15.stable_oracle_index(rewards, policy)
        headroom = float(rewards[oracle] - rewards[policy])
        if decision <= outcome["first_violation_step"] and headroom > threshold:
            eligible.append((decision, path, row, rewards, policy, oracle, headroom))
    if not eligible:
        return None
    decision, path, row, rewards, policy, oracle, headroom = min(
        eligible, key=lambda value: (value[0], str(value[1]))
    )
    return {
        "scene_id": str(row["rollout_scene_id"]),
        "decision_step": decision,
        "state_step": int(row["state_step"]),
        "policy_index": policy,
        "oracle_index": oracle,
        "headroom": headroom,
        "first_violation_step": int(outcome["first_violation_step"]),
        "baseline_score": float(outcome["score"]),
        "baseline_no_at_fault_collisions": float(outcome["no_at_fault_collisions"]),
        "baseline_drivable_area_compliance": float(outcome["drivable_area_compliance"]),
        "candidate_trajectories_8": np.asarray(
            row["candidate_trajectories_8"], dtype=np.float32
        ).tolist(),
        "candidate_rewards": rewards.astype(np.float32).tolist(),
        "current_logits": np.asarray(row["current_logits"], dtype=np.float32).tolist(),
        "source_record": str(path.resolve()),
        "source_record_sha256": r15.sha256_file(path),
    }


def write_scenario_file(path: Path, source: dict, scene_ids: list[str]):
    missing = [scene for scene in scene_ids if scene not in source]
    if missing:
        raise RuntimeError(f"target scenes absent from source scenario file: {missing}")
    payload = {scene: source[scene] for scene in scene_ids}
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def padded_scenes(targets: list[str], source_order: list[str], minimum: int = 8):
    values = list(dict.fromkeys(targets))
    for scene in source_order:
        if len(values) >= minimum:
            break
        if scene not in values:
            values.append(scene)
    if len(values) < minimum:
        raise RuntimeError("cannot construct an eight-scenario intervention collection")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, action="append", required=True)
    parser.add_argument("--source-scenarios", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--headroom-threshold", type=float, default=0.02)
    args = parser.parse_args()
    if len(args.baseline_audit) != 2:
        raise RuntimeError("R1.5 target freezing requires exactly seed0 and seed1 baselines")
    if args.headroom_threshold < 0:
        raise ValueError("headroom threshold must be non-negative")
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "target_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite immutable targets: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = args.source_scenarios.expanduser().resolve()
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    if not isinstance(source, dict) or not source:
        raise RuntimeError("invalid source scenario file")
    source_order = [str(scene) for scene in source]

    loaded = [load_audit(path) for path in args.baseline_audit]
    source_sha = r15.sha256_file(source_path)
    if any(row.get("scenario_file_sha256") != source_sha for _, row in loaded):
        raise RuntimeError("baseline/source scenario SHA256 drifted")
    if {int(row["behavior_policy_train_seed"]) for _, row in loaded} != {0, 1}:
        raise RuntimeError("baseline audits must be scalar V3 seed0 and seed1")
    targets = []
    collections = {}
    baselines = []
    for audit_path, audit in sorted(
        loaded, key=lambda value: int(value[1]["behavior_policy_train_seed"])
    ):
        seed = int(audit["behavior_policy_train_seed"])
        root = Path(audit["rollout_root"]).expanduser().resolve()
        metrics_path = Path(audit["closed_loop_outcome"]["metrics_csv"])
        outcomes = load_outcomes(metrics_path)
        by_scene = {}
        for record_path in record_paths(root):
            with record_path.open("rb") as stream:
                row = pickle.load(stream)
            scene = str(row["rollout_scene_id"])
            by_scene.setdefault(scene, []).append((record_path, row))
        seed_targets = []
        for scene in sorted(outcomes):
            target = target_for_scene(
                by_scene.get(scene, []), outcomes[scene], args.headroom_threshold
            )
            if target is not None:
                target["train_seed"] = seed
                target["baseline_audit"] = str(audit_path)
                seed_targets.append(target)
                targets.append(target)
        target_scenes = [row["scene_id"] for row in seed_targets]
        formal_scenes = padded_scenes(target_scenes, source_order)
        smoke_targets = target_scenes[:1]
        smoke_scenes = padded_scenes(smoke_targets, source_order)
        formal_path = output_root / f"intervention_scenarios_seed{seed}.pkl"
        smoke_path = output_root / f"intervention_smoke_seed{seed}.pkl"
        write_scenario_file(formal_path, source, formal_scenes)
        write_scenario_file(smoke_path, source, smoke_scenes)
        collections[str(seed)] = {
            "target_count": len(seed_targets),
            "target_scenes": target_scenes,
            "formal_scenario_file": str(formal_path),
            "formal_scenario_file_sha256": r15.sha256_file(formal_path),
            "formal_scenario_count": len(formal_scenes),
            "smoke_scenario_file": str(smoke_path),
            "smoke_scenario_file_sha256": r15.sha256_file(smoke_path),
            "smoke_scenario_count": len(smoke_scenes),
        }
        baselines.append({
            "train_seed": seed,
            "audit": str(audit_path),
            "audit_sha256": r15.sha256_file(audit_path),
            "rollout_root": str(root),
            "metrics_csv": str(metrics_path.resolve()),
            "metrics_csv_sha256": r15.sha256_file(metrics_path),
        })
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": TARGET_METHOD,
        "selection_rule": "earliest_decision_at_or_before_first_violation_with_headroom_gt_threshold",
        "headroom_threshold_strictly_greater_than": float(args.headroom_threshold),
        "source_scenario_file": str(source_path),
        "source_scenario_file_sha256": source_sha,
        "baseline_collections": baselines,
        "collections": collections,
        "target_count": len(targets),
        "targets": targets,
        "scientific_role": "diagnostic_causal_intervention_not_deployable_oracle",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "output": str(manifest_path),
        "target_count": len(targets),
        "target_count_by_seed": {
            seed: row["target_count"] for seed, row in collections.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
