#!/usr/bin/env python3
"""Build immutable V4 policy-sentinel and 20-arm treatment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import v4_causal_cache_common as common


STAGES = ("pilot64", "expand192", "dev64")


def treatment_payload(
    *,
    target_path: Path,
    target_sha: str,
    protocol_path: Path,
    scenario_path: Path,
    stage: str,
    collection_id: str,
    source_split: str,
    treatment_kind: str,
    treatment_index: int | None,
    targets: list[dict],
) -> dict:
    rows = []
    for target in targets:
        index = (
            int(target["policy_index"])
            if treatment_kind == "policy_sentinel"
            else int(treatment_index)
        )
        rows.append({
            "scene_id": target["scene_id"],
            "decision_step": int(target["decision_step"]),
            "treatment_index": index,
        })
    return {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.TREATMENT_METHOD,
        "scientific_role": "method_neutral_one_step_causal_value_collection",
        "collection_id": collection_id,
        "stage": stage,
        "source_split": source_split,
        "treatment_kind": treatment_kind,
        "treatment_index": treatment_index,
        "future_outcome_used_to_choose_treatment": False,
        "method_training_performed": False,
        "training_data_consumed": False,
        "development_consumed": False,
        "test_consumed": False,
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "target_manifest": str(target_path),
        "target_manifest_sha256": target_sha,
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": common.sha256_file(protocol_path),
        "scenario_file": str(scenario_path),
        "scenario_file_sha256": common.sha256_file(scenario_path),
        "target_count": len(rows),
        "targets": rows,
    }


def verify_master(path: Path) -> Path:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != common.MASTER_METHOD
        or payload.get("design_version") != common.DESIGN_VERSION
    ):
        raise RuntimeError(f"invalid existing V4 master manifest: {path}")
    if common.sha256_file(payload["target_manifest"]) != payload["target_manifest_sha256"]:
        raise RuntimeError("existing V4 master target provenance drifted")
    for row in payload["collections"].values():
        if common.sha256_file(row["path"]) != row["sha256"]:
            raise RuntimeError(f"existing V4 treatment drifted: {row['path']}")
        common.load_treatment(row["path"], row["sha256"])
    return path


def build(args) -> Path:
    output_root = args.output_root.expanduser().resolve()
    master_path = output_root / "master_manifest.json"
    if master_path.exists():
        return verify_master(master_path)
    # A missing master means deterministic incomplete files may be regenerated.

    target_path, target_payload = common.load_target_manifest(args.target_manifest)
    target_sha = common.sha256_file(target_path)
    protocol_path = args.protocol.expanduser().resolve()
    if common.sha256_file(protocol_path) != target_payload["protocol_file_sha256"]:
        raise RuntimeError("V4 treatment protocol differs from frozen target protocol")
    targets_by_stage = {
        stage: [
            row for row in target_payload["targets"] if row.get("stage") == stage
        ]
        for stage in STAGES
    }
    expected = {
        "pilot64": common.PILOT_TARGETS,
        "expand192": common.TRAIN_TARGETS - common.PILOT_TARGETS,
        "dev64": common.DEV_TARGETS,
    }
    if {key: len(value) for key, value in targets_by_stage.items()} != expected:
        raise RuntimeError("V4 target stage counts drifted")

    scenario_paths = {
        "pilot64": Path(target_payload["pilot64_scenario_file"]),
        "expand192": Path(target_payload["expand192_scenario_file"]),
        "dev64": Path(target_payload["dev64_scenario_file"]),
    }
    pilot_source = common.load_pickle(scenario_paths["pilot64"])
    sentinel_targets = targets_by_stage["pilot64"][:8]
    sentinel_path = output_root / "sentinel8_scenarios.pkl"

    output_root.mkdir(parents=True, exist_ok=True)
    common.atomic_pickle(
        sentinel_path,
        {row["scene_id"]: pilot_source[row["scene_id"]] for row in sentinel_targets},
    )
    collections = {}

    def write(payload: dict) -> None:
        path = output_root / f"{payload['collection_id']}.json"
        common.atomic_json(path, payload)
        sha = common.sha256_file(path)
        common.load_treatment(path, sha)
        collections[payload["collection_id"]] = {
            "path": str(path.resolve()),
            "sha256": sha,
            "stage": payload["stage"],
            "source_split": payload["source_split"],
            "treatment_index": payload["treatment_index"],
        }

    write(treatment_payload(
        target_path=target_path,
        target_sha=target_sha,
        protocol_path=protocol_path,
        scenario_path=sentinel_path,
        stage="sentinel",
        collection_id="sentinel_policy",
        source_split="train",
        treatment_kind="policy_sentinel",
        treatment_index=None,
        targets=sentinel_targets,
    ))
    for stage in STAGES:
        source_split = "validation" if stage == "dev64" else "train"
        for index in range(common.NUM_CANDIDATES):
            collection_id = f"{stage}_arm_{index:02d}"
            write(treatment_payload(
                target_path=target_path,
                target_sha=target_sha,
                protocol_path=protocol_path,
                scenario_path=scenario_paths[stage],
                stage=stage,
                collection_id=collection_id,
                source_split=source_split,
                treatment_kind="fixed_candidate",
                treatment_index=index,
                targets=targets_by_stage[stage],
            ))

    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.MASTER_METHOD,
        "scientific_role": "immutable_method_neutral_v4_collection_design",
        "target_manifest": str(target_path),
        "target_manifest_sha256": target_sha,
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": common.sha256_file(protocol_path),
        "sentinel_scenario_file": str(sentinel_path.resolve()),
        "sentinel_scenario_file_sha256": common.sha256_file(sentinel_path),
        "collection_count": len(collections),
        "collections": dict(sorted(collections.items())),
        "method_training_authorized": False,
        "development_consumed": False,
        "test_consumed": False,
    }
    common.atomic_json(master_path, payload)
    return verify_master(master_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = build(args)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
