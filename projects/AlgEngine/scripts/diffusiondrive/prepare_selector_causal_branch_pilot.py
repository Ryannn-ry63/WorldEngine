#!/usr/bin/env python3
"""Prepare and audit the immutable 147-scene train-only causal-pilot source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import causal_branch_pilot_common as common
import oracle_r15_common as r15


def prepare(source_audit_path: Path, output_root: Path) -> Path:
    source_audit_path = source_audit_path.expanduser().resolve()
    source_audit = json.loads(source_audit_path.read_text())
    if (
        source_audit.get("status") != "PASS"
        or source_audit.get("method") != common.SOURCE_AUDIT_METHOD
        or int(source_audit.get("num_scenarios", -1)) != 147
        or int(source_audit.get("num_lanes", -1)) != 3
        or int(source_audit.get("navtest_overlap_logs", -1)) != 0
        or int(source_audit.get("navtest_overlap_tokens", -1)) != 0
    ):
        raise RuntimeError("source BWM scenario audit is not the frozen train-only contract")

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_path = output_root / "train_only_147_scenarios.pkl"
    smoke_scenario_path = output_root / "pipeline_smoke8_scenarios.pkl"
    prepared_audit_path = output_root / "source_audit.json"
    prepared_paths = (scenario_path, smoke_scenario_path, prepared_audit_path)
    if any(path.exists() for path in prepared_paths):
        if not all(path.exists() for path in prepared_paths):
            raise RuntimeError("partial immutable source preparation exists")
        prepared = json.loads(prepared_audit_path.read_text())
        if (
            prepared.get("status") != "PASS"
            or prepared.get("method") != common.PREPARED_METHOD
            or r15.sha256_file(scenario_path) != prepared["scenario_file_sha256"]
            or r15.sha256_file(smoke_scenario_path)
            != prepared["pipeline_smoke_scenario_file_sha256"]
            or r15.sha256_file(source_audit_path) != prepared["source_audit_sha256"]
        ):
            raise RuntimeError("prepared source contract drifted")
        return prepared_audit_path

    merged = {}
    source_shards = []
    for lane in sorted(source_audit["lanes"], key=lambda row: int(row["lane"])):
        path = Path(lane["scenario_file"]).expanduser().resolve()
        actual_sha = r15.sha256_file(path)
        if actual_sha != lane["scenario_file_sha256"]:
            raise RuntimeError(f"source shard SHA256 drifted: {path}")
        payload = common.load_pickle(path)
        if not isinstance(payload, dict) or len(payload) != int(lane["num_scenarios"]):
            raise RuntimeError(f"source shard count/type drifted: {path}")
        overlap = set(merged) & set(payload)
        if overlap:
            raise RuntimeError(f"duplicate scenes across shards: {sorted(overlap)}")
        merged.update(payload)
        source_shards.append({
            "lane": int(lane["lane"]),
            "path": str(path),
            "sha256": actual_sha,
            "num_scenarios": len(payload),
        })

    if len(merged) != 147 or set(merged) != {str(scene["id"]) for scene in merged.values()}:
        raise RuntimeError("merged scenario identity/count drifted")
    metadata = {scene_id: common.scenario_metadata(scene) for scene_id, scene in merged.items()}
    source_counts = Counter(row["source_kind"] for row in metadata.values())
    pairing_counts = Counter(row["pairing_method"] for row in metadata.values())
    origin_tokens = {row["origin_token"] for row in metadata.values()}
    if dict(sorted(source_counts.items())) != source_audit["source_counts"]:
        raise RuntimeError("source-kind counts drifted")
    if dict(sorted(pairing_counts.items())) != source_audit["pairing_counts"]:
        raise RuntimeError("pairing-method counts drifted")
    if len(origin_tokens) != int(source_audit["num_origin_tokens"]):
        raise RuntimeError("origin-token count drifted")

    smoke_rows = [
        {"scene_id": scene_id, **metadata[scene_id]}
        for scene_id in sorted(merged)
    ]
    smoke_selection = common.smoke_subset(
        smoke_rows, count=8, salt="pipeline-contract-smoke8"
    )
    if len(smoke_selection) != 8:
        raise RuntimeError("pipeline smoke requires exactly 8 train-only scenes")
    smoke_scene_ids = [row["scene_id"] for row in smoke_selection]
    smoke_scenarios = {
        scene_id: merged[scene_id] for scene_id in smoke_scene_ids
    }

    common.write_pickle(scenario_path, merged)
    common.write_pickle(smoke_scenario_path, smoke_scenarios)
    prepared = {
        "schema_version": 1,
        "status": "PASS",
        "method": common.PREPARED_METHOD,
        "scientific_role": "train_only_causal_diagnostic_no_new_training_data",
        "source_audit": str(source_audit_path),
        "source_audit_sha256": r15.sha256_file(source_audit_path),
        "scenario_file": str(scenario_path),
        "scenario_file_sha256": r15.sha256_file(scenario_path),
        "num_scenarios": len(merged),
        "pipeline_smoke_scenario_file": str(smoke_scenario_path),
        "pipeline_smoke_scenario_file_sha256": r15.sha256_file(
            smoke_scenario_path
        ),
        "pipeline_smoke_scene_ids": smoke_scene_ids,
        "pipeline_smoke_scenario_count": len(smoke_scenarios),
        "num_origin_tokens": len(origin_tokens),
        "num_origin_logs": len({row["origin_log"] for row in metadata.values()}),
        "navtest_overlap_logs": 0,
        "navtest_overlap_tokens": 0,
        "source_counts": dict(sorted(source_counts.items())),
        "pairing_counts": dict(sorted(pairing_counts.items())),
        "source_shards": source_shards,
    }
    common.write_json(prepared_audit_path, prepared)
    return prepared_audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = prepare(args.source_audit, args.output_root)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

