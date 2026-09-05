#!/usr/bin/env python3
"""Combine the immutable pilot64 and authorized expand192 V4 train caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import v4_causal_cache_common as common


def combine(args) -> tuple[Path, Path]:
    output = args.output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    if audit_output.exists():
        if not output.exists():
            raise RuntimeError("V4 train-cache audit exists without its cache")
        audit = json.loads(audit_output.read_text())
        if common.sha256_file(output) != audit["cache_file_sha256"]:
            raise RuntimeError("existing V4 train cache drifted")
        return output, audit_output

    pilot_path = args.pilot_cache.expanduser().resolve()
    expand_path = args.expand_cache.expanduser().resolve()
    pilot = common.load_pickle(pilot_path)
    expand = common.load_pickle(expand_path)
    expected = ((pilot, "pilot64", 64), (expand, "expand192", 192))
    for payload, stage, count in expected:
        if (
            payload.get("status") != "PASS"
            or payload.get("method") != common.CACHE_METHOD
            or payload.get("stage") != stage
            or int(payload.get("num_rows", -1)) != count
        ):
            raise RuntimeError(f"invalid V4 {stage} cache")
    provenance_keys = (
        "target_manifest_sha256",
        "master_manifest_sha256",
        "checkpoint_sha256",
        "selector_state_sha256",
        "candidate_noise_namespace",
        "rollout_implementation_sha256",
        "collection_code_sha",
    )
    if any(pilot[key] != expand[key] for key in provenance_keys):
        raise RuntimeError("V4 pilot/expand provenance drifted")
    pilot_ids = {row["scene_id"] for row in pilot["rows"]}
    expand_ids = {row["scene_id"] for row in expand["rows"]}
    if pilot_ids & expand_ids or len(pilot_ids | expand_ids) != common.TRAIN_TARGETS:
        raise RuntimeError("V4 train-cache target partition drifted")
    rows = list(pilot["rows"]) + list(expand["rows"])
    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.CACHE_METHOD,
        "scientific_role": "complete_v4_train_only_causal_label_cache",
        "stage": "train256",
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "training_data_consumed": False,
        "development_consumed": False,
        "test_consumed": False,
        **{key: pilot[key] for key in provenance_keys},
        "num_candidates": common.NUM_CANDIDATES,
        "num_rows": len(rows),
        "rows": rows,
    }
    common.atomic_pickle(output, payload)
    audit = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.CACHE_METHOD,
        "stage": "train256",
        "cache_file": str(output),
        "cache_file_sha256": common.sha256_file(output),
        "num_rows": len(rows),
        "pilot_cache": str(pilot_path),
        "pilot_cache_sha256": common.sha256_file(pilot_path),
        "expand_cache": str(expand_path),
        "expand_cache_sha256": common.sha256_file(expand_path),
        "development_consumed": False,
        "method_training_performed": False,
    }
    common.atomic_json(audit_output, audit)
    return output, audit_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-cache", type=Path, required=True)
    parser.add_argument("--expand-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    cache, audit = combine(args)
    print(json.dumps({
        "status": "PASS", "cache": str(cache), "audit": str(audit)
    }, sort_keys=True))


if __name__ == "__main__":
    main()
