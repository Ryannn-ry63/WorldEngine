#!/usr/bin/env python3
"""Create immutable log-disjoint CL-dev/cert token filters from the RAPG manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached_rare_rollout as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.data_manifest.expanduser().resolve()
    pool_path = args.hard_pool.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or not manifest.get("log_disjoint_split"):
        raise RuntimeError("RAPG data manifest is not log-disjoint and audited")
    if str(pool_path) != str(Path(manifest["hard_pool"]).resolve()):
        raise RuntimeError("hard-pool path disagrees with data manifest")
    rows = trainer.load_hard_pool(pool_path, manifest["hard_pool_sha256"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    split_logs = {}
    for split in ("development", "certification"):
        selected = [
            row
            for row in rows
            if row["split"] == split and row["hard_kind"] == "real_rare"
        ]
        tokens = sorted({str(row["rare_token"]) for row in selected})
        logs = sorted({str(row["log_name"]) for row in selected})
        expected = manifest["split_counts"][split]
        if len(selected) != int(expected["real_rare"]) or len(tokens) != len(selected):
            raise RuntimeError(f"{split} real-rare token count drifted")
        if len(logs) != int(expected["logs"]):
            raise RuntimeError(f"{split} log count drifted")
        filter_path = output_dir / f"{split}_tokens.yaml"
        # JSON is valid YAML and avoids environment-specific YAML emitters.
        filter_path.write_text(json.dumps({"tokens": tokens}, indent=2) + "\n")
        splits[split] = {
            "tokens": len(tokens),
            "logs": len(logs),
            "token_filter": str(filter_path),
            "token_filter_sha256": common.sha256_file(filter_path),
        }
        split_logs[split] = set(logs)
    if split_logs["development"].intersection(split_logs["certification"]):
        raise RuntimeError("closed-loop development/certification logs overlap")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_log_disjoint_closed_loop_splits_v1",
        "selection_split": "development",
        "certification_policy": "consume_once_after_method_lock",
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(pool_path),
        "hard_pool_sha256": common.sha256_file(pool_path),
        "splits": splits,
        "logs_disjoint": True,
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "manifest": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
