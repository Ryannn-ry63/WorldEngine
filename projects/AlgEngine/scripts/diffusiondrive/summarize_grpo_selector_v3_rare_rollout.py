#!/usr/bin/env python3
"""Aggregate the paired three-seed V3 rare-rollout formal comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_grpo_selector_v3_rare_original as base


FAMILIES = (
    ("epoch100", "base"),
    ("common_v3_progress_fix", "common_v3"),
    ("rare_original_frozen", "rare_frozen"),
    ("rare_original_tuned", "rare_tuned"),
    ("rare_rollout_v1", "rare_rollout"),
)
MAIN_METRICS = (
    ("openloop_navtest.score", "OP-PDMS navtest"),
    ("openloop_failures.score", "OP-PDMS rare"),
    ("success_rate", "CL valid rate"),
    ("closedloop_reactive.score", "CL-PDMS"),
)


def markdown(report: dict) -> str:
    lines = [
        "# DiffusionDrive V3 rare-rollout formal comparison",
        "",
        "All values are paired-seed means (seeds 0/1/2), shown on a 0–100 scale.",
        "",
        "| Method | " + " | ".join(label for _, label in MAIN_METRICS) + " |",
        "|---|" + "---:|" * len(MAIN_METRICS),
    ]
    for label, _ in FAMILIES:
        cells = [
            f'{100.0 * report["aggregate"][label][metric]["mean"]:.2f}'
            for metric, _ in MAIN_METRICS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Primary causal comparison: `rare_rollout_v1 - rare_original_frozen`. "
            "Both start from the same immutable epoch-100 DiffusionDrive checkpoint, "
            "train a fresh exact-zero V3 selector, and use the same 304,272 examples, "
            "4,800 optimizer steps, hyperparameters, and paired evaluation seeds. The "
            "difference is the hard half of the training distribution: rare-original "
            "uses real rare frames, while rare-rollout adds senior-v1-filtered online "
            "synthetic frames and keeps real rare frames.",
            "",
            "`rare_original_tuned` is a 4×-compute secondary reference and is not the "
            "compute-matched causal baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--common-v3", action="append", required=True)
    parser.add_argument("--rare-frozen", action="append", required=True)
    parser.add_argument("--rare-tuned", action="append", required=True)
    parser.add_argument("--rare-rollout", action="append", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "epoch100": base.parse_seed_paths(args.base, "base"),
        "common_v3_progress_fix": base.parse_seed_paths(args.common_v3, "common-v3"),
        "rare_original_frozen": base.parse_seed_paths(args.rare_frozen, "rare-frozen"),
        "rare_original_tuned": base.parse_seed_paths(args.rare_tuned, "rare-tuned"),
        "rare_rollout_v1": base.parse_seed_paths(args.rare_rollout, "rare-rollout"),
    }
    rows = {
        label: base.load_summaries(seed_paths, label)
        for label, seed_paths in paths.items()
    }
    data_path = args.data_manifest.expanduser().resolve()
    data = json.loads(data_path.read_text())
    if data.get("status") != "PASS" or data.get("method") != (
        "diffusiondrive_v3_rare_rollout_mixture_v1"
    ):
        raise RuntimeError("rare-rollout data manifest did not pass")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "comparison": (
            "epoch100_vs_common_v3_vs_rare_original_frozen_vs_"
            "rare_original_tuned_vs_compute_matched_rare_rollout_v1"
        ),
        "primary_causal_comparison": "rare_rollout_v1_minus_rare_original_frozen",
        "aggregate": {
            label: base.aggregate_metrics(method_rows)
            for label, method_rows in rows.items()
        },
        "paired_summary_deltas": {
            "rare_rollout_minus_rare_frozen": base.summary_deltas(
                rows["rare_rollout_v1"], rows["rare_original_frozen"]
            ),
            "rare_rollout_minus_common_v3": base.summary_deltas(
                rows["rare_rollout_v1"], rows["common_v3_progress_fix"]
            ),
            "rare_rollout_minus_epoch100": base.summary_deltas(
                rows["rare_rollout_v1"], rows["epoch100"]
            ),
        },
        "paired_scenario_deltas": {
            "rare_rollout_minus_rare_frozen": base.paired_scenario_deltas(
                rows["rare_rollout_v1"], rows["rare_original_frozen"]
            ),
            "rare_rollout_minus_common_v3": base.paired_scenario_deltas(
                rows["rare_rollout_v1"], rows["common_v3_progress_fix"]
            ),
        },
        "data_manifest": {
            "path": str(data_path),
            "sha256": base.sha256_file(data_path),
            "hard_pool_rows": data["hard_pool_rows"],
            "hard_kind_counts": data["hard_kind_counts"],
            "filtered_synthetic_records": data["filtered_synthetic_records"],
        },
        "summaries": {
            label: {
                str(seed): {"path": str(path), "sha256": base.sha256_file(path)}
                for seed, path in sorted(seed_paths.items())
            }
            for label, seed_paths in paths.items()
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
