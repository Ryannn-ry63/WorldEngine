#!/usr/bin/env python3
"""Aggregate the complete rare-original V3 causal comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_grpo_selector_v3_rare_original as base


FAMILY_ARGS = (
    ("epoch100", "base"),
    ("common_v3_progress_fix", "common_v3"),
    ("paired_common", "paired_common"),
    ("rare_frozen", "rare_frozen"),
    ("rare_tuned", "rare_tuned"),
)
MAIN_COLUMNS = (
    ("openloop_navtest.score", "OP-PDMS（navtest）"),
    ("openloop_failures.score", "OP-PDMS（rare）"),
    ("success_rate", "CL - Valid Rate"),
    ("closedloop_reactive.score", "CL - PDMS"),
)
DETAIL_COLUMNS = (
    "openloop_navtest.ade",
    "openloop_navtest.fde",
    "closedloop_nonreactive.score",
    "closedloop_reactive.score",
)


def markdown(report):
    lines = [
        "# DiffusionDrive V3 Rare Original Formal Comparison",
        "",
        "与 result.md 一致的四列（3 个 paired eval seeds 的均值，百分制）：",
        "",
        "| Model | "
        + " | ".join(label for _, label in MAIN_COLUMNS)
        + " |",
        "|---|" + "---:|" * len(MAIN_COLUMNS),
    ]
    for label, _ in FAMILY_ARGS:
        cells = [
            f'{100.0 * report["aggregate"][label][metric]["mean"]:.2f}'
            for metric, _ in MAIN_COLUMNS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "完整审计值（均值 ± population std，原始 0–1/米单位）：",
            "",
            "| Model | " + " | ".join(DETAIL_COLUMNS) + " |",
            "|---|" + "---:|" * len(DETAIL_COLUMNS),
        ]
    )
    for label, _ in FAMILY_ARGS:
        cells = [
            f'{report["aggregate"][label][metric]["mean"]:.6f} ± '
            f'{report["aggregate"][label][metric]["population_std"]:.6f}'
            for metric in DETAIL_COLUMNS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "主因果对照是 rare_frozen 与 paired_common：rare_frozen 使用 "
            "50/50 rare/common balanced sampling，paired_common 使用 "
            "common-only sampling；二者使用同一配对 token 池、相同 304,272 "
            "examples、4,800 optimizer steps 和相同 eval seeds。rare_tuned "
            "沿用 50/50 rare/common sampling，但使用 rare development 选择的 "
            "epoch 64，共 1,217,088 examples 和 19,200 optimizer steps，是 "
            "secondary tuned result，不是 compute-matched 因果对照。"
            "common_v3_progress_fix 是与既有 V3 的连续性对照。",
            "",
        ]
    )
    return "\n".join(lines)


def read_audited_json(path, label):
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{label} did not pass: {path}")
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "payload": payload,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--common-v3", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--paired-common", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--rare-frozen", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--rare-tuned", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--tuning-split-audit", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "epoch100": base.parse_seed_paths(args.base, "base"),
        "common_v3_progress_fix": base.parse_seed_paths(args.common_v3, "common-v3"),
        "paired_common": base.parse_seed_paths(args.paired_common, "paired-common"),
        "rare_frozen": base.parse_seed_paths(args.rare_frozen, "rare-frozen"),
        "rare_tuned": base.parse_seed_paths(args.rare_tuned, "rare-tuned"),
    }
    rows = {
        label: base.load_summaries(seed_paths, label)
        for label, seed_paths in paths.items()
    }
    aggregate = {
        label: base.aggregate_metrics(method_rows)
        for label, method_rows in rows.items()
    }
    report = {
        "schema_version": 2,
        "status": "PASS",
        "comparison": (
            "epoch100_vs_corrected_common_v3_vs_compute_matched_paired_common_"
            "vs_rare_original_frozen_vs_rare_original_tuned"
        ),
        "main_table_definition": {
            "op_pdms_navtest": "metrics.openloop_navtest.score",
            "op_pdms_rare": "metrics.openloop_failures.score",
            "cl_valid_rate": (
                "reactive episode fraction with no_at_fault_collisions == 1 "
                "and drivable_area_compliance == 1"
            ),
            "cl_pdms": "metrics.closedloop_reactive.score",
        },
        "aggregate": aggregate,
        "paired_summary_deltas": {
            "rare_frozen_minus_paired_common": base.summary_deltas(
                rows["rare_frozen"], rows["paired_common"]
            ),
            "rare_tuned_minus_paired_common": base.summary_deltas(
                rows["rare_tuned"], rows["paired_common"]
            ),
            "rare_frozen_minus_common_v3": base.summary_deltas(
                rows["rare_frozen"], rows["common_v3_progress_fix"]
            ),
            "rare_tuned_minus_common_v3": base.summary_deltas(
                rows["rare_tuned"], rows["common_v3_progress_fix"]
            ),
        },
        "paired_scenario_deltas": {
            "rare_frozen_minus_paired_common": base.paired_scenario_deltas(
                rows["rare_frozen"], rows["paired_common"]
            ),
            "rare_tuned_minus_paired_common": base.paired_scenario_deltas(
                rows["rare_tuned"], rows["paired_common"]
            ),
        },
        "data_contracts": {
            "rare_data": read_audited_json(
                args.rare_data_audit, "rare-data audit"
            ),
            "tuning_split": read_audited_json(
                args.tuning_split_audit, "tuning split audit"
            ),
            "certification": read_audited_json(
                args.certification, "rare certification"
            ),
        },
        "summaries": {
            label: {
                str(seed): {
                    "path": str(path),
                    "sha256": base.sha256_file(path),
                }
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
