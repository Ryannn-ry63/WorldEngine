#!/usr/bin/env python3
"""Aggregate arbitrary paired-seed V2 rare experiment families."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import summarize_grpo_selector_v3_rare_original as base


MAIN_METRICS = (
    ("openloop_navtest.score", "OP-PDMS navtest"),
    ("openloop_failures.score", "OP-PDMS rare"),
    ("success_rate", "CL valid rate"),
    ("closedloop_reactive.score", "CL-PDMS"),
)


def parse_input(value: str):
    try:
        family_seed, raw_path = value.split("=", 1)
        family, raw_seed = family_seed.rsplit(":", 1)
        seed = int(raw_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected FAMILY:SEED=SUMMARY") from error
    return family, seed, Path(raw_path).expanduser().resolve()


def markdown(report: dict) -> str:
    lines = [
        "# DiffusionDrive selector V2 rare comparison",
        "",
        "Paired seeds 0/1/2; values below use the 0–100 scale.",
        "",
        "| Method | " + " | ".join(label for _, label in MAIN_METRICS) + " |",
        "|---|" + "---:|" * len(MAIN_METRICS),
    ]
    for family in report["family_order"]:
        cells = [
            f'{100.0 * report["aggregate"][family][metric]["mean"]:.2f}'
            for metric, _ in MAIN_METRICS
        ]
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`rare_original_fixed` and `rare_rollout_fixed` are the fixed-compute "
            "architecture/data attribution rows. Families ending in `_tuned` use "
            "the same predeclared development-only search budget and are secondary "
            "best-tuned results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=parse_input, required=True,
        metavar="FAMILY:SEED=SUMMARY",
    )
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = defaultdict(dict)
    family_order = []
    for family, seed, path in args.input:
        if family not in paths:
            family_order.append(family)
        if seed in paths[family]:
            raise RuntimeError(f"duplicate family/seed input: {family}:{seed}")
        paths[family][seed] = path
    for family, seed_paths in paths.items():
        if set(seed_paths) != {0, 1, 2}:
            raise RuntimeError(f"{family} does not contain exact seeds 0/1/2")
    rows = {
        family: base.load_summaries(seed_paths, family)
        for family, seed_paths in paths.items()
    }
    provenance = []
    for raw_path in args.provenance:
        path = raw_path.expanduser().resolve()
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS":
            raise RuntimeError(f"provenance did not pass: {path}")
        provenance.append(
            {"path": str(path), "sha256": base.sha256_file(path), "payload": payload}
        )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "family_order": family_order,
        "aggregate": {
            family: base.aggregate_metrics(method_rows)
            for family, method_rows in rows.items()
        },
        "paired_summary_deltas": {
            f"{left}_minus_{right}": base.summary_deltas(rows[left], rows[right])
            for left, right in zip(family_order[1:], family_order[:-1])
        },
        "summaries": {
            family: {
                str(seed): {"path": str(path), "sha256": base.sha256_file(path)}
                for seed, path in sorted(seed_paths.items())
            }
            for family, seed_paths in paths.items()
        },
        "provenance": provenance,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
