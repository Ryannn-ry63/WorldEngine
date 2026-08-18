#!/usr/bin/env python3
"""Compare corrected V3 formal results with pre-fix V3 and epoch-100."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


METRICS = {
    "openloop_navtest_pdm": ("metrics", "openloop_navtest", "score"),
    "openloop_navtest_ade": ("metrics", "openloop_navtest", "ade"),
    "openloop_navtest_fde": ("metrics", "openloop_navtest", "fde"),
    "openloop_failures_pdm": ("metrics", "openloop_failures", "score"),
    "closedloop_nr_pdm": ("metrics", "closedloop_nonreactive", "score"),
    "closedloop_nr_ep": ("metrics", "closedloop_nonreactive", "ego_progress"),
    "closedloop_r_pdm": ("metrics", "closedloop_reactive", "score"),
    "closedloop_r_ep": ("metrics", "closedloop_reactive", "ego_progress"),
    "success_rate": ("metrics", "success_rate"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--old-v3-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fixed-prefix", required=True)
    parser.add_argument("--old-v3-prefix", default="e2e_diffusiondrive_grpo_selector_v3")
    parser.add_argument("--reference-prefix", default="e2e_diffusiondrive_reference_paired")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(payload, keys):
    value = payload
    for key in keys:
        value = value[key]
    return float(value)


def load_family(root, prefix):
    rows = []
    inputs = []
    for seed in range(3):
        path = root / "formal_eval" / f"{prefix}_s{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS" or payload.get("eval_seed") != seed:
            raise RuntimeError(f"invalid formal summary: {path}")
        rows.append(
            {
                "seed": seed,
                **{name: nested(payload, keys) for name, keys in METRICS.items()},
            }
        )
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return rows, inputs


def summarize(rows):
    output = {}
    for name in METRICS:
        values = [row[name] for row in rows]
        output[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return output


def deltas(left, right):
    return {
        name: {
            "mean": left[name]["mean"] - right[name]["mean"],
            "per_seed": [
                left[name]["values"][seed] - right[name]["values"][seed]
                for seed in range(3)
            ],
        }
        for name in METRICS
    }


def markdown(report):
    labels = {
        "openloop_navtest_pdm": "OpenLoop-navtest PDM",
        "openloop_navtest_ade": "OpenLoop-navtest ADE",
        "openloop_navtest_fde": "OpenLoop-navtest FDE",
        "openloop_failures_pdm": "OpenLoop-failures PDM",
        "closedloop_nr_pdm": "CL-NR PDM",
        "closedloop_nr_ep": "CL-NR EP",
        "closedloop_r_pdm": "CL-R PDM",
        "closedloop_r_ep": "CL-R EP",
        "success_rate": "Success Rate",
    }
    lines = [
        "# DiffusionDrive V3 Progress Normalization Fix",
        "",
        "| Metric | Epoch-100 | V3 pre-fix | V3 fixed | Fixed - pre-fix | Fixed - epoch100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in METRICS:
        reference = report["aggregate"]["epoch100"][name]["mean"]
        old = report["aggregate"]["v3_pre_fix"][name]["mean"]
        fixed = report["aggregate"]["v3_progress_fix"][name]["mean"]
        lines.append(
            f"| {labels[name]} | {reference:.6f} | {old:.6f} | {fixed:.6f} | "
            f"{fixed - old:+.6f} | {fixed - reference:+.6f} |"
        )
    lines.extend(
        [
            "",
            "三组均值使用相同 eval seed 0/1/2；JSON 同时保存标准差、逐 seed 数值和输入 SHA256。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    fixed_rows, fixed_inputs = load_family(args.fixed_root.resolve(), args.fixed_prefix)
    old_rows, old_inputs = load_family(args.old_v3_root.resolve(), args.old_v3_prefix)
    reference_rows, reference_inputs = load_family(
        args.reference_root.resolve(), args.reference_prefix
    )
    aggregate = {
        "epoch100": summarize(reference_rows),
        "v3_pre_fix": summarize(old_rows),
        "v3_progress_fix": summarize(fixed_rows),
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "aggregate": aggregate,
        "deltas": {
            "fixed_minus_pre_fix": deltas(
                aggregate["v3_progress_fix"], aggregate["v3_pre_fix"]
            ),
            "fixed_minus_epoch100": deltas(
                aggregate["v3_progress_fix"], aggregate["epoch100"]
            ),
        },
        "rows": {
            "epoch100": reference_rows,
            "v3_pre_fix": old_rows,
            "v3_progress_fix": fixed_rows,
        },
        "inputs": {
            "epoch100": reference_inputs,
            "v3_pre_fix": old_inputs,
            "v3_progress_fix": fixed_inputs,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(report))
    print(json.dumps({"status": "PASS", "output": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
