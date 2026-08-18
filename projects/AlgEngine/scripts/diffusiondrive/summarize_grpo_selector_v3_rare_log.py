#!/usr/bin/env python3
"""Aggregate paired three-seed rare-log, original V3, and epoch-100 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


BLOCK_FILES = (
    "openloop_navtest_pdm",
    "openloop_failures_pdm",
    "closedloop_nr",
    "closedloop_r",
)
SCENARIO_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "score",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seed_paths(values, label):
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"{label}: expected SEED=SUMMARY, got {value}")
        seed = int(seed_text)
        path = Path(path_text).expanduser().resolve()
        if seed in result:
            raise ValueError(f"{label}: duplicate seed {seed}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[seed] = path
    if set(result) != {0, 1, 2}:
        raise ValueError(f"{label}: expected seeds 0,1,2; got {sorted(result)}")
    return result


def flatten_numeric(prefix, value, output):
    if isinstance(value, dict):
        for key, nested in sorted(value.items()):
            flatten_numeric(f"{prefix}.{key}" if prefix else key, nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def load_summaries(paths, label):
    output = {}
    for seed, path in sorted(paths.items()):
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS":
            raise RuntimeError(f"{label} seed {seed} summary did not pass")
        if int(payload.get("eval_seed", -1)) != seed:
            raise RuntimeError(f"{label} summary seed provenance drifted: {path}")
        if payload.get("checkpoint_sha256") != sha256_file(
            Path(payload["checkpoint"])
        ):
            raise RuntimeError(f"{label} checkpoint SHA256 drifted: {path}")
        flat = {}
        flatten_numeric("", payload["metrics"], flat)
        output[seed] = {"path": path, "payload": payload, "metrics": flat}
    keys = [set(row["metrics"]) for row in output.values()]
    if not all(key_set == keys[0] for key_set in keys[1:]):
        raise RuntimeError(f"{label} metric schema drifted across seeds")
    return output


def aggregate_metrics(rows):
    keys = sorted(next(iter(rows.values()))["metrics"])
    return {
        key: {
            "mean": statistics.fmean(rows[seed]["metrics"][key] for seed in rows),
            "population_std": statistics.pstdev(
                rows[seed]["metrics"][key] for seed in rows
            ),
            "by_seed": {
                str(seed): rows[seed]["metrics"][key] for seed in sorted(rows)
            },
        }
        for key in keys
    }


def summary_deltas(left, right):
    output = {}
    for seed in sorted(left):
        if set(left[seed]["metrics"]) != set(right[seed]["metrics"]):
            raise RuntimeError("paired summary metric schemas disagree")
        output[str(seed)] = {
            key: left[seed]["metrics"][key] - right[seed]["metrics"][key]
            for key in sorted(left[seed]["metrics"])
        }
    return output


def read_scenario_rows(path):
    rows = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            token = str(row.get("token", ""))
            if token == "average":
                continue
            if not token:
                raise RuntimeError(f"scenario CSV has no token column: {path}")
            if token in rows:
                raise RuntimeError(f"duplicate scenario token in {path}: {token}")
            rows[token] = {
                metric: float(row[metric]) for metric in SCENARIO_METRICS
            }
    if not rows:
        raise RuntimeError(f"empty scenario CSV: {path}")
    return rows


def input_file(summary, block):
    record = summary["payload"]["input_files"][block]
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file() or record["sha256"] != sha256_file(path):
        raise RuntimeError(f"summary input provenance drifted: {path}")
    return path


def paired_scenario_deltas(left, right):
    output = {}
    for seed in sorted(left):
        output[str(seed)] = {}
        for block in BLOCK_FILES:
            left_rows = read_scenario_rows(input_file(left[seed], block))
            right_rows = read_scenario_rows(input_file(right[seed], block))
            if set(left_rows) != set(right_rows):
                raise RuntimeError(
                    f"paired scenario token sets disagree for seed {seed} block {block}"
                )
            tokens = sorted(left_rows)
            score_deltas = [
                left_rows[token]["score"] - right_rows[token]["score"]
                for token in tokens
            ]
            output[str(seed)][block] = {
                "num_scenarios": len(tokens),
                "mean_delta": {
                    metric: statistics.fmean(
                        left_rows[token][metric] - right_rows[token][metric]
                        for token in tokens
                    )
                    for metric in SCENARIO_METRICS
                },
                "score_wins": sum(value > 0.0 for value in score_deltas),
                "score_ties": sum(value == 0.0 for value in score_deltas),
                "score_losses": sum(value < 0.0 for value in score_deltas),
            }
    return output


def render_markdown(report):
    selected = (
        "openloop_navtest.ade",
        "openloop_navtest.fde",
        "openloop_navtest.score",
        "openloop_failures.score",
        "closedloop_nonreactive.score",
        "closedloop_reactive.score",
        "success_rate",
    )
    lines = [
        "# DiffusionDrive V3 rare-log formal comparison",
        "",
        "| Method | " + " | ".join(selected) + " |",
        "|---|" + "---:|" * len(selected),
    ]
    for method in ("base", "common_v3", "rare_log_v3"):
        cells = [
            f'{report["aggregate"][method][metric]["mean"]:.6f} ± '
            f'{report["aggregate"][method][metric]["population_std"]:.6f}'
            for metric in selected
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "All values are mean ± population standard deviation over paired "
            "evaluation seeds 0, 1, and 2. Scenario-level paired deltas and "
            "input SHA256 provenance are stored in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rare", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--common", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--base", action="append", required=True, metavar="SEED=SUMMARY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "rare_log_v3": parse_seed_paths(args.rare, "rare"),
        "common_v3": parse_seed_paths(args.common, "common"),
        "base": parse_seed_paths(args.base, "base"),
    }
    rows = {
        label: load_summaries(seed_paths, label)
        for label, seed_paths in paths.items()
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "comparison": "rare_log_v3_vs_original_common_v3_and_epoch100",
        "aggregate": {
            label: aggregate_metrics(method_rows)
            for label, method_rows in rows.items()
        },
        "paired_summary_deltas": {
            "rare_minus_common": summary_deltas(
                rows["rare_log_v3"], rows["common_v3"]
            ),
            "rare_minus_base": summary_deltas(
                rows["rare_log_v3"], rows["base"]
            ),
        },
        "paired_scenario_deltas": {
            "rare_minus_common": paired_scenario_deltas(
                rows["rare_log_v3"], rows["common_v3"]
            ),
            "rare_minus_base": paired_scenario_deltas(
                rows["rare_log_v3"], rows["base"]
            ),
        },
        "summaries": {
            label: {
                str(seed): {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for seed, path in sorted(seed_paths.items())
            }
            for label, seed_paths in paths.items()
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report))
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
