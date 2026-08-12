#!/usr/bin/env python3
"""Build one auditable four-block result row from explicit evaluator files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


PDM_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--openloop-navtest-ade", type=Path, required=True)
    parser.add_argument("--openloop-navtest-pdm", type=Path, required=True)
    parser.add_argument("--openloop-failures-pdm", type=Path, required=True)
    parser.add_argument("--closedloop-nr", type=Path, required=True)
    parser.add_argument("--closedloop-r", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with path.open("r", newline="") as stream:
        return list(csv.DictReader(stream))


def means(path, require_valid=False):
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"empty metric CSV: {path}")
    if require_valid:
        invalid = [row for row in rows if row.get("valid", "").lower() != "true"]
        if invalid:
            raise RuntimeError(f"{path} contains {len(invalid)} invalid rows")
    result = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in PDM_KEYS
    }
    return rows, result


def ade_fde(path):
    rows = read_rows(path)
    average = [row for row in rows if row.get("token") == "average"]
    if len(average) != 1:
        raise RuntimeError(f"expected one average row in {path}")
    return float(average[0]["ade_4s"]), float(average[0]["fde_4s"])


def format_value(value):
    return f"{value:.6f}"


def main():
    args = parse_args()
    paths = {
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "openloop_navtest_ade": args.openloop_navtest_ade.expanduser().resolve(),
        "openloop_navtest_pdm": args.openloop_navtest_pdm.expanduser().resolve(),
        "openloop_failures_pdm": args.openloop_failures_pdm.expanduser().resolve(),
        "closedloop_nr": args.closedloop_nr.expanduser().resolve(),
        "closedloop_r": args.closedloop_r.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    ade, fde = ade_fde(paths["openloop_navtest_ade"])
    navtest_rows, navtest = means(
        paths["openloop_navtest_pdm"], require_valid=True
    )
    failure_rows, failures = means(
        paths["openloop_failures_pdm"], require_valid=True
    )
    nr_rows, nr = means(paths["closedloop_nr"])
    r_rows, reactive = means(paths["closedloop_r"])
    if len(navtest_rows) < 10000:
        raise RuntimeError("OpenLoop-navtest has unexpectedly few rows")
    if len(failure_rows) < 250:
        raise RuntimeError("OpenLoop failures has unexpectedly few rows")
    if len(nr_rows) < 250 or len(r_rows) < 250:
        raise RuntimeError("closed-loop failures evaluation is incomplete")

    episode_successes = sum(
        float(row["no_at_fault_collisions"]) == 1.0
        and float(row["drivable_area_compliance"]) == 1.0
        for row in r_rows
    )
    success_rate = episode_successes / len(r_rows)
    product_of_means = (
        reactive["no_at_fault_collisions"]
        * reactive["drivable_area_compliance"]
    )

    values = [ade, fde]
    for block in (navtest, failures, nr, reactive):
        values.extend(block[key] for key in PDM_KEYS)
    values.append(success_rate)
    row = [args.model_name, args.note] + [
        format_value(value) for value in values
    ]

    report = {
        "schema_version": 1,
        "status": "PASS",
        "model_name": args.model_name,
        "note": args.note,
        "eval_seed": args.eval_seed,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "input_files": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
            if key != "checkpoint"
        },
        "counts": {
            "openloop_navtest": len(navtest_rows),
            "openloop_failures": len(failure_rows),
            "closedloop_nr": len(nr_rows),
            "closedloop_r": len(r_rows),
            "reactive_episode_successes": episode_successes,
        },
        "metrics": {
            "openloop_navtest": {"ade": ade, "fde": fde, **navtest},
            "openloop_failures": failures,
            "closedloop_nonreactive": nr,
            "closedloop_reactive": reactive,
            "success_rate": success_rate,
            "success_rate_definition": (
                "fraction of reactive episodes with "
                "no_at_fault_collisions == 1 and "
                "drivable_area_compliance == 1"
            ),
            "product_of_mean_nc_and_dac_diagnostic_only": product_of_means,
        },
        "tsv_row": "\t".join(row),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(output.suffix + ".tsv").write_text(
        report["tsv_row"] + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
