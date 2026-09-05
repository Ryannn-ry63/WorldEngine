#!/usr/bin/env python3
"""Build the frozen CPV E2 common-coverage arms from existing navtrain data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
import yaml


METHOD = "cpv_e2_existing_navtrain_common_coverage_v1"
DEFAULT_SEED = 20260830


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_rank(seed: int, *values: object) -> str:
    payload = "\x1f".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_filter(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"filter is not a mapping: {path}")
    return payload


def render_filter(base: dict, tokens: list[str]) -> str:
    payload = dict(base)
    payload["tokens"] = sorted(tokens)
    if "scenario_tokens" in payload:
        payload["scenario_tokens"] = sorted(tokens)
    return yaml.safe_dump(payload, sort_keys=False)


def load_index(index_root: Path) -> dict[str, str]:
    csvs = sorted((index_root / "metadata").glob("*.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"expected one metric-cache CSV under {index_root}")
    token_to_log: dict[str, str] = {}
    with csvs[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            path = Path(row["file_name"])
            token = path.parent.name
            log_name = path.parents[2].name
            previous = token_to_log.setdefault(token, log_name)
            if previous != log_name:
                raise RuntimeError(f"token belongs to multiple logs: {token}")
    return token_to_log


def load_metadata_rows(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    rows = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported metadata payload: {path}")
    return rows


def command_id(value) -> int:
    vector = np.asarray(value)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise RuntimeError("invalid driving_command metadata")
    return int(vector.argmax())


def stratified_nested_order(rows: list[dict], seed: int) -> list[dict]:
    """Round-robin (log, map, command) cells into one nested deterministic order."""

    cells: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["log_name"], row["map_location"], row["command"])].append(row)
    ordered_cells = sorted(
        cells,
        key=lambda cell: deterministic_rank(seed, "cell", *cell),
    )
    for cell, values in cells.items():
        values.sort(
            key=lambda row: deterministic_rank(seed, "token", *cell, row["token"])
        )
    result = []
    for depth in range(max(len(values) for values in cells.values())):
        for cell in ordered_cells:
            if depth < len(cells[cell]):
                result.append(cells[cell][depth])
    if len(result) != len(rows) or len({row["token"] for row in result}) != len(rows):
        raise RuntimeError("nested common ordering lost or duplicated tokens")
    return result


def token_digest(tokens: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(tokens)) + "\n").encode()).hexdigest()


def arm_summary(
    label: str,
    rows: list[dict],
    d0_tokens: set[str],
    rare_tokens: set[str],
    heldout_logs: set[str],
) -> dict:
    tokens = [row["token"] for row in rows]
    logs = {row["log_name"] for row in rows}
    strata = Counter(
        f"{row['map_location']}/command_{row['command']}" for row in rows
    )
    return {
        "label": label,
        "source_kind": "existing_navtrain_strict_common",
        "unique_tokens": len(set(tokens)),
        "rows": len(tokens),
        "duplicate_rows": len(tokens) - len(set(tokens)),
        "logs": len(logs),
        "strata": dict(sorted(strata.items())),
        "tokens_sha256": token_digest(tokens),
        "overlap_with_d0": len(set(tokens) & d0_tokens),
        "overlap_with_rare": len(set(tokens) & rare_tokens),
        "heldout_log_overlap": len(logs & heldout_logs),
    }


def write_arm(root: Path, label: str, rows: list[dict], base_filter: dict) -> dict:
    arm_root = root / label
    arm_root.mkdir(parents=True)
    token_records = arm_root / "tokens.jsonl"
    token_records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    nav_filter = arm_root / "common_tokens.yaml"
    nav_filter.write_text(render_filter(base_filter, [row["token"] for row in rows]))
    return {
        "token_records": str(token_records),
        "token_records_sha256": sha256_file(token_records),
        "nav_filter": str(nav_filter),
        "nav_filter_sha256": sha256_file(nav_filter),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-cache-index", type=Path, required=True)
    parser.add_argument("--rare-pairs", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-navtrain-tokens", type=int, default=103288)
    parser.add_argument("--expected-train-logs", type=int, default=743)
    parser.add_argument("--expected-common-population", type=int, default=75810)
    parser.add_argument("--expected-d0-unique", type=int, default=5267)
    parser.add_argument("--d2-size", type=int, default=20000)
    return parser.parse_args()


def build(args) -> dict:
    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "metric_cache_index": args.metric_cache_index,
            "rare_pairs": args.rare_pairs,
            "rare_data_audit": args.rare_data_audit,
            "hard_pool": args.hard_pool,
            "data_manifest": args.data_manifest,
            "base_filter": args.base_filter,
            "metadata_root": args.metadata_root,
        }.items()
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if min(args.expected_d0_unique, args.d2_size) <= 0:
        raise ValueError("arm sizes must be positive")

    rare_audit = json.loads(paths["rare_data_audit"].read_text())
    data_manifest = json.loads(paths["data_manifest"].read_text())
    if rare_audit.get("status") != "PASS" or data_manifest.get("status") != "PASS":
        raise RuntimeError("upstream rare/rollout data did not pass")
    hard_rows = read_jsonl(paths["hard_pool"])
    if sha256_file(paths["hard_pool"]) != data_manifest["hard_pool_sha256"]:
        raise RuntimeError("hard-pool SHA256 drifted")
    train_rows = [row for row in hard_rows if row["split"] == "train"]
    train_logs = {row["log_name"] for row in train_rows}
    development_logs = {
        row["log_name"] for row in hard_rows if row["split"] == "development"
    }
    certification_logs = {
        row["log_name"] for row in hard_rows if row["split"] == "certification"
    }
    heldout_logs = development_logs | certification_logs
    if train_logs & heldout_logs:
        raise RuntimeError("hard-pool log split leaked")
    if len(train_logs) != args.expected_train_logs:
        raise RuntimeError(
            f"train log count {len(train_logs)} != {args.expected_train_logs}"
        )

    pairs = read_jsonl(paths["rare_pairs"])
    rare_tokens = {row["rare_token"] for row in pairs}
    token_to_log = load_index(paths["metric_cache_index"])
    if len(token_to_log) != args.expected_navtrain_tokens:
        raise RuntimeError("full navtrain index coverage drifted")
    if len(rare_tokens) != int(rare_audit["rare_count"]):
        raise RuntimeError("rare token count drifted")
    strict_common = set(token_to_log) - rare_tokens
    population_tokens = {
        token for token in strict_common if token_to_log[token] in train_logs
    }
    if len(population_tokens) != args.expected_common_population:
        raise RuntimeError(
            f"train common population {len(population_tokens)} "
            f"!= {args.expected_common_population}"
        )
    d0_draw_tokens = [row["paired_common_token"] for row in train_rows]
    d0_tokens = set(d0_draw_tokens)
    if len(d0_tokens) != args.expected_d0_unique:
        raise RuntimeError(
            f"D0 unique common {len(d0_tokens)} != {args.expected_d0_unique}"
        )
    if not d0_tokens <= population_tokens:
        raise RuntimeError("D0 contains a non-train or non-common token")
    if args.d2_size > len(population_tokens):
        raise RuntimeError("D2 exceeds the available strict-common population")

    metadata_rows = []
    remaining = set(population_tokens)
    for log_name in sorted(train_logs):
        path = paths["metadata_root"] / f"{log_name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in load_metadata_rows(path):
            token = str(row["token"])
            if token not in remaining:
                continue
            if str(row["log_name"]) != log_name:
                raise RuntimeError(f"metadata log mismatch for {token}")
            metadata_rows.append(
                {
                    "token": token,
                    "log_name": log_name,
                    "map_location": str(row["map_location"]),
                    "command": command_id(row["driving_command"]),
                }
            )
            remaining.remove(token)
    if remaining:
        raise RuntimeError(f"metadata omitted {len(remaining)} train common tokens")
    nested = stratified_nested_order(metadata_rows, args.seed)
    by_token = {row["token"]: row for row in metadata_rows}
    d0 = [by_token[token] for token in sorted(d0_tokens)]
    d1 = nested[: args.expected_d0_unique]
    d2 = nested[: args.d2_size]
    if not {row["token"] for row in d1} <= {row["token"] for row in d2}:
        raise RuntimeError("D1 is not nested within D2")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        base_filter = load_filter(paths["base_filter"])
        population_path = temporary / "population.jsonl"
        population_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in nested)
        )
        arm_rows = {"D0_paired_common": d0, "D1_diverse_matched": d1, "D2_diverse_20k": d2}
        arms = {}
        for label, rows in arm_rows.items():
            summary = arm_summary(label, rows, d0_tokens, rare_tokens, heldout_logs)
            summary.update(write_arm(temporary, label, rows, base_filter))
            summary["token_records"] = str(output / label / "tokens.jsonl")
            summary["nav_filter"] = str(output / label / "common_tokens.yaml")
            arms[label] = summary
        arms["D0_paired_common"].update(
            {
                "training_draw_rows": len(d0_draw_tokens),
                "training_duplicate_draws": len(d0_draw_tokens) - len(d0_tokens),
                "maximum_training_multiplicity": max(Counter(d0_draw_tokens).values()),
                "role": "frozen_E1_A3_reference",
            }
        )
        arms["D1_diverse_matched"]["role"] = "paired_common_selection_bias_test"
        arms["D2_diverse_20k"]["role"] = "unique_common_coverage_test"
        for label, arm in arms.items():
            if arm["overlap_with_rare"] or arm["heldout_log_overlap"]:
                raise RuntimeError(f"{label} failed leakage audit")
        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "method": METHOD,
            "seed": args.seed,
            "selection": (
                "nested round-robin over existing (log_name,map_location,driving_command) "
                "cells; metadata are sampling/audit fields only and are not model inputs"
            ),
            "existing_dataset_only": True,
            "new_annotations": False,
            "training_labels_changed": False,
            "population": {
                "strict_common_tokens": len(population_tokens),
                "train_logs": len(train_logs),
                "development_logs_excluded": len(development_logs),
                "certification_logs_excluded": len(certification_logs),
                "population_records": str(output / "population.jsonl"),
                "population_records_sha256": sha256_file(population_path),
            },
            "arms": arms,
            "nesting": {
                "D1_is_subset_of_D2": True,
                "D1_D2_overlap": len(
                    {row["token"] for row in d1} & {row["token"] for row in d2}
                ),
            },
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in paths.items()
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return json.loads((output / "manifest.json").read_text())


def main():
    args = parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_dir.expanduser().resolve()),
                "D0": manifest["arms"]["D0_paired_common"]["unique_tokens"],
                "D1": manifest["arms"]["D1_diverse_matched"]["unique_tokens"],
                "D2": manifest["arms"]["D2_diverse_20k"]["unique_tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
