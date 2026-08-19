#!/usr/bin/env python3
"""Build deterministic log-disjoint tuning splits for rare-original V3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import prepare_grpo_selector_v3_rare_original_data as prepare


SPLIT_NAMES = ("train", "development", "certification")


def load_pair_rows(pair_manifest):
    pair_manifest = Path(pair_manifest).expanduser().resolve()
    rows = []
    for line_number, line in enumerate(pair_manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"rare_token", "common_token", "log_name", "rare_votes"}
        if not required.issubset(row):
            raise RuntimeError(
                f"pair row {line_number} missing keys {sorted(required - set(row))}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"empty pair manifest: {pair_manifest}")
    return rows


def assign_logs_to_splits(rows, split_seed, train_fraction, development_fraction):
    fractions = {
        "train": float(train_fraction),
        "development": float(development_fraction),
        "certification": 1.0 - float(train_fraction) - float(development_fraction),
    }
    if any(value <= 0.0 for value in fractions.values()):
        raise ValueError("train/development/certification fractions must be positive")

    by_log = defaultdict(list)
    for row in rows:
        by_log[str(row["log_name"])].append(row)
    if len(by_log) < len(SPLIT_NAMES):
        raise RuntimeError("log-disjoint tuning split requires at least three rare logs")

    targets = {name: len(rows) * fractions[name] for name in SPLIT_NAMES}
    assigned = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    ordered_logs = sorted(
        by_log,
        key=lambda log_name: (
            -len(by_log[log_name]),
            prepare.deterministic_rank(split_seed, "log_split", log_name),
        ),
    )
    for log_index, log_name in enumerate(ordered_logs):
        remaining_logs = len(ordered_logs) - log_index
        empty = [name for name in SPLIT_NAMES if not assigned[name]]
        candidates = empty if len(empty) == remaining_logs else SPLIT_NAMES

        def assignment_score(name):
            hypothetical = dict(counts)
            hypothetical[name] += len(by_log[log_name])
            normalized_error = sum(
                ((hypothetical[key] - targets[key]) / max(targets[key], 1.0)) ** 2
                for key in SPLIT_NAMES
            )
            return (
                normalized_error,
                prepare.deterministic_rank(
                    split_seed, "split_tie", log_name, name
                ),
            )

        selected = min(candidates, key=assignment_score)
        assigned[selected].append(log_name)
        counts[selected] += len(by_log[log_name])
    return assigned, fractions


def build_tuning_split(
    pair_manifest,
    rare_data_audit,
    base_filter,
    output_dir,
    split_seed,
    train_fraction,
    development_fraction,
):
    pair_manifest = Path(pair_manifest).expanduser().resolve()
    rare_data_audit = Path(rare_data_audit).expanduser().resolve()
    base_filter = Path(base_filter).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (pair_manifest, rare_data_audit, base_filter):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(rare_data_audit.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("schema_version") != 1
        or audit.get("method")
        != "diffusiondrive_v3_full_navtrain_rare_original_v1"
    ):
        raise RuntimeError("rare-data audit contract did not pass")
    if audit.get("outputs", {}).get("pairs_sha256") != prepare.sha256_file(
        pair_manifest
    ):
        raise RuntimeError("rare/common pair manifest SHA256 drifted")
    rows = load_pair_rows(pair_manifest)
    if len(rows) != int(audit.get("rare_count", -1)):
        raise RuntimeError("pair count disagrees with rare-data audit")

    assignments, fractions = assign_logs_to_splits(
        rows, split_seed, train_fraction, development_fraction
    )
    base_payload = prepare.load_yaml(base_filter)
    split_reports = {}
    seen_logs = set()
    seen_rare = set()
    for split_name in SPLIT_NAMES:
        log_set = set(assignments[split_name])
        if seen_logs.intersection(log_set):
            raise RuntimeError("log leakage across tuning splits")
        seen_logs.update(log_set)
        split_rows = [
            row for row in rows if str(row["log_name"]) in log_set
        ]
        rare_tokens = {str(row["rare_token"]) for row in split_rows}
        common_tokens = {str(row["common_token"]) for row in split_rows}
        if not split_rows:
            raise RuntimeError(f"empty {split_name} tuning split")
        if seen_rare.intersection(rare_tokens):
            raise RuntimeError("rare-token leakage across tuning splits")
        seen_rare.update(rare_tokens)
        if rare_tokens.intersection(common_tokens):
            raise RuntimeError(f"rare/common overlap in {split_name} split")

        split_root = output_dir / split_name
        pair_path = split_root / "pairs.jsonl"
        rare_filter = split_root / "rare_tokens.yaml"
        common_filter = split_root / "common_tokens.yaml"
        union_filter = split_root / "rare_common_union.yaml"
        prepare.write_immutable_text(
            pair_path,
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in split_rows
            ),
        )
        prepare.write_immutable_text(
            rare_filter, prepare.render_filter(base_payload, rare_tokens)
        )
        prepare.write_immutable_text(
            common_filter, prepare.render_filter(base_payload, common_tokens)
        )
        prepare.write_immutable_text(
            union_filter,
            prepare.render_filter(base_payload, rare_tokens | common_tokens),
        )
        subset_audit_path = split_root / "rare_data_audit.json"
        subset_audit = {
            "schema_version": 1,
            "status": "PASS",
            "method": "diffusiondrive_v3_full_navtrain_rare_original_v1",
            "source_kind": "log_disjoint_tuning_subset",
            "parent_rare_data_audit": str(rare_data_audit),
            "parent_rare_data_audit_sha256": prepare.sha256_file(rare_data_audit),
            "split": split_name,
            "split_seed": int(split_seed),
            "rare_count": len(rare_tokens),
            "paired_common_unique": len(common_tokens),
            "union_count": len(rare_tokens | common_tokens),
            "outputs": {
                "pairs": str(pair_path),
                "pairs_sha256": prepare.sha256_file(pair_path),
                "rare_filter": str(rare_filter),
                "rare_filter_sha256": prepare.sha256_file(rare_filter),
                "common_filter": str(common_filter),
                "common_filter_sha256": prepare.sha256_file(common_filter),
                "union_filter": str(union_filter),
                "union_filter_sha256": prepare.sha256_file(union_filter),
            },
        }
        prepare.write_immutable_text(
            subset_audit_path,
            json.dumps(subset_audit, indent=2, sort_keys=True) + "\n",
        )
        split_reports[split_name] = {
            "logs": len(log_set),
            "log_names_sha256": prepare.digest_values(log_set),
            "pair_rows": len(split_rows),
            "rare_tokens": len(rare_tokens),
            "common_tokens": len(common_tokens),
            "union_tokens": len(rare_tokens | common_tokens),
            "pairs": str(pair_path),
            "pairs_sha256": prepare.sha256_file(pair_path),
            "rare_filter": str(rare_filter),
            "rare_filter_sha256": prepare.sha256_file(rare_filter),
            "common_filter": str(common_filter),
            "common_filter_sha256": prepare.sha256_file(common_filter),
            "union_filter": str(union_filter),
            "union_filter_sha256": prepare.sha256_file(union_filter),
        }
    expected_rare = {str(row["rare_token"]) for row in rows}
    if seen_rare != expected_rare:
        raise RuntimeError("tuning splits did not account for every rare token")

    split_audit = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_original_log_disjoint_tuning_v1",
        "split_seed": int(split_seed),
        "requested_fractions": fractions,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": prepare.sha256_file(pair_manifest),
        "rare_data_audit": str(rare_data_audit),
        "rare_data_audit_sha256": prepare.sha256_file(rare_data_audit),
        "log_disjoint": True,
        "all_rare_rows_accounted": True,
        "splits": split_reports,
    }
    output = output_dir / "split_audit.json"
    prepare.write_immutable_text(
        output, json.dumps(split_audit, indent=2, sort_keys=True) + "\n"
    )
    return split_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--development-fraction", type=float, default=0.1)
    args = parser.parse_args()
    report = build_tuning_split(
        args.pair_manifest,
        args.rare_data_audit,
        args.base_filter,
        args.output_dir,
        args.split_seed,
        args.train_fraction,
        args.development_fraction,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
