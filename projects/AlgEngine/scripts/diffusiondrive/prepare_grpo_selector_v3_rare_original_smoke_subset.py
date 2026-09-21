#!/usr/bin/env python3
"""Build a deterministic single-log subset for the rare-original H100 smoke."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import prepare_grpo_selector_v3_rare_original_data as prepare


def build_subset(
    source_index,
    annotation_file,
    base_filter,
    output_dir,
    num_tokens,
    expected_source_tokens,
    seed,
):
    source_index = Path(source_index).expanduser().resolve()
    annotation_file = Path(annotation_file).expanduser().resolve()
    base_filter = Path(base_filter).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (source_index, annotation_file, base_filter):
        if not path.exists():
            raise FileNotFoundError(path)
    source_audit_path = source_index / "index_audit.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError(source_audit_path)
    source_audit = json.loads(source_audit_path.read_text())
    if source_audit.get("status") != "PASS":
        raise RuntimeError("full-navtrain source index did not pass")

    cache_paths = prepare.load_cache_index(source_index)
    if len(cache_paths) != expected_source_tokens:
        raise RuntimeError(
            f"source index contains {len(cache_paths)} != {expected_source_tokens}"
        )
    token_to_log, _ = prepare.load_annotation(annotation_file)
    missing = set(cache_paths) - set(token_to_log)
    if missing:
        raise RuntimeError(f"annotation omitted {len(missing)} source tokens")

    tokens_by_log = defaultdict(list)
    for token in cache_paths:
        tokens_by_log[token_to_log[token]].append(token)
    eligible = [
        (len(tokens), log_name)
        for log_name, tokens in tokens_by_log.items()
        if len(tokens) >= num_tokens
    ]
    if not eligible:
        raise RuntimeError(
            f"no navtrain log contains at least {num_tokens} cache tokens"
        )
    _, selected_log = max(eligible, key=lambda row: (row[0], row[1]))
    selected = sorted(
        tokens_by_log[selected_log],
        key=lambda token: prepare.deterministic_rank(
            seed, "rare-original-smoke", selected_log, token
        ),
    )[:num_tokens]
    if len(selected) != num_tokens or len(set(selected)) != num_tokens:
        raise RuntimeError("smoke subset coverage drifted")

    metadata = output_dir / "metadata" / "metric_cache.csv"
    prepare.write_immutable_text(
        metadata,
        "\n".join(
            ["file_name", *(str(cache_paths[token]) for token in sorted(selected))]
        )
        + "\n",
    )
    base_payload = prepare.load_yaml(base_filter)
    subset_filter = output_dir / "subset_filter.yaml"
    prepare.write_immutable_text(
        subset_filter, prepare.render_filter(base_payload, selected)
    )
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "single_log_full_navtrain_smoke_subset",
        "source_index": str(source_index),
        "source_index_sha256": prepare.sha256_file(
            prepare.metadata_csv(source_index)
        ),
        "source_tokens": len(cache_paths),
        "selected_log": selected_log,
        "num_tokens": len(selected),
        "tokens_sha256": prepare.digest_values(selected),
        "selection_seed": int(seed),
        "metadata_csv": str(metadata),
        "metadata_sha256": prepare.sha256_file(metadata),
        "subset_filter": str(subset_filter),
        "subset_filter_sha256": prepare.sha256_file(subset_filter),
        "annotation_file": str(annotation_file),
        "annotation_sha256": prepare.sha256_file(annotation_file),
    }
    prepare.write_immutable_text(
        output_dir / "subset_audit.json",
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--expected-source-tokens", type=int, default=103288)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.num_tokens < 4:
        raise ValueError("--num-tokens must be at least 4")
    report = build_subset(
        args.source_index,
        args.annotation_file,
        args.base_filter,
        args.output_dir,
        args.num_tokens,
        args.expected_source_tokens,
        args.seed,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
