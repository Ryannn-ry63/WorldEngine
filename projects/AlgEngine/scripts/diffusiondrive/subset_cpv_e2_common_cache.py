#!/usr/bin/env python3
"""Materialize a verified token subset from a schema-v2 selector cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import grpo_selector_v3_cached_common as common


def load_tokens(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tokens = [str(row["token"]) for row in rows]
    if not tokens or len(tokens) != len(set(tokens)):
        raise RuntimeError("selection token records are empty or duplicated")
    return tokens


def subset_cache(cache: dict, tokens: list[str]) -> dict:
    token_map = {str(token): index for index, token in enumerate(cache["tokens"])}
    missing = set(tokens) - set(token_map)
    if missing:
        raise RuntimeError(f"source cache omitted {len(missing)} selection tokens")
    ordered_tokens = sorted(tokens)
    indices = torch.tensor([token_map[token] for token in ordered_tokens], dtype=torch.long)
    source_count = len(cache["tokens"])
    result = {}
    for key, value in cache.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == source_count:
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == source_count:
            result[key] = [value[index] for index in indices.tolist()]
        else:
            result[key] = value
    if result["tokens"] != ordered_tokens:
        raise RuntimeError("subset token ordering drifted")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--token-records", type=Path, required=True)
    parser.add_argument("--nav-filter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("D1_diverse_matched",), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = args.source_cache.expanduser().resolve()
    records_path = args.token_records.expanduser().resolve()
    nav_filter = args.nav_filter.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (source_path, records_path, nav_filter):
        if not path.is_file():
            raise FileNotFoundError(path)
    source, source_manifest = common.load_cache(source_path, "train")
    tokens = load_tokens(records_path)
    cache = subset_cache(source, tokens)
    output.mkdir(parents=True)
    cache_path = output / "cache.pt"
    torch.save(cache, cache_path)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "status": "PASS",
            "method": "cpv_e2_common_cache_subset_v1",
            "arm": args.arm,
            "num_tokens": len(tokens),
            "num_scenes": len(set(cache["scenes"])),
            "nav_filter": str(nav_filter),
            "nav_filter_sha256": common.sha256_file(nav_filter),
            "token_records": str(records_path),
            "token_records_sha256": common.sha256_file(records_path),
            "source_cache": str(source_path),
            "source_cache_sha256": common.sha256_file(source_path),
            "source_manifest_sha256": common.sha256_file(
                source_path.parent / "manifest.json"
            ),
            "cache": str(cache_path),
            "cache_sha256": common.sha256_file(cache_path),
            "tensor_shapes": {
                key: list(value.shape)
                for key, value in cache.items()
                if torch.is_tensor(value)
            },
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    loaded, _ = common.load_cache(cache_path, "train")
    if loaded["tokens"] != sorted(tokens):
        raise RuntimeError("written subset cache failed identity validation")
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "tokens": len(tokens),
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
