#!/usr/bin/env python3
"""Prepare immutable BWM scenario shards for DiffusionDrive online rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import yaml


METHOD = "diffusiondrive_v3_rare_rollout_bwm_scenario_contract_v3"
RECORDS_PER_SCENE = 8
EXPECTED_SOURCE_NAMES = (
    "bwm_collision",
    "bwm_low_ep",
    "bwm_offroad",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH") from error
    if name not in EXPECTED_SOURCE_NAMES:
        raise argparse.ArgumentTypeError(
            f"source must be one of {EXPECTED_SOURCE_NAMES}"
        )
    return name, Path(path).expanduser().resolve()


def load_pairs(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("rare/common pair manifest is empty")
    pair_by_rare = {}
    common_by_log = defaultdict(set)
    for row in rows:
        rare = str(row["rare_token"])
        if rare in pair_by_rare:
            raise RuntimeError(f"duplicate rare pair token: {rare}")
        pair_by_rare[rare] = row
        common_by_log[str(row["log_name"])].add(str(row["common_token"]))
    return pair_by_rare, {
        log_name: sorted(tokens) for log_name, tokens in common_by_log.items()
    }


def navtest_membership(path: Path) -> tuple[set[str], set[str]]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("navtest filter is not a mapping")
    tokens = {
        str(value)
        for value in payload.get("tokens", payload.get("scenario_tokens", []))
    }
    logs = {
        str(value)
        for value in payload.get("log_names", payload.get("logs", []))
    }
    return tokens, logs


def parse_scenario_identity(scene_id: str) -> tuple[str, str, str]:
    parts = scene_id.rsplit("-", 2)
    if len(parts) != 3:
        raise RuntimeError(f"unexpected BWM scenario id: {scene_id}")
    log_name, origin_token, variant = parts
    if not log_name or not origin_token or not variant.isdigit():
        raise RuntimeError(f"invalid BWM scenario identity: {scene_id}")
    return log_name, origin_token, variant


def deterministic_common(
    source_name: str,
    scene_id: str,
    log_name: str,
    origin_token: str,
    pair_by_rare: dict,
    common_by_log: dict[str, list[str]],
) -> tuple[str, str]:
    if origin_token in pair_by_rare:
        pair = pair_by_rare[origin_token]
        if str(pair["log_name"]) != log_name:
            raise RuntimeError(f"direct rare pair log drifted: {scene_id}")
        return str(pair["common_token"]), "direct_diffusiondrive_rare_pair"
    candidates = common_by_log.get(log_name, [])
    if not candidates:
        raise RuntimeError(f"BWM scenario log has no common pairing pool: {scene_id}")
    digest = hashlib.sha256(
        f"diffusiondrive-bwm-v1:{source_name}:{scene_id}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(candidates)
    return candidates[index], "deterministic_same_log_common"


def write_pickle_atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def prepare(
    sources: list[tuple[str, Path]],
    pair_manifest: Path,
    navtest_filter: Path,
    output_dir: Path,
    num_lanes: int,
) -> dict:
    if tuple(sorted(name for name, _ in sources)) != tuple(
        sorted(EXPECTED_SOURCE_NAMES)
    ):
        raise RuntimeError("exactly the three BWM generalization sources are required")
    if len({name for name, _ in sources}) != len(sources):
        raise RuntimeError("BWM source names are repeated")
    pair_by_rare, common_by_log = load_pairs(pair_manifest)
    navtest_tokens, navtest_logs = navtest_membership(navtest_filter)

    lanes = [dict() for _ in range(num_lanes)]
    source_counts = Counter()
    source_origin_tokens = defaultdict(set)
    pairing_counts = Counter()
    seen_scenes = set()
    seen_source_variants = set()
    source_rows = []
    mapping_digest = hashlib.sha256()

    for source_name, source_path in sorted(sources):
        with source_path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"BWM source is empty or invalid: {source_path}")
        source_rows.append(
            {
                "source_kind": source_name,
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "num_scenarios": len(payload),
            }
        )
        for scene_id, raw_scene in sorted(payload.items()):
            scene_id = str(scene_id)
            if scene_id in seen_scenes:
                raise RuntimeError(f"BWM sources repeat scenario id: {scene_id}")
            if not isinstance(raw_scene, dict) or str(raw_scene.get("id")) != scene_id:
                raise RuntimeError(f"BWM scenario key/id mismatch: {scene_id}")
            log_name, origin_token, variant = parse_scenario_identity(scene_id)
            source_variant = (source_name, origin_token, variant)
            if source_variant in seen_source_variants:
                raise RuntimeError(f"BWM source repeats origin variant: {scene_id}")
            if origin_token in navtest_tokens or log_name in navtest_logs:
                raise RuntimeError(f"BWM scenario overlaps navtest: {scene_id}")
            common_token, pairing_method = deterministic_common(
                source_name,
                scene_id,
                log_name,
                origin_token,
                pair_by_rare,
                common_by_log,
            )
            scene = dict(raw_scene)
            metadata = dict(scene.get("metadata", {}))
            additions = {
                "rollout_origin_token": origin_token,
                "rollout_sidecar_prefix": f"{origin_token}-{variant}",
                "rollout_source_kind": source_name,
                "rollout_log_name": log_name,
                "paired_common_token": common_token,
                "common_pairing_method": pairing_method,
                "bwm_source_file": str(source_path),
                "bwm_source_file_sha256": source_rows[-1]["sha256"],
            }
            for key, value in additions.items():
                if key in metadata and str(metadata[key]) != str(value):
                    raise RuntimeError(f"BWM metadata collision for {key}: {scene_id}")
                metadata[key] = value
            scene["metadata"] = metadata
            lane = int.from_bytes(
                hashlib.sha256(
                    f"diffusiondrive-bwm-lane-v1:{scene_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            ) % num_lanes
            lanes[lane][scene_id] = scene
            seen_scenes.add(scene_id)
            seen_source_variants.add(source_variant)
            source_counts[source_name] += 1
            source_origin_tokens[source_name].add(origin_token)
            pairing_counts[pairing_method] += 1
            mapping_digest.update(
                f"{source_name}\t{scene_id}\t{origin_token}\t"
                f"{common_token}\t{origin_token}-{variant}\t{lane}\n".encode("utf-8")
            )

    if any(not lane for lane in lanes):
        raise RuntimeError("deterministic BWM sharding produced an empty lane")
    output_dir.mkdir(parents=True, exist_ok=True)
    lane_rows = []
    for lane_index, payload in enumerate(lanes):
        path = output_dir / f"scenario_shard_{lane_index:02d}_of_{num_lanes:02d}.pkl"
        write_pickle_atomic(path, dict(sorted(payload.items())))
        lane_rows.append(
            {
                "lane": lane_index,
                "scenario_file": str(path.resolve()),
                "scenario_file_sha256": sha256_file(path),
                "num_scenarios": len(payload),
                "records_per_scene": RECORDS_PER_SCENE,
            }
        )

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": METHOD,
        "source_kind": "senior_published_bwm_worlds_replayed_by_diffusiondrive",
        "source_policy": "immutable_epoch100_diffusiondrive",
        "num_scenarios": len(seen_scenes),
        "num_origin_tokens": len(
            set().union(*source_origin_tokens.values())
        ),
        "num_lanes": num_lanes,
        "expected_records_per_scene": RECORDS_PER_SCENE,
        "source_counts": dict(sorted(source_counts.items())),
        "source_origin_counts": {
            key: len(value) for key, value in sorted(source_origin_tokens.items())
        },
        "pairing_counts": dict(sorted(pairing_counts.items())),
        "scenario_mapping_sha256": mapping_digest.hexdigest(),
        "navtest_overlap_tokens": 0,
        "navtest_overlap_logs": 0,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "navtest_filter": str(navtest_filter),
        "navtest_filter_sha256": sha256_file(navtest_filter),
        "sources": source_rows,
        "lanes": lane_rows,
    }
    output = output_dir / "bwm_scenario_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--navtest-filter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-lanes", type=int, default=3)
    args = parser.parse_args()
    if args.num_lanes != 3:
        raise ValueError("formal BWM rollout requires exactly three lanes")
    prepare(
        args.source,
        args.pair_manifest.expanduser().resolve(),
        args.navtest_filter.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.num_lanes,
    )


if __name__ == "__main__":
    main()
