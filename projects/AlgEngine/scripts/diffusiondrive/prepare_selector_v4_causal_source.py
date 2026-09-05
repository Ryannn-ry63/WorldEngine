#!/usr/bin/env python3
"""Freeze train/development source pools for the V4 causal cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import v4_causal_cache_common as common


def validate_split_contract(source_root: Path) -> tuple[dict, dict, dict[str, set[str]]]:
    split_audit_path = source_root / "split_audit.json"
    token_split_path = source_root / "token_split_map.json"
    audit = json.loads(split_audit_path.read_text())
    token_splits = json.loads(token_split_path.read_text())
    if (
        audit.get("revision") != common.SOURCE_REVISION
        or audit.get("seed") != common.SOURCE_SEED
        or int(audit.get("num_logs", -1)) != 1190
    ):
        raise RuntimeError("upstream DiffusionDrive V4 split contract drifted")
    log_sets = {
        split: set(audit["splits"][split]["log_names"])
        for split in ("train", "validation", "test")
    }
    if any(
        log_sets[left] & log_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("upstream split has origin-log leakage")
    expected_counts = {"train": 832, "validation": 178, "test": 180}
    if {key: len(value) for key, value in log_sets.items()} != expected_counts:
        raise RuntimeError("upstream split log counts drifted")
    return audit, token_splits, log_sets


def validate_scene_id(
    scene_id: str, split: str, token_splits: dict, log_sets: dict[str, set[str]]
) -> None:
    token = common.scene_token(scene_id)
    origin_log = common.scene_origin_log(scene_id)
    if token_splits.get(token) != split:
        raise RuntimeError(f"token split drifted for {scene_id}")
    if origin_log not in log_sets[split]:
        raise RuntimeError(f"origin log split drifted for {scene_id}")
    for other_split, values in log_sets.items():
        if other_split != split and origin_log in values:
            raise RuntimeError(f"origin log leakage for {scene_id}")


def select_train_family(
    source_root: Path,
    family: str,
    count: int,
    maximum_per_log: int,
    token_splits: dict,
    log_sets: dict[str, set[str]],
) -> tuple[dict, dict]:
    index_path = source_root / "p2_shards_v1" / family / "index.json"
    index = json.loads(index_path.read_text())
    if (
        index.get("revision") != "diffusiondrive_grpo_v4_p2_shards_v1"
        or index.get("scenario_family") != family
        or index.get("scenario_variant") != "original"
        or int(index.get("num_scenarios", -1)) != sum(
            int(row["num_scenarios"]) for row in index.get("shards", [])
        )
    ):
        raise RuntimeError(f"invalid train shard index for {family}")
    scene_to_shard: dict[str, Path] = {}
    indexed_count = 0
    shard_rows: dict[Path, list[str]] = defaultdict(list)
    for shard in index["shards"]:
        path = Path(shard["path"]).expanduser().resolve()
        ids = [str(value) for value in shard["scenario_ids"]]
        if len(ids) != int(shard["num_scenarios"]):
            raise RuntimeError(f"shard index count drifted: {path}")
        payload = common.load_pickle(path)
        if not isinstance(payload, dict) or set(payload) != set(ids):
            raise RuntimeError(f"train shard payload/index drifted: {path}")
        indexed_count += len(ids)
        for scene_id in ids:
            if scene_id in scene_to_shard:
                raise RuntimeError(f"duplicate indexed train scene: {scene_id}")
            validate_scene_id(scene_id, "train", token_splits, log_sets)
            scenario = payload[scene_id]
            if str(scenario.get("id")) != scene_id:
                raise RuntimeError(f"scenario identity drifted: {scene_id}")
            if int(scenario.get("log_length", -1)) >= 20:
                scene_to_shard[scene_id] = path
    selected = common.deterministic_cap(
        scene_to_shard,
        count,
        maximum_per_log,
        f"{common.SOURCE_SELECTION_SALT}:train:{family}",
    )
    for scene_id in selected:
        shard_rows[scene_to_shard[scene_id]].append(scene_id)
    result = {}
    source_shards = []
    for path in sorted(shard_rows, key=str):
        payload = common.load_pickle(path)
        expected = set(shard_rows[path])
        missing = expected - set(payload)
        if missing:
            raise RuntimeError(f"selected scenes missing from shard {path}: {sorted(missing)}")
        for scene_id in shard_rows[path]:
            scenario = payload[scene_id]
            if str(scenario.get("id")) != scene_id:
                raise RuntimeError(f"scenario identity drifted: {scene_id}")
            result[scene_id] = common.annotate_scenario(scenario, family, "train")
        source_shards.append({
            "path": str(path),
            "sha256": common.sha256_file(path),
            "selected_count": len(expected),
        })
    if set(result) != set(selected):
        raise RuntimeError(f"failed to materialize complete train pool for {family}")
    ordered = {scene_id: result[scene_id] for scene_id in selected}
    return ordered, {
        "index_file": str(index_path.resolve()),
        "index_file_sha256": common.sha256_file(index_path),
        "source_count": indexed_count,
        "collectable_source_count": len(scene_to_shard),
        "selected_count": len(ordered),
        "origin_log_count": len({common.scene_origin_log(value) for value in ordered}),
        "source_shards": source_shards,
    }


def select_validation_family(
    source_root: Path,
    family: str,
    count: int,
    maximum_per_log: int,
    token_splits: dict,
    log_sets: dict[str, set[str]],
) -> tuple[dict, dict]:
    source_path = source_root / "original" / family / "validation" / "all_scenarios.pkl"
    payload = common.load_pickle(source_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid validation source for {family}")
    for scene_id in payload:
        validate_scene_id(str(scene_id), "validation", token_splits, log_sets)
    collectable = {
        scene_id: scenario for scene_id, scenario in payload.items()
        if int(scenario.get("log_length", -1)) >= 20
    }
    selected = common.deterministic_cap(
        collectable,
        count,
        maximum_per_log,
        f"{common.SOURCE_SELECTION_SALT}:validation:{family}",
    )
    result = {}
    for scene_id in selected:
        scenario = collectable[scene_id]
        if str(scenario.get("id")) != scene_id:
            raise RuntimeError(f"scenario identity drifted: {scene_id}")
        result[scene_id] = common.annotate_scenario(scenario, family, "validation")
    return result, {
        "source_file": str(source_path.resolve()),
        "source_file_sha256": common.sha256_file(source_path),
        "source_count": len(payload),
        "collectable_source_count": len(collectable),
        "selected_count": len(result),
        "origin_log_count": len({common.scene_origin_log(value) for value in result}),
    }


def verify_existing(output_root: Path) -> Path | None:
    audit_path = output_root / "source_audit.json"
    train_path = output_root / "train_pool_1024.pkl"
    validation_path = output_root / "validation_pool_256.pkl"
    present = [path.exists() for path in (audit_path, train_path, validation_path)]
    if not audit_path.exists():
        return None
    if not all(present):
        raise RuntimeError("partial V4 source preparation exists")
    _, audit = common.load_source_audit(audit_path)
    if (
        audit["train_scenario_file"] != str(train_path.resolve())
        or audit["validation_scenario_file"] != str(validation_path.resolve())
    ):
        raise RuntimeError("existing V4 source audit points to another run")
    return audit_path


def prepare(args) -> Path:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    existing = verify_existing(output_root)
    if existing is not None:
        return existing
    audit, token_splits, log_sets = validate_split_contract(source_root)

    pools = {"train": {}, "validation": {}}
    family_audits = {"train": {}, "validation": {}}
    for family in common.ALLOWED_FAMILIES:
        rows, row_audit = select_train_family(
            source_root, family, args.train_per_family, args.maximum_per_log,
            token_splits, log_sets,
        )
        overlap = set(pools["train"]) & set(rows)
        if overlap:
            raise RuntimeError(f"train family scene overlap: {sorted(overlap)}")
        pools["train"].update(rows)
        family_audits["train"][family] = row_audit

        rows, row_audit = select_validation_family(
            source_root, family, args.validation_per_family, args.maximum_per_log,
            token_splits, log_sets,
        )
        overlap = set(pools["validation"]) & set(rows)
        if overlap:
            raise RuntimeError(f"validation family scene overlap: {sorted(overlap)}")
        pools["validation"].update(rows)
        family_audits["validation"][family] = row_audit

    if set(pools["train"]) & set(pools["validation"]):
        raise RuntimeError("train/validation scene leakage in V4 source pools")
    train_logs = {common.scene_origin_log(value) for value in pools["train"]}
    validation_logs = {common.scene_origin_log(value) for value in pools["validation"]}
    if train_logs & validation_logs:
        raise RuntimeError("train/validation origin-log leakage in V4 source pools")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train_pool_1024.pkl"
    validation_path = output_root / "validation_pool_256.pkl"
    common.atomic_pickle(train_path, pools["train"])
    common.atomic_pickle(validation_path, pools["validation"])
    split_audit_path = source_root / "split_audit.json"
    token_split_path = source_root / "token_split_map.json"
    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.SOURCE_METHOD,
        "scientific_role": "frozen_original_train_and_sealed_development_source",
        "training_data_kind_added": False,
        "allowed_families": list(common.ALLOWED_FAMILIES),
        "forbidden_sources": ["test", "augmented", "broad_common", "BWM", "prior_ccv_33"],
        "source_root": str(source_root),
        "source_split_audit": str(split_audit_path),
        "source_split_audit_sha256": common.sha256_file(split_audit_path),
        "token_split_map": str(token_split_path),
        "token_split_map_sha256": common.sha256_file(token_split_path),
        "source_revision": audit["revision"],
        "source_seed": audit["seed"],
        "selection_salt": common.SOURCE_SELECTION_SALT,
        "maximum_scenarios_per_origin_log_per_family": args.maximum_per_log,
        "train_scenario_file": str(train_path.resolve()),
        "train_scenario_file_sha256": common.sha256_file(train_path),
        "train_scenario_count": len(pools["train"]),
        "train_origin_log_count": len(train_logs),
        "validation_scenario_file": str(validation_path.resolve()),
        "validation_scenario_file_sha256": common.sha256_file(validation_path),
        "validation_scenario_count": len(pools["validation"]),
        "validation_origin_log_count": len(validation_logs),
        "test_consumed": False,
        "development_consumed": False,
        "origin_log_overlap": 0,
        "family_audits": family_audits,
        "source_family_counts": {
            split: dict(sorted(Counter(
                common.source_metadata(scene)["scenario_family"]
                for scene in pool.values()
            ).items()))
            for split, pool in pools.items()
        },
    }
    audit_path = output_root / "source_audit.json"
    common.atomic_json(audit_path, payload)
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-per-family", type=int, default=common.TRAIN_POOL_PER_FAMILY)
    parser.add_argument("--validation-per-family", type=int, default=common.VALIDATION_POOL_PER_FAMILY)
    parser.add_argument("--maximum-per-log", type=int, default=common.MAXIMUM_POOL_SCENES_PER_LOG)
    args = parser.parse_args()
    if args.train_per_family <= 0 or args.validation_per_family <= 0 or args.maximum_per_log <= 0:
        raise ValueError("all source selection counts must be positive")
    output = prepare(args)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
