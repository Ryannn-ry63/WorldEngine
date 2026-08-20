#!/usr/bin/env python3
"""Build the audited real-rare + filtered-synthetic rare-rollout contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer
from audit_grpo_selector_v3_rare_rollout_collection import (
    BASELINE_SHA256,
    COMPONENT_NAMES,
    senior_v1_filter,
)


METHOD = "diffusiondrive_v3_rare_rollout_mixture_v1"
EXPECTED_REAL_CACHE_SEEDS = (0, 1, 2)
TENSOR_KEYS = (
    "candidate_features",
    "candidate_trajectories_8",
    "route_bev_features",
    "status_tokens",
    "ego_queries",
    "agents_queries",
    "candidate_rewards",
    "candidate_reward_components",
    "candidate_reward_valid_mask",
    "reference_logits",
)


def parse_seed_path(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text).expanduser().resolve()
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected SEED=PATH") from error


def sha256_file(path: Path) -> str:
    return common.sha256_file(path)


def log_split(log_name: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(
            ("diffusiondrive-v3-rare-rollout-v1:" + log_name).encode("utf-8")
        ).digest()[:8],
        "big",
    ) % 10000
    if bucket < 8500:
        return "train"
    if bucket < 9400:
        return "development"
    return "certification"


def rollout_record_paths(root: Path) -> list[Path]:
    candidate = root / "WE_output/openscene_format/diffusiondrive_rollout_records"
    paths = sorted(candidate.glob("*_reward.pkl")) if candidate.is_dir() else []
    if not paths:
        raise RuntimeError(f"no merged DiffusionDrive records under {root}")
    return paths


def checked_array(row: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(row[key])
    if value.shape != shape or not np.isfinite(value).all():
        raise RuntimeError(f"{key} is invalid: shape={value.shape}, expected={shape}")
    return value


def load_collection_audit(path: Path, root: Path) -> dict:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method")
        != "diffusiondrive_v3_rare_rollout_collection_audit_v1"
        or payload.get("layout") != "merged"
        or payload.get("source_policy") != "immutable_epoch100_diffusiondrive"
        or payload.get("checkpoint_sha256") != BASELINE_SHA256
    ):
        raise RuntimeError(f"collection audit contract did not pass: {path}")
    if int(payload.get("num_records", -1)) != len(rollout_record_paths(root)):
        raise RuntimeError(f"collection audit/file count drifted: {path}")
    return payload


def load_record(path: Path, pair_by_rare: dict[str, dict]) -> dict:
    with path.open("rb") as stream:
        row = pickle.load(stream)
    if row.get("schema_version") != 2 or row.get("record_type") != (
        "diffusiondrive_closed_loop_candidate_reward"
    ):
        raise RuntimeError(f"invalid rollout record: {path}")
    if row.get("checkpoint_sha256") != BASELINE_SHA256:
        raise RuntimeError(f"rollout behavior checkpoint drifted: {path}")
    origin = str(row.get("rollout_origin_token", ""))
    if origin not in pair_by_rare:
        raise RuntimeError(f"rollout origin is not in the rare set: {path}")
    pair = pair_by_rare[origin]
    scene = str(row["rollout_scene_id"])
    if not scene.startswith(str(pair["log_name"]) + "-"):
        raise RuntimeError(f"rollout scene/log pairing drifted: {path}")
    values = {
        "candidate_features": checked_array(row, "candidate_features", (20, 256)),
        "candidate_trajectories_8": checked_array(
            row, "candidate_trajectories_8", (20, 8, 3)
        ),
        "route_bev_features": checked_array(
            row, "route_bev_features", (20, 8, 256)
        ),
        "status_tokens": checked_array(row, "status_tokens", (1, 256)),
        "ego_queries": checked_array(row, "ego_queries", (1, 256)),
        "agents_queries": checked_array(row, "agents_queries", (30, 256)),
        "candidate_rewards": checked_array(row, "candidate_rewards", (20,)),
        "candidate_reward_components": checked_array(
            row, "candidate_reward_components", (20, 6)
        ),
        "reference_logits": checked_array(row, "reference_logits", (20,)),
    }
    valid = np.asarray(row["candidate_reward_valid_mask"], dtype=np.bool_)
    current = checked_array(row, "current_logits", (20,))
    if valid.shape != (20,) or not valid.all():
        raise RuntimeError(f"invalid candidate mask: {path}")
    if float(np.max(np.abs(current - values["reference_logits"]))) > 1e-5:
        raise RuntimeError(f"record was not generated by the epoch-100 behavior: {path}")
    if tuple(row.get("reward_component_names", ())) != COMPONENT_NAMES:
        raise RuntimeError(f"reward component order drifted: {path}")
    selected = int(row["selected_index"])
    if selected != int(values["reference_logits"].argmax()):
        raise RuntimeError(f"selected/reference action drifted: {path}")
    keep, deployed_failure, low_ep = senior_v1_filter(
        values["candidate_rewards"],
        values["candidate_reward_components"],
        selected,
    )
    identity = f"{scene}:{int(row['worldengine_step']):04d}"
    return {
        "identity": identity,
        "scene": scene,
        "worldengine_step": int(row["worldengine_step"]),
        "origin_rare_token": origin,
        "paired_common_token": str(pair["common_token"]),
        "log_name": str(pair["log_name"]),
        "split": log_split(str(pair["log_name"])),
        "record_path": str(path.resolve()),
        "raw_observation_path": str(row["raw_observation_path"]),
        "selected_index": selected,
        "selected_reward": float(values["candidate_rewards"][selected]),
        "oracle_reward": float(values["candidate_rewards"].max()),
        "eligible": bool(keep),
        "deployed_failure": bool(deployed_failure),
        "low_ep": bool(low_ep),
        "values": {**values, "candidate_reward_valid_mask": valid},
    }


def load_real_caches(seed_paths: list[tuple[int, Path]]):
    paths = dict(seed_paths)
    if set(paths) != set(EXPECTED_REAL_CACHE_SEEDS) or len(paths) != len(seed_paths):
        raise RuntimeError("exact real-cache seeds 0,1,2 are required")
    loaded = [common.load_cache(paths[seed], "train") for seed in EXPECTED_REAL_CACHE_SEEDS]
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, "train", EXPECTED_REAL_CACHE_SEEDS)
    for manifest in manifests:
        if manifest.get("checkpoint_sha256") != BASELINE_SHA256:
            raise RuntimeError("real context cache baseline drifted")
    return paths, caches, manifests


def empty_array(shape, dtype):
    return np.empty(shape, dtype=dtype)


def build_synthetic_cache(rows: list[dict], real_cache: dict) -> dict:
    count = len(rows)
    shapes = {
        "candidate_features": ((count, 20, 256), np.float16),
        "candidate_trajectories_8": ((count, 20, 8, 3), np.float32),
        "route_bev_features": ((count, 20, 8, 256), np.float16),
        "status_tokens": ((count, 1, 256), np.float16),
        "ego_queries": ((count, 1, 256), np.float16),
        "agents_queries": ((count, 30, 256), np.float16),
        "candidate_rewards": ((count, 20), np.float32),
        "candidate_reward_components": ((count, 20, 6), np.float32),
        "candidate_reward_valid_mask": ((count, 20), np.bool_),
        "reference_logits": ((count, 20), np.float32),
    }
    tensors = {}
    for key, (shape, dtype) in shapes.items():
        if rows:
            array = np.stack([row["values"][key] for row in rows]).astype(
                dtype, copy=False
            )
        else:
            array = empty_array(shape, dtype)
        tensors[key] = torch.from_numpy(array)
    return {
        "schema_version": 3,
        "source_kind": "base_policy_rollout",
        "tokens": [row["identity"] for row in rows],
        "scenes": [row["scene"] for row in rows],
        "origin_rare_tokens": [row["origin_rare_token"] for row in rows],
        "paired_common_tokens": [row["paired_common_token"] for row in rows],
        "log_names": [row["log_name"] for row in rows],
        "data_splits": [row["split"] for row in rows],
        "record_paths": [row["record_path"] for row in rows],
        "raw_observation_paths": [row["raw_observation_path"] for row in rows],
        "selected_indices": torch.tensor(
            [row["selected_index"] for row in rows], dtype=torch.long
        ),
        "baseline_selector_state": real_cache["baseline_selector_state"],
        "scene_selector_config": real_cache["scene_selector_config"],
        **tensors,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_data(
    lane_roots: list[Path],
    lane_audits: list[Path],
    real_cache_paths: list[tuple[int, Path]],
    pair_manifest: Path,
    rare_data_audit: Path,
    output_dir: Path,
    minimum_synthetic: int,
    expected_lanes: int = 3,
) -> dict:
    if len(lane_roots) != expected_lanes or len(lane_audits) != expected_lanes:
        raise RuntimeError(f"exactly {expected_lanes} collection lanes are required")
    pair_rows, rare_audit, pair_path, rare_audit_path = rare_trainer.load_pair_contract(
        pair_manifest, rare_data_audit
    )
    pair_by_rare = {str(row["rare_token"]): row for row in pair_rows}
    paths, real_caches, real_manifests = load_real_caches(real_cache_paths)
    rare_trainer.validate_cache_pair_alignment(
        real_caches, real_manifests, pair_rows, rare_audit
    )

    audits = [
        load_collection_audit(path, root)
        for path, root in zip(lane_audits, lane_roots)
    ]
    if len({row["code_sha"] for row in audits}) != 1:
        raise RuntimeError("collection lanes used different code revisions")
    if len({row["config_sha256"] for row in audits}) != 1:
        raise RuntimeError("collection lanes used different rollout configs")
    if len({row["candidate_noise_namespace"] for row in audits}) != 1:
        raise RuntimeError("collection lanes used different noise namespaces")
    if expected_lanes == 3:
        if any(
            int(row.get("records_per_scene", -1)) != 8
            or int(row.get("num_workers", -1)) != 8
            for row in audits
        ):
            raise RuntimeError("formal collection must use 8 records and 8 workers per lane")
        if len({row.get("scenario_file_sha256") for row in audits}) != expected_lanes:
            raise RuntimeError("formal collection lanes do not use distinct scenario shards")
        if sum(int(row["num_scenarios"]) for row in audits) != len(pair_rows):
            raise RuntimeError("formal collection scenario count does not cover all rare rows")
        if sum(int(row["num_records"]) for row in audits) != 8 * len(pair_rows):
            raise RuntimeError("formal collection record count does not cover all rare rows")

    all_rows = []
    seen = set()
    source_digest = hashlib.sha256()
    for lane, root in enumerate(lane_roots):
        for path in rollout_record_paths(root):
            row = load_record(path, pair_by_rare)
            if row["identity"] in seen:
                raise RuntimeError(f"duplicate record across lanes: {row['identity']}")
            seen.add(row["identity"])
            row["lane"] = lane
            all_rows.append(row)
            source_digest.update(path.name.encode("utf-8"))
            source_digest.update(sha256_file(path).encode("ascii"))
    all_rows.sort(key=lambda row: row["identity"])
    collected_origins = {row["origin_rare_token"] for row in all_rows}
    if expected_lanes == 3 and collected_origins != set(pair_by_rare):
        raise RuntimeError(
            "formal collection does not exactly cover every origin rare token: "
            f"missing={len(set(pair_by_rare) - collected_origins)} "
            f"extra={len(collected_origins - set(pair_by_rare))}"
        )
    eligible_rows = [row for row in all_rows if row["eligible"]]
    if len(eligible_rows) < minimum_synthetic:
        raise RuntimeError(
            f"senior-v1 filter retained {len(eligible_rows)} synthetic rows; "
            f"minimum is {minimum_synthetic}"
        )

    hard_rows = []
    for row in pair_rows:
        log_name = str(row["log_name"])
        hard_rows.append(
            {
                "hard_id": "real:" + str(row["rare_token"]),
                "hard_kind": "real_rare",
                "rare_token": str(row["rare_token"]),
                "paired_common_token": str(row["common_token"]),
                "log_name": log_name,
                "split": log_split(log_name),
            }
        )
    for synthetic_index, row in enumerate(eligible_rows):
        hard_rows.append(
            {
                "hard_id": "synthetic:" + row["identity"],
                "hard_kind": "synthetic_rollout",
                "synthetic_index": synthetic_index,
                "rare_token": row["origin_rare_token"],
                "paired_common_token": row["paired_common_token"],
                "log_name": row["log_name"],
                "split": row["split"],
                "record_path": row["record_path"],
                "raw_observation_path": row["raw_observation_path"],
            }
        )
    if len({row["hard_id"] for row in hard_rows}) != len(hard_rows):
        raise RuntimeError("hard-pool identities are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_cache_path = output_dir / "synthetic_cache.pt"
    synthetic_cache = build_synthetic_cache(eligible_rows, real_caches[0])
    torch.save(synthetic_cache, synthetic_cache_path)
    hard_pool_path = output_dir / "hard_pool.jsonl"
    write_jsonl(hard_pool_path, hard_rows)

    filter_counts = Counter()
    for row in all_rows:
        filter_counts["all"] += 1
        filter_counts["oracle_feasible"] += int(row["oracle_reward"] >= 0.9)
        filter_counts["deployed_failure"] += int(row["deployed_failure"])
        filter_counts["low_ep"] += int(row["low_ep"])
        filter_counts["eligible"] += int(row["eligible"])
    hard_kind_counts = Counter(row["hard_kind"] for row in hard_rows)
    split_counts = {
        split: {
            "hard": sum(row["split"] == split for row in hard_rows),
            "real_rare": sum(
                row["split"] == split and row["hard_kind"] == "real_rare"
                for row in hard_rows
            ),
            "synthetic": sum(
                row["split"] == split
                and row["hard_kind"] == "synthetic_rollout"
                for row in hard_rows
            ),
            "logs": len({row["log_name"] for row in hard_rows if row["split"] == split}),
        }
        for split in ("train", "development", "certification")
    }
    split_logs = {
        split: {row["log_name"] for row in hard_rows if row["split"] == split}
        for split in split_counts
    }
    if any(
        split_logs[left].intersection(split_logs[right])
        for left, right in (("train", "development"), ("train", "certification"), ("development", "certification"))
    ):
        raise RuntimeError("rare-rollout log split is not disjoint")

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": METHOD,
        "source_policy": "immutable_epoch100_diffusiondrive",
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "behavior_policy_is_trained_v3": False,
        "selector_architecture": "scene_conditioned_v3_zero_initialized",
        "senior_reference": {
            "dataset_class": "NavSimOpenSceneE2EFineTuneSynthetic",
            "customized_filter": "v1",
            "include_real_failures": True,
            "normal_ratio": 1,
        },
        "mixture_contract": {
            "overall_common_fraction": 0.5,
            "overall_hard_fraction": 0.5,
            "hard_contains": ["real_rare", "filtered_synthetic_rollout"],
            "common_pairing": "deterministic_same_log_pair_inherited_from_origin_rare",
            "formal_sampling_uses_all_log_splits": True,
            "log_split_purpose": "audit_and_future_tuning_only",
        },
        "rare_pair_rows": len(pair_rows),
        "hard_pool_rows": len(hard_rows),
        "hard_kind_counts": dict(sorted(hard_kind_counts.items())),
        "common_pool_rows_per_balanced_cycle": len(hard_rows),
        "unique_paired_common_tokens": len(
            {row["paired_common_token"] for row in hard_rows}
        ),
        "raw_rollout_records": len(all_rows),
        "complete_rare_origin_coverage": collected_origins == set(pair_by_rare),
        "collected_origin_rare_tokens": len(collected_origins),
        "filtered_synthetic_records": len(eligible_rows),
        "filter_counts": dict(sorted(filter_counts.items())),
        "split_counts": split_counts,
        "log_disjoint_split": True,
        "collection_code_sha": audits[0]["code_sha"],
        "collection_config_sha256": audits[0]["config_sha256"],
        "collection_noise_namespace": audits[0]["candidate_noise_namespace"],
        "rollout_record_set_sha256": source_digest.hexdigest(),
        "synthetic_cache": str(synthetic_cache_path),
        "synthetic_cache_sha256": sha256_file(synthetic_cache_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": sha256_file(hard_pool_path),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "rare_data_audit": str(rare_audit_path),
        "rare_data_audit_sha256": sha256_file(rare_audit_path),
        "real_caches": {
            str(seed): {
                "cache": str(paths[seed]),
                "cache_sha256": sha256_file(paths[seed]),
                "manifest": str(paths[seed].parent / "manifest.json"),
                "manifest_sha256": sha256_file(paths[seed].parent / "manifest.json"),
            }
            for seed in EXPECTED_REAL_CACHE_SEEDS
        },
        "collection_lanes": [
            {
                "root": str(root),
                "audit": str(path),
                "audit_sha256": sha256_file(path),
                "num_scenarios": audit["num_scenarios"],
                "num_records": audit["num_records"],
            }
            for root, path, audit in zip(lane_roots, lane_audits, audits)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-root", action="append", type=Path, required=True)
    parser.add_argument("--lane-audit", action="append", type=Path, required=True)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-synthetic", type=int, default=1)
    parser.add_argument("--expected-lanes", type=int, default=3)
    args = parser.parse_args()
    if args.minimum_synthetic < 0:
        raise ValueError("--minimum-synthetic must be non-negative")
    build_data(
        [path.expanduser().resolve() for path in args.lane_root],
        [path.expanduser().resolve() for path in args.lane_audit],
        args.real_cache,
        args.pair_manifest.expanduser().resolve(),
        args.rare_data_audit.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.minimum_synthetic,
        args.expected_lanes,
    )


if __name__ == "__main__":
    main()
