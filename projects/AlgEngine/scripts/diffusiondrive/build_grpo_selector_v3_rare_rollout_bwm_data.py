#!/usr/bin/env python3
"""Merge frozen reactive rollout data with audited BWM rollout records."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import build_grpo_selector_v3_rare_rollout_data as base
import grpo_selector_v3_cached_common as common
from audit_grpo_selector_v3_rare_rollout_collection import (
    BASELINE_SHA256,
    COMPONENT_NAMES,
    senior_v1_filter,
)
from prepare_grpo_selector_v3_rare_rollout_bwm_scenarios import (
    METHOD as SCENARIO_METHOD,
    RECORDS_PER_SCENE,
)


METHOD = "diffusiondrive_v3_rare_rollout_bwm_mixture_v1"
CACHE_SOURCE_KIND = "base_policy_rollout_multi_source_bwm"


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"input manifest did not pass: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_scenario_contract(audit_path: Path) -> tuple[dict, dict[str, dict]]:
    audit = load_json(audit_path)
    if (
        audit.get("schema_version") != 3
        or audit.get("method") != SCENARIO_METHOD
        or audit.get("num_lanes") != 3
    ):
        raise RuntimeError("BWM scenario audit contract drifted")
    contracts = {}
    for lane in audit["lanes"]:
        path = Path(lane["scenario_file"]).resolve()
        if common.sha256_file(path) != lane["scenario_file_sha256"]:
            raise RuntimeError(f"BWM scenario shard SHA256 drifted: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if len(payload) != int(lane["num_scenarios"]):
            raise RuntimeError(f"BWM scenario shard count drifted: {path}")
        for scene_id, scene in payload.items():
            metadata = scene.get("metadata", {})
            required = (
                "rollout_origin_token",
                "rollout_sidecar_prefix",
                "rollout_source_kind",
                "rollout_log_name",
                "paired_common_token",
            )
            if not isinstance(metadata, dict) or any(
                metadata.get(key) in (None, "") for key in required
            ):
                raise RuntimeError(f"BWM scenario metadata is incomplete: {scene_id}")
            if scene_id in contracts:
                raise RuntimeError(f"BWM scenario is repeated across lanes: {scene_id}")
            contracts[str(scene_id)] = {key: str(metadata[key]) for key in required}
    if len(contracts) != int(audit["num_scenarios"]):
        raise RuntimeError("BWM scenario coverage drifted")
    return audit, contracts


def load_record(path: Path, contracts: dict[str, dict]) -> dict:
    with path.open("rb") as stream:
        row = pickle.load(stream)
    if row.get("schema_version") != 2 or row.get("record_type") != (
        "diffusiondrive_closed_loop_candidate_reward"
    ):
        raise RuntimeError(f"invalid BWM rollout record: {path}")
    if row.get("checkpoint_sha256") != BASELINE_SHA256:
        raise RuntimeError(f"BWM rollout behavior checkpoint drifted: {path}")
    scene = str(row["rollout_scene_id"])
    if scene not in contracts:
        raise RuntimeError(f"BWM record is outside scenario contract: {path}")
    contract = contracts[scene]
    expected = {
        "rollout_origin_token": contract["rollout_origin_token"],
        "rollout_sidecar_prefix": contract["rollout_sidecar_prefix"],
        "rollout_source_kind": contract["rollout_source_kind"],
        "rollout_log_name": contract["rollout_log_name"],
        "paired_common_token": contract["paired_common_token"],
    }
    for key, value in expected.items():
        if str(row.get(key)) != value:
            raise RuntimeError(f"BWM record {key} drifted: {path}")
    values = {
        "candidate_features": base.checked_array(row, "candidate_features", (20, 256)),
        "candidate_trajectories_8": base.checked_array(
            row, "candidate_trajectories_8", (20, 8, 3)
        ),
        "route_bev_features": base.checked_array(
            row, "route_bev_features", (20, 8, 256)
        ),
        "status_tokens": base.checked_array(row, "status_tokens", (1, 256)),
        "ego_queries": base.checked_array(row, "ego_queries", (1, 256)),
        "agents_queries": base.checked_array(row, "agents_queries", (30, 256)),
        "candidate_rewards": base.checked_array(row, "candidate_rewards", (20,)),
        "candidate_reward_components": base.checked_array(
            row, "candidate_reward_components", (20, 6)
        ),
        "reference_logits": base.checked_array(row, "reference_logits", (20,)),
    }
    valid = np.asarray(row["candidate_reward_valid_mask"], dtype=np.bool_)
    current = base.checked_array(row, "current_logits", (20,))
    if valid.shape != (20,) or not valid.all():
        raise RuntimeError(f"BWM candidate validity drifted: {path}")
    if float(np.max(np.abs(current - values["reference_logits"]))) > 1e-5:
        raise RuntimeError(f"BWM rollout did not use the epoch-100 policy: {path}")
    if tuple(row.get("reward_component_names", ())) != COMPONENT_NAMES:
        raise RuntimeError(f"BWM reward component order drifted: {path}")
    selected = int(row["selected_index"])
    if selected != int(values["reference_logits"].argmax()):
        raise RuntimeError(f"BWM selected/reference action drifted: {path}")
    keep, deployed_failure, low_ep = senior_v1_filter(
        values["candidate_rewards"],
        values["candidate_reward_components"],
        selected,
    )
    step = int(row["worldengine_step"])
    source = contract["rollout_source_kind"]
    return {
        "identity": f"{source}:{scene}:{step:04d}",
        "scene": scene,
        "worldengine_step": step,
        "origin_rare_token": contract["rollout_origin_token"],
        "paired_common_token": contract["paired_common_token"],
        "log_name": contract["rollout_log_name"],
        "split": base.log_split(contract["rollout_log_name"]),
        "source_kind": source,
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


def concatenate_caches(base_cache: dict, new_rows: list[dict]) -> dict:
    new_cache = base.build_synthetic_cache(new_rows, base_cache)
    tensor_keys = (*base.TENSOR_KEYS, "selected_indices")
    list_keys = (
        "tokens",
        "scenes",
        "origin_rare_tokens",
        "paired_common_tokens",
        "log_names",
        "data_splits",
        "record_paths",
        "raw_observation_paths",
    )
    result = {
        "schema_version": 3,
        "source_kind": CACHE_SOURCE_KIND,
        "baseline_selector_state": base_cache["baseline_selector_state"],
        "scene_selector_config": base_cache["scene_selector_config"],
        "source_kinds": ["base_reactive"] * len(base_cache["tokens"])
        + [row["source_kind"] for row in new_rows],
    }
    for key in tensor_keys:
        result[key] = torch.cat((base_cache[key], new_cache[key]), dim=0)
    for key in list_keys:
        result[key] = list(base_cache[key]) + list(new_cache[key])
    return result


def build_data(
    base_manifest_path: Path,
    scenario_audit_path: Path,
    lane_roots: list[Path],
    lane_audits: list[Path],
    real_cache_paths: list[tuple[int, Path]],
    output_dir: Path,
    minimum_new_synthetic: int,
    formal: bool = True,
) -> dict:
    expected_lanes = 3 if formal else 1
    if len(lane_roots) != expected_lanes or len(lane_audits) != expected_lanes:
        raise RuntimeError(f"exactly {expected_lanes} BWM collection lanes are required")
    base_manifest = load_json(base_manifest_path)
    if (
        base_manifest.get("method") != base.METHOD
        or base_manifest.get("baseline_checkpoint_sha256") != BASELINE_SHA256
    ):
        raise RuntimeError("frozen reactive rollout manifest drifted")
    base_cache_path = Path(base_manifest["synthetic_cache"]).resolve()
    base_pool_path = Path(base_manifest["hard_pool"]).resolve()
    if (
        common.sha256_file(base_cache_path) != base_manifest["synthetic_cache_sha256"]
        or common.sha256_file(base_pool_path) != base_manifest["hard_pool_sha256"]
    ):
        raise RuntimeError("frozen reactive rollout data SHA256 drifted")
    base_cache = torch.load(base_cache_path, map_location="cpu")
    base_rows = load_jsonl(base_pool_path)
    if len(base_rows) != int(base_manifest["hard_pool_rows"]):
        raise RuntimeError("frozen reactive hard-pool count drifted")

    paths, real_caches, real_manifests = base.load_real_caches(real_cache_paths)
    token_maps = [
        {str(token): index for index, token in enumerate(cache["tokens"])}
        for cache in real_caches
    ]
    scenario_audit, contracts = load_scenario_contract(scenario_audit_path)
    audits = [
        base.load_collection_audit(path, root)
        for path, root in zip(lane_audits, lane_roots)
    ]
    if len({row["code_sha"] for row in audits}) != 1:
        raise RuntimeError("BWM collection lanes used different code revisions")
    if len({row["config_sha256"] for row in audits}) != 1:
        raise RuntimeError("BWM collection lanes used different rollout configs")
    if len({row["candidate_noise_namespace"] for row in audits}) != 1:
        raise RuntimeError("BWM collection lanes used different noise namespaces")
    expected_workers = 8 if formal else 1
    if any(
        int(row.get("records_per_scene", -1)) != RECORDS_PER_SCENE
        or int(row.get("num_workers", -1)) != expected_workers
        for row in audits
    ):
        raise RuntimeError(
            "BWM collection record/worker contract drifted: "
            f"records_per_scene={RECORDS_PER_SCENE} workers={expected_workers}"
        )
    expected_records = sum(int(row["num_records"]) for row in audits)

    all_rows = []
    seen = set()
    source_digest = hashlib.sha256()
    for root in lane_roots:
        for path in base.rollout_record_paths(root):
            row = load_record(path, contracts)
            if row["identity"] in seen:
                raise RuntimeError(f"duplicate BWM rollout frame: {row['identity']}")
            for token_map in token_maps:
                if row["paired_common_token"] not in token_map:
                    raise RuntimeError(
                        f"BWM row has no common cache token: {row['identity']}"
                    )
            seen.add(row["identity"])
            all_rows.append(row)
            source_digest.update(path.name.encode("utf-8"))
            source_digest.update(common.sha256_file(path).encode("ascii"))
    all_rows.sort(key=lambda row: row["identity"])
    if len(all_rows) != expected_records:
        raise RuntimeError("BWM record file coverage drifted")
    observed_scenes = {row["scene"] for row in all_rows}
    if formal and observed_scenes != set(contracts):
        raise RuntimeError("formal BWM records do not cover every scenario")
    if not formal and not observed_scenes.issubset(contracts):
        raise RuntimeError("smoke BWM records escaped the scenario contract")
    eligible = [row for row in all_rows if row["eligible"]]
    if len(eligible) < minimum_new_synthetic:
        raise RuntimeError(
            f"senior-v1 retained {len(eligible)} BWM rows; minimum is "
            f"{minimum_new_synthetic}"
        )

    offset = len(base_cache["tokens"])
    hard_rows = list(base_rows)
    for local_index, row in enumerate(eligible):
        hard_rows.append(
            {
                "hard_id": "synthetic_bwm:" + row["identity"],
                "hard_kind": "synthetic_rollout",
                "synthetic_index": offset + local_index,
                "rare_token": row["origin_rare_token"],
                "paired_common_token": row["paired_common_token"],
                "log_name": row["log_name"],
                "split": row["split"],
                "source_kind": row["source_kind"],
                "record_path": row["record_path"],
                "raw_observation_path": row["raw_observation_path"],
            }
        )
    if len({row["hard_id"] for row in hard_rows}) != len(hard_rows):
        raise RuntimeError("BWM hard-pool identities are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "synthetic_cache.pt"
    cache = concatenate_caches(base_cache, eligible)
    torch.save(cache, cache_path)
    pool_path = output_dir / "hard_pool.jsonl"
    base.write_jsonl(pool_path, hard_rows)

    filter_counts = Counter()
    source_counts = Counter()
    eligible_source_counts = Counter()
    for row in all_rows:
        filter_counts["all"] += 1
        filter_counts["oracle_feasible"] += int(row["oracle_reward"] >= 0.9)
        filter_counts["deployed_failure"] += int(row["deployed_failure"])
        filter_counts["low_ep"] += int(row["low_ep"])
        filter_counts["eligible"] += int(row["eligible"])
        source_counts[row["source_kind"]] += 1
        eligible_source_counts[row["source_kind"]] += int(row["eligible"])
    hard_source_counts = Counter(
        row.get("source_kind", row["hard_kind"]) for row in hard_rows
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": METHOD,
        "source_policy": "immutable_epoch100_diffusiondrive",
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "behavior_policy_is_trained_v3": False,
        "selector_architecture": "scene_conditioned_v3_zero_initialized",
        "synthetic_cache_source_kind": CACHE_SOURCE_KIND,
        "senior_reference": {
            "dataset_class": "NavSimOpenSceneE2EFineTuneSynthetic",
            "customized_filter": "v1",
            "include_real_failures": True,
            "normal_ratio": 1,
            "generalization_sources": [
                "base_reactive",
                "bwm_collision",
                "bwm_low_ep",
                "bwm_offroad",
            ],
        },
        "mixture_contract": {
            "overall_common_fraction": 0.5,
            "overall_hard_fraction": 0.5,
            "hard_contains": [
                "real_rare",
                "filtered_base_reactive_rollout",
                "filtered_bwm_rollout",
            ],
            "hard_sampling": "uniform_rows_without_source_reweighting",
        },
        "base_reactive_manifest": str(base_manifest_path),
        "base_reactive_manifest_sha256": common.sha256_file(base_manifest_path),
        "bwm_scenario_audit": str(scenario_audit_path),
        "bwm_scenario_audit_sha256": common.sha256_file(scenario_audit_path),
        "rare_pair_rows": int(base_manifest["rare_pair_rows"]),
        "base_filtered_synthetic_records": int(
            base_manifest["filtered_synthetic_records"]
        ),
        "new_bwm_raw_records": len(all_rows),
        "new_bwm_filtered_synthetic_records": len(eligible),
        "filtered_synthetic_records": len(cache["tokens"]),
        "hard_pool_rows": len(hard_rows),
        "hard_source_counts": dict(sorted(hard_source_counts.items())),
        "filter_counts": dict(sorted(filter_counts.items())),
        "bwm_raw_source_counts": dict(sorted(source_counts.items())),
        "bwm_filtered_source_counts": dict(sorted(eligible_source_counts.items())),
        "rollout_record_set_sha256": source_digest.hexdigest(),
        "synthetic_cache": str(cache_path),
        "synthetic_cache_sha256": common.sha256_file(cache_path),
        "hard_pool": str(pool_path),
        "hard_pool_sha256": common.sha256_file(pool_path),
        "pair_manifest": base_manifest["pair_manifest"],
        "pair_manifest_sha256": base_manifest["pair_manifest_sha256"],
        "rare_data_audit": base_manifest["rare_data_audit"],
        "rare_data_audit_sha256": base_manifest["rare_data_audit_sha256"],
        "real_caches": {
            str(seed): {
                "cache": str(paths[seed]),
                "cache_sha256": common.sha256_file(paths[seed]),
                "manifest": str(paths[seed].parent / "manifest.json"),
                "manifest_sha256": common.sha256_file(
                    paths[seed].parent / "manifest.json"
                ),
            }
            for seed in base.EXPECTED_REAL_CACHE_SEEDS
        },
        "collection_code_sha": audits[0]["code_sha"],
        "collection_config_sha256": audits[0]["config_sha256"],
        "collection_noise_namespace": audits[0]["candidate_noise_namespace"],
        "collection_lanes": [
            {
                "root": str(root),
                "audit": str(path),
                "audit_sha256": common.sha256_file(path),
                "num_scenarios": audit["num_scenarios"],
                "num_records": audit["num_records"],
            }
            for root, path, audit in zip(lane_roots, lane_audits, audits)
        ],
        "navtest_overlap_tokens": scenario_audit["navtest_overlap_tokens"],
        "navtest_overlap_logs": scenario_audit["navtest_overlap_logs"],
        "formal_collection": formal,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--scenario-audit", type=Path, required=True)
    parser.add_argument("--lane-root", action="append", type=Path, required=True)
    parser.add_argument("--lane-audit", action="append", type=Path, required=True)
    parser.add_argument("--real-cache", action="append", type=base.parse_seed_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-new-synthetic", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.minimum_new_synthetic < 0:
        raise ValueError("--minimum-new-synthetic must be non-negative")
    build_data(
        args.base_manifest.expanduser().resolve(),
        args.scenario_audit.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.lane_root],
        [path.expanduser().resolve() for path in args.lane_audit],
        args.real_cache,
        args.output_dir.expanduser().resolve(),
        args.minimum_new_synthetic,
        formal=not args.smoke,
    )


if __name__ == "__main__":
    main()
