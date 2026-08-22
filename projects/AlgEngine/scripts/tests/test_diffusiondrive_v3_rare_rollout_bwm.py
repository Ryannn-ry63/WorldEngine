from __future__ import annotations

import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
prepare = importlib.import_module(
    "prepare_grpo_selector_v3_rare_rollout_bwm_scenarios"
)
builder = importlib.import_module("build_grpo_selector_v3_rare_rollout_bwm_data")
SIDECAR_CONTRACT_PATH = (
    SCRIPT_DIR.parents[2]
    / "SimEngine/worldengine/manager/diffusiondrive_sidecar_contract.py"
)
sidecar_spec = importlib.util.spec_from_file_location(
    "diffusiondrive_sidecar_contract_for_test", SIDECAR_CONTRACT_PATH
)
sidecar_contract = importlib.util.module_from_spec(sidecar_spec)
assert sidecar_spec.loader is not None
sidecar_spec.loader.exec_module(sidecar_contract)


def test_bwm_scenarios_are_disjoint_paired_and_deterministic(tmp_path):
    source_paths = []
    pair_rows = []
    for source_index, source_name in enumerate(prepare.EXPECTED_SOURCE_NAMES):
        payload = {}
        for local_index in range(12):
            index = source_index * 12 + local_index
            log_name = f"log_{index:03d}"
            direct_origin = f"direct_{index:03d}"
            origin = direct_origin if index % 2 == 0 else f"other_{index:03d}"
            scene_id = f"{log_name}-{origin}-{local_index:03d}"
            payload[scene_id] = {
                "id": scene_id,
                "name": scene_id,
                "token": f"published-{origin}",
                "dataset": "bwm.nuplan",
                "log_length": 21,
                "metadata": {},
            }
            pair_rows.append(
                {
                    "rare_token": direct_origin,
                    "common_token": f"common_{index:03d}",
                    "log_name": log_name,
                }
            )
        path = tmp_path / f"{source_name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
        source_paths.append((source_name, path))
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows)
    )
    navtest = tmp_path / "navtest.yaml"
    navtest.write_text("tokens: []\nlog_names: []\n")

    first = prepare.prepare(
        source_paths, pair_path, navtest, tmp_path / "first", 3
    )
    second = prepare.prepare(
        source_paths, pair_path, navtest, tmp_path / "second", 3
    )
    assert first["status"] == second["status"] == "PASS"
    assert first["num_scenarios"] == 36
    assert first["expected_records_per_scene"] == 8
    assert {row["records_per_scene"] for row in first["lanes"]} == {8}
    assert first["pairing_counts"] == {
        "deterministic_same_log_common": 18,
        "direct_diffusiondrive_rare_pair": 18,
    }
    assert first["scenario_mapping_sha256"] == second["scenario_mapping_sha256"]
    assert sum(row["num_scenarios"] for row in first["lanes"]) == 36
    seen = set()
    for row in first["lanes"]:
        with Path(row["scenario_file"]).open("rb") as stream:
            payload = pickle.load(stream)
        assert not seen.intersection(payload)
        seen.update(payload)
        for scene in payload.values():
            metadata = scene["metadata"]
            assert metadata["rollout_source_kind"] in prepare.EXPECTED_SOURCE_NAMES
            assert metadata["paired_common_token"].startswith("common_")
            log_name, origin, variant = prepare.parse_scenario_identity(scene["id"])
            assert metadata["rollout_log_name"] == log_name
            assert metadata["rollout_sidecar_prefix"] == f"{origin}-{variant}"


def test_bwm_explicit_sidecar_prefix_precedes_legacy_ambiguous_names():
    scene = {
        "id": "log-rare_token-005",
        "token": "log-rare_token-goal_conditional_copy_with_noise",
        "metadata": {"rollout_sidecar_prefix": "rare_token-005"},
    }
    prefixes = sidecar_contract.sidecar_prefixes(scene)
    assert prefixes[0] == "rare_token-005"
    assert "005" in prefixes
    assert "goal_conditional_copy_with_noise" in prefixes


def test_sidecar_prefix_rejects_path_traversal():
    try:
        sidecar_contract.sidecar_prefixes(
            {"id": "scene", "metadata": {"rollout_sidecar_prefix": "../bad"}}
        )
    except RuntimeError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("unsafe sidecar prefix was accepted")


def test_bwm_prepare_rejects_navtest_overlap(tmp_path):
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(
        json.dumps(
            {"rare_token": "rare", "common_token": "common", "log_name": "log"}
        )
        + "\n"
    )
    sources = []
    for source_name in prepare.EXPECTED_SOURCE_NAMES:
        scene_id = f"log-rare-{len(sources):03d}"
        path = tmp_path / f"{source_name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(
                {
                    scene_id: {
                        "id": scene_id,
                        "name": scene_id,
                        "log_length": 21,
                        "metadata": {},
                    }
                },
                stream,
            )
        sources.append((source_name, path))
    navtest = tmp_path / "navtest.yaml"
    navtest.write_text("tokens: [rare]\n")
    try:
        prepare.prepare(sources, pair_path, navtest, tmp_path / "output", 3)
    except RuntimeError as error:
        assert "overlaps navtest" in str(error)
    else:
        raise AssertionError("navtest-overlapping BWM scenario was accepted")


def tiny_cache(count: int) -> dict:
    return {
        "schema_version": 3,
        "source_kind": "base_policy_rollout",
        "tokens": [f"base:{index}" for index in range(count)],
        "scenes": [f"scene:{index}" for index in range(count)],
        "origin_rare_tokens": [f"rare:{index}" for index in range(count)],
        "paired_common_tokens": [f"common:{index}" for index in range(count)],
        "log_names": [f"log:{index}" for index in range(count)],
        "data_splits": ["train"] * count,
        "record_paths": [f"record:{index}" for index in range(count)],
        "raw_observation_paths": [f"raw:{index}" for index in range(count)],
        "selected_indices": torch.zeros(count, dtype=torch.long),
        "baseline_selector_state": {},
        "scene_selector_config": {"feature_dim": 256},
        "candidate_features": torch.zeros(count, 20, 256, dtype=torch.float16),
        "candidate_trajectories_8": torch.zeros(count, 20, 8, 3),
        "route_bev_features": torch.zeros(count, 20, 8, 256, dtype=torch.float16),
        "status_tokens": torch.zeros(count, 1, 256, dtype=torch.float16),
        "ego_queries": torch.zeros(count, 1, 256, dtype=torch.float16),
        "agents_queries": torch.zeros(count, 30, 256, dtype=torch.float16),
        "candidate_rewards": torch.zeros(count, 20),
        "candidate_reward_components": torch.zeros(count, 20, 6),
        "candidate_reward_valid_mask": torch.ones(count, 20, dtype=torch.bool),
        "reference_logits": torch.zeros(count, 20),
    }


def test_multi_source_cache_preserves_base_indices_and_appends_bwm():
    base_cache = tiny_cache(2)
    values = {
        key: value[0].numpy()
        for key, value in tiny_cache(1).items()
        if isinstance(value, torch.Tensor)
        and key not in ("selected_indices",)
    }
    row = {
        "identity": "bwm:scene:0004",
        "scene": "bwm-scene",
        "origin_rare_token": "origin",
        "paired_common_token": "common",
        "log_name": "log",
        "split": "train",
        "source_kind": "bwm_low_ep",
        "record_path": "record",
        "raw_observation_path": "raw",
        "selected_index": 0,
        "values": values,
    }
    merged = builder.concatenate_caches(base_cache, [row])
    assert merged["source_kind"] == builder.CACHE_SOURCE_KIND
    assert merged["tokens"] == ["base:0", "base:1", "bwm:scene:0004"]
    assert merged["source_kinds"] == [
        "base_reactive",
        "base_reactive",
        "bwm_low_ep",
    ]
    assert tuple(merged["candidate_features"].shape) == (3, 20, 256)
