from __future__ import annotations

import importlib
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
audit = importlib.import_module(
    "audit_grpo_selector_v3_rare_rollout_collection"
)
builder = importlib.import_module("build_grpo_selector_v3_rare_rollout_data")
trainer = importlib.import_module("train_grpo_selector_v3_cached_rare_rollout")
rare = trainer.rare


def candidate_values(count=1):
    return {
        "candidate_features": torch.zeros(count, 20, 256, dtype=torch.float16),
        "candidate_trajectories_8": torch.zeros(count, 20, 8, 3),
        "route_bev_features": torch.zeros(
            count, 20, 8, 256, dtype=torch.float16
        ),
        "status_tokens": torch.zeros(count, 1, 256, dtype=torch.float16),
        "ego_queries": torch.zeros(count, 1, 256, dtype=torch.float16),
        "agents_queries": torch.zeros(count, 30, 256, dtype=torch.float16),
        "candidate_rewards": torch.zeros(count, 20),
        "candidate_reward_components": torch.zeros(count, 20, 6),
        "candidate_reward_valid_mask": torch.ones(count, 20, dtype=torch.bool),
        "reference_logits": torch.zeros(count, 20),
    }


def test_senior_v1_filter_matches_hydramdp_boundaries():
    rewards = np.full(20, 0.5, dtype=np.float64)
    components = np.ones((20, 6), dtype=np.float64)
    rewards[3] = 0.9
    rewards[0] = 2.0 / 3.0
    keep, failure, low_ep = audit.senior_v1_filter(rewards, components, 0)
    assert keep and failure and not low_ep

    rewards[0] = 0.8
    components[0, 2] = np.nextafter(np.float64(0.2), np.float64(0.0))
    keep, failure, low_ep = audit.senior_v1_filter(rewards, components, 0)
    assert keep and not failure and low_ep

    components[0, 2] = 0.2
    keep, failure, low_ep = audit.senior_v1_filter(rewards, components, 0)
    assert not keep and not failure and not low_ep

    rewards[3] = np.nextafter(np.float64(0.9), np.float64(0.0))
    rewards[0] = 0.5
    keep, _, _ = audit.senior_v1_filter(rewards, components, 0)
    assert not keep


def test_log_split_is_deterministic_and_log_disjoint():
    names = [f"log_{index}" for index in range(1000)]
    first = {name: builder.log_split(name) for name in names}
    second = {name: builder.log_split(name) for name in reversed(names)}
    assert first == second
    assert set(first.values()) == {"train", "development", "certification"}
    buckets = {
        split: {name for name, value in first.items() if value == split}
        for split in set(first.values())
    }
    assert not buckets["train"].intersection(buckets["development"])
    assert not buckets["train"].intersection(buckets["certification"])
    assert not buckets["development"].intersection(buckets["certification"])


def test_synthetic_cache_preserves_model_independent_and_raw_provenance():
    values = {
        key: tensor[0].numpy()
        for key, tensor in candidate_values().items()
    }
    row = {
        "identity": "scene:0004",
        "scene": "log-rare",
        "origin_rare_token": "rare",
        "paired_common_token": "common",
        "log_name": "log",
        "split": "train",
        "record_path": "/records/scene.pkl",
        "raw_observation_path": "/frames/scene.pkl",
        "selected_index": 0,
        "values": values,
    }
    real = {
        "baseline_selector_state": {"0.weight": torch.zeros(1)},
        "scene_selector_config": {
            "feature_dim": 256,
            "model_dim": 256,
            "route_bev_dim": 256,
            "context_dim": 256,
            "geometry_dim": 58,
            "geometry_hidden_dim": 128,
            "num_heads": 4,
            "feedforward_dim": 512,
            "num_route_steps": 8,
            "num_set_layers": 1,
            "use_set_attention": True,
            "use_trajectory_geometry": True,
            "use_route_bev": True,
            "use_scene_context": True,
        },
    }
    cache = builder.build_synthetic_cache([row], real)
    assert cache["tokens"] == ["scene:0004"]
    assert cache["origin_rare_tokens"] == ["rare"]
    assert cache["paired_common_tokens"] == ["common"]
    assert cache["record_paths"] == ["/records/scene.pkl"]
    assert cache["raw_observation_paths"] == ["/frames/scene.pkl"]
    assert tuple(cache["route_bev_features"].shape) == (1, 20, 8, 256)


def test_mixed_batch_is_exactly_source_aware():
    real = candidate_values(2)
    synthetic = candidate_values(1)
    real["reference_logits"][0] = 1.0
    real["reference_logits"][1] = 2.0
    synthetic["reference_logits"][0] = 3.0
    rows = [
        {
            "hard_kind": "real_rare",
            "rare_token": "rare",
            "paired_common_token": "common",
        },
        {
            "hard_kind": "synthetic_rollout",
            "synthetic_index": 0,
            "rare_token": "rare",
            "paired_common_token": "common",
        },
    ]
    inputs, reference, rewards, valid, kinds = trainer.mixed_batch(
        [("hard", 0), ("hard", 1), ("common", 0), ("common", 1)],
        rows,
        real,
        {"rare": 0, "common": 1},
        synthetic,
        torch.device("cpu"),
    )
    assert inputs["candidate_features"].shape[0] == 4
    assert sorted(reference[:, 0].tolist()) == [1.0, 2.0, 2.0, 3.0]
    assert rewards.shape == (4, 20)
    assert valid.all()
    assert kinds == {
        "hard_real_rare": 1,
        "hard_synthetic": 1,
        "common": 2,
    }


def write_collection_fixture(tmp_path: Path):
    scene = "2021.01.01.00.00.00_veh-01_00000_00001-rare"
    scenario_file = tmp_path / "scenarios.pkl"
    with scenario_file.open("wb") as stream:
        pickle.dump({scene: {"id": scene}}, stream)
    report = tmp_path / "runner_report_1.json"
    report.write_text(
        json.dumps(
            [
                {
                    "scenario_name": scene,
                    "log_name": "split_0",
                    "succeeded": True,
                }
            ]
        )
    )
    completed = tmp_path / "split_0/completed_scenarios/completed_split_0.txt"
    completed.parent.mkdir(parents=True)
    completed.write_text(scene + "\n")
    raw = tmp_path / "raw.pkl"
    raw.write_bytes(b"raw")
    sidecar_dir = tmp_path / "split_0" / "diffusiondrive_candidate_sidecars"
    sidecar_dir.mkdir(parents=True)
    record_dir = (
        tmp_path
        / "split_0/WE_output/openscene_format/diffusiondrive_rollout_records"
    )
    record_dir.mkdir(parents=True)
    for step in range(4, 12):
        sidecar = sidecar_dir / f"rare_{step}.pkl"
        sidecar.write_bytes(b"sidecar")
        rewards = np.full(20, 0.5, dtype=np.float32)
        rewards[1] = 0.95
        components = np.ones((20, 6), dtype=np.float32)
        reference = np.arange(20, dtype=np.float32)
        row = {
            "schema_version": 2,
            "record_type": "diffusiondrive_closed_loop_candidate_reward",
            "checkpoint_sha256": audit.BASELINE_SHA256,
            "candidate_noise_namespace": "rare-rollout-test",
            "rollout_scene_id": scene,
            "rollout_origin_token": "rare",
            "worldengine_step": step,
            "candidate_rewards": rewards,
            "candidate_reward_components": components,
            "candidate_reward_valid_mask": np.ones(20, dtype=np.bool_),
            "reward_component_names": audit.COMPONENT_NAMES,
            "reference_logits": reference,
            "current_logits": reference.copy(),
            "selected_index": 19,
            "deployed_candidate_parity_max_abs_error": 0.0,
            "raw_observation_path": str(raw),
            "sidecar_path": str(sidecar),
            "config_sha256": "config",
            "resolved_config_sha256": "resolved",
            "code_sha": "code",
        }
        with (record_dir / f"rare_{step}_reward.pkl").open("wb") as stream:
            pickle.dump(row, stream)
    return scene, scenario_file


def test_resume_audit_accepts_success_after_prior_failure(tmp_path):
    scene, scenario_file = write_collection_fixture(tmp_path)
    # A worker process can be interrupted after it appends the durable
    # completed ledger but before the aggregate runner report is written.
    # Resume therefore trusts the ledger + complete records, not the presence
    # of a success row in a process-lifetime report.
    (tmp_path / "runner_report_1.json").write_text("[]")
    prior = tmp_path / "runner_report_0.json"
    prior.write_text(
        json.dumps(
            [
                {
                    "scenario_name": scene,
                    "log_name": "split_0",
                    "succeeded": False,
                }
            ]
        )
    )
    report = audit.audit_lane(
        tmp_path,
        scenario_file,
        "split",
        "rare-rollout-test",
        "code",
        8,
        1,
        None,
    )
    assert report["status"] == "PASS"
    assert report["num_records"] == 8
    assert report["scenarios_with_prior_failed_attempts"] == 1


def test_formal_compute_contract_matches_rare_original():
    counts = [
        rare_count + common_count
        for block in range(16 * 3)
        for rare_count, common_count in [rare.block_class_counts(6339, block)]
    ]
    assert sum(counts) == trainer.FORMAL_TOTAL_EXAMPLES == 304272
    assert 16 * 3 * ((6339 + 64 - 1) // 64) == (
        trainer.FORMAL_OPTIMIZER_STEPS
    )


def test_smoke_limiter_uses_hydra_struct_append_override():
    runner = SCRIPT_DIR / "run_grpo_selector_v3_rare_rollout_collect_h100.sh"
    source = runner.read_text()
    assert 'MAX_ARGUMENT=("+max_successful_scenarios=${MAX_SCENARIOS}")' in source


def test_formal_builder_requires_exact_origin_rare_coverage(tmp_path, monkeypatch):
    pair_rows = [
        {"rare_token": token, "common_token": f"common_{token}", "log_name": "log"}
        for token in ("rare_a", "rare_b", "rare_c")
    ]
    pair_path = tmp_path / "pairs.jsonl"
    audit_path = tmp_path / "rare_audit.json"
    pair_path.write_text("pairs\n")
    audit_path.write_text("{}\n")
    monkeypatch.setattr(
        builder.rare_trainer,
        "load_pair_contract",
        lambda *_: (pair_rows, {}, pair_path, audit_path),
    )
    monkeypatch.setattr(
        builder,
        "load_real_caches",
        lambda *_: (
            {seed: tmp_path / f"cache_{seed}.pt" for seed in range(3)},
            ({}, {}, {}),
            ({}, {}, {}),
        ),
    )
    monkeypatch.setattr(
        builder.rare_trainer, "validate_cache_pair_alignment", lambda *_: None
    )
    lane_roots = [tmp_path / f"lane{lane}" for lane in range(3)]
    lane_audits = [tmp_path / f"lane{lane}.json" for lane in range(3)]
    audit_rows = {
        str(path): {
            "code_sha": "code",
            "config_sha256": "config",
            "candidate_noise_namespace": "namespace",
            "records_per_scene": 8,
            "num_workers": 8,
            "scenario_file_sha256": f"scenario_{lane}",
            "num_scenarios": 1,
            "num_records": 8,
        }
        for lane, path in enumerate(lane_audits)
    }
    monkeypatch.setattr(
        builder, "load_collection_audit", lambda path, _root: audit_rows[str(path)]
    )
    monkeypatch.setattr(
        builder,
        "rollout_record_paths",
        lambda root: [root / f"record_{index}.pkl" for index in range(8)],
    )
    origin_by_lane = {"lane0": "rare_a", "lane1": "rare_a", "lane2": "rare_b"}
    monkeypatch.setattr(
        builder,
        "load_record",
        lambda path, _pairs: {
            "identity": f"{path.parent.name}:{path.stem}",
            "origin_rare_token": origin_by_lane[path.parent.name],
        },
    )
    monkeypatch.setattr(builder, "sha256_file", lambda _path: "sha")

    with np.testing.assert_raises_regex(
        RuntimeError, "does not exactly cover every origin rare token"
    ):
        builder.build_data(
            lane_roots,
            lane_audits,
            [(seed, tmp_path / f"cache_{seed}.pt") for seed in range(3)],
            pair_path,
            audit_path,
            tmp_path / "output",
            minimum_synthetic=0,
            expected_lanes=3,
        )


def test_tiny_training_starts_from_zero_and_balances_common_hard(
    tmp_path, monkeypatch
):
    config = {
        "feature_dim": 256,
        "model_dim": 256,
        "route_bev_dim": 256,
        "context_dim": 256,
        "geometry_dim": 58,
        "geometry_hidden_dim": 128,
        "num_heads": 4,
        "feedforward_dim": 512,
        "num_route_steps": 8,
        "num_set_layers": 1,
        "use_set_attention": True,
        "use_trajectory_geometry": True,
        "use_route_bev": True,
        "use_scene_context": True,
    }
    real_caches = []
    for seed in range(3):
        cache = {
            "schema_version": 2,
            "tokens": ["rare", "common"],
            "scenes": ["scene", "scene"],
            "scene_selector_config": config,
            **candidate_values(2),
        }
        cache["candidate_rewards"] = torch.rand(2, 20)
        real_caches.append(cache)
    synthetic = {
        "schema_version": 3,
        "source_kind": "base_policy_rollout",
        "tokens": ["synthetic"],
        "scenes": ["scene"],
        "origin_rare_tokens": ["rare"],
        "scene_selector_config": config,
        **candidate_values(1),
    }
    synthetic["candidate_rewards"] = torch.rand(1, 20)
    hard_rows = [
        {
            "hard_id": "real:rare",
            "hard_kind": "real_rare",
            "rare_token": "rare",
            "paired_common_token": "common",
            "split": "train",
        },
        {
            "hard_id": "synthetic:scene:0004",
            "hard_kind": "synthetic_rollout",
            "synthetic_index": 0,
            "rare_token": "rare",
            "paired_common_token": "common",
            "split": "train",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    pool_path = tmp_path / "hard_pool.jsonl"
    synthetic_path = tmp_path / "synthetic.pt"
    manifest_path.write_text("{}")
    pool_path.write_text("{}\n")
    torch.save(synthetic, synthetic_path)
    monkeypatch.setattr(
        trainer,
        "load_contract",
        lambda args: (
            manifest_path,
            {},
            pool_path,
            hard_rows,
            tuple(real_caches),
            tuple({"noise_seed": seed} for seed in range(3)),
            tuple({"rare": 0, "common": 1} for _ in range(3)),
            synthetic_path,
            synthetic,
        ),
    )
    args = argparse.Namespace(
        real_cache=[],
        synthetic_cache=synthetic_path,
        data_manifest=manifest_path,
        hard_pool=pool_path,
        output_dir=tmp_path / "train",
        temperature=1.0,
        learning_rate=1e-4,
        kl_weight=1e-3,
        seed=0,
        epochs=1,
        examples_per_cache_epoch=4,
        checkpoint_epochs="1",
        batch_size=4,
        device="cpu",
        method_name=trainer.METHOD,
        formal_contract=False,
        smoke_limit_hard_pool=None,
    )
    report = trainer.train(args)
    assert report["initial_max_abs_delta"] == 0.0
    assert report["sampling"]["class_examples"] == {"hard": 6, "common": 6}
    assert report["sampling"]["total_optimizer_steps"] == 3
    state = torch.load(
        tmp_path / "train/epoch_1_scene_selector.pt", map_location="cpu"
    )
    assert len(state["scene_selector_state"]) == 54
