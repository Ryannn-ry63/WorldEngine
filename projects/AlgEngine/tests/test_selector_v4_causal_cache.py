import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "diffusiondrive"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("v4_causal_cache_common")
objectives = load_module("selector_v4_objectives")
source_builder = load_module("prepare_selector_v4_causal_source")
oracle_common = load_module("oracle_r15_common")
target_builder = load_module("build_selector_v4_causal_targets")


def test_scene_identity_and_deterministic_log_cap():
    rows = [
        f"log-{log:02d}-{token:016x}"
        for log in range(10)
        for token in range(log * 10, log * 10 + 5)
    ]
    selected_a = common.deterministic_cap(rows, 20, 2, "unit")
    selected_b = common.deterministic_cap(reversed(rows), 20, 2, "unit")
    assert selected_a == selected_b
    counts = {}
    for scene_id in selected_a:
        counts[common.scene_origin_log(scene_id)] = counts.get(common.scene_origin_log(scene_id), 0) + 1
    assert len(selected_a) == 20
    assert max(counts.values()) == 2


def test_source_annotation_is_copy_and_strict():
    source = {
        "id": "source-log-0123456789abcdef",
        "token": "source-log-0123456789abcdef",
        "metadata": {
            "x": 1,
            "nuplan_lidar_pc_tokens": ["0123456789abcdef"],
            "openscene_data_infos_dict": {
                "0123456789abcdef": {"log_name": "source-log"}
            },
        },
    }
    annotated = common.annotate_scenario(source, "rare_union", "train")
    assert "v4_causal_source" not in source["metadata"]
    assert common.source_metadata(annotated)["origin_log"] == "source-log"
    assert common.source_metadata(annotated)["scenario_family"] == "rare_union"
    with pytest.raises(ValueError):
        common.annotate_scenario(source, "broad_common", "train")


def test_ipct_preserves_causally_optimal_incumbent():
    logits = torch.tensor([[3.0, 0.0, -2.0]], requires_grad=True)
    q = torch.tensor([[0.995, 1.0, 0.2]])
    loss, audit = objectives.incumbent_preserving_causal_top1_loss(logits, q, logits.detach(), epsilon=0.01)
    assert audit.target_index.tolist() == [0]
    assert audit.incumbent_preserved.tolist() == [True]
    loss.backward()
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 2] > 0.0


def test_ipct_target_uses_best_v3_supported_causal_optimum():
    new_logits = torch.tensor([[5.0, -8.0, -7.0, 0.0]], requires_grad=True)
    v3 = torch.tensor([[4.0, 2.0, 3.0, 0.0]])
    q = torch.tensor([[0.1, 1.0, 0.995, 0.0]])
    loss, audit = objectives.incumbent_preserving_causal_top1_loss(new_logits, q, v3, epsilon=0.01)
    assert audit.target_index.tolist() == [2]
    assert audit.incumbent_preserved.tolist() == [False]
    loss.backward()
    # Candidate 2 is severely suppressed but still receives a strong update.
    assert new_logits.grad[0, 2] < -0.9
    assert new_logits.grad[0, 0] > 0.0


def test_ipct_flat_surface_is_exact_differentiable_zero():
    logits = torch.randn(2, 20, requires_grad=True)
    q = torch.ones(2, 20)
    frozen = torch.randn(2, 20)
    loss, audit = objectives.incumbent_preserving_causal_top1_loss(logits, q, frozen)
    assert loss.item() == 0.0
    assert audit.active_comparisons.tolist() == [0, 0]
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_ipct_is_candidate_permutation_equivariant():
    torch.manual_seed(4)
    logits = torch.randn(3, 20)
    q = torch.rand(3, 20)
    frozen = torch.randn(3, 20)
    permutation = torch.randperm(20)
    original, _ = objectives.incumbent_preserving_causal_top1_loss(logits, q, frozen, reduction="none")
    permuted, _ = objectives.incumbent_preserving_causal_top1_loss(
        logits[:, permutation], q[:, permutation], frozen[:, permutation], reduction="none"
    )
    assert torch.allclose(original, permuted, atol=1e-7, rtol=0.0)


def test_ipct_not_probability_suppressed_like_exact_group_control():
    logits_ipct = torch.tensor([[12.0, -12.0, 0.0]], requires_grad=True)
    logits_grpo = logits_ipct.detach().clone().requires_grad_(True)
    labels = torch.tensor([[0.0, 1.0, 0.1]])
    frozen = logits_ipct.detach()
    ipct, audit = objectives.incumbent_preserving_causal_top1_loss(logits_ipct, labels, frozen)
    grpo = objectives.exact_group_grpo_loss(logits_grpo, labels)
    ipct.backward()
    grpo.backward()
    assert audit.target_index.tolist() == [1]
    assert abs(float(logits_ipct.grad[0, 1])) > 0.9
    assert abs(float(logits_grpo.grad[0, 1])) < 1e-8


def test_treatment_manifest_rejects_future_outcome_selection(tmp_path):
    source_record = tmp_path / "record.pkl"
    with source_record.open("wb") as stream:
        pickle.dump({}, stream)
    row = {
        "scene_id": "source-log-0123456789abcdef",
        "split": "train",
        "scenario_family": "rare_union",
        "origin_log": "source-log",
        "origin_token": "0123456789abcdef",
        "outcome_stratum": "solved",
        "decision_step": 4,
        "state_step": 3,
        "policy_index": 0,
        "candidate_trajectories_8": np.zeros((20, 8, 3)).tolist(),
        "current_logits": np.arange(20).tolist(),
        "candidate_rewards": np.ones(20).tolist(),
        "source_record_a": str(source_record),
        "source_record_a_sha256": common.sha256_file(source_record),
    }
    targets = tmp_path / "targets.json"
    common.atomic_json(targets, {
        "status": "PASS", "method": common.TARGET_METHOD,
        "schema_version": 1, "design_version": common.DESIGN_VERSION,
        "intervention_outcomes_observed_before_freeze": False,
        "target_count": 1, "targets": [row],
    })
    treatment = tmp_path / "treatment.json"
    common.atomic_json(treatment, {
        "status": "PASS", "method": common.TREATMENT_METHOD,
        "schema_version": 1, "design_version": common.DESIGN_VERSION,
        "future_outcome_used_to_choose_treatment": True,
        "treatment_index": 0, "target_count": 1,
        "target_manifest": str(targets),
        "target_manifest_sha256": common.sha256_file(targets),
        "targets": [{"scene_id": row["scene_id"], "decision_step": 4, "treatment_index": 0}],
    })
    with pytest.raises(RuntimeError):
        common.load_treatment(treatment)


def test_train_source_selection_excludes_short_scenarios(tmp_path):
    source_root = tmp_path / "source"
    shard_root = source_root / "p2_shards_v1" / "rare_union"
    shard_root.mkdir(parents=True)
    rows = {}
    token_splits = {}
    train_logs = set()
    for index, length in enumerate((24, 19, 24, 18, 24, 24)):
        token = f"{index:016x}"
        log = f"train-log-{index:02d}"
        scene_id = f"{log}-{token}"
        rows[scene_id] = {
            "id": scene_id, "token": scene_id, "log_length": length,
            "metadata": {
                "nuplan_lidar_pc_tokens": [token],
                "openscene_data_infos_dict": {
                    token: {"log_name": log}
                },
            },
        }
        token_splits[token] = "train"
        train_logs.add(log)
    shard_path = shard_root / "shard_000.pkl"
    common.atomic_pickle(shard_path, rows)
    common.atomic_json(shard_root / "index.json", {
        "revision": "diffusiondrive_grpo_v4_p2_shards_v1",
        "scenario_family": "rare_union",
        "scenario_variant": "original",
        "num_scenarios": len(rows),
        "shards": [{
            "path": str(shard_path), "num_scenarios": len(rows),
            "scenario_ids": list(rows),
        }],
    })
    selected, audit = source_builder.select_train_family(
        source_root, "rare_union", 4, 4, token_splits,
        {"train": train_logs, "validation": set(), "test": set()},
    )
    assert len(selected) == 4
    assert all(scene["log_length"] >= 20 for scene in selected.values())
    assert audit["source_count"] == 6
    assert audit["collectable_source_count"] == 4


def test_target_freezes_baseline_a_reward_without_selecting_on_rerun_drift(tmp_path):
    scene_id = "source-log-0123456789abcdef"
    record_path_a = tmp_path / "record_a.pkl"
    record_path_b = tmp_path / "record_b.pkl"
    record_path_a.write_bytes(b"a")
    record_path_b.write_bytes(b"b")
    candidates = np.zeros((20, 8, 3), dtype=np.float32)
    logits = np.arange(20, dtype=np.float32)
    rewards_a = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    rewards_b = rewards_a.copy()
    rewards_b[3] += 0.1
    base = {
        "candidate_trajectories_8": candidates,
        "current_logits": logits,
        "policy_selected_index": 19,
        "candidate_reward_components": np.zeros((20, 6), dtype=np.float32),
        "state_step": 3,
    }
    record_a = dict(base, candidate_rewards=rewards_a)
    record_b = dict(base, candidate_rewards=rewards_b)
    target = target_builder.make_target(
        {
            "scene_id": scene_id,
            "split": "train",
            "scenario_family": "rare_union",
            "origin_log": "source-log",
            "origin_token": "0123456789abcdef",
            "outcome_stratum": "solved",
            "decision_step": 4,
            "outcome_a": {
                "score": 1.0,
                "success": True,
                "ego_progress": 1.0,
                "first_violation_step": None,
            },
        },
        {scene_id: {4: (record_path_a, record_a)}},
        {scene_id: {4: (record_path_b, record_b)}},
    )
    assert np.allclose(target["candidate_rewards"], rewards_a)
    assert target["local_reward_rerun_max_abs_error"] > 0.09
    assert target["local_reward_rerun_stable_at_array_tolerance"] is False


def test_v4_runtime_is_opt_in_and_merge_isolated():
    default_runner = (
        SCRIPT_ROOT.parents[2] / "SimEngine" / "worldengine" / "configs"
        / "default_runner.yaml"
    ).read_text()
    base_env = (
        SCRIPT_ROOT.parents[2] / "SimEngine" / "worldengine" / "envs"
        / "base_env.py"
    ).read_text()
    merge = (
        SCRIPT_ROOT.parents[2] / "SimEngine" / "scripts"
        / "merge_simulation_results.py"
    ).read_text()
    config = (
        SCRIPT_ROOT.parents[1] / "configs" / "diffusiondrive"
        / "e2e_diffusiondrive_grpo_selector_v3_v4_causal_cache.py"
    ).read_text()
    assert "diffusiondrive_v4_causal_cache: false" in default_runner
    assert "DiffusionDriveV4CausalCacheManager" in base_env
    assert "diffusiondrive_v4_causal_records" in merge
    assert "if re.fullmatch(" in config
    assert "valid_collection = re.fullmatch(" not in config
