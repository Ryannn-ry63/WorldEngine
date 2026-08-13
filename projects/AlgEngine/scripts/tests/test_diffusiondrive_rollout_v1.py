from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ALGENGINE_ROOT = Path(__file__).resolve().parents[2]
WORLDENGINE_ROOT = ALGENGINE_ROOT.parents[1]
SIMENGINE_ROOT = WORLDENGINE_ROOT / "projects/SimEngine"
DIFFUSION_SCRIPTS = ALGENGINE_ROOT / "scripts/diffusiondrive"
for path in (ALGENGINE_ROOT, SIMENGINE_ROOT, DIFFUSION_SCRIPTS):
    sys.path.insert(0, str(path))


def rollout_context():
    return {
        "schema_version": 1,
        "candidate_features": np.zeros((20, 256), dtype=np.float16),
        "candidate_trajectories_8": np.zeros((20, 8, 3), dtype=np.float32),
        "route_bev_features": np.zeros((20, 8, 256), dtype=np.float16),
        "status_tokens": np.zeros((1, 256), dtype=np.float16),
        "ego_queries": np.zeros((1, 256), dtype=np.float16),
        "agents_queries": np.zeros((30, 256), dtype=np.float16),
        "reference_logits": np.arange(20, dtype=np.float32),
        "current_logits": np.arange(20, dtype=np.float32),
        "selected_indices": np.asarray(19, dtype=np.int64),
    }


def reward_record():
    context = rollout_context()
    return {
        "schema_version": 2,
        "record_type": "diffusiondrive_closed_loop_candidate_reward",
        "rollout_scene_id": "scene-a",
        "rollout_scene_token": "scene-token-a",
        "worldengine_step": 3,
        "planner_step": 4,
        "source_sample_token": "sample-a",
        "selected_index": 19,
        "checkpoint_sha256": "baseline-sha",
        "candidate_noise_namespace": "diffusiondrive_rollout_v1_seed0",
        "config_sha256": "config-file-sha",
        "resolved_config_sha256": "resolved-config-sha",
        "code_sha": "code-sha",
        "reward_component_names": (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ),
        "candidate_rewards": np.linspace(0, 1, 20, dtype=np.float32),
        "candidate_reward_components": np.ones((20, 6), dtype=np.float32),
        "candidate_reward_valid_mask": np.ones(20, dtype=np.bool_),
        "deployed_candidate_parity_max_abs_error": 0.0,
        **{key: value for key, value in context.items() if key != "schema_version"},
    }


def test_expand_candidates_matches_diffusiondrive_endpoint_contract():
    from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
        expand_candidates_to_40,
    )

    rng = np.random.default_rng(7)
    candidates = rng.normal(size=(20, 8, 3)).astype(np.float32)
    candidates[..., 2] = np.linspace(-3.0, 3.0, 8)
    expanded = expand_candidates_to_40(candidates)
    assert expanded.shape == (20, 40, 3)
    np.testing.assert_allclose(expanded[:, 4::5], candidates, atol=1e-6)
    assert np.max(np.abs(expanded[..., 2])) <= np.pi


def test_pairwise_reward_uses_reference_candidate_progress_normalization():
    from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
        pairwise_official_scores,
    )

    multi = np.ones((2, 21), dtype=np.float64)
    multi[0, 2] = 0.0
    weighted = np.ones((5, 21), dtype=np.float64)
    progress_raw = np.linspace(10.0, 30.0, 21)
    scorer = SimpleNamespace(
        _multi_metrics=multi,
        _weighted_metrics=weighted,
        _progress_raw=progress_raw,
    )
    scores, components = pairwise_official_scores(scorer)
    assert scores.shape == (20,)
    assert components.shape == (20, 6)
    # Candidate 0 progresses 11m against a 10m reference, so its normalized
    # progress is 1.0. Candidate 1 is collision-gated and therefore scores 0.
    assert components[0, 2] == 1.0
    assert scores[1] == 0.0
    # A later 20m candidate also normalizes against max(10, 20), not against
    # the best proposal among all 20 candidates.
    assert components[9, 2] == 1.0


def test_rollout_residual_selector_initializes_at_exact_zero():
    import grpo_selector_v3_cached_common as common

    selector = common.SceneConditionedTrajectorySetSelector().eval()
    output = selector(
        torch.randn(1, 20, 256),
        torch.randn(1, 20, 8, 3),
        route_bev_features=torch.randn(1, 20, 8, 256),
        status_token=torch.randn(1, 1, 256),
        ego_query=torch.randn(1, 1, 256),
        agents_query=torch.randn(1, 30, 256),
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_forward_test_exports_the_same_candidate_context():
    from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import (
        DiffusionGRPOOnlineSelectorPlanningHead,
    )

    candidate_features = torch.randn(1, 20, 256)
    candidates = torch.randn(1, 20, 8, 3)
    context = {
        "route_bev_features": torch.randn(1, 20, 8, 256),
        "status_token": torch.randn(1, 1, 256),
        "ego_query": torch.randn(1, 1, 256),
        "agents_query": torch.randn(1, 30, 256),
    }

    class FakeHead:
        export_rollout_context = True

        def _generate_frozen_candidates(self, *unused):
            return candidates, candidate_features

        def _scene_selector_context(self, *unused):
            return context

        def _selector_outputs(self, *unused, context=None):
            assert context is not None
            logits = torch.arange(20, dtype=torch.float32).unsqueeze(0)
            return logits, logits.clone()

        def _build_result(self, candidate_set, current, reference):
            return {
                "selected_indices": current.argmax(dim=-1),
                "candidate_trajectories_8": candidate_set,
            }

    result = DiffusionGRPOOnlineSelectorPlanningHead._forward_test(
        FakeHead(), None, None, None, None
    )
    exported = result["diffusiondrive_rollout_context"]
    assert torch.equal(exported["candidate_features"], candidate_features)
    assert torch.equal(exported["candidate_trajectories_8"], candidates)
    assert torch.equal(exported["route_bev_features"], context["route_bev_features"])


def test_sidecar_is_atomic_and_contains_exact_context(tmp_path):
    from closed_loop.sim_test import save_diffusiondrive_rollout_sidecar

    cfg = {
        "selector_rollout_contract": {
            "deployed_action_parity_required": True,
        }
    }
    cfg = type("ConfigLike", (dict,), {})(cfg)
    cfg.sim = SimpleNamespace(diffusiondrive_rollout_sidecar_path=str(tmp_path))
    result = {
        "token": "sample-a",
        "chosen_ind": 19,
        "trajectory": np.zeros((40, 3), dtype=np.float32),
        "diffusiondrive_rollout_context": rollout_context(),
    }
    path = save_diffusiondrive_rollout_sidecar(
        result,
        cfg,
        SimpleNamespace(prefix="scene-a"),
        4,
        {
            "checkpoint_sha256": "baseline-sha",
            "config_sha256": "config-sha",
            "resolved_config_sha256": "resolved-sha",
            "candidate_noise_namespace": "diffusiondrive_rollout_v1_seed0",
            "code_sha": "code-sha",
        },
    )
    assert path.is_file()
    assert not path.with_suffix(".pkl.tmp").exists()
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["selected_index"] == 19
    assert payload["current_reference_logits_max_abs_error"] == 0.0
    assert payload["candidate_features"].shape == (20, 256)


def test_rollout_record_and_schema_v3_cache_load(tmp_path):
    import build_grpo_selector_rollout_v1_cache as builder
    import grpo_selector_v3_cached_common as common

    record_path = tmp_path / "scene-a_4_reward.pkl"
    with record_path.open("wb") as stream:
        pickle.dump(reward_record(), stream)
    row = builder.load_record(
        record_path,
        "baseline-sha",
        "diffusiondrive_rollout_v1_seed0",
    )
    assert row["token"] == "scene-a:0003"
    assert row["feature"].shape == (20, 256)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = {
        "schema_version": 3,
        "source_kind": "base_policy_rollout",
        "tokens": ["scene-a:0003"],
        "scenes": ["scene-a"],
        "candidate_features": torch.zeros(1, 20, 256, dtype=torch.float16),
        "candidate_trajectories_8": torch.zeros(1, 20, 8, 3),
        "route_bev_features": torch.zeros(1, 20, 8, 256, dtype=torch.float16),
        "status_tokens": torch.zeros(1, 1, 256, dtype=torch.float16),
        "ego_queries": torch.zeros(1, 1, 256, dtype=torch.float16),
        "agents_queries": torch.zeros(1, 30, 256, dtype=torch.float16),
        "candidate_rewards": torch.zeros(1, 20),
        "candidate_reward_components": torch.zeros(1, 20, 6),
        "candidate_reward_valid_mask": torch.ones(1, 20, dtype=torch.bool),
        "reference_logits": torch.zeros(1, 20),
        "baseline_selector_state": {},
        "scene_selector_config": {},
    }
    cache_path = cache_dir / "cache.pt"
    torch.save(cache, cache_path)
    manifest = {
        "schema_version": 3,
        "status": "PASS",
        "source_kind": "base_policy_rollout",
        "split": "train",
        "cache_sha256": common.sha256_file(cache_path),
    }
    import json

    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    loaded, loaded_manifest = common.load_cache(cache_path, "train")
    assert loaded["source_kind"] == "base_policy_rollout"
    assert loaded_manifest["schema_version"] == 3


def test_single_gpu_launcher_preserves_the_rollout_contract():
    launcher = (
        ALGENGINE_ROOT
        / "scripts/diffusiondrive/run_grpo_selector_rollout_v1_1gpu_h100.sh"
    ).read_text()
    rollout = (
        ALGENGINE_ROOT
        / "scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh"
    ).read_text()
    merger = (
        SIMENGINE_ROOT / "scripts/merge_simulation_results.py"
    ).read_text()
    assert "DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1" in launcher
    assert "DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=1" in launcher
    assert 'split_id<ROLLOUT_GPU_COUNT' in rollout
    assert '--num-splits "${ROLLOUT_GPU_COUNT}"' in rollout
    assert 'default=8' in merger


def test_merge_simulation_results_accepts_one_split(tmp_path, monkeypatch):
    import importlib.util
    import pandas as pd

    split_root = tmp_path / "split_0"
    plan_root = split_root / "plan_traj"
    openscene_root = split_root / "WE_output/openscene_format"
    record_root = openscene_root / "diffusiondrive_rollout_records"
    plan_root.mkdir(parents=True)
    record_root.mkdir(parents=True)
    pd.DataFrame([{"token": "scene-a"}]).to_csv(
        plan_root / "plan_idx.csv", index=False
    )
    pd.DataFrame(
        [
            {"token": "scene-a", "pdm_score": 0.5},
            {"token": "overall_average", "pdm_score": 0.5},
        ]
    ).to_csv(openscene_root / "all_scenes_pdm_averages_NR.csv", index=False)
    record_path = record_root / "scene-a_4_reward.pkl"
    with record_path.open("wb") as stream:
        pickle.dump(reward_record(), stream)

    script = SIMENGINE_ROOT / "scripts/merge_simulation_results.py"
    spec = importlib.util.spec_from_file_location("merge_one_split", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--test_path",
            str(tmp_path),
            "--react_type",
            "NR",
            "--num-splits",
            "1",
        ],
    )
    module.main()

    assert (tmp_path / "plan_traj/plan_idx.csv").is_file()
    assert (
        tmp_path
        / "WE_output/openscene_format/diffusiondrive_rollout_records"
        / record_path.name
    ).is_file()


def test_rollout_smoke_and_pilot_limit_input_scenes_deterministically():
    from worldengine.runner.run_simulation import limit_input_scenes

    scenes = {f"scene-{index}": {"index": index} for index in range(12)}
    assert limit_input_scenes(scenes, None) is scenes
    assert list(limit_input_scenes(scenes, 1)) == ["scene-0"]
    assert list(limit_input_scenes(scenes, 8)) == [
        f"scene-{index}" for index in range(8)
    ]


def test_dynamic_reward_waits_until_first_planner_action_exists():
    from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
        should_load_rollout_sidecar,
    )

    assert not should_load_rollout_sidecar(2, num_history=4, buffer_size=9)
    # Step 3 creates the fourth observation needed by the first planner
    # forward; no sidecar can exist until that observation has been written.
    assert not should_load_rollout_sidecar(3, num_history=4, buffer_size=9)
    assert should_load_rollout_sidecar(4, num_history=4, buffer_size=9)
    assert should_load_rollout_sidecar(11, num_history=4, buffer_size=9)
    assert not should_load_rollout_sidecar(12, num_history=4, buffer_size=9)


def test_rollout_execution_audit_rejects_failed_scenarios(tmp_path):
    import audit_grpo_selector_rollout_execution as execution_audit

    report_dir = tmp_path / "__WORKER_ID__/WE_output"
    report_dir.mkdir(parents=True)
    report = report_dir / "runner_report_test.json"
    report.write_text(
        '{"invalid": true}'
    )
    with __import__("pytest").raises(RuntimeError, match="invalid WorldEngine"):
        execution_audit.load_reports(tmp_path)

    report.write_text(
        '[{"scenario_name": "scene-a", "log_name": "split_0", '
        '"succeeded": false, "error_message": "sidecar missing"}]'
    )
    paths, rows = execution_audit.load_reports(tmp_path)
    assert paths == [report]
    assert rows[0]["error_message"] == "sidecar missing"
    with __import__("pytest").raises(RuntimeError, match="sidecar missing"):
        execution_audit.validate_reports(
            rows, minimum_scenarios=1, maximum_scenarios=1
        )


def test_rollout_execution_audit_allows_one_filtered_pilot_scene():
    import audit_grpo_selector_rollout_execution as execution_audit

    rows = [{"succeeded": True} for _ in range(7)]
    execution_audit.validate_reports(
        rows, minimum_scenarios=7, maximum_scenarios=8
    )
    with __import__("pytest").raises(RuntimeError, match="minimum is 8"):
        execution_audit.validate_reports(
            rows, minimum_scenarios=8, maximum_scenarios=8
        )


def test_formal_collection_launcher_is_frozen_and_eight_gpu_only():
    launcher = (WORLDENGINE_ROOT / "run_diffusiondrive_rollout_v1_collect_8h100.sh").read_text()
    assert "formal-collection-ready-20260813" in launcher
    assert "git status --porcelain --untracked-files=no" in launcher
    assert 'exec "${LAUNCHER}" collect' in launcher
    assert "DIFFUSIONDRIVE_EXPECTED_GPU_COUNT" not in launcher
    assert "DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT" not in launcher
    assert "RUN_ID=r1full" in launcher
