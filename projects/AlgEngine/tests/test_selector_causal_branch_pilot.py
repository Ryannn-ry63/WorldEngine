from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve()
ALGENGINE = HERE.parents[1]
SCRIPTS = ALGENGINE / "scripts/diffusiondrive"
SIMENGINE = ALGENGINE.parent / "SimEngine"
sys.path.insert(0, str(SCRIPTS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load(
    "causal_branch_pilot_common_test",
    SCRIPTS / "causal_branch_pilot_common.py",
)
builder = load(
    "build_selector_causal_branch_pilot_targets_test",
    SCRIPTS / "build_selector_causal_branch_pilot_targets.py",
)
analyzer = load(
    "analyze_selector_causal_branch_pilot_test",
    SCRIPTS / "analyze_selector_causal_branch_pilot.py",
)


def candidate_set():
    trajectories = np.zeros((20, 8, 3), dtype=np.float64)
    for index in range(20):
        trajectories[index, :, 0] = index / 10.0
    return trajectories


def reward_set(policy=0, oracle=10):
    rewards = np.full(20, -0.2, dtype=np.float64)
    rewards[policy] = 0.50
    rewards[oracle] = 0.80
    rewards[9] = 0.504
    rewards[2] = 0.503
    return rewards


def frame(scene, decision, rewards=None):
    logits = np.linspace(-1.0, -0.1, 20)
    logits[0] = 1.0
    return {
        "rollout_scene_id": scene,
        "decision_step": decision,
        "state_step": decision - 1,
        "policy_selected_index": 0,
        "candidate_trajectories_8": candidate_set(),
        "candidate_rewards": reward_set() if rewards is None else rewards,
        "current_logits": logits,
    }


def scenario(scene, log="log-a", source="bwm_offroad", pairing="direct_diffusiondrive_rare_pair"):
    return {
        "id": scene,
        "token": scene + "-token",
        "metadata": {
            "rollout_log_name": log,
            "rollout_origin_token": log + "-token",
            "rollout_source_kind": source,
            "common_pairing_method": pairing,
        },
    }


def test_matched_control_is_nonimproving_and_magnitude_matched():
    trajectories = candidate_set()
    rewards_a = reward_set()
    rewards_b = reward_set()
    index, distance, oracle_distance, absolute_error, relative_error = (
        common.choose_matched_index(
            trajectories, rewards_a, rewards_b, policy_index=0, oracle_index=10
        )
    )
    assert index == 9
    assert rewards_a[index] <= rewards_a[0] + 0.005
    assert np.isclose(absolute_error, abs(distance - oracle_distance))
    assert absolute_error <= 0.5
    assert relative_error <= 0.5


def test_matched_control_requires_both_baselines_to_be_nonimproving():
    rewards_a = reward_set()
    rewards_b = reward_set()
    rewards_b[9] = 0.7
    index, _, _, _, _ = common.choose_matched_index(
        candidate_set(), rewards_a, rewards_b, policy_index=0, oracle_index=10
    )
    assert index != 9
    assert rewards_a[index] <= rewards_a[0] + 0.005
    assert rewards_b[index] <= rewards_b[0] + 0.005


def test_failed_target_is_strictly_before_first_violation(tmp_path):
    scene_id = "scene-a"
    frames_a, frames_b = {}, {}
    for decision in range(4, 12):
        path_a, path_b = tmp_path / f"a{decision}", tmp_path / f"b{decision}"
        path_a.write_bytes(b"a")
        path_b.write_bytes(b"b")
        row_a = frame(scene_id, decision)
        row_b = frame(scene_id, decision)
        if decision < 6:
            row_a["candidate_rewards"] = np.full(20, 0.5)
            row_b["candidate_rewards"] = np.full(20, 0.5)
        frames_a[decision] = (path_a, row_a)
        frames_b[decision] = (path_b, row_b)
    outcome = {
        "success": False,
        "score": 0.2,
        "first_violation_step": 6,
    }
    target, reason = builder.target_for_scene(
        scene_id,
        scenario(scene_id),
        frames_a,
        frames_b,
        outcome,
        outcome,
        0.02,
        0.005,
    )
    assert target is None
    assert reason == "no_eligible_preaction_frame"



def test_target_freezer_skips_early_frame_outside_caliper(tmp_path):
    scene_id = "scene-caliper"
    frames_a, frames_b = {}, {}
    for decision in range(4, 12):
        path_a, path_b = tmp_path / f"a{decision}", tmp_path / f"b{decision}"
        path_a.write_bytes(b"a")
        path_b.write_bytes(b"b")
        row_a = frame(scene_id, decision)
        row_b = frame(scene_id, decision)
        if decision == 4:
            poor = np.zeros((20, 8, 3), dtype=np.float64)
            poor[10, :, 0] = 1.0
            row_a["candidate_trajectories_8"] = poor
            row_b["candidate_trajectories_8"] = poor.copy()
        frames_a[decision] = (path_a, row_a)
        frames_b[decision] = (path_b, row_b)
    outcome = {"success": True, "score": 1.0, "first_violation_step": None}
    rejections = {}
    from collections import Counter
    rejection_counts = Counter(rejections)
    target, reason = builder.target_for_scene(
        scene_id,
        scenario(scene_id),
        frames_a,
        frames_b,
        outcome,
        outcome,
        frame_rejections=rejection_counts,
    )
    assert reason == "eligible"
    assert target["decision_step"] == 5
    assert rejection_counts["matched_control_outside_magnitude_caliper"] == 1
    assert target["ade_match_error"] <= 0.5
    assert target["relative_ade_match_error"] <= 0.5

def test_target_freezer_rejects_unstable_closed_loop_label(tmp_path):
    target, reason = builder.target_for_scene(
        "scene-a",
        scenario("scene-a"),
        {},
        {},
        {"success": False, "score": 0.0, "first_violation_step": 6},
        {"success": True, "score": 1.0, "first_violation_step": None},
        0.02,
        0.005,
    )
    assert target is None
    assert reason == "unstable_success"


def test_stratified_selection_caps_origin_log_exposure():
    rows = []
    for index in range(12):
        rows.append({
            "scene_id": f"scene-{index}",
            "origin_log": f"log-{index // 3}",
            "source_kind": "rare" if index % 2 else "common",
            "pairing_method": "paired",
        })
    selected = common.stratified_scene_cap(rows, limit=8, max_per_origin_log=2)
    counts = {}
    for row in selected:
        counts[row["origin_log"]] = counts.get(row["origin_log"], 0) + 1
    assert len(selected) == 8
    assert max(counts.values()) <= 2


def test_cluster_bootstrap_resamples_logs_not_scenes():
    rows = [
        {"origin_log": "a", "value": 1.0},
        {"origin_log": "a", "value": 1.0},
        {"origin_log": "b", "value": -1.0},
    ]
    report = common.cluster_bootstrap(rows, "value", repetitions=200, seed=3)
    assert report["num_logs"] == 2
    assert np.isclose(report["estimate"], 1.0 / 3.0)


def test_source_contract_names_the_causal_estimand_and_matched_arm():
    config = (
        ALGENGINE
        / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_causal_branch_pilot.py"
    ).read_text()
    manager = (
        SIMENGINE
        / "worldengine/manager/diffusiondrive_preaction_oracle_manager.py"
    ).read_text()
    protocol = (ALGENGINE.parents[1] / "DIFFUSIONDRIVE_SELECTOR_CAUSAL_BRANCH_PILOT_PROTOCOL_20260903.md").read_text()
    assert 'schema_version=7' in config
    assert 'primary_causal_estimand="one_step_local_oracle_minus_no_intervention_v3_baseline"' in config
    assert 'specificity_causal_estimand="one_step_local_oracle_minus_magnitude_matched_nonimproving_control"' in config
    assert 'matched_control_max_absolute_ade_error_m=0.5' in config
    assert 'matched_control_max_relative_ade_error=0.5' in config
    assert 'mode not in {"observe_only", "one_shot_oracle", "one_shot_matched"}' in config
    assert '"matched_index"' in manager
    assert 'self.intervention_mode.startswith("one_shot_")' in manager
    assert "FROZEN AFTER BASELINE A/B AND BEFORE ANY INTERVENTION OUTCOME" in protocol
    assert "treatment_magnitude_match_not_directional_geometry_match" not in protocol
    assert "not claimed to match trajectory direction or full geometry" in protocol




def test_matched_control_rejects_absolute_caliper_failure():
    trajectories = np.zeros((20, 8, 3), dtype=np.float64)
    trajectories[10, :, 0] = 1.0
    rewards = np.full(20, -0.2, dtype=np.float64)
    rewards[0] = 0.5
    rewards[10] = 0.8
    with pytest.raises(ValueError, match="matched_control_outside_magnitude_caliper"):
        common.choose_matched_index(
            trajectories, rewards, rewards, policy_index=0, oracle_index=10
        )


def test_matched_control_rejects_relative_caliper_failure():
    trajectories = np.zeros((20, 8, 3), dtype=np.float64)
    trajectories[1, :, 0] = 0.1
    trajectories[2:, :, 0] = 0.2
    rewards = np.full(20, -0.2, dtype=np.float64)
    rewards[0] = 0.5
    rewards[1] = 0.8
    with pytest.raises(ValueError, match="matched_control_outside_magnitude_caliper"):
        common.choose_matched_index(
            trajectories, rewards, rewards, policy_index=0, oracle_index=1
        )


def test_perturbation_diagnostics_wrap_yaw_and_report_xy_magnitude():
    policy = np.zeros((8, 3), dtype=np.float64)
    candidate = np.zeros((8, 3), dtype=np.float64)
    candidate[:, 0] = np.linspace(0.1, 0.8, 8)
    candidate[:, 2] = 2 * np.pi - 0.1
    report = common.trajectory_perturbation_diagnostics(candidate, policy)
    assert np.isclose(report["xy_ade_m"], 0.45)
    assert np.isclose(report["xy_final_m"], 0.8)
    assert np.isclose(report["yaw_abs_mean_rad"], 0.1)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"coverage_pass": False}, "INSUFFICIENT_CAUSAL_BRANCH_COVERAGE"),
        ({"reproducible": False}, "INVALID_CAUSAL_BRANCH_REPRODUCIBILITY"),
        ({"primary_rescue_rate": 0.1}, "STOP_LOCAL_HEADROOM_CAUSAL_ROUTE"),
        (
            {"specificity_rescue_rate": 0.1},
            "REDIRECT_TO_REWARD_HORIZON_OR_ACTION_SENSITIVITY",
        ),
        ({"specificity_lower_95": 0.0}, "EXPAND_CAUSAL_BRANCH_SEED1"),
        ({}, "AUTHORIZE_DIFFUSION_SET_SELECTOR_METHOD"),
    ],
)
def test_causal_v2_decision_contract(overrides, expected):
    inputs = {
        "coverage_pass": True,
        "reproducible": True,
        "primary_rescue_rate": 0.25,
        "oracle_rescue_origin_logs": 4,
        "primary_lower_95": 0.01,
        "specificity_rescue_rate": 0.25,
        "differential_rescue_origin_logs": 4,
        "specificity_lower_95": 0.01,
    }
    inputs.update(overrides)
    assert analyzer.decide_causal_route(**inputs)["decision"] == expected
