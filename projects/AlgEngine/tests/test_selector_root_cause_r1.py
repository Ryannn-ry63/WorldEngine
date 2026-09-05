from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np


ALGENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ALGENGINE_ROOT / "scripts/diffusiondrive"
CONFIG = (
    ALGENGINE_ROOT
    / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_on_policy_diagnostic.py"
)
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_selector_root_cause_r1 as analyzer  # noqa: E402
import audit_selector_root_cause_r1_collection as auditor  # noqa: E402
import selector_root_cause_metrics as metrics  # noqa: E402


def interval(lower, point, upper):
    return {"lower95": lower, "point": point, "upper95": upper}


def failure_stratum(headroom, recoverable, sensitivity=None):
    if sensitivity is None:
        sensitivity = interval(0.1, 0.2, 0.3)
    return {
        "scenario_count": 5,
        "scenario_bootstrap_95": {
            "headroom": headroom,
            "recoverable_0p005": recoverable,
            "recoverable_0p02": sensitivity,
        },
    }


def test_r1_gate_pass_borderline_and_clear_failure():
    passed = analyzer.decide_headroom(
        failure_stratum(
            interval(0.011, 0.02, 0.03), interval(0.26, 0.4, 0.5)
        )
    )
    assert passed["decision"] == "AUTHORIZE_R1_SCALAR_SEED1_SEED2"

    borderline = analyzer.decide_headroom(
        failure_stratum(
            interval(0.008, 0.02, 0.03), interval(0.26, 0.4, 0.5)
        )
    )
    assert borderline["decision"] == "AUTHORIZE_R1_SCALAR_SEED1_BORDERLINE"

    failed = analyzer.decide_headroom(
        failure_stratum(
            interval(0.001, 0.004, 0.009), interval(0.05, 0.1, 0.2)
        )
    )
    assert failed["decision"] == "STOP_SELECTOR_ONLY_NO_ON_POLICY_HEADROOM"


def test_frame_metrics_measure_fixed20_probability_support_and_topk():
    logits = np.zeros((2, 20), dtype=np.float64)
    rewards = np.zeros((2, 20), dtype=np.float64)
    components = np.zeros((2, 20, 6), dtype=np.float64)
    logits[0, 0] = 10.0
    rewards[0, 19] = 1.0
    logits[1] = np.arange(20, dtype=np.float64)
    rewards[1, 19] = 1.0
    values = metrics.frame_metric_arrays(logits, rewards, components)
    assert values["headroom"].tolist() == [1.0, 0.0]
    assert values["oracle_rank"].tolist() == [20.0, 1.0]
    assert values["oracle_in_top5"].tolist() == [0.0, 1.0]
    assert values["oracle_probability_below_1e-4"].tolist() == [1.0, 0.0]


def test_outcome_loader_ignores_simulator_aggregate_footer(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "token,no_at_fault_collisions,drivable_area_compliance,ego_progress,score\n"
        "scene-a,1,1,0.5,0.8\n"
        "overall_average,0.9,0.8,0.4,0.7\n"
    )
    outcomes = metrics.load_outcomes(csv_path)
    assert set(outcomes) == {"scene-a"}
    assert outcomes["scene-a"]["success"] is True


def test_scenario_bootstrap_aggregates_frames_before_resampling():
    scenes = ["log-a-scene-1", "log-a-scene-1", "log-b-scene-2"]
    values = {
        key: np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
        for key in metrics.CI_METRICS
    }
    result = metrics.summarize_stratum(
        values,
        scenes,
        np.ones(3, dtype=np.bool_),
        bootstrap_repetitions=100,
        bootstrap_seed=7,
    )
    assert result["frame_count"] == 3
    assert result["scenario_count"] == 2
    assert result["scenario_means"]["headroom"] == 0.75
    assert result["frame_means"]["headroom"] == 2.0 / 3.0


def test_config_contract_matches_collection_auditor(monkeypatch):
    environment = {
        "DIFFUSIONDRIVE_BEHAVIOR_POLICY_FAMILY": "scalar_v3",
        "DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED": "0",
        "DIFFUSIONDRIVE_DIAGNOSTIC_SPLIT": "cl_dev58",
        "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256": "checkpoint-sha",
        "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256": "manifest-sha",
        "DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256": "implementation-sha",
        "DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE": "r1-namespace",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    loaded = runpy.run_path(str(CONFIG))
    expected = auditor.expected_contract(
        family="scalar_v3",
        seed=0,
        checkpoint_sha="checkpoint-sha",
        manifest_sha="manifest-sha",
        split="cl_dev58",
        namespace="r1-namespace",
        implementation_sha="implementation-sha",
    )
    assert loaded["selector_rollout_contract"] == expected
    assert loaded["model"]["planning_head"]["export_rollout_context"] is True
    assert loaded["model"]["planning_head"]["online_reward"] is None


def test_config_rejects_unknown_policy(monkeypatch):
    environment = {
        "DIFFUSIONDRIVE_BEHAVIOR_POLICY_FAMILY": "unknown",
        "DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED": "0",
        "DIFFUSIONDRIVE_DIAGNOSTIC_SPLIT": "cl_dev58",
        "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256": "checkpoint-sha",
        "DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256": "manifest-sha",
        "DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256": "implementation-sha",
        "DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE": "r1-namespace",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    try:
        runpy.run_path(str(CONFIG))
    except RuntimeError as error:
        assert "unsupported R1 behavior policy" in str(error)
    else:
        raise AssertionError("unknown behavior policy was accepted")

