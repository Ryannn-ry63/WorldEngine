from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np


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


common = load("ccv_sweep_common_test", SCRIPTS / "ccv_sweep_common.py")
analyzer = load("analyze_selector_ccv_sweep_test", SCRIPTS / "analyze_selector_ccv_sweep.py")


def gate(passed=False, point=False):
    return {"pass": passed, "point_pass": point or passed}


def test_rank_statistics_are_tie_aware_and_monotonic():
    left = np.asarray([0.0, 0.0, 1.0, 2.0])
    assert np.isclose(common.spearman(left, left), 1.0)
    assert np.isclose(common.kendall_tau_b(left, left), 1.0)
    assert common.spearman(left, -left) < 0.0
    assert common.kendall_tau_b(left, -left) < 0.0


def test_regret_decomposition_identity():
    q_policy = 0.1
    q_reward_oracle = 0.4
    q_reward_set = 0.6
    q_star = 0.9
    left = q_star - q_policy
    right = (
        (q_star - q_reward_set)
        + (q_reward_set - q_reward_oracle)
        + (q_reward_oracle - q_policy)
    )
    assert np.isclose(left, right, atol=1e-8)


def test_gap_gate_requires_magnitude_logs_and_positive_bootstrap():
    strong = [
        {"origin_log": f"log-{index}", "gap": 0.2}
        for index in range(6)
    ]
    report = common.gap_gate(strong, "gap", seed=4)
    assert report["point_pass"]
    assert report["bootstrap_pass"]
    assert report["pass"]
    concentrated = [
        {"origin_log": "same-log", "gap": 0.2}
        for _ in range(9)
    ]
    assert not common.gap_gate(concentrated, "gap", seed=4)["point_pass"]


def test_geometry_knn_recovers_smooth_candidate_values():
    trajectories = np.zeros((20, 8, 3), dtype=np.float64)
    for index in range(20):
        trajectories[index, :, 0] = index / 10.0
    values = np.linspace(0.0, 1.0, 20)
    result = common.knn_geometry_prediction(trajectories, values)
    assert result["spearman"] > 0.9
    assert result["top3_recall_best"]


def test_decision_routes_are_mutually_ordered():
    geometry = {"pass": False}
    decision, _ = analyzer.decide({
        "reward_ranking_horizon_gap": gate(True),
        "reward_tie_gap": gate(True),
        "selector_gap": gate(True),
    }, geometry)
    assert decision == "AUTHORIZE_LONG_HORIZON_CAUSAL_ADVANTAGE"
    decision, _ = analyzer.decide({
        "reward_ranking_horizon_gap": gate(False),
        "reward_tie_gap": gate(True),
        "selector_gap": gate(True),
    }, geometry)
    assert decision == "AUTHORIZE_SCALAR_COARSE_DIFFUSION_SET_TIE_BREAKER"
    decision, _ = analyzer.decide({
        "reward_ranking_horizon_gap": gate(False),
        "reward_tie_gap": gate(False),
        "selector_gap": gate(True),
    }, geometry)
    assert decision == "RETAIN_SCALAR_REWARD_AND_IMPROVE_SELECTOR"
    decision, _ = analyzer.decide({
        "reward_ranking_horizon_gap": gate(False, point=True),
        "reward_tie_gap": gate(False),
        "selector_gap": gate(False),
    }, geometry)
    assert decision == "EXPAND_CCV_TARGETS_OR_FROZEN_NOISE"


def test_source_contract_is_isolated_and_never_authorizes_training():
    config = (
        ALGENGINE
        / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_ccv_sweep.py"
    ).read_text()
    manager = (
        SIMENGINE
        / "worldengine/manager/diffusiondrive_candidate_sweep_manager.py"
    ).read_text()
    base_env = (SIMENGINE / "worldengine/envs/base_env.py").read_text()
    protocol = (ALGENGINE.parents[1] / "DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_PROTOCOL.md").read_text()
    amendment = (ALGENGINE.parents[1] / "DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_SENTINEL_AMENDMENT.md").read_text()
    analysis_note = (ALGENGINE.parents[1] / "DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_ANALYSIS_NOTE.md").read_text()
    analyzer_source = (SCRIPTS / "analyze_selector_ccv_sweep.py").read_text()
    verifier_source = (SCRIPTS / "verify_selector_ccv_sentinel.py").read_text()
    runner = (SCRIPTS / "run_selector_ccv_sweep_8hopper.sh").read_text()
    merger = (SIMENGINE / "scripts/merge_simulation_results.py").read_text()
    assert 'intervention_mode="one_shot_manifest"' in config
    assert 'causal_value_authorized_for_training=False' in config
    assert "diffusiondrive_candidate_sweep" in base_env
    assert "self.engine.external_actions = deployed_action" in manager
    assert 'target["treatment_index"]' in manager
    assert "33 train-only targets" in protocol
    assert "not consumed as training data" in protocol
    assert "STALE_CCV_PARTIAL_PROVENANCE" in runner
    assert "FROZEN AFTER SENTINEL AND BEFORE FORMAL ARMS" in amendment
    assert "FROZEN AFTER FORMAL COLLECTION AND BEFORE CAUSAL GAP ANALYSIS" in analysis_note
    assert "reward_recomputation_role" in verifier_source
    assert "CCV policy branch did not reproduce baseline" in analyzer_source
    assert "FORMAL_POLICY_BASELINE_CONTINUOUS_EQUIVALENCE = 1e-3" in analyzer_source
    assert '"all_nc_equal"' in analyzer_source
    assert "--analysis-note" in runner
    assert '(f"{openscene_base}/diffusiondrive_ccv_records", "*_ccv.pkl", False)' in merger


def test_analysis_requires_arm_and_sentinel_provenance_match():
    source = inspect.getsource(analyzer.load_arm)
    assert "rollout_implementation_sha256" in source
    assert "checkpoint_manifest_sha256" in source
    assert "code_sha" not in source
