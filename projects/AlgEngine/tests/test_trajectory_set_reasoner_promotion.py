import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/audit_trajectory_set_reasoner_promotion.py"
)
SPEC = importlib.util.spec_from_file_location("reasoner_promotion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def thresholds():
    return SimpleNamespace(
        minimum_closed_loop_gain=0.01,
        minimum_positive_seeds=2,
        maximum_success_drop=0.035,
        maximum_component_drop=0.02,
        maximum_navtest_drop=0.01,
        maximum_seed_closed_loop_drop=0.03,
    )


def metrics(cl_delta=0.0, success_delta=0.0, component_delta=0.0, nav_delta=0.0):
    return {
        str(seed): {
            "cl_nonreactive_pdm": 0.75 + cl_delta,
            "cl_reactive_pdm": 0.74 + cl_delta,
            "success_rate": 0.90 + success_delta,
            "no_at_fault_collisions": 0.95 + component_delta,
            "drivable_area_compliance": 0.96 + component_delta,
            "navtest_pdm": 0.87 + nav_delta,
        }
        for seed in (0, 1, 2)
    }


def test_closed_loop_gain_within_guardrails_is_eligible():
    row = MODULE.audit_candidate(
        "full",
        Path("full.json"),
        metrics(0.02, -0.03, -0.01, -0.005),
        metrics(),
        thresholds(),
    )
    assert row["eligible"]
    assert row["positive_closed_loop_seeds"] == 3


def test_high_closed_loop_pareto_result_can_fail_success_guardrail():
    row = MODULE.audit_candidate(
        "pareto",
        Path("pareto.json"),
        metrics(0.04, -0.04),
        metrics(),
        thresholds(),
    )
    assert row["mean_closed_loop_pdm_delta"] > 0.01
    assert not row["gates"]["success_guardrail"]
    assert not row["eligible"]


def test_one_catastrophic_seed_blocks_promotion():
    baseline = metrics()
    candidate = metrics(0.02)
    candidate["2"]["cl_nonreactive_pdm"] -= 0.06
    candidate["2"]["cl_reactive_pdm"] -= 0.06
    row = MODULE.audit_candidate(
        "unstable", Path("unstable.json"), candidate, baseline, thresholds()
    )
    assert not row["gates"]["no_catastrophic_seed"]
    assert not row["eligible"]
