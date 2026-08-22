from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
promotion = importlib.import_module("audit_grpo_selector_v3_candidate_promotion")


def summary(path: Path, seed: int, **updates) -> Path:
    values = {
        "openloop_navtest_pdm": 0.85,
        "reactive_pdm": 0.75,
        "reactive_ep": 0.50,
        "reactive_nc": 0.97,
        "reactive_dac": 0.98,
        "reactive_success_rate": 0.95,
    }
    values.update(updates)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "eval_seed": seed,
        "model_name": path.stem,
        "metrics": {
            "openloop_navtest": {"score": values["openloop_navtest_pdm"]},
            "closedloop_reactive": {
                "score": values["reactive_pdm"],
                "ego_progress": values["reactive_ep"],
                "no_at_fault_collisions": values["reactive_nc"],
                "drivable_area_compliance": values["reactive_dac"],
            },
            "success_rate": values["reactive_success_rate"],
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_bwm_gate_promotes_only_with_reactive_gain_and_safety(tmp_path):
    baseline = summary(tmp_path / "baseline.json", 0)
    good = summary(
        tmp_path / "good.json",
        0,
        reactive_pdm=0.751,
        reactive_success_rate=0.95,
        openloop_navtest_pdm=0.84,
    )
    report = promotion.audit("bwm_generalization", [(0, baseline)], [(0, good)])
    assert report["decision"] == "PROMOTE"

    bad = summary(
        tmp_path / "bad.json",
        0,
        reactive_pdm=0.76,
        reactive_success_rate=0.949,
    )
    report = promotion.audit("bwm_generalization", [(0, baseline)], [(0, bad)])
    assert report["decision"] == "HOLD_INCUMBENT"


def test_gate_conditioned_requires_ep_gain_and_noninferiority(tmp_path):
    baselines = []
    candidates = []
    for seed in range(3):
        baselines.append((seed, summary(tmp_path / f"baseline_{seed}.json", seed)))
        candidates.append(
            (
                seed,
                summary(
                    tmp_path / f"candidate_{seed}.json",
                    seed,
                    reactive_ep=0.511,
                    reactive_pdm=0.746,
                    reactive_success_rate=0.946,
                    reactive_nc=0.966,
                    reactive_dac=0.976,
                ),
            )
        )
    report = promotion.audit("gate_conditioned", baselines, candidates)
    assert report["decision"] == "PROMOTE"
    assert report["seed_count"] == 3


def test_candidate_seed_mismatch_is_rejected(tmp_path):
    baseline = summary(tmp_path / "baseline.json", 0)
    candidate = summary(tmp_path / "candidate.json", 1)
    try:
        promotion.audit("bwm_generalization", [(0, baseline)], [(1, candidate)])
    except RuntimeError as error:
        assert "seed sets" in str(error)
    else:
        raise AssertionError("mismatched seed sets were accepted")
