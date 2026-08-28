import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/collect_trajectory_set_reasoner_formal_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("reasoner_collector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(seed, checkpoint_sha):
    return {
        "schema_version": 1,
        "status": "PASS",
        "eval_seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "metrics": {
            "closedloop_nonreactive": {"score": 0.70 + seed * 0.01},
            "closedloop_reactive": {
                "score": 0.69 + seed * 0.01,
                "no_at_fault_collisions": 0.90,
                "drivable_area_compliance": 0.91,
            },
            "success_rate": 0.88,
            "openloop_navtest": {"score": 0.85},
        },
    }


def run_collector(tmp_path, monkeypatch, checkpoint_shas):
    summaries = []
    for seed, checkpoint_sha in enumerate(checkpoint_shas):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(summary(seed, checkpoint_sha)))
        summaries.append(path)
    output = tmp_path / "metrics.json"
    argv = [str(MODULE_PATH)]
    for path in summaries:
        argv.extend(("--summary", str(path)))
    argv.extend(("--output", str(output)))
    monkeypatch.setattr(sys, "argv", argv)
    MODULE.main()
    return json.loads(output.read_text())


def test_collector_accepts_three_independently_trained_checkpoints(tmp_path, monkeypatch):
    payload = run_collector(tmp_path, monkeypatch, ("sha0", "sha1", "sha2"))
    assert payload["independent_training_seeds"] == [0, 1, 2]
    assert payload["checkpoint_sha256_by_seed"] == {
        "0": "sha0", "1": "sha1", "2": "sha2"
    }
    assert set(payload["seeds"]) == {"0", "1", "2"}


def test_collector_rejects_reused_checkpoint(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="independently trained"):
        run_collector(tmp_path, monkeypatch, ("same", "same", "same"))
