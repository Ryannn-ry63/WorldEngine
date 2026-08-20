from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
from pathlib import Path

import pytest


DIFFUSIONDRIVE_SCRIPTS = Path(__file__).resolve().parents[1] / "diffusiondrive"
SIMENGINE_SCRIPTS = Path(__file__).resolve().parents[3] / "SimEngine" / "scripts"
sys.path.insert(0, str(DIFFUSIONDRIVE_SCRIPTS))
sys.path.insert(0, str(SIMENGINE_SCRIPTS))
tuning = importlib.import_module("grpo_selector_v3_rare_clpdms_tuning")
merge_results = importlib.import_module("merge_simulation_results")


def make_scene(scene_id: str, log_name: str) -> dict:
    return {
        "id": scene_id,
        "name": scene_id,
        "token": scene_id,
        "metadata": {
            "openscene_data_infos_dict": {
                "frame": {"log_name": log_name},
            }
        },
    }


def write_synthetic_source(path: Path, num_logs: int = 10) -> None:
    scenes = {}
    for index in range(num_logs):
        log_name = f"log_{index:02d}"
        scene_id = f"{log_name}-{index:016x}"
        scenes[scene_id] = make_scene(scene_id, log_name)
    with path.open("wb") as stream:
        pickle.dump(scenes, stream)


def split_args(source: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source=source,
        output_root=output,
        split_seed=20260820,
        development_fraction=0.2,
        code_commit="test-commit",
        expected_source_scenes=10,
        expected_source_logs=10,
        expected_development_scenes=2,
        expected_development_logs=2,
        expected_confirmation_scenes=8,
        expected_confirmation_logs=8,
    )


def test_log_disjoint_split_is_deterministic_complete_and_reusable(tmp_path):
    source = tmp_path / "all_scenarios.pkl"
    output = tmp_path / "split"
    write_synthetic_source(source)

    first = tuning.build_split(split_args(source, output))
    second = tuning.build_split(split_args(source, output))
    assert first == second
    assert first["status"] == "PASS"
    assert first["log_disjoint"] is True
    assert first["token_disjoint"] is True
    assert first["splits"]["development"]["num_scenarios"] == 2
    assert first["splits"]["development"]["num_logs"] == 2
    assert first["splits"]["confirmation"]["num_scenarios"] == 8
    assert first["splits"]["confirmation"]["num_logs"] == 8
    assert first["splits"]["smoke"]["num_scenarios"] == 1

    development = set(first["splits"]["development"]["tokens"])
    confirmation = set(first["splits"]["confirmation"]["tokens"])
    assert not development.intersection(confirmation)
    assert len(development | confirmation) == 10


def test_reusable_split_rejects_code_commit_drift(tmp_path):
    source = tmp_path / "all_scenarios.pkl"
    output = tmp_path / "split"
    write_synthetic_source(source)
    tuning.build_split(split_args(source, output))

    drifted = split_args(source, output)
    drifted.code_commit = "different-commit"
    with pytest.raises(RuntimeError, match="code commit drifted"):
        tuning.build_split(drifted)


def test_scene_log_metadata_mismatch_fails_closed():
    scene_id = "log_a-0000000000000001"
    scene = make_scene(scene_id, "log_b")
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        tuning.scene_log_name(scene_id, scene)


def write_metric_csv(path: Path, tokens: list[str]) -> None:
    fields = ["token", *tuning.PDM_KEYS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, token in enumerate(tokens):
            score = 0.7 + 0.1 * index
            writer.writerow(
                {
                    "token": token,
                    "no_at_fault_collisions": 1.0,
                    "drivable_area_compliance": 1.0,
                    "ego_progress": score,
                    "time_to_collision_within_bound": 1.0,
                    "comfort": 1.0,
                    "score": score,
                }
            )
        writer.writerow(
            {
                "token": "overall_average",
                **{key: 0.75 for key in tuning.PDM_KEYS},
            }
        )


def test_split_metrics_exclude_footer_and_require_exact_tokens(tmp_path):
    source = tmp_path / "all_scenarios.pkl"
    output = tmp_path / "split"
    write_synthetic_source(source)
    audit = tuning.build_split(split_args(source, output))
    tokens = audit["splits"]["development"]["tokens"]
    metric_csv = tmp_path / "metrics.csv"
    write_metric_csv(metric_csv, tokens)
    report_path = tmp_path / "metrics.json"
    args = argparse.Namespace(
        split_audit=output / "split_audit.json",
        split="development",
        csv=metric_csv,
        model="candidate",
        seed=0,
        checkpoint_manifest=None,
        allow_superset=False,
        output=report_path,
    )
    report = tuning.build_metrics(args)
    assert report["metrics"]["num_scenarios"] == 2
    assert report["source_csv_rows"] == 2
    assert report["metrics"]["score"] == pytest.approx(0.75)
    assert report["metrics"]["success_rate"] == 1.0


def aggregate(name: str, scores: list[float]) -> dict:
    return {
        "name": name,
        "by_seed": {
            str(seed): {"score": score} for seed, score in enumerate(scores)
        },
        "mean": {
            "score": sum(scores) / len(scores),
            "success_rate": 1.0,
        },
    }


def test_three_seed_promotion_gate_keeps_incumbent_for_small_gain():
    aggregates = {
        "rare_tuned": aggregate("rare_tuned", [0.75, 0.75, 0.75]),
        "small": aggregate("small", [0.754, 0.754, 0.754]),
        "robust": aggregate("robust", [0.76, 0.755, 0.751]),
    }
    selected, diagnostics = tuning.choose_final(
        aggregates,
        ["small", "robust"],
        "rare_tuned",
        minimum_improvement=0.005,
        minimum_nonnegative_seeds=2,
    )
    assert selected == "robust"
    by_name = {row["name"]: row for row in diagnostics}
    assert by_name["small"]["promotion_gate_passed"] is False
    assert by_name["robust"]["promotion_gate_passed"] is True


def test_merge_results_accepts_one_split(tmp_path, monkeypatch):
    test_path = tmp_path / "closed_loop"
    split = test_path / "split_0"
    (split / "plan_traj").mkdir(parents=True)
    (split / "WE_output" / "openscene_format").mkdir(parents=True)
    (split / "plan_traj" / "plan_idx.csv").write_text("token,index\na,0\n")
    write_metric_csv(
        split
        / "WE_output"
        / "openscene_format"
        / "all_scenes_pdm_averages_R.csv",
        ["log_a-0000000000000001"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_simulation_results.py",
            "--test_path",
            str(test_path),
            "--react_type",
            "R",
            "--num-splits",
            "1",
        ],
    )
    merge_results.main()
    merged = (
        test_path
        / "WE_output"
        / "openscene_format"
        / "all_scenes_pdm_averages_R.csv"
    )
    rows = list(csv.DictReader(merged.open()))
    assert rows[0]["token"] == "log_a-0000000000000001"
    assert rows[-1]["token"] == "overall_average"
