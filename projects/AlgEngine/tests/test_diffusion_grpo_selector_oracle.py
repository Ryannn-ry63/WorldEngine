import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))

import finalize_grpo_selector_oracle as finalizer
from grpo_selector_oracle_common import (
    COMPONENT_NAMES,
    candidate_diversity,
    classify_oracle_gain,
    load_records,
    paired_bootstrap_ci,
    summarize_records,
    write_records,
)


def components(score):
    return {name: float(score) for name in COMPONENT_NAMES}


def make_record(token, reference, current, oracle):
    return {
        "token": token,
        "valid_candidate_count": 20,
        "reference_reward": reference,
        "current_reward": current,
        "oracle_reward": oracle,
        "reference_oracle_match": reference == oracle,
        "current_oracle_match": current == oracle,
        "oracle_oracle_match": True,
        "reference_components": components(reference),
        "current_components": components(current),
        "oracle_components": components(oracle),
        "selection_disagreement": current != reference,
        "reward_spread": oracle - reference,
        "reward_std": 0.01,
        "mean_pairwise_ade": 0.5,
        "mean_pairwise_fde": 1.0,
        "max_pairwise_fde": 2.0,
        "unique_candidate_count_1e4": 20,
        "current_expected_reward": current - 0.001,
        "reference_expected_reward": reference - 0.001,
    }


def write_official(path, records, selection):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["token", "valid", *COMPONENT_NAMES, "score"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            score = row[f"{selection}_reward"]
            writer.writerow(
                {
                    "token": row["token"],
                    "valid": "True",
                    **components(score),
                    "score": score,
                }
            )


def test_candidate_helpers_and_record_io(tmp_path):
    candidates = np.zeros((20, 8, 3), dtype=np.float32)
    candidates[:, :, 0] = np.arange(20, dtype=np.float32)[:, None]
    diversity = candidate_diversity(candidates)
    assert diversity["unique_candidate_count_1e4"] == 20
    assert diversity["mean_pairwise_ade"] > 0

    records = [
        make_record("a", 0.80, 0.81, 0.84),
        make_record("b", 0.90, 0.90, 0.92),
    ]
    path = tmp_path / "records.jsonl.gz"
    write_records(path, records)
    assert load_records(path) == records
    summary = summarize_records(records)
    assert summary["num_candidate_scores"] == 40
    assert np.isclose(summary["oracle_reference_gain"], 0.03)
    assert np.isclose(summary["current_reference_gain"], 0.005)
    assert np.isclose(summary["oracle_capture_rate"], 1.0 / 6.0)


def test_bootstrap_and_classification_are_deterministic():
    gains = [0.01, 0.02, 0.03, 0.04]
    first = paired_bootstrap_ci(gains, num_resamples=200, seed=7)
    assert first == paired_bootstrap_ci(gains, num_resamples=200, seed=7)
    assert classify_oracle_gain(0.006, 0.010) == "selector_limited"
    assert classify_oracle_gain(-0.001, 0.002) == "generator_limited"
    assert classify_oracle_gain(0.001, 0.006) == "mixed_or_inconclusive"


def test_three_seed_finalizer_end_to_end(tmp_path, monkeypatch):
    tokens = ["a", "b", "c"]
    failures = ["a", "c"]
    failure_filter = tmp_path / "failures.yaml"
    failure_filter.write_text(yaml.safe_dump({"tokens": failures}))
    seed_dirs = []
    reference_summaries = []
    current_summaries = []

    for seed in range(3):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir()
        seed_dirs.append(seed_dir)
        records = [
            make_record(
                token,
                0.80 + 0.01 * index,
                0.805 + 0.01 * index,
                0.82 + 0.01 * index,
            )
            for index, token in enumerate(tokens)
        ]
        records_path = seed_dir / "records.jsonl.gz"
        write_records(records_path, records)
        (seed_dir / "audit.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "noise_seed": seed,
                    "train_seed": seed,
                    "records": str(records_path),
                    "paths": {"checkpoint": f"/checkpoint/seed{seed}.pth"},
                    "checkpoint_sha256": f"sha{seed}",
                }
            )
        )
        for selection in ("reference", "current", "oracle"):
            write_official(
                seed_dir
                / f"{selection}_official_pdms"
                / "pdm_scores_merged.csv",
                records,
                selection,
            )
        for selection, destination in (
            ("reference", reference_summaries),
            ("current", current_summaries),
        ):
            score = float(
                np.mean([row[f"{selection}_reward"] for row in records])
            )
            path = tmp_path / f"{selection}_seed{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "eval_seed": seed,
                        "metrics": {"openloop_navtest": {"score": score}},
                    }
                )
            )
            destination.append(path)

    output = tmp_path / "final" / "oracle_audit.json"
    argv = ["finalize_grpo_selector_oracle.py"]
    for seed_dir in seed_dirs:
        argv.extend(["--seed-dir", str(seed_dir)])
    argv.extend(["--failures-filter", str(failure_filter)])
    for path in reference_summaries:
        argv.extend(["--formal-reference-summary", str(path)])
    for path in current_summaries:
        argv.extend(["--formal-current-summary", str(path)])
    argv.extend(
        [
            "--output",
            str(output),
            "--expected-num-tokens",
            "3",
            "--expected-num-failures",
            "2",
            "--bootstrap-resamples",
            "100",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    finalizer.main()

    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["seeds"] == [0, 1, 2]
    assert report["num_candidate_scores"] == 180
    assert report["diagnosis"]["classification"] == "selector_limited"
    assert np.isclose(
        report["aggregate"]["navtest"]["gains"]["oracle_reference"], 0.02
    )
    assert output.with_suffix(".tsv").is_file()
    assert "Oracle 使用日志未来信息" in output.with_suffix(".md").read_text()


def test_audit_uses_formal_test_path():
    source = (SCRIPT_DIR / "audit_grpo_selector_oracle.py").read_text()
    assert "copy.deepcopy(cfg.data.test)" in source
    assert "model(return_loss=False, rescale=True, **data)" in source
    assert "formal_navtest_seed" in source
    assert "candidate_trajectories_8" in source
    assert "original_forward_test = head._forward_test" in source
    assert "cfg.data.train" not in source
