from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
prepare = importlib.import_module("prepare_grpo_selector_v3_rare_original_data")
smoke_subset = importlib.import_module(
    "prepare_grpo_selector_v3_rare_original_smoke_subset"
)
trainer = importlib.import_module("train_grpo_selector_v3_cached_rare_original")
splitter = importlib.import_module("split_grpo_selector_v3_rare_original")


def write_yaml(path, payload):
    path.write_text(prepare.yaml.safe_dump(payload, sort_keys=False))


def write_score(path, rows):
    fieldnames = ["token", "valid", *prepare.PDM_COMPONENTS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for token, values in rows.items():
            writer.writerow({"token": token, "valid": "True", **values})


def component_row(naf=1.0, dac=1.0, progress=8.0):
    return {
        "no_at_fault_collisions": naf,
        "drivable_area_compliance": dac,
        "ego_progress": progress,
        "time_to_collision_within_bound": 1.0,
        "comfort": 1.0,
        "driving_direction_compliance": 1.0,
        "score": naf * dac * progress / 10.0,
    }


@pytest.fixture
def small_full_navtrain(tmp_path):
    physical = tmp_path / "physical"
    token_logs = {
        "a0": "log_a",
        "a1": "log_a",
        "a2": "log_a",
        "b0": "log_b",
        "b1": "log_b",
        "b2": "log_b",
        "t0": "log_test",
    }
    for token, log_name in token_logs.items():
        path = physical / log_name / "unknown" / token / "metric_cache.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"cache")

    navtrain = tmp_path / "navtrain.yaml"
    navtest = tmp_path / "navtest.yaml"
    write_yaml(
        navtrain,
        {
            "_target_": "navsim.common.dataclasses.SceneFilter",
            "num_history_frames": 4,
            "num_future_frames": 10,
            "log_names": ["log_a", "log_b"],
        },
    )
    write_yaml(navtest, {"log_names": ["log_test"]})

    annotation = tmp_path / "annotation.pkl"
    infos = [
        {
            "token": token,
            "log_name": log_name,
            "scene_token": f"scene_{log_name}",
        }
        for token, log_name in token_logs.items()
        if token != "t0"
    ]
    with annotation.open("wb") as stream:
        pickle.dump({"infos": infos}, stream)

    index = tmp_path / "index"
    audit = prepare.build_full_index(
        physical,
        navtrain,
        navtest,
        annotation,
        index,
        expected_tokens=6,
        expected_logs=2,
        expected_physical_tokens=7,
    )
    return {
        "physical": physical,
        "navtrain": navtrain,
        "navtest": navtest,
        "annotation": annotation,
        "index": index,
        "index_audit": audit,
        "token_logs": token_logs,
    }


def test_full_index_is_complete_disjoint_and_immutable(small_full_navtrain):
    fixture = small_full_navtrain
    indexed = prepare.load_cache_index(fixture["index"])
    assert set(indexed) == {"a0", "a1", "a2", "b0", "b1", "b2"}
    assert fixture["index_audit"]["navtest_overlap"] == 0
    assert fixture["index_audit"]["navtrain_logs"] == 2

    repeated = prepare.build_full_index(
        fixture["physical"],
        fixture["navtrain"],
        fixture["navtest"],
        fixture["annotation"],
        fixture["index"],
        expected_tokens=6,
        expected_logs=2,
        expected_physical_tokens=7,
    )
    assert repeated == fixture["index_audit"]


def test_smoke_subset_uses_one_deterministic_dense_log(
    small_full_navtrain, tmp_path
):
    output = tmp_path / "subset"
    audit = smoke_subset.build_subset(
        small_full_navtrain["index"],
        small_full_navtrain["annotation"],
        small_full_navtrain["navtrain"],
        output,
        num_tokens=2,
        expected_source_tokens=6,
        seed=99,
    )
    tokens = set(prepare.load_cache_index(output))
    assert audit["status"] == "PASS"
    assert audit["selected_log"] == "log_b"
    assert tokens.issubset({"b0", "b1", "b2"})
    assert len(tokens) == 2


def test_rare_union_and_same_log_common_pairing(small_full_navtrain, tmp_path):
    rows = {
        "a0": component_row(naf=0.0, progress=0.0),
        "a1": component_row(progress=10.0),
        "a2": component_row(progress=1.0),
        "b0": component_row(dac=0.0, progress=0.0),
        "b1": component_row(progress=9.0),
        "b2": component_row(progress=2.0),
    }
    score_paths = []
    for seed in range(3):
        path = tmp_path / f"score_{seed}.csv"
        write_score(path, rows)
        score_paths.append(f"{seed}={path}")

    output = tmp_path / "rare"
    audit = prepare.build_rare_dataset(
        score_paths,
        small_full_navtrain["index"],
        small_full_navtrain["annotation"],
        small_full_navtrain["navtrain"],
        output,
        expected_tokens=6,
        ep_percentile=50.0,
        pair_seed=17,
    )
    assert audit["status"] == "PASS"
    assert audit["rare_count"] == 4
    assert audit["paired_common_unique"] == 2
    assert audit["paired_common_reuse_count"] == 2
    assert audit["union_count"] == 6
    assert audit["rare_vote_histogram"] == {"0": 2, "3": 4}

    pairs = [
        json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()
    ]
    assert {row["rare_token"] for row in pairs} == {"a0", "a2", "b0", "b2"}
    for row in pairs:
        rare_original = small_full_navtrain["token_logs"][row["rare_token"]]
        common_log = small_full_navtrain["token_logs"][row["common_token"]]
        assert rare_original == common_log == row["log_name"]

    loaded, loaded_audit, _, _ = trainer.load_pair_contract(
        output / "pairs.jsonl", output / "rare_data_audit.json"
    )
    assert loaded == pairs
    assert loaded_audit["union_count"] == 6


def test_pairing_fails_when_a_rare_original_has_no_strict_common(
    small_full_navtrain, tmp_path
):
    rows = {
        "a0": component_row(naf=0.0, progress=0.0),
        "a1": component_row(naf=0.0, progress=0.0),
        "a2": component_row(naf=0.0, progress=0.0),
        "b0": component_row(dac=0.0, progress=0.0),
        "b1": component_row(progress=9.0),
        "b2": component_row(progress=2.0),
    }
    score_paths = []
    for seed in range(3):
        path = tmp_path / f"bad_score_{seed}.csv"
        write_score(path, rows)
        score_paths.append(f"{seed}={path}")

    with pytest.raises(RuntimeError, match="no strict-common token"):
        prepare.build_rare_dataset(
            score_paths,
            small_full_navtrain["index"],
            small_full_navtrain["annotation"],
            small_full_navtrain["navtrain"],
            tmp_path / "bad_rare",
            expected_tokens=6,
            ep_percentile=50.0,
            pair_seed=17,
        )


def test_formal_sampling_budget_is_exact_and_sampler_covers_before_reuse():
    counts = [
        trainer.block_class_counts(6339, block) for block in range(16 * 3)
    ]
    assert sum(rare for rare, _ in counts) == 152136
    assert sum(common for _, common in counts) == 152136
    assert sum(rare + common for rare, common in counts) == 304272
    assert 16 * 3 * ((6339 + 64 - 1) // 64) == 4800

    sampler = trainer.CyclingPairSampler(size=5, seed=123)
    first = sampler.take(7)
    second = sampler.take(3)
    assert len(first) == 7
    assert len(second) == 3
    assert set(first[:5]) == set(range(5))
    assert sampler.visited == set(range(5))

def test_low_progress_threshold_is_strictly_below_percentile():
    rows = {
        "p1": component_row(progress=1.0),
        "p2": component_row(progress=2.0),
        "p3": component_row(progress=3.0),
    }
    modes, threshold = prepare.classify_rare(rows, 50.0)
    assert threshold == 2.0
    assert modes["low_ego_progress"] == {"p1"}


def test_log_disjoint_tuning_split_is_complete_and_audited(tmp_path):
    rows = [
        {
            "rare_token": f"rare_{index}",
            "common_token": f"common_{index}",
            "log_name": f"log_{index}",
            "scene": f"scene_{index}",
            "rare_votes": 1 + index % 3,
            "failure_modes_by_seed": {"0": ["offroad"], "1": [], "2": []},
        }
        for index in range(9)
    ]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    rare_audit = tmp_path / "rare_data_audit.json"
    rare_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "method": "diffusiondrive_v3_full_navtrain_rare_original_v1",
                "rare_count": len(rows),
                "outputs": {
                    "pairs_sha256": prepare.sha256_file(pairs),
                },
            }
        )
    )
    base_filter = tmp_path / "base.yaml"
    write_yaml(
        base_filter,
        {
            "_target_": "navsim.common.dataclasses.SceneFilter",
            "num_history_frames": 4,
            "num_future_frames": 10,
        },
    )
    output = tmp_path / "split"
    audit = splitter.build_tuning_split(
        pairs,
        rare_audit,
        base_filter,
        output,
        split_seed=20260819,
        train_fraction=0.6,
        development_fraction=0.2,
    )
    assert audit["status"] == "PASS"
    assert audit["log_disjoint"] is True
    assert sum(
        row["pair_rows"] for row in audit["splits"].values()
    ) == len(rows)
    log_sets = []
    for split_name in splitter.SPLIT_NAMES:
        split_rows = splitter.load_pair_rows(output / split_name / "pairs.jsonl")
        log_sets.append({row["log_name"] for row in split_rows})
        loaded, subset_audit, _, _ = trainer.load_pair_contract(
            output / split_name / "pairs.jsonl",
            output / split_name / "rare_data_audit.json",
        )
        assert loaded == split_rows
        assert subset_audit["source_kind"] == "log_disjoint_tuning_subset"
    assert not log_sets[0].intersection(log_sets[1])
    assert not log_sets[0].intersection(log_sets[2])
    assert not log_sets[1].intersection(log_sets[2])


@pytest.mark.parametrize(
    ("sampling_mode", "method"),
    [
        ("rare_balanced", trainer.FORMAL_RARE_METHOD),
        ("paired_common", trainer.FORMAL_PAIRED_COMMON_METHOD),
    ],
)
def test_formal_contract_covers_rare_and_paired_common(sampling_mode, method):
    args = argparse.Namespace(
        temperature=1.0,
        learning_rate=1e-4,
        kl_weight=1e-3,
        epochs=16,
        examples_per_cache_epoch=6339,
        batch_size=64,
        method_name=method,
        sampling_mode=sampling_mode,
        ablation="full",
        train_cache=[Path("a"), Path("b"), Path("c")],
    )
    trainer.validate_formal_contract(args)
    assert trainer.expected_method(sampling_mode) == method


def test_trainer_provenance_includes_corrected_pdm_reward():
    provenance = trainer.implementation_provenance()
    assert any(
        path.endswith("diffusiondrive_online_pdm_reward.py")
        for path in provenance
    )
