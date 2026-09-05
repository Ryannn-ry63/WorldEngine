import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/diffusiondrive"


def load(name, filename, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_cache(tokens, markers):
    count = len(tokens)
    candidates = 2
    marker = torch.tensor(markers, dtype=torch.float32).view(count, 1, 1)
    return {
        "tokens": list(tokens),
        "candidate_features": marker.expand(count, candidates, 1).clone(),
        "candidate_trajectories_8": torch.zeros(count, candidates, 8, 3),
        "route_bev_features": torch.zeros(count, candidates, 8, 1),
        "status_tokens": torch.zeros(count, 1, 1),
        "ego_queries": torch.zeros(count, 1, 1),
        "agents_queries": torch.zeros(count, 1, 1),
        "reference_logits": marker[:, :, 0].expand(count, candidates).clone(),
        "candidate_rewards": torch.zeros(count, candidates),
        "candidate_reward_components": torch.zeros(count, candidates, 6),
        "candidate_reward_valid_mask": torch.ones(
            count, candidates, dtype=torch.bool
        ),
    }


def test_e2_sidecar_replaces_only_common_rows_and_preserves_batch_labels(monkeypatch):
    trainer = load(
        "cpv_e2_trainer", "train_grpo_selector_v3_cached_rare_rollout.py", monkeypatch
    )
    real = tiny_cache(["pair0", "rare0"], [1.0, 10.0])
    sidecar = tiny_cache(["c0", "c1"], [100.0, 101.0])
    synthetic = tiny_cache(["synthetic0"], [200.0])
    hard_rows = [
        {
            "hard_kind": "real_rare",
            "rare_token": "rare0",
            "paired_common_token": "pair0",
        },
        {"hard_kind": "synthetic_rollout", "synthetic_index": 0},
    ]
    rows = [("common", 1), ("hard", 0), ("common", 0), ("hard", 1)]
    inputs, _, _, _, _, kinds = trainer.mixed_batch(
        rows,
        hard_rows,
        real,
        {token: index for index, token in enumerate(real["tokens"])},
        sidecar,
        {token: index for index, token in enumerate(sidecar["tokens"])},
        ["c0", "c1"],
        synthetic,
        torch.device("cpu"),
    )
    assert inputs["candidate_features"][:, 0, 0].tolist() == [
        10.0,
        101.0,
        100.0,
        200.0,
    ]
    assert trainer.ordered_batch_source_labels(
        rows, hard_rows, separate_common=True
    ) == ["hard_real_rare", "common", "common", "hard_synthetic"]
    assert kinds == {
        "common": 2,
        "hard_real_rare": 1,
        "hard_synthetic": 1,
    }


def test_legacy_paired_common_path_is_unchanged(monkeypatch):
    trainer = load(
        "cpv_e2_legacy_trainer",
        "train_grpo_selector_v3_cached_rare_rollout.py",
        monkeypatch,
    )
    real = tiny_cache(["pair0", "rare0"], [1.0, 10.0])
    hard_rows = [
        {
            "hard_kind": "real_rare",
            "rare_token": "rare0",
            "paired_common_token": "pair0",
        }
    ]
    inputs, *_ = trainer.mixed_batch(
        [("common", 0), ("hard", 0)],
        hard_rows,
        real,
        {token: index for index, token in enumerate(real["tokens"])},
        None,
        None,
        None,
        tiny_cache(["synthetic0"], [200.0]),
        torch.device("cpu"),
    )
    assert inputs["candidate_features"][:, 0, 0].tolist() == [1.0, 10.0]


def test_common_decision_pool_size_is_independent_of_hard_pool(monkeypatch):
    trainer = load(
        "cpv_e2_pool_trainer",
        "train_grpo_selector_v3_cached_rare_rollout.py",
        monkeypatch,
    )
    hard_rows = [
        {"hard_kind": "real_rare"},
        {"hard_kind": "synthetic_rollout"},
    ]
    pools = trainer.decision_source_specs(hard_rows, common_count=7)
    assert len(pools["common"]) == 7
    assert len(pools["hard_real_rare"]) == 1
    assert len(pools["hard_synthetic"]) == 1


def test_nested_common_order_is_deterministic_balanced_and_leak_auditable(
    monkeypatch,
):
    prepare = load(
        "cpv_e2_prepare", "prepare_cpv_e2_common_coverage.py", monkeypatch
    )
    rows = [
        {
            "token": f"t{index}",
            "log_name": log,
            "map_location": map_name,
            "command": command,
        }
        for index, (log, map_name, command) in enumerate(
            [
                ("l0", "m0", 0),
                ("l0", "m0", 0),
                ("l1", "m0", 1),
                ("l1", "m0", 1),
                ("l2", "m1", 2),
                ("l2", "m1", 2),
            ]
        )
    ]
    first = prepare.stratified_nested_order([dict(row) for row in rows], 19)
    second = prepare.stratified_nested_order([dict(row) for row in rows], 19)
    assert [row["token"] for row in first] == [row["token"] for row in second]
    assert len(
        {(row["log_name"], row["map_location"], row["command"]) for row in first[:3]}
    ) == 3
    summary = prepare.arm_summary(
        "D1", first[:3], {"t0"}, {"rare"}, {"heldout"}
    )
    assert summary["duplicate_rows"] == 0
    assert summary["overlap_with_rare"] == 0
    assert summary["heldout_log_overlap"] == 0


def test_cache_subset_reorders_every_example_aligned_field(monkeypatch):
    subset = load(
        "cpv_e2_subset", "subset_cpv_e2_common_cache.py", monkeypatch
    )
    source = tiny_cache(["c", "a", "b"], [3.0, 1.0, 2.0])
    source["scenes"] = ["scene-c", "scene-a", "scene-b"]
    source["constant"] = "kept"
    result = subset.subset_cache(source, ["c", "a"])
    assert result["tokens"] == ["a", "c"]
    assert result["scenes"] == ["scene-a", "scene-c"]
    assert result["candidate_features"][:, 0, 0].tolist() == [1.0, 3.0]
    assert result["constant"] == "kept"


def valid_manifest():
    return {
        "status": "PASS",
        "method": "cpv_e2_existing_navtrain_common_coverage_v1",
        "existing_dataset_only": True,
        "new_annotations": False,
        "training_labels_changed": False,
        "nesting": {"D1_is_subset_of_D2": True},
        "arms": {
            arm: {
                "unique_tokens": unique,
                "rows": unique,
                "duplicate_rows": 0,
                "overlap_with_rare": 0,
                "heldout_log_overlap": 0,
            }
            for arm, unique in {
                "D1_diverse_matched": 5267,
                "D2_diverse_20k": 20000,
            }.items()
        },
    }


def valid_arm(contract, arm, manifest_sha="manifest-sha"):
    row = dict(contract.BASE_TRAINING_CONTRACT)
    row.update(
        method=f"cpv_e2_{arm.lower()}_source_sign_pair_regret",
        sampling_mode="cpv_decision_source_sign",
        common_arm=arm,
        common_unique_tokens=contract.ARM_UNIQUE[arm],
        common_data_manifest_sha256=manifest_sha,
        common_caches={
            str(seed): {"path": f"seed{seed}.pt", "sha256": f"sha{seed}"}
            for seed in range(3)
        },
    )
    return row


def test_gate_rejects_data_or_checkpoint_identity_drift(monkeypatch):
    contract = load(
        "cpv_e2_gate_contract", "cpv_e2_gate_contract.py", monkeypatch
    )
    contract.validate_data_manifest(valid_manifest())
    row = valid_arm(contract, "D1_diverse_matched")
    contract.validate_arm(row, "D1_diverse_matched", "manifest-sha")
    row["common_unique_tokens"] = 5266
    with pytest.raises(RuntimeError, match="contract drifted"):
        contract.validate_arm(row, "D1_diverse_matched", "manifest-sha")


@pytest.mark.parametrize(
    ("passed", "flags", "decision", "followup"),
    [
        (
            {"D1": True, "D2": True},
            (False, False, False, False, False),
            "D1_GATE_PASS",
            "closed_loop_development",
        ),
        (
            {"D1": False, "D2": True},
            (False, False, False, False, False),
            "D2_GATE_PASS",
            "closed_loop_development",
        ),
        (
            {"D1": False, "D2": False},
            (True, True, False, False, False),
            "PAIRED_COMMON_SELECTION_BIAS",
            "stop_E2_retain_V3",
        ),
        (
            {"D1": False, "D2": False},
            (False, False, True, True, True),
            "COVERAGE_SIGNAL_INCOMPLETE",
            "E2_full_common_once",
        ),
        (
            {"D1": False, "D2": False},
            (False, False, False, False, False),
            "CURRENT_CPV_METHOD_BOTTLENECK",
            "stop_CPV_retain_V3",
        ),
    ],
)
def test_all_preregistered_e2_outcome_branches(
    monkeypatch, passed, flags, decision, followup
):
    contract = load(
        f"cpv_e2_gate_{decision}", "cpv_e2_gate_contract.py", monkeypatch
    )
    observed = contract.choose_decision(passed, *flags)
    assert observed[0] == decision
    assert observed[2] == followup
