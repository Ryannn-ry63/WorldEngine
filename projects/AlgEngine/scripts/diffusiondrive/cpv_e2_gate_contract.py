"""Frozen identities and outcome branches for the CPV E2 development gate."""

from __future__ import annotations


DATA_METHOD = "cpv_e2_existing_navtrain_common_coverage_v1"
PROPOSAL32_SHA256 = "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
ARM_UNIQUE = {
    "D1_diverse_matched": 5267,
    "D2_diverse_20k": 20000,
}


BASE_TRAINING_CONTRACT = {
    "selector_architecture": "proposal_conditioned_regret_arbitration",
    "objective": "official_pdm_proposal_regret_arbitration",
    "arbiter_loss": "regret",
    "arbiter_risk": "source_sign",
    "use_decision_context": False,
    "proposal_checkpoint_sha256": PROPOSAL32_SHA256,
    "reward_components_consumed": False,
    "training_data_split": "train",
    "training_epochs": 16,
    "training_examples_per_cache_epoch": 6339,
    "training_batch_size": 64,
    "formal_contract": True,
    "temperature": 1.0,
    "learning_rate": 1e-4,
    "kl_weight": 1e-3,
    "override_threshold": 0.0,
    "train_seed": 0,
    "epoch": 16,
}


def assert_fields(row: dict, expected: dict, label: str) -> None:
    drift = {
        key: {"actual": row.get(key), "expected": value}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{label} contract drifted: {drift}")


def validate_data_manifest(payload: dict) -> None:
    expected = {
        "status": "PASS",
        "method": DATA_METHOD,
        "existing_dataset_only": True,
        "new_annotations": False,
        "training_labels_changed": False,
    }
    assert_fields(payload, expected, "E2 data manifest")
    arms = payload.get("arms", {})
    for arm, unique in ARM_UNIQUE.items():
        row = arms.get(arm)
        if not isinstance(row, dict):
            raise RuntimeError(f"E2 data manifest omitted {arm}")
        assert_fields(
            row,
            {
                "unique_tokens": unique,
                "rows": unique,
                "duplicate_rows": 0,
                "overlap_with_rare": 0,
                "heldout_log_overlap": 0,
            },
            arm,
        )
    if payload.get("nesting", {}).get("D1_is_subset_of_D2") is not True:
        raise RuntimeError("E2 D1/D2 nesting contract drifted")


def validate_d0(row: dict) -> None:
    expected = dict(BASE_TRAINING_CONTRACT)
    expected.update(
        method="cpv_v1_source_sign_pair_regret",
        common_arm=None,
    )
    # The frozen E1 evaluation predates sampling_mode propagation. Every other
    # immutable identity is checked; E1's own gate SHA-audited its checkpoint.
    assert_fields(row, expected, "D0 frozen E1 A3")


def validate_arm(row: dict, arm: str, data_manifest_sha256: str) -> None:
    if arm not in ARM_UNIQUE:
        raise RuntimeError(f"unknown E2 arm: {arm}")
    expected = dict(BASE_TRAINING_CONTRACT)
    expected.update(
        method=f"cpv_e2_{arm.lower()}_source_sign_pair_regret",
        sampling_mode="cpv_decision_source_sign",
        common_arm=arm,
        common_unique_tokens=ARM_UNIQUE[arm],
        common_data_manifest_sha256=data_manifest_sha256,
    )
    assert_fields(row, expected, arm)
    caches = row.get("common_caches")
    if not isinstance(caches, dict) or set(caches) != {"0", "1", "2"}:
        raise RuntimeError(f"{arm} common-cache seed identities drifted")
    for seed, cache in caches.items():
        if not isinstance(cache, dict) or not cache.get("path") or not cache.get("sha256"):
            raise RuntimeError(f"{arm} common cache {seed} provenance is incomplete")


def choose_decision(
    passed: dict[str, bool],
    d1_reliable: bool,
    d1_equal_preserved: bool,
    d2_reliable: bool,
    d2_adds_reliably: bool,
    d2_equal_improved: bool,
) -> tuple[str, str | None, str, str]:
    """Return the single pre-registered follow-up for observed E2 evidence."""

    if passed["D1"]:
        return (
            "D1_GATE_PASS",
            "D1",
            "closed_loop_development",
            "run_D1_closed_loop_development",
        )
    if passed["D2"]:
        return (
            "D2_GATE_PASS",
            "D2",
            "closed_loop_development",
            "run_D2_closed_loop_development",
        )
    if d1_reliable and d1_equal_preserved and not d2_adds_reliably:
        return (
            "PAIRED_COMMON_SELECTION_BIAS",
            "D1",
            "stop_E2_retain_V3",
            "record_D1_data_recipe_without_promotion",
        )
    if d2_reliable and d2_equal_improved:
        return (
            "COVERAGE_SIGNAL_INCOMPLETE",
            None,
            "E2_full_common_once",
            "implement_one_full_common_run",
        )
    return (
        "CURRENT_CPV_METHOD_BOTTLENECK",
        None,
        "stop_CPV_retain_V3",
        "choose_pairwise_or_temporal_representation_fork",
    )
