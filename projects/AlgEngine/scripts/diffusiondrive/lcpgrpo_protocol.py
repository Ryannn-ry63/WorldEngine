#!/usr/bin/env python3
"""Locked split and statistical helpers for the LC-PGRPO experiment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METHOD = "lineage_consistent_proximal_selector_v1"
EVALUATION_METHOD = "lineage_consistent_proximal_evaluation_v1"
MECHANISM_METHOD = "lineage_consistent_mechanism_audit_v1"
FOLD_SALT = "20260902"
NUM_FOLDS = 5
TRAIN_NOISE_SEEDS = (0, 1, 2)
FRESH_NOISE_SEEDS = (9, 10, 11)
DEVELOPMENT_NOISE_SEEDS = (3, 4, 5)
CERTIFICATION_NOISE_SEEDS = (6, 7, 8)
TARGET_KL_GRID = (0.01, 0.03, 0.10)
CHECKPOINT_EPOCHS = (1, 2, 4, 8)

THRESHOLDS = {
    "equal_stratum_hard_gain": 0.002,
    "common_hard_gain": 0.0,
    "rare_hard_gain": 0.0,
    "per_noise_seed_floor": -0.001,
    "solved_subset_floor": -0.001,
    "suppressed_recoverable_regret_reduction": 0.005,
    "positive_fold_count": 4,
    "mean_current_to_v3_kl_ceiling": 0.9,
    "paired_log_bootstrap_lower": 0.0,
    "method_margin": 0.001,
}


def log_fold(log_name: str, salt: str = FOLD_SALT, folds: int = NUM_FOLDS) -> int:
    """Map a complete log to a deterministic fold without token leakage."""

    if folds < 2:
        raise ValueError("fold count must be at least two")
    name = str(log_name)
    if not name:
        raise ValueError("log name must be non-empty")
    digest = hashlib.sha256(f"{salt}:{name}".encode("utf-8")).hexdigest()
    return int(digest, 16) % int(folds)


def build_fold_contract(pair_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Build and validate the immutable five-fold log assignment."""

    if not pair_rows:
        raise ValueError("pair rows must be non-empty")
    log_to_fold: Dict[str, int] = {}
    fold_pair_positions: Dict[int, List[int]] = {
        fold: [] for fold in range(NUM_FOLDS)
    }
    tokens_by_fold = {fold: set() for fold in range(NUM_FOLDS)}
    for position, row in enumerate(pair_rows):
        log_name = str(row["log_name"])
        fold = log_fold(log_name)
        previous = log_to_fold.setdefault(log_name, fold)
        if previous != fold:
            raise RuntimeError("one log was assigned to multiple folds")
        fold_pair_positions[fold].append(position)
        for key in ("rare_token", "common_token"):
            token = str(row[key])
            if not token:
                raise RuntimeError("empty token in pair contract")
            tokens_by_fold[fold].add(token)

    if any(not positions for positions in fold_pair_positions.values()):
        raise RuntimeError("deterministic split produced an empty fold")
    for left in range(NUM_FOLDS):
        for right in range(left + 1, NUM_FOLDS):
            if tokens_by_fold[left].intersection(tokens_by_fold[right]):
                raise RuntimeError("token leakage across log folds")

    return {
        "salt": FOLD_SALT,
        "hash": "sha256(salt + ':' + log_name) interpreted as a base-16 integer",
        "num_folds": NUM_FOLDS,
        "num_logs": len(log_to_fold),
        "num_pairs": len(pair_rows),
        "log_to_fold": dict(sorted(log_to_fold.items())),
        "folds": {
            str(fold): {
                "num_logs": sum(value == fold for value in log_to_fold.values()),
                "num_pairs": len(fold_pair_positions[fold]),
                "num_tokens": len(tokens_by_fold[fold]),
                "pair_positions": fold_pair_positions[fold],
            }
            for fold in range(NUM_FOLDS)
        },
    }


def pair_positions_for_training(
    fold_contract: Mapping[str, object], heldout_fold: int
) -> List[int]:
    if heldout_fold == -1:
        return list(range(int(fold_contract["num_pairs"])))
    if heldout_fold not in range(NUM_FOLDS):
        raise ValueError("heldout fold must be -1 or one of 0..4")
    heldout = set(fold_contract["folds"][str(heldout_fold)]["pair_positions"])
    positions = [
        position
        for position in range(int(fold_contract["num_pairs"]))
        if position not in heldout
    ]
    if set(positions).intersection(heldout):
        raise RuntimeError("training/heldout pair leakage")
    return positions


def pair_positions_for_evaluation(
    fold_contract: Mapping[str, object], heldout_fold: int
) -> List[int]:
    if heldout_fold == -1:
        return list(range(int(fold_contract["num_pairs"])))
    if heldout_fold not in range(NUM_FOLDS):
        raise ValueError("heldout fold must be -1 or one of 0..4")
    return list(fold_contract["folds"][str(heldout_fold)]["pair_positions"])


def rows_by_key(rows: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, int], dict]:
    output: Dict[Tuple[str, int], dict] = {}
    for row in rows:
        key = (str(row["token"]), int(row["noise_seed"]))
        if key in output:
            raise RuntimeError(f"duplicate evaluation key: {key}")
        output[key] = dict(row)
    if not output:
        raise RuntimeError("empty evaluation records")
    return output


def paired_log_bootstrap(
    left_rows: Iterable[Mapping[str, object]],
    right_rows: Optional[Iterable[Mapping[str, object]]] = None,
    *,
    repetitions: int = 10000,
    seed: int = 20260902,
) -> Dict[str, object]:
    """Equal-stratum paired-log bootstrap of hard PDM improvement.

    With no reference rows, ``hard_gain`` is compared with zero.  With a
    reference, current selected rewards are paired token/draw-wise.
    """

    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    left = rows_by_key(left_rows)
    right = rows_by_key(right_rows) if right_rows is not None else None
    if right is not None and set(left) != set(right):
        raise RuntimeError("paired methods cover different token/noise keys")

    by_log = defaultdict(lambda: defaultdict(list))
    invariant_names = ("scene", "log", "stratum", "noise_seed", "anchor_reward")
    for key in sorted(left):
        row = left[key]
        if right is None:
            delta = float(row["hard_gain"])
        else:
            reference = right[key]
            if any(row[name] != reference[name] for name in invariant_names):
                raise RuntimeError(f"paired invariant drift at {key}")
            delta = float(row["current_reward"]) - float(reference["current_reward"])
        by_log[str(row["log"])][str(row["stratum"])].append(delta)

    per_log = []
    for log_name in sorted(by_log):
        groups = by_log[log_name]
        if not groups["rare"] or not groups["common"]:
            raise RuntimeError(f"log lacks a paired stratum: {log_name}")
        per_log.append(
            0.5
            * (
                np.mean(groups["rare"], dtype=np.float64)
                + np.mean(groups["common"], dtype=np.float64)
            )
        )
    values = np.asarray(per_log, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        samples = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[samples].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "paired_log",
        "num_logs": int(len(values)),
        "repetitions": int(repetitions),
    }


def conditional_mean(values: np.ndarray, mask: np.ndarray):
    selected = values[mask]
    return float(selected.mean()) if selected.size else None


def aggregate_records(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_noise_seeds: Sequence[int],
    fold_gains: Optional[Mapping[int, float]] = None,
    bootstrap_repetitions: int = 10000,
    seed: int = 20260902,
) -> Dict[str, object]:
    """Compute every pre-registered efficacy statistic from raw records."""

    if not rows:
        raise RuntimeError("cannot aggregate empty records")
    strata = {}
    for label in ("common", "rare"):
        subset = [row for row in rows if str(row["stratum"]) == label]
        if not subset:
            raise RuntimeError(f"missing {label} evaluation stratum")
        hard = np.asarray([float(row["hard_gain"]) for row in subset])
        solved = np.asarray([bool(row["solved"]) for row in subset])
        suppressed = np.asarray(
            [bool(row["suppressed_recoverable"]) for row in subset]
        )
        strata[label] = {
            "num_records": len(subset),
            "hard_gain": float(hard.mean()),
            "solved_subset_hard_gain": conditional_mean(hard, solved),
            "suppressed_recoverable_regret_reduction": conditional_mean(
                hard, suppressed
            ),
            "mean_current_to_v3_kl": float(
                np.mean([float(row["current_to_v3_kl"]) for row in subset])
            ),
        }

    by_seed = {}
    for noise_seed in expected_noise_seeds:
        means = []
        for label in ("common", "rare"):
            values = [
                float(row["hard_gain"])
                for row in rows
                if int(row["noise_seed"]) == int(noise_seed)
                and str(row["stratum"]) == label
            ]
            if not values:
                raise RuntimeError(f"missing seed {noise_seed}/{label} records")
            means.append(float(np.mean(values, dtype=np.float64)))
        by_seed[str(noise_seed)] = 0.5 * (means[0] + means[1])

    def equal_conditional(name):
        values = [strata[label][name] for label in ("common", "rare")]
        return None if any(value is None for value in values) else 0.5 * sum(values)

    summary = {
        "equal_stratum_hard_gain": 0.5
        * (strata["common"]["hard_gain"] + strata["rare"]["hard_gain"]),
        "common_hard_gain": strata["common"]["hard_gain"],
        "rare_hard_gain": strata["rare"]["hard_gain"],
        "equal_stratum_by_noise_seed_hard_gain": by_seed,
        "equal_solved_subset_hard_gain": equal_conditional(
            "solved_subset_hard_gain"
        ),
        "equal_suppressed_recoverable_oracle_regret_reduction": equal_conditional(
            "suppressed_recoverable_regret_reduction"
        ),
        "mean_current_to_v3_kl": 0.5
        * (
            strata["common"]["mean_current_to_v3_kl"]
            + strata["rare"]["mean_current_to_v3_kl"]
        ),
        "paired_log_bootstrap_hard_gain": paired_log_bootstrap(
            rows, repetitions=bootstrap_repetitions, seed=seed
        ),
    }
    if fold_gains is not None:
        summary["fold_equal_stratum_hard_gain"] = {
            str(key): float(value) for key, value in sorted(fold_gains.items())
        }
        summary["positive_fold_count"] = sum(value > 0.0 for value in fold_gains.values())
    return {"summary": summary, "strata": strata}


def efficacy_gates(summary: Mapping[str, object], *, require_folds: bool) -> Dict[str, bool]:
    suppressed = summary["equal_suppressed_recoverable_oracle_regret_reduction"]
    solved = summary["equal_solved_subset_hard_gain"]
    gates = {
        "equal_stratum_gain_at_least_0002": (
            float(summary["equal_stratum_hard_gain"])
            >= THRESHOLDS["equal_stratum_hard_gain"]
        ),
        "paired_log_bootstrap_lower_positive": (
            float(summary["paired_log_bootstrap_hard_gain"]["lower_95"])
            > THRESHOLDS["paired_log_bootstrap_lower"]
        ),
        "common_nonnegative": (
            float(summary["common_hard_gain"]) >= THRESHOLDS["common_hard_gain"]
        ),
        "rare_nonnegative": (
            float(summary["rare_hard_gain"]) >= THRESHOLDS["rare_hard_gain"]
        ),
        "every_noise_seed_floor_minus_0001": (
            min(summary["equal_stratum_by_noise_seed_hard_gain"].values())
            >= THRESHOLDS["per_noise_seed_floor"]
        ),
        "solved_subset_drop_at_most_0001": (
            solved is not None and float(solved) >= THRESHOLDS["solved_subset_floor"]
        ),
        "suppressed_recoverable_reduction_at_least_0005": (
            suppressed is not None
            and float(suppressed)
            >= THRESHOLDS["suppressed_recoverable_regret_reduction"]
        ),
        "mean_current_to_v3_kl_at_most_09": (
            float(summary["mean_current_to_v3_kl"])
            <= THRESHOLDS["mean_current_to_v3_kl_ceiling"]
        ),
    }
    if require_folds:
        gates["at_least_four_of_five_positive_folds"] = (
            int(summary.get("positive_fold_count", -1))
            >= THRESHOLDS["positive_fold_count"]
        )
    return gates
