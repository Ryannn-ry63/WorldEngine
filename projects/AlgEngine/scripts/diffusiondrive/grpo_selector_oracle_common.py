#!/usr/bin/env python3
"""Pure helpers for DiffusionDrive best-of-20 oracle audits."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
SELECTION_NAMES = ("reference", "current", "oracle")


def candidate_diversity(candidates: np.ndarray) -> Dict[str, float]:
    """Summarize geometric diversity for one (20, 8, 3) candidate set."""

    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (8, 3):
        raise ValueError(
            "candidate diversity expects shape (num_candidates, 8, 3), "
            f"got {tuple(candidates.shape)}"
        )
    if candidates.shape[0] < 2:
        raise ValueError("candidate diversity requires at least two candidates")

    xy = candidates[..., :2]
    pairwise_pose_distance = np.linalg.norm(
        xy[:, None, :, :] - xy[None, :, :, :], axis=-1
    )
    pairwise_ade = pairwise_pose_distance.mean(axis=-1)
    pairwise_fde = pairwise_pose_distance[..., -1]
    upper = np.triu_indices(candidates.shape[0], k=1)
    flattened = np.round(candidates, decimals=4).reshape(candidates.shape[0], -1)
    return {
        "mean_pairwise_ade": float(pairwise_ade[upper].mean()),
        "mean_pairwise_fde": float(pairwise_fde[upper].mean()),
        "max_pairwise_fde": float(pairwise_fde[upper].max()),
        "unique_candidate_count_1e4": int(
            np.unique(flattened, axis=0).shape[0]
        ),
    }


def load_records(path: Path) -> List[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_records(path: Path, records: Iterable[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_records(records: Sequence[Mapping]) -> Dict[str, object]:
    if not records:
        raise ValueError("oracle audit summary requires records")

    summary: Dict[str, object] = {
        "num_tokens": len(records),
        "num_candidate_scores": int(
            sum(int(row["valid_candidate_count"]) for row in records)
        ),
    }
    for selection in SELECTION_NAMES:
        summary[f"{selection}_reward"] = mean(
            [float(row[f"{selection}_reward"]) for row in records]
        )
        summary[f"{selection}_oracle_match"] = mean(
            [bool(row[f"{selection}_oracle_match"]) for row in records]
        )
        components = {}
        for component in COMPONENT_NAMES:
            components[component] = mean(
                [
                    float(row[f"{selection}_components"][component])
                    for row in records
                ]
            )
        summary[f"{selection}_components"] = components

    oracle_reference = np.asarray(
        [
            float(row["oracle_reward"]) - float(row["reference_reward"])
            for row in records
        ],
        dtype=np.float64,
    )
    current_reference = np.asarray(
        [
            float(row["current_reward"]) - float(row["reference_reward"])
            for row in records
        ],
        dtype=np.float64,
    )
    summary.update(
        {
            "oracle_reference_gain": float(oracle_reference.mean()),
            "current_reference_gain": float(current_reference.mean()),
            "selection_disagreement": mean(
                [bool(row["selection_disagreement"]) for row in records]
            ),
            "oracle_strictly_better_fraction": float(
                np.mean(oracle_reference > 1e-8)
            ),
            "oracle_gap_gt_0.01_fraction": float(
                np.mean(oracle_reference > 0.01)
            ),
            "oracle_gap_gt_0.05_fraction": float(
                np.mean(oracle_reference > 0.05)
            ),
            "oracle_gap_gt_0.10_fraction": float(
                np.mean(oracle_reference > 0.10)
            ),
            "oracle_gap_q50": float(np.quantile(oracle_reference, 0.50)),
            "oracle_gap_q75": float(np.quantile(oracle_reference, 0.75)),
            "oracle_gap_q90": float(np.quantile(oracle_reference, 0.90)),
            "reward_spread": mean(
                [float(row["reward_spread"]) for row in records]
            ),
            "reward_std": mean(
                [float(row["reward_std"]) for row in records]
            ),
            "mean_pairwise_ade": mean(
                [float(row["mean_pairwise_ade"]) for row in records]
            ),
            "mean_pairwise_fde": mean(
                [float(row["mean_pairwise_fde"]) for row in records]
            ),
            "max_pairwise_fde": mean(
                [float(row["max_pairwise_fde"]) for row in records]
            ),
            "unique_candidate_count_1e4": mean(
                [float(row["unique_candidate_count_1e4"]) for row in records]
            ),
            "current_expected_reward": mean(
                [float(row["current_expected_reward"]) for row in records]
            ),
            "reference_expected_reward": mean(
                [float(row["reference_expected_reward"]) for row in records]
            ),
        }
    )
    summary["expected_reward_gain"] = float(
        summary["current_expected_reward"]
        - summary["reference_expected_reward"]
    )
    oracle_gain = float(summary["oracle_reference_gain"])
    summary["oracle_capture_rate"] = (
        float(summary["current_reference_gain"]) / oracle_gain
        if oracle_gain > 1e-12
        else None
    )
    return summary


def read_official_csv(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = str(row["token"])
            if token == "average":
                continue
            if token in rows:
                raise RuntimeError(f"duplicate official PDM token: {token}")
            if str(row.get("valid", "True")).lower() not in ("true", "1"):
                raise RuntimeError(f"invalid official PDM row for token {token}")
            rows[token] = {
                key: float(row[key]) for key in (*COMPONENT_NAMES, "score")
            }
    return rows


def summarize_official_rows(
    rows: Mapping[str, Mapping[str, float]], tokens: Iterable[str]
) -> Dict[str, float]:
    selected = [rows[token] for token in tokens]
    if not selected:
        raise ValueError("official PDM summary selected no tokens")
    return {
        key: mean([float(row[key]) for row in selected])
        for key in (*COMPONENT_NAMES, "score")
    }


def paired_bootstrap_ci(
    token_gains: Sequence[float],
    *,
    num_resamples: int = 10000,
    seed: int = 20260809,
    batch_size: int = 100,
) -> Tuple[float, float]:
    gains = np.asarray(token_gains, dtype=np.float64)
    if gains.ndim != 1 or gains.size < 2:
        raise ValueError("paired bootstrap requires at least two token gains")
    if num_resamples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    rng = np.random.default_rng(seed)
    means = np.empty(num_resamples, dtype=np.float64)
    offset = 0
    while offset < num_resamples:
        count = min(batch_size, num_resamples - offset)
        indices = rng.integers(0, gains.size, size=(count, gains.size))
        means[offset : offset + count] = gains[indices].mean(axis=1)
        offset += count
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def classify_oracle_gain(
    ci_low: float,
    ci_high: float,
    *,
    selector_limited_threshold: float = 0.005,
    generator_limited_threshold: float = 0.002,
) -> str:
    if ci_low >= selector_limited_threshold:
        return "selector_limited"
    if ci_high <= generator_limited_threshold:
        return "generator_limited"
    return "mixed_or_inconclusive"
