#!/usr/bin/env python3
"""Confirm the diffusion-lineage problem before LC-PGRPO model selection.

This audit answers three separate questions with immutable evidence:

1. Is the historical Direct result exactly reproducible?
2. Does candidate index carry a stable plan-anchor lineage across noise draws?
3. Does leave-one-draw lineage utility predict held-out draw quality?

It never trains or selects a new method.  Historical development evidence is
used only to establish that Direct's fresh-noise improvement did not transfer.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--historical-direct-checkpoint", type=Path, required=True)
    parser.add_argument("--reproduced-direct-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-fresh-evaluation", type=Path, required=True)
    parser.add_argument("--direct-development-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="diagnostic subset only; a subset can never authorize formal CV",
    )
    return parser.parse_args()


def load_json(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def selector_state(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    state = payload.get("scene_selector_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint lacks selector state: {path}")
    return path, payload, state


def exact_state_reproduction(historical: Path, reproduced: Path):
    old_path, old_payload, old_state = selector_state(historical)
    new_path, new_payload, new_state = selector_state(reproduced)
    same_keys = tuple(old_state) == tuple(new_state)
    differing = []
    if set(old_state) == set(new_state):
        for key in old_state:
            if not torch.equal(old_state[key], new_state[key]):
                differing.append(key)
    else:
        differing = sorted(set(old_state).symmetric_difference(new_state))
    exact = same_keys and not differing
    return {
        "exact": exact,
        "same_ordered_keys": same_keys,
        "num_tensors": len(old_state),
        "differing_tensors": differing,
        "historical_checkpoint": str(old_path),
        "historical_checkpoint_sha256": common.sha256_file(old_path),
        "historical_method": old_payload.get("method"),
        "historical_arm": old_payload.get("arm"),
        "reproduced_checkpoint": str(new_path),
        "reproduced_checkpoint_sha256": common.sha256_file(new_path),
        "reproduced_method": new_payload.get("method"),
        "reproduced_arm": new_payload.get("arm"),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray):
    if left.size < 2 or right.size != left.size:
        return None
    if float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def trajectory_alignment(caches, indices, batch_size):
    pair_results = {}
    identity_numerator = 0
    identity_denominator = 0
    all_same_ade = []
    for left_draw, right_draw in combinations(range(len(caches)), 2):
        matches = 0
        total = 0
        same_ade = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            left = caches[left_draw]["candidate_trajectories_8"][batch_indices, :, :, :2].float()
            right = caches[right_draw]["candidate_trajectories_8"][batch_indices, :, :, :2].float()
            # [B, K_left, K_right, T]
            distance = torch.linalg.vector_norm(
                left[:, :, None] - right[:, None, :], dim=-1
            ).mean(dim=-1)
            nearest = distance.argmin(dim=-1)
            expected = torch.arange(left.shape[1])[None].expand_as(nearest)
            matches += int(nearest.eq(expected).sum())
            total += int(nearest.numel())
            diagonal = distance.diagonal(dim1=1, dim2=2).reshape(-1)
            same_ade.append(diagonal)
        values = torch.cat(same_ade).numpy()
        label = f"{left_draw}-{right_draw}"
        pair_results[label] = {
            "nearest_candidate_same_index_fraction": matches / total,
            "same_lineage_ade_mean_m": float(values.mean()),
            "same_lineage_ade_median_m": float(np.median(values)),
            "num_candidate_comparisons": total,
        }
        identity_numerator += matches
        identity_denominator += total
        all_same_ade.append(values)
    concatenated = np.concatenate(all_same_ade)
    return {
        "pairs": pair_results,
        "pooled_nearest_candidate_same_index_fraction": (
            identity_numerator / identity_denominator
        ),
        "pooled_same_lineage_ade_mean_m": float(concatenated.mean()),
        "pooled_same_lineage_ade_median_m": float(np.median(concatenated)),
    }


def reward_lineage_correlation(caches, indices):
    pair_values = {}
    pooled_left, pooled_right = [], []
    for left_draw, right_draw in combinations(range(len(caches)), 2):
        left_reward = caches[left_draw]["candidate_rewards"][indices].float()
        right_reward = caches[right_draw]["candidate_rewards"][indices].float()
        left_valid = caches[left_draw]["candidate_reward_valid_mask"][indices].bool()
        right_valid = caches[right_draw]["candidate_reward_valid_mask"][indices].bool()
        valid = left_valid & right_valid & torch.isfinite(left_reward) & torch.isfinite(right_reward)
        left = left_reward[valid].double().numpy()
        right = right_reward[valid].double().numpy()
        pair_values[f"{left_draw}-{right_draw}"] = {
            "correlation": safe_correlation(left, right),
            "num_lineage_pairs": int(left.size),
        }
        pooled_left.append(left)
        pooled_right.append(right)
    left = np.concatenate(pooled_left)
    right = np.concatenate(pooled_right)
    return {
        "pairs": pair_values,
        "pooled_correlation": safe_correlation(left, right),
        "num_pooled_lineage_pairs": int(left.size),
    }


def heldout_lineage_prediction(caches, indices):
    rewards = torch.stack(
        [cache["candidate_rewards"][indices].float() for cache in caches], dim=1
    )
    valid = torch.stack(
        [cache["candidate_reward_valid_mask"][indices].bool() for cache in caches],
        dim=1,
    )
    valid = valid & torch.isfinite(rewards)
    exact_oracle_index = []
    oracle_index_in_predicted_top4 = []
    predicted_index_in_heldout_top4 = []
    tie_aware_top1 = []
    within_oracle_margin_0005 = []
    predicted_rewards, oracle_rewards = [], []
    for heldout in range(rewards.shape[1]):
        other_draws = [draw for draw in range(rewards.shape[1]) if draw != heldout]
        other_valid = valid[:, other_draws]
        other_safe = torch.where(
            other_valid, rewards[:, other_draws], torch.zeros_like(rewards[:, other_draws])
        )
        count = other_valid.sum(dim=1)
        lineage_valid = valid[:, heldout] & (count > 0)
        utility = other_safe.sum(dim=1) / count.clamp_min(1).float()
        predicted_order = utility.masked_fill(~lineage_valid, -torch.inf).argsort(
            dim=-1, descending=True
        )
        predicted = predicted_order[:, 0]
        heldout_reward = rewards[:, heldout]
        heldout_valid = valid[:, heldout]
        oracle = heldout_reward.masked_fill(~heldout_valid, -torch.inf).argmax(dim=-1)
        heldout_order = heldout_reward.masked_fill(
            ~heldout_valid, -torch.inf
        ).argsort(dim=-1, descending=True)
        selected_reward = heldout_reward.gather(1, predicted[:, None]).squeeze(1)
        oracle_reward = heldout_reward.gather(1, oracle[:, None]).squeeze(1)
        exact_oracle_index.append(predicted.eq(oracle))
        oracle_index_in_predicted_top4.append(
            predicted_order[:, :4].eq(oracle[:, None]).any(dim=-1)
        )
        predicted_index_in_heldout_top4.append(
            heldout_order[:, :4].eq(predicted[:, None]).any(dim=-1)
        )
        # PDM has genuine reward ties (especially at 1.0). Equality to the
        # first argmax is therefore not a well-defined top-1 quality metric.
        tie_aware_top1.append(selected_reward >= oracle_reward - 1e-7)
        within_oracle_margin_0005.append(selected_reward >= oracle_reward - 0.005)
        predicted_rewards.append(selected_reward)
        oracle_rewards.append(oracle_reward)
    exact_oracle_index = torch.stack(exact_oracle_index, dim=1).float()
    oracle_index_in_predicted_top4 = torch.stack(
        oracle_index_in_predicted_top4, dim=1
    ).float()
    predicted_index_in_heldout_top4 = torch.stack(
        predicted_index_in_heldout_top4, dim=1
    ).float()
    tie_aware_top1 = torch.stack(tie_aware_top1, dim=1).float()
    within_oracle_margin_0005 = torch.stack(
        within_oracle_margin_0005, dim=1
    ).float()
    predicted_rewards = torch.stack(predicted_rewards, dim=1)
    oracle_rewards = torch.stack(oracle_rewards, dim=1)
    return {
        "predicted_top1_is_first_heldout_argmax_fraction": float(
            exact_oracle_index.mean()
        ),
        "predicted_top1_is_tie_aware_heldout_top1_fraction": float(
            tie_aware_top1.mean()
        ),
        "predicted_top1_is_in_heldout_top4_fraction": float(
            predicted_index_in_heldout_top4.mean()
        ),
        "first_heldout_argmax_is_in_predicted_top4_fraction": float(
            oracle_index_in_predicted_top4.mean()
        ),
        "predicted_top1_within_0005_of_heldout_oracle_fraction": float(
            within_oracle_margin_0005.mean()
        ),
        "mean_selected_reward": float(predicted_rewards.mean()),
        "mean_oracle_reward": float(oracle_rewards.mean()),
        "mean_oracle_gap": float((oracle_rewards - predicted_rewards).mean()),
        "num_token_draws": int(exact_oracle_index.numel()),
    }


def validate_direct_evidence(fresh_payload, development_payload):
    fresh_summary = fresh_payload.get("summary", {})
    development_summary = development_payload.get("summary", {})
    required = ("equal_stratum_hard_gain", "common_hard_gain", "rare_hard_gain")
    if any(key not in fresh_summary or key not in development_summary for key in required):
        raise RuntimeError("Direct evaluation summary lacks locked fields")
    return {
        "fresh_noise": {key: float(fresh_summary[key]) for key in required},
        "log_disjoint_development": {
            key: float(development_summary[key]) for key in required
        },
        "fresh_method": fresh_payload.get("method"),
        "fresh_split": fresh_payload.get("split"),
        "development_method": development_payload.get("method"),
        "development_split": development_payload.get("split"),
    }


def main():
    args = parse_args()
    if args.batch_size <= 0 or (args.max_tokens is not None and args.max_tokens <= 0):
        raise ValueError("audit budgets must be positive")
    loaded = [common.load_cache(path, "train") for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(
        caches, manifests, "train", protocol.FRESH_NOISE_SEEDS
    )
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
    common_indices = np.asarray(sorted(set(common_indices)), dtype=np.int64)
    if args.max_tokens is not None:
        # A deterministic balanced prefix is useful for smoke tests but cannot
        # generate an authorization gate.
        half = max(1, args.max_tokens // 2)
        rare_indices = rare_indices[:half]
        common_indices = common_indices[:half]
    all_indices = np.asarray(
        sorted(set(rare_indices.tolist()) | set(common_indices.tolist())),
        dtype=np.int64,
    )

    reproduction = exact_state_reproduction(
        args.historical_direct_checkpoint, args.reproduced_direct_checkpoint
    )
    fresh_path, fresh_payload = load_json(args.direct_fresh_evaluation)
    development_path, development_payload = load_json(
        args.direct_development_evaluation
    )
    direct = validate_direct_evidence(fresh_payload, development_payload)
    alignment = trajectory_alignment(caches, all_indices, args.batch_size)
    reward_stability = {
        "common": reward_lineage_correlation(caches, common_indices),
        "rare": reward_lineage_correlation(caches, rare_indices),
    }
    prediction = {
        "common": heldout_lineage_prediction(caches, common_indices),
        "rare": heldout_lineage_prediction(caches, rare_indices),
    }

    formal_coverage = args.max_tokens is None and len(all_indices) == len(caches[0]["tokens"])
    gates = {
        "direct_state_exactly_reproduced": bool(reproduction["exact"]),
        "candidate_index_is_stable_lineage": (
            alignment["pooled_nearest_candidate_same_index_fraction"] >= 0.90
        ),
        "same_lineage_reward_correlation_common_at_least_075": (
            reward_stability["common"]["pooled_correlation"] is not None
            and reward_stability["common"]["pooled_correlation"] >= 0.75
        ),
        "same_lineage_reward_correlation_rare_at_least_075": (
            reward_stability["rare"]["pooled_correlation"] is not None
            and reward_stability["rare"]["pooled_correlation"] >= 0.75
        ),
        "leave_one_draw_selected_lineage_in_heldout_top4_common_at_least_090": (
            prediction["common"][
                "predicted_top1_is_in_heldout_top4_fraction"
            ]
            >= 0.90
        ),
        "leave_one_draw_selected_lineage_in_heldout_top4_rare_at_least_090": (
            prediction["rare"][
                "predicted_top1_is_in_heldout_top4_fraction"
            ]
            >= 0.90
        ),
        "direct_fresh_gain_positive": (
            direct["fresh_noise"]["equal_stratum_hard_gain"] >= 0.002
        ),
        "direct_log_disjoint_gain_negative": (
            direct["log_disjoint_development"]["equal_stratum_hard_gain"] < 0.0
        ),
        "full_train_union_coverage": formal_coverage,
    }
    decision = (
        "AUTHORIZE_LCPGRPO_LOG_CV"
        if all(gates.values())
        else "STOP_LCPGRPO_MECHANISM_NOT_CONFIRMED"
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "status": "PASS",
        "method": protocol.MECHANISM_METHOD,
        "decision": decision,
        "gates": gates,
        "gate_scope": "mechanism existence only; no new method was trained or selected",
        "state_reproduction": reproduction,
        "direct_generalization_evidence": direct,
        "trajectory_lineage_alignment": alignment,
        "same_lineage_reward_stability": reward_stability,
        "leave_one_draw_prediction": prediction,
        "prediction_metric_contract": {
            "prediction": "argmax of mean reward from the other two draws at the same candidate index",
            "primary_top4": "predicted lineage index belongs to the four highest-reward valid candidates in the held-out draw",
            "tie_aware_top1": "predicted lineage reward equals the held-out maximum within 1e-7",
            "first_argmax_diagnostic": "predicted index equals the first index returned by held-out argmax; diagnostic only because PDM can tie",
            "inverse_top4_diagnostic": "the first held-out argmax index belongs to the four highest other-draw utilities; not used as a gate",
            "oracle_margin_diagnostic": "predicted held-out reward is within 0.005 of the held-out maximum",
            "gate": {
                "metric": "predicted_top1_is_in_heldout_top4_fraction",
                "minimum_per_stratum": 0.9,
            },
        },
        "num_tokens": int(len(all_indices)),
        "num_common_tokens": int(len(common_indices)),
        "num_rare_tokens": int(len(rare_indices)),
        "noise_seeds": list(protocol.FRESH_NOISE_SEEDS),
        "cache_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "direct_fresh_evaluation": str(fresh_path),
        "direct_fresh_evaluation_sha256": common.sha256_file(fresh_path),
        "direct_development_evaluation": str(development_path),
        "direct_development_evaluation_sha256": common.sha256_file(development_path),
        "scientific_contract": {
            "generator_frozen": True,
            "selector_only": True,
            "official_scalar_pdm_only": True,
            "new_training_data_consumed": False,
            "new_method_trained_or_selected": False,
            "historical_direct_development_evidence_consumed": True,
            "development_consumed_for_new_method_selection": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(protocol.__file__).resolve()): common.sha256_file(
                Path(protocol.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "decision": decision, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
