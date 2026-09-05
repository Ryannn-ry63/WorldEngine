#!/usr/bin/env python3
"""Train-only decision-consistent top-1 reranking audit.

Rare-tuned V3 first locks a valid top-5 and its rank-1 incumbent.  One compact
evaluator scores each of the four possible overrides against that incumbent.
The audit compares historical pairwise verification with a cost-sensitive
structured top-1 objective whose training decision matches deployment argmax.
Rewards and reward components are labels/metrics only, never model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grpo_selector_v3_cached_common as common
import probe_hard_pair_information as hard
import train_trajectory_set_reasoner_grpo as trainer


NUM_FOLDS = 5
TOPK = 5
ALTERNATIVES = torch.arange(1, TOPK, dtype=torch.long)
ARMS = (
    "P0_pair_token",
    "P1_pair_relation",
    "T0_structured_token",
    "T1_structured_relation",
    "C1_structured_relation_shuffle",
)
PAIRWISE_ARMS = {"P0_pair_token", "P1_pair_relation"}
STRUCTURED_ARMS = set(ARMS) - PAIRWISE_ARMS


class Top1Comparator(nn.Module):
    """Capacity-matched directed comparator, antisymmetric by construction."""

    def __init__(self, candidate_dim=257, relation_dim=41, hidden_dim=128):
        super().__init__()
        self.candidate_norm = nn.LayerNorm(candidate_dim)
        self.relation_norm = nn.LayerNorm(relation_dim)
        width = 2 * candidate_dim + relation_dim
        self.network = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def ordered(self, first, second, relation):
        values = torch.cat(
            (
                self.candidate_norm(first),
                self.candidate_norm(second),
                self.relation_norm(relation),
            ),
            dim=-1,
        )
        return self.network(values).squeeze(-1)

    def forward(self, challenger, incumbent, relation_ci, relation_ic):
        forward = self.ordered(challenger, incumbent, relation_ci)
        reverse = self.ordered(incumbent, challenger, relation_ic)
        return 0.5 * (forward - reverse)


def relation_for_arm(group, arm):
    if arm in {"P0_pair_token", "T0_structured_token"}:
        return torch.zeros_like(group["relation"])
    if arm == "C1_structured_relation_shuffle":
        return group["shuffled_relation"]
    if arm in {"P1_pair_relation", "T1_structured_relation"}:
        return group["relation"]
    raise ValueError(f"unknown DCSR arm: {arm}")


def alternative_evidence(model, base, relation):
    """Return four challenger-vs-incumbent evidence values per group."""
    batch_size = base.shape[0]
    challenger = base[:, 1:]
    incumbent = base[:, :1].expand(-1, TOPK - 1, -1)
    relation_ci = relation[:, 1:, 0]
    relation_ic = relation[:, 0, 1:]
    shape = (batch_size * (TOPK - 1), -1)
    return model(
        challenger.reshape(shape),
        incumbent.reshape(shape),
        relation_ci.reshape(shape),
        relation_ic.reshape(shape),
    ).reshape(batch_size, TOPK - 1)


def structured_top1_loss(alternative_scores, rewards):
    """Cost-sensitive structured hinge aligned with final top-1 argmax."""
    if alternative_scores.ndim != 2 or alternative_scores.shape[1] != TOPK - 1:
        raise ValueError("alternative scores must have shape (B, 4)")
    if rewards.shape != (alternative_scores.shape[0], TOPK):
        raise ValueError("structured rewards must have shape (B, 5)")
    active = rewards.max(dim=1).values.sub(rewards.min(dim=1).values).gt(1e-6)
    if not bool(active.any()):
        return alternative_scores.sum() * 0.0, active
    scores = torch.cat(
        (torch.zeros_like(alternative_scores[:, :1]), alternative_scores), dim=1
    )
    best_reward, target = rewards.max(dim=1)
    cost = best_reward[:, None] - rewards
    target_score = scores.gather(1, target[:, None]).squeeze(1)
    losses = (scores + cost).max(dim=1).values - target_score
    return losses[active].mean(), active


def build_groups(caches, manifests, model, token_meta, device, batch_size):
    groups = []
    module = common.SELECTOR_MODULE
    for cache, manifest in zip(caches, manifests):
        noise_seed = int(manifest["noise_seed"])
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = torch.arange(start, stop, dtype=torch.long)
            inputs = hard.model_inputs(cache, indices, device)
            reference = cache["reference_logits"][indices].to(
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                tokens = model.encode_tokens(**inputs)
                logits = reference + model.delta_head(tokens).squeeze(-1)
            valid = cache["candidate_reward_valid_mask"][indices].to(
                device=device, dtype=torch.bool
            )
            if bool(valid.sum(dim=1).lt(TOPK).any()):
                raise RuntimeError("DCSR requires at least five valid candidates")
            order = torch.argsort(
                logits.masked_fill(~valid, -torch.inf),
                dim=1,
                descending=True,
                stable=True,
            )[:, :TOPK].cpu()

            base_all = torch.cat((tokens.cpu(), logits.cpu().unsqueeze(-1)), dim=-1)
            top_base = base_all.gather(
                1, order[..., None].expand(-1, -1, base_all.shape[-1])
            )
            trajectories = cache["candidate_trajectories_8"][indices].float()
            top_trajectories = trajectories.gather(
                1,
                order[..., None, None].expand(-1, -1, 8, 3),
            )
            relations = module.pairwise_trajectory_relations(top_trajectories)
            for local, cache_index in enumerate(range(start, stop)):
                token = str(cache["tokens"][cache_index])
                ranks = order[local]
                relation = relations[local]
                generator = torch.Generator().manual_seed(
                    hard.stable_seed(token, noise_seed, "dcsr_relation_shuffle")
                )
                permutation = torch.randperm(TOPK, generator=generator)
                rewards = cache["candidate_rewards"][cache_index].float()
                components = cache["candidate_reward_components"][cache_index].float()
                if not bool(torch.isfinite(rewards[ranks]).all()):
                    raise RuntimeError("DCSR top5 contains a non-finite reward label")
                if not bool(torch.isfinite(components[ranks]).all()):
                    raise RuntimeError("DCSR top5 contains a non-finite component label")
                valid_rewards = rewards.masked_fill(
                    ~cache["candidate_reward_valid_mask"][cache_index].bool(),
                    -torch.inf,
                )
                meta = token_meta[token]
                groups.append(
                    {
                        "token": token,
                        "log_name": meta["log_name"],
                        "stratum": meta["stratum"],
                        "noise_seed": noise_seed,
                        "base": top_base[local],
                        "relation": relation,
                        "shuffled_relation": relation[permutation][:, permutation],
                        "top_indices": ranks.long(),
                        "top_rewards": rewards[ranks],
                        "top_components": components[ranks],
                        "all_rewards": valid_rewards,
                        "v3_index": int(ranks[0]),
                    }
                )
    return groups


def tensor_dataset(groups, group_indices, arm):
    base = torch.stack([groups[index]["base"] for index in group_indices])
    relation = torch.stack(
        [relation_for_arm(groups[index], arm) for index in group_indices]
    )
    rewards = torch.stack([groups[index]["top_rewards"] for index in group_indices])
    return base, relation, rewards


def fit_model(groups, arm, train_folds, fold_assignment, args, seed):
    group_indices = [
        index
        for index, group in enumerate(groups)
        if fold_assignment[group["log_name"]] in train_folds
    ]
    base, relation, rewards = tensor_dataset(groups, group_indices, arm)
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = Top1Comparator(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3
    )
    generator = torch.Generator().manual_seed(seed + 1)

    if arm in PAIRWISE_ARMS:
        delta = rewards[:, 1:] - rewards[:, :1]
        active = delta.abs().gt(1e-6)
        if not bool(active.any()):
            raise RuntimeError("pairwise DCSR arm has no informative boundary pairs")
        challenger = base[:, 1:][active]
        incumbent = base[:, :1].expand(-1, TOPK - 1, -1)[active]
        relation_ci = relation[:, 1:, 0][active]
        relation_ic = relation[:, 0, 1:][active]
        targets = delta[active].gt(0).float()
        weights = delta[active].abs()
        weights = weights / weights.mean().clamp_min(1e-6)
        count = len(targets)
        for _ in range(args.optimizer_steps):
            index = torch.randint(
                0, count, (args.pair_batch_size,), generator=generator
            )
            optimizer.zero_grad(set_to_none=True)
            evidence = model(
                challenger[index].to(device),
                incumbent[index].to(device),
                relation_ci[index].to(device),
                relation_ic[index].to(device),
            )
            loss = (
                F.binary_cross_entropy_with_logits(
                    evidence, targets[index].to(device), reduction="none"
                )
                * weights[index].to(device)
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite pairwise DCSR loss")
            loss.backward()
            optimizer.step()
    else:
        active = rewards.max(dim=1).values.sub(rewards.min(dim=1).values).gt(1e-6)
        base, relation, rewards = base[active], relation[active], rewards[active]
        if not len(rewards):
            raise RuntimeError("structured DCSR arm has no informative groups")
        for _ in range(args.optimizer_steps):
            index = torch.randint(
                0,
                len(rewards),
                (args.structured_batch_size,),
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            evidence = alternative_evidence(
                model, base[index].to(device), relation[index].to(device)
            )
            loss, _ = structured_top1_loss(evidence, rewards[index].to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite structured DCSR loss")
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_scores(model, groups, arm, group_indices, device, batch_size):
    output = {}
    for start in range(0, len(group_indices), batch_size):
        indices = group_indices[start : start + batch_size]
        base, relation, _ = tensor_dataset(groups, indices, arm)
        evidence = alternative_evidence(
            model, base.to(device), relation.to(device)
        ).cpu()
        if not bool(torch.isfinite(evidence).all()):
            raise RuntimeError("non-finite DCSR OOF evidence")
        scores = torch.cat((torch.zeros_like(evidence[:, :1]), evidence), dim=1)
        output.update(
            {group_id: scores[local] for local, group_id in enumerate(indices)}
        )
    return output


def selections_at_threshold(scores, group_indices, threshold):
    selected = {}
    for group_id in group_indices:
        alternatives = scores[group_id][1:]
        challenger = int(alternatives.argmax()) + 1
        selected[group_id] = (
            challenger if float(alternatives[challenger - 1]) > threshold else 0
        )
    return selected


def calibration_summary(groups, selected, group_indices):
    rewards = torch.stack([groups[index]["top_rewards"] for index in group_indices])
    components = torch.stack(
        [groups[index]["top_components"] for index in group_indices]
    )
    ranks = torch.tensor([selected[index] for index in group_indices], dtype=torch.long)
    row = torch.arange(len(group_indices))
    chosen_reward = rewards[row, ranks]
    baseline_reward = rewards[:, 0]
    chosen_safety = components[row, ranks, :2].prod(dim=1)
    baseline_safety = components[:, 0, :2].prod(dim=1)
    common = torch.tensor(
        [groups[index]["stratum"] == "common" for index in group_indices]
    )
    rare = ~common

    def mean(values, mask):
        return float(values[mask].mean())

    common_pdm = mean(chosen_reward, common)
    rare_pdm = mean(chosen_reward, rare)
    base_common = mean(baseline_reward, common)
    base_rare = mean(baseline_reward, rare)
    return {
        "common_pdm": common_pdm,
        "rare_pdm": rare_pdm,
        "equal_stratum_pdm": 0.5 * (common_pdm + rare_pdm),
        "v3_common_pdm": base_common,
        "v3_rare_pdm": base_rare,
        "v3_equal_stratum_pdm": 0.5 * (base_common + base_rare),
        "common_hard_safety": mean(chosen_safety, common),
        "rare_hard_safety": mean(chosen_safety, rare),
        "v3_common_hard_safety": mean(baseline_safety, common),
        "v3_rare_hard_safety": mean(baseline_safety, rare),
        "common_degraded_fraction": float(
            chosen_reward[common].lt(baseline_reward[common] - 1e-6).float().mean()
        ),
        "override_coverage": float(ranks.ne(0).float().mean()),
    }


def calibration_feasible(metrics):
    return (
        metrics["common_pdm"] >= metrics["v3_common_pdm"] - 0.002
        and metrics["rare_pdm"] >= metrics["v3_rare_pdm"]
        and metrics["common_hard_safety"]
        >= metrics["v3_common_hard_safety"] - 0.002
        and metrics["rare_hard_safety"]
        >= metrics["v3_rare_hard_safety"] - 0.002
        and metrics["common_degraded_fraction"] <= 0.05
    )


def calibrate_threshold(scores, groups, calibration_indices):
    maximum = torch.stack([scores[index][1:].max() for index in calibration_indices])
    quantiles = torch.quantile(
        maximum.clamp_min(0.0), torch.linspace(0.0, 1.0, 101)
    )
    thresholds = sorted({0.0, *(float(value) for value in quantiles)})
    thresholds.append(float("inf"))
    best = None
    for threshold in thresholds:
        selected = selections_at_threshold(scores, calibration_indices, threshold)
        metrics = calibration_summary(groups, selected, calibration_indices)
        if not calibration_feasible(metrics):
            continue
        candidate = (metrics["equal_stratum_pdm"], threshold, metrics)
        if (
            best is None
            or candidate[0] > best[0] + 1e-12
            or (abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] > best[1])
        ):
            best = candidate
    if best is None:
        raise RuntimeError("no feasible DCSR threshold, including keep-all")
    return best[1], {
        "threshold": None if math.isinf(best[1]) else best[1],
        "keep_all": math.isinf(best[1]),
        "candidate_thresholds": len(thresholds),
        "selection": best[2],
    }


def agreement(mapping):
    hits, count = 0, 0
    for values in mapping.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                hits += int(values[left] == values[right])
                count += 1
    if not count:
        raise RuntimeError("noise-view agreement requires repeated tokens")
    return float(hits / count)


def selection_metrics(groups, selections, repetitions, seed):
    rows = []
    per_log = defaultdict(list)
    selected_by_token = defaultdict(list)
    v3_by_token = defaultdict(list)
    for group_id, group in enumerate(groups):
        rank = selections[group_id]
        reward = float(group["top_rewards"][rank])
        baseline = float(group["top_rewards"][0])
        safety = float(group["top_components"][rank, :2].prod())
        baseline_safety = float(group["top_components"][0, :2].prod())
        oracle_rank = int(group["top_rewards"].argmax())
        row = {
            "reward": reward,
            "baseline": baseline,
            "gain": reward - baseline,
            "safety": safety,
            "baseline_safety": baseline_safety,
            "override": rank != 0,
            "oracle_match": rank == oracle_rank,
            "available_positive_regret": max(
                0.0, float(group["top_rewards"].max()) - baseline
            ),
            "stratum": group["stratum"],
        }
        rows.append(row)
        per_log[group["log_name"]].append(row["gain"])
        selected_by_token[group["token"]].append(int(group["top_indices"][rank]))
        v3_by_token[group["token"]].append(group["v3_index"])

    def stratum_metrics(stratum):
        subset = [row for row in rows if row["stratum"] == stratum]
        overrides = [row for row in subset if row["override"]]
        available = sum(row["available_positive_regret"] for row in subset)
        recovered = sum(max(0.0, row["gain"]) for row in subset)
        return {
            "pdm": float(np.mean([row["reward"] for row in subset])),
            "v3_pdm": float(np.mean([row["baseline"] for row in subset])),
            "hard_safety": float(np.mean([row["safety"] for row in subset])),
            "v3_hard_safety": float(
                np.mean([row["baseline_safety"] for row in subset])
            ),
            "override_coverage": len(overrides) / len(subset),
            "override_precision": (
                float(np.mean([row["gain"] > 1e-6 for row in overrides]))
                if overrides
                else None
            ),
            "degraded_fraction": float(
                np.mean([row["gain"] < -1e-6 for row in subset])
            ),
            "false_accept_regret": float(
                sum(max(0.0, -row["gain"]) for row in overrides)
            ),
            "positive_regret_recovery": recovered / available if available > 0 else 0.0,
            "top5_oracle_match": float(
                np.mean([row["oracle_match"] for row in subset])
            ),
        }

    common_row = stratum_metrics("common")
    rare_row = stratum_metrics("rare")
    common_pdm, rare_pdm = common_row["pdm"], rare_row["pdm"]
    base_common, base_rare = common_row["v3_pdm"], rare_row["v3_pdm"]
    log_gain = {name: float(np.mean(values)) for name, values in per_log.items()}
    return {
        "common": common_row,
        "rare": rare_row,
        "common_pdm": common_pdm,
        "rare_pdm": rare_pdm,
        "equal_stratum_pdm": 0.5 * (common_pdm + rare_pdm),
        "v3_common_pdm": base_common,
        "v3_rare_pdm": base_rare,
        "v3_equal_stratum_pdm": 0.5 * (base_common + base_rare),
        "gain_equal_stratum_pdm": 0.5
        * ((common_pdm - base_common) + (rare_pdm - base_rare)),
        "common_hard_safety": common_row["hard_safety"],
        "rare_hard_safety": rare_row["hard_safety"],
        "v3_common_hard_safety": common_row["v3_hard_safety"],
        "v3_rare_hard_safety": rare_row["v3_hard_safety"],
        "common_degraded_fraction": common_row["degraded_fraction"],
        "selected_noise_view_agreement": agreement(selected_by_token),
        "v3_noise_view_agreement": agreement(v3_by_token),
        "log_bootstrap_gain": hard.bootstrap_mean(
            log_gain.values(), repetitions, seed
        ),
        "per_log_gain": log_gain,
    }


def eligibility_gates(metrics):
    return {
        "equal_stratum_gain_at_least_0005": metrics[
            "gain_equal_stratum_pdm"
        ]
        >= 0.005,
        "common_floor": metrics["common_pdm"]
        >= metrics["v3_common_pdm"] - 0.005,
        "rare_floor": metrics["rare_pdm"] >= metrics["v3_rare_pdm"],
        "common_hard_safety_floor": metrics["common_hard_safety"]
        >= metrics["v3_common_hard_safety"] - 0.002,
        "rare_hard_safety_floor": metrics["rare_hard_safety"]
        >= metrics["v3_rare_hard_safety"] - 0.002,
        "common_degraded_fraction_at_most_010": metrics[
            "common_degraded_fraction"
        ]
        <= 0.10,
        "log_bootstrap_gain_lower_positive": metrics["log_bootstrap_gain"][
            "lower_95"
        ]
        > 0.0,
        "noise_agreement_floor": metrics["selected_noise_view_agreement"]
        >= metrics["v3_noise_view_agreement"] - 0.02,
    }


def boundary_metrics(groups, score_by_group):
    pooled_scores = defaultdict(list)
    pooled_targets = defaultdict(list)
    per_log_scores = defaultdict(list)
    per_log_targets = defaultdict(list)
    for group_id, group in enumerate(groups):
        delta = group["top_rewards"][1:] - group["top_rewards"][0]
        active = delta.abs().gt(1e-6)
        scores = score_by_group[group_id][1:][active]
        targets = delta[active].gt(0).float()
        for key in ("overall", group["stratum"]):
            pooled_scores[key].append(scores)
            pooled_targets[key].append(targets)
        per_log_scores[group["log_name"]].append(scores)
        per_log_targets[group["log_name"]].append(targets)
    auc = {}
    for key in ("overall", "common", "rare"):
        value = hard.binary_auc(
            torch.cat(pooled_scores[key]), torch.cat(pooled_targets[key])
        )
        if value is None:
            raise RuntimeError(f"boundary AUC is undefined for {key}")
        auc[key] = value
    per_log = {}
    for name in per_log_scores:
        value = hard.binary_auc(
            torch.cat(per_log_scores[name]), torch.cat(per_log_targets[name])
        )
        if value is not None:
            per_log[name] = value
    return auc, per_log


def log_bootstrap_delta(target, reference, repetitions, seed):
    shared = sorted(set(target) & set(reference))
    if not shared:
        raise RuntimeError("paired-log comparison has no shared logs")
    return hard.bootstrap_mean(
        [target[name] - reference[name] for name in shared], repetitions, seed
    )


def selection_comparison(reports, target, reference, repetitions, seed):
    target_selection = reports[target]["selection"]
    reference_selection = reports[reference]["selection"]
    gain = (
        target_selection["equal_stratum_pdm"]
        - reference_selection["equal_stratum_pdm"]
    )
    bootstrap = log_bootstrap_delta(
        target_selection["per_log_gain"],
        reference_selection["per_log_gain"],
        repetitions,
        seed,
    )
    return {
        "equal_stratum_pdm_gain": gain,
        "paired_log_bootstrap": bootstrap,
        "passed": gain >= 0.002 and bootstrap["lower_95"] > 0.0,
    }


def relation_signal(reports, repetitions, seed):
    comparisons = {}
    passed = True
    for offset, reference in enumerate(
        ("T0_structured_token", "C1_structured_relation_shuffle")
    ):
        target_auc = reports["T1_structured_relation"]["boundary_auc"]
        reference_auc = reports[reference]["boundary_auc"]
        bootstrap = log_bootstrap_delta(
            reports["T1_structured_relation"]["per_log_boundary_auc"],
            reports[reference]["per_log_boundary_auc"],
            repetitions,
            seed + offset,
        )
        gates = {
            "overall_auc_gain_at_least_001": target_auc["overall"]
            - reference_auc["overall"]
            >= 0.01,
            "paired_log_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
            "common_delta_floor": target_auc["common"]
            - reference_auc["common"]
            >= -0.01,
            "rare_delta_floor": target_auc["rare"] - reference_auc["rare"]
            >= -0.01,
        }
        comparisons[reference] = {
            "delta_overall": target_auc["overall"] - reference_auc["overall"],
            "delta_common": target_auc["common"] - reference_auc["common"],
            "delta_rare": target_auc["rare"] - reference_auc["rare"],
            "paired_log_bootstrap": bootstrap,
            "gates": gates,
        }
        passed = passed and all(gates.values())
    return {"passed": passed, "comparisons": comparisons}


def choose_decision(reports, signal, comparisons):
    t1_eligible = all(reports["T1_structured_relation"]["gates"].values())
    if (
        t1_eligible
        and signal["passed"]
        and all(
            comparisons[name]["passed"]
            for name in ("T1_over_P1", "T1_over_T0", "T1_over_C1")
        )
    ):
        return "AUTHORIZE_DCSR_RELATIONAL_V4", "T1_structured_relation"
    t0_eligible = all(reports["T0_structured_token"]["gates"].values())
    if t0_eligible and comparisons["T0_over_P0"]["passed"]:
        return "AUTHORIZE_DCSR_TOKEN_V4", "T0_structured_token"
    return "STOP_CURRENT_FRAME_SELECTOR_AND_AUDIT_HISTORY", None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--prior-hard-pair-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=1024)
    parser.add_argument("--structured-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = (
        args.optimizer_steps,
        args.pair_batch_size,
        args.structured_batch_size,
        args.feature_batch_size,
        args.hidden_dim,
        args.bootstrap_repetitions,
    )
    if min(budgets) <= 0:
        raise ValueError("all DCSR budgets must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, "train") for path in args.train_cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    if len(caches) != 3 or [int(row["noise_seed"]) for row in manifests] != [0, 1, 2]:
        raise RuntimeError("DCSR requires train noise seeds 0,1,2")
    if any(cache.get("schema_version") != 4 for cache in caches):
        raise RuntimeError("DCSR requires the existing schema-v4 train caches")
    pair_rows, rare_audit, pair_path, audit_path = trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    token_meta = {}
    for row in pair_rows:
        for key, stratum in (("rare_token", "rare"), ("common_token", "common")):
            token = str(row[key])
            value = {"stratum": stratum, "log_name": str(row["log_name"])}
            if token in token_meta and token_meta[token] != value:
                raise RuntimeError(f"ambiguous token provenance: {token}")
            token_meta[token] = value

    prior_path = args.prior_hard_pair_gate.expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    if (
        prior.get("status") != "PASS"
        or prior.get("decision") != "AUTHORIZE_LOCKED_TOP1_OBJECTIVE"
        or prior.get("selected_arm") is not None
        or prior.get("development_or_certification_consumed") is not False
    ):
        raise RuntimeError("prior hard-pair gate does not authorize DCSR")

    model, checkpoint_payload, checkpoint_path, checkpoint_sha = hard.load_checkpoint(
        args.v3_checkpoint,
        args.v3_checkpoint_sha256,
        caches[0],
        device,
    )
    groups = build_groups(
        caches,
        manifests,
        model,
        token_meta,
        device,
        args.feature_batch_size,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fold_assignment = hard.assign_log_folds(groups)
    reports = {
        arm: {"folds": [], "scores": {}, "selected": {}} for arm in ARMS
    }

    for test_fold in range(NUM_FOLDS):
        calibration_fold = (test_fold + 1) % NUM_FOLDS
        train_folds = set(range(NUM_FOLDS)) - {test_fold, calibration_fold}
        calibration_indices = [
            index
            for index, group in enumerate(groups)
            if fold_assignment[group["log_name"]] == calibration_fold
        ]
        test_indices = [
            index
            for index, group in enumerate(groups)
            if fold_assignment[group["log_name"]] == test_fold
        ]
        for arm_index, arm in enumerate(ARMS):
            model = fit_model(
                groups,
                arm,
                train_folds,
                fold_assignment,
                args,
                args.seed + test_fold * 1000 + arm_index,
            )
            scores = predict_scores(
                model,
                groups,
                arm,
                calibration_indices + test_indices,
                device,
                args.feature_batch_size,
            )
            threshold, calibration = calibrate_threshold(
                scores, groups, calibration_indices
            )
            selected = selections_at_threshold(scores, test_indices, threshold)
            reports[arm]["scores"].update(
                {index: scores[index] for index in test_indices}
            )
            reports[arm]["selected"].update(selected)
            reports[arm]["folds"].append(
                {
                    "test_fold": test_fold,
                    "calibration_fold": calibration_fold,
                    "train_folds": sorted(train_folds),
                    "test_groups": len(test_indices),
                    "calibration": calibration,
                }
            )
            del model

    for arm_index, arm in enumerate(ARMS):
        if len(reports[arm]["scores"]) != len(groups):
            raise RuntimeError(f"{arm} did not produce complete OOF scores")
        score_list = {
            index: reports[arm]["scores"][index] for index in range(len(groups))
        }
        selected = {
            index: reports[arm]["selected"][index] for index in range(len(groups))
        }
        boundary_auc, per_log_auc = boundary_metrics(groups, score_list)
        reports[arm]["boundary_auc"] = boundary_auc
        reports[arm]["per_log_boundary_auc"] = per_log_auc
        reports[arm]["selection"] = selection_metrics(
            groups,
            selected,
            args.bootstrap_repetitions,
            args.seed + 20000 + arm_index,
        )
        reports[arm]["gates"] = eligibility_gates(reports[arm]["selection"])
        reports[arm].pop("scores")
        reports[arm].pop("selected")

    comparisons = {
        "T0_over_P0": selection_comparison(
            reports,
            "T0_structured_token",
            "P0_pair_token",
            args.bootstrap_repetitions,
            args.seed + 30000,
        ),
        "T1_over_P1": selection_comparison(
            reports,
            "T1_structured_relation",
            "P1_pair_relation",
            args.bootstrap_repetitions,
            args.seed + 30001,
        ),
        "T1_over_T0": selection_comparison(
            reports,
            "T1_structured_relation",
            "T0_structured_token",
            args.bootstrap_repetitions,
            args.seed + 30002,
        ),
        "T1_over_C1": selection_comparison(
            reports,
            "T1_structured_relation",
            "C1_structured_relation_shuffle",
            args.bootstrap_repetitions,
            args.seed + 30003,
        ),
    }
    signal = relation_signal(
        reports, args.bootstrap_repetitions, args.seed + 40000
    )
    decision, selected_arm = choose_decision(reports, signal, comparisons)

    v3 = float(np.mean([float(group["top_rewards"][0]) for group in groups]))
    oracle5 = float(
        np.mean([float(group["top_rewards"].max()) for group in groups])
    )
    oracle20 = float(
        np.mean([float(group["all_rewards"].max()) for group in groups])
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "dcsr_locked_top1_train_only_log_cv_audit_v1",
        "development_or_certification_consumed": False,
        "decision": decision,
        "selected_arm": selected_arm,
        "historical_nonduplication": {
            "ivps_pcra_role": "non-promotable conceptual controls only",
            "difference": "DCSR has no independent proposal: one top5 evidence head both nominates and accepts an override of rare-V3 rank1",
            "old_rollout_or_synthetic_data_consumed": False,
        },
        "locked_contract": {
            "incumbent": "rare_tuned_V3_rank1",
            "topk": TOPK,
            "reward_or_components_as_input": False,
            "inference": "argmax challenger evidence followed by one global calibrated threshold",
            "interaction_features": False,
            "primary_metric": "PDM",
            "cached_hard_safety": "no_at_fault_collisions * drivable_area_compliance",
        },
        "headroom": {
            "v3_selected_pdm": v3,
            "oracle5_pdm": oracle5,
            "oracle20_pdm": oracle20,
            "top5_headroom_fraction": (
                (oracle5 - v3) / (oracle20 - v3) if oracle20 > v3 else 0.0
            ),
        },
        "budgets": {
            "groups": len(groups),
            "folds": NUM_FOLDS,
            "train_folds_per_outer_fold": 3,
            "separate_calibration_fold": True,
            "optimizer_steps_per_arm_fold": args.optimizer_steps,
            "pair_batch_size": args.pair_batch_size,
            "structured_group_batch_size": args.structured_batch_size,
            "candidate_edges_per_structured_group": TOPK - 1,
            "hidden_dim": args.hidden_dim,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "arms": reports,
        "relation_signal": signal,
        "method_comparisons": comparisons,
        "decision_semantics": {
            "AUTHORIZE_DCSR_RELATIONAL_V4": "materialize only the structured relation arm",
            "AUTHORIZE_DCSR_TOKEN_V4": "materialize only the structured frozen-V3-token arm",
            "STOP_CURRENT_FRAME_SELECTOR_AND_AUDIT_HISTORY": "permanently stop current-frame selector/objective variants and test temporal information",
        },
        "provenance": {
            "v3_checkpoint": str(checkpoint_path),
            "v3_checkpoint_sha256": checkpoint_sha,
            "v3_checkpoint_method": checkpoint_payload.get("method"),
            "prior_hard_pair_gate": str(prior_path),
            "prior_hard_pair_gate_sha256": common.sha256_file(prior_path),
            "pair_manifest": str(pair_path),
            "pair_manifest_sha256": common.sha256_file(pair_path),
            "rare_data_audit": str(audit_path),
            "rare_data_audit_sha256": common.sha256_file(audit_path),
            "train_manifests": list(manifests),
            "implementation_files": {
                str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
                str(Path(common.__file__).resolve()): common.sha256_file(
                    Path(common.__file__).resolve()
                ),
                str(Path(hard.__file__).resolve()): common.sha256_file(
                    Path(hard.__file__).resolve()
                ),
            },
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "selected_arm": selected_arm,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
