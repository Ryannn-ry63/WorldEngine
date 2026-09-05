#!/usr/bin/env python3
"""Train-only hard-top5 information and selective-reranking audit.

The frozen rare-tuned V3 defines the candidate shortlist.  Rewards are used
only as labels/metrics after the shortlist has been locked; they are never
model inputs.  All learned results are out-of-fold by audited log, with a
separate calibration fold for the selective override threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grpo_selector_v3_cached_common as common
import train_trajectory_set_reasoner_grpo as trainer


NUM_FOLDS = 5
TOPK = 5
PAIR_I, PAIR_J = torch.triu_indices(TOPK, TOPK, offset=1)
ARMS = ("H0_token", "H1_relation", "H2_interaction", "S1_relation_shuffle", "S2_interaction_shuffle")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def topk_pair_ranks(k: int = TOPK):
    """Return the locked rank pairs; deliberately has no reward argument."""
    return torch.triu_indices(k, k, offset=1)


class AntisymmetricComparator(nn.Module):
    """Capacity-matched ordered-pair MLP with antisymmetry by construction."""

    def __init__(self, candidate_dim=257, relation_dim=41, interaction_dim=48, hidden=128):
        super().__init__()
        self.candidate_norm = nn.LayerNorm(candidate_dim)
        self.relation_norm = nn.LayerNorm(relation_dim)
        self.interaction_norm = nn.LayerNorm(interaction_dim)
        width = 2 * candidate_dim + relation_dim + 2 * interaction_dim
        self.network = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def ordered(self, left, right, relation, left_interaction, right_interaction):
        values = torch.cat(
            (
                self.candidate_norm(left), self.candidate_norm(right),
                self.relation_norm(relation),
                self.interaction_norm(left_interaction), self.interaction_norm(right_interaction),
            ), dim=-1,
        )
        return self.network(values).squeeze(-1)

    def forward(self, left, right, relation_lr, relation_rl, left_interaction, right_interaction):
        forward = self.ordered(left, right, relation_lr, left_interaction, right_interaction)
        reverse = self.ordered(right, left, relation_rl, right_interaction, left_interaction)
        return 0.5 * (forward - reverse)


def binary_auc(scores: torch.Tensor, targets: torch.Tensor):
    scores = scores.detach().double().cpu()
    targets = targets.detach().bool().cpu()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    starts = counts.cumsum(0) - counts
    average_ranks = starts.double() + (counts.double() + 1.0) / 2.0
    positive_rank_sum = average_ranks[inverse][targets[order]].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def graph_challenger(evidence: torch.Tensor, incumbent: int = 0) -> int:
    """Borda winner from a complete antisymmetric preference graph."""
    if evidence.ndim != 2 or evidence.shape[0] != evidence.shape[1]:
        raise ValueError("evidence must be a square matrix")
    probability = torch.sigmoid(evidence)
    scores = probability.sum(dim=1) - 0.5
    maximum = scores.max()
    winners = torch.nonzero(scores.eq(maximum), as_tuple=False).flatten().tolist()
    return incumbent if incumbent in winners else int(min(winners))


def bootstrap_mean(values, repetitions, seed):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        raise RuntimeError("bootstrap requires at least one audited log")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[samples].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "repetitions": int(repetitions),
        "num_logs": int(len(values)),
    }


def assign_log_folds(groups):
    counts = Counter(group["log_name"] for group in groups)
    loads = [0] * NUM_FOLDS
    assignment = {}
    for name in sorted(counts, key=lambda value: (-counts[value], value)):
        fold = min(range(NUM_FOLDS), key=lambda value: (loads[value], value))
        assignment[name] = fold
        loads[fold] += counts[name]
    if any(value == 0 for value in loads):
        raise RuntimeError("hard-pair audit requires five non-empty log folds")
    return assignment


def load_checkpoint(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"checkpoint SHA256 drifted: {path}")
    payload = torch.load(path, map_location="cpu")
    architecture = payload.get("selector_architecture", "scene_conditioned_v3")
    if architecture not in {"scene_conditioned_v3", "v3"}:
        raise RuntimeError(f"incumbent must be V3, got {architecture}")
    config = payload.get("scene_selector_config")
    if config:
        model = common.model_from_config(config)
    else:
        model, config = common.model_from_cache(cache, "full", "scene_conditioned_v3")
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model.to(device).eval()
    return model, payload, path, actual_sha


def model_inputs(cache, indices, device):
    values = common.batch_inputs(cache, indices, device)
    values.pop("frozen_track_states", None)
    values.pop("frozen_track_classes", None)
    values.pop("frozen_track_mask", None)
    return values


def build_groups(caches, manifests, model, token_meta, device, batch_size):
    groups = []
    module = common.SELECTOR_MODULE
    for cache, manifest in zip(caches, manifests):
        noise_seed = int(manifest["noise_seed"])
        strata_indices = defaultdict(list)
        for index, token in enumerate(cache["tokens"]):
            strata_indices[token_meta[str(token)]["stratum"]].append(index)
        donors = {}
        for stratum, indices in strata_indices.items():
            ordered = sorted(indices, key=lambda index: str(cache["tokens"][index]))
            if len(ordered) < 2:
                raise RuntimeError(f"cannot build matched interaction shuffle for {stratum}")
            for position, index in enumerate(ordered):
                offset = 1 + stable_seed(cache["tokens"][index], noise_seed, "track_donor") % (len(ordered) - 1)
                donors[index] = ordered[(position + offset) % len(ordered)]

        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = torch.arange(start, stop, dtype=torch.long)
            inputs = model_inputs(cache, indices, device)
            reference = cache["reference_logits"][indices].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                tokens = model.encode_tokens(**inputs)
                logits = reference + model.delta_head(tokens).squeeze(-1)
            valid = cache["candidate_reward_valid_mask"][indices].to(device=device, dtype=torch.bool)
            if bool(valid.sum(dim=1).lt(8).any()):
                raise RuntimeError("hard-pair audit requires at least eight valid candidates")
            masked_logits = logits.masked_fill(~valid, -torch.inf)
            order8 = torch.argsort(masked_logits, dim=1, descending=True, stable=True)[:, :8].cpu()
            tokens = tokens.cpu()
            logits = logits.cpu()
            for local, cache_index in enumerate(range(start, stop)):
                token = str(cache["tokens"][cache_index])
                ranks8 = order8[local]
                ranks = ranks8[:TOPK]
                trajectories = cache["candidate_trajectories_8"][cache_index, ranks].float().unsqueeze(0)
                relation = module.pairwise_trajectory_relations(trajectories).squeeze(0)
                tracks = cache["frozen_track_states"][cache_index].float().unsqueeze(0)
                track_mask = cache["frozen_track_mask"][cache_index].bool().unsqueeze(0)
                interaction = module.interaction_probe_features(trajectories, tracks, track_mask).squeeze(0)
                donor = donors[cache_index]
                shuffled_interaction = module.interaction_probe_features(
                    trajectories,
                    cache["frozen_track_states"][donor].float().unsqueeze(0),
                    cache["frozen_track_mask"][donor].bool().unsqueeze(0),
                ).squeeze(0)
                generator = torch.Generator().manual_seed(stable_seed(token, noise_seed, "relation_shuffle"))
                permutation = torch.randperm(TOPK, generator=generator)
                shuffled_relation = relation[permutation][:, permutation]
                base = torch.cat((tokens[local, ranks], logits[local, ranks, None]), dim=-1)
                rewards = cache["candidate_rewards"][cache_index].float()
                reward_valid = cache["candidate_reward_valid_mask"][cache_index].bool()
                components = cache["candidate_reward_components"][cache_index].float()
                meta = token_meta[token]
                groups.append(
                    {
                        "token": token, "log_name": meta["log_name"], "stratum": meta["stratum"],
                        "noise_seed": noise_seed, "base": base, "relation": relation,
                        "interaction": interaction, "shuffled_relation": shuffled_relation,
                        "shuffled_interaction": shuffled_interaction,
                        "top_indices": ranks.long(), "top_rewards": rewards[ranks],
                        "top_components": components[ranks],
                        "top8_rewards": rewards[ranks8],
                        "all_rewards": rewards.masked_fill(~reward_valid, -torch.inf),
                        "v3_index": int(ranks[0]),
                    }
                )
    return groups


def arm_blocks(group, arm):
    zero_relation = torch.zeros_like(group["relation"])
    zero_interaction = torch.zeros_like(group["interaction"])
    if arm == "H0_token":
        return zero_relation, zero_interaction
    if arm == "H1_relation":
        return group["relation"], zero_interaction
    if arm == "H2_interaction":
        return group["relation"], group["interaction"]
    if arm == "S1_relation_shuffle":
        return group["shuffled_relation"], zero_interaction
    if arm == "S2_interaction_shuffle":
        return group["relation"], group["shuffled_interaction"]
    raise ValueError(arm)


def pair_examples(groups, arm, allowed_folds, fold_assignment):
    pieces = defaultdict(list)
    for group_id, group in enumerate(groups):
        if fold_assignment[group["log_name"]] not in allowed_folds:
            continue
        relation, interaction = arm_blocks(group, arm)
        delta = group["top_rewards"][PAIR_I] - group["top_rewards"][PAIR_J]
        active = delta.abs().gt(1e-6)
        if not bool(active.any()):
            continue
        i, j = PAIR_I[active], PAIR_J[active]
        pieces["left"].append(group["base"][i])
        pieces["right"].append(group["base"][j])
        pieces["relation_lr"].append(relation[i, j])
        pieces["relation_rl"].append(relation[j, i])
        pieces["left_interaction"].append(interaction[i])
        pieces["right_interaction"].append(interaction[j])
        pieces["target"].append(delta[active].gt(0).float())
        pieces["weight"].append(delta[active].abs())
        pieces["group_id"].append(torch.full((int(active.sum()),), group_id, dtype=torch.long))
    if not pieces:
        raise RuntimeError("no informative locked-top5 pairs")
    return {key: torch.cat(value) for key, value in pieces.items()}


def comparator_call(model, examples, indices, device):
    keys = ("left", "right", "relation_lr", "relation_rl", "left_interaction", "right_interaction")
    return model(*(examples[key][indices].to(device) for key in keys))


def fit_comparator(groups, arm, train_folds, fold_assignment, args, seed):
    examples = pair_examples(groups, arm, train_folds, fold_assignment)
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = AntisymmetricComparator(hidden=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    weights = examples["weight"] / examples["weight"].mean().clamp_min(1e-6)
    for _ in range(args.epochs):
        order = torch.randperm(len(weights), generator=generator)
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = comparator_call(model, examples, index, device)
            loss = (F.binary_cross_entropy_with_logits(logits, examples["target"][index].to(device), reduction="none") * weights[index].to(device)).mean()
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def group_evidence(model, group, arm, device):
    relation, interaction = arm_blocks(group, arm)
    left = group["base"][:, None].expand(TOPK, TOPK, -1).reshape(TOPK * TOPK, -1)
    right = group["base"][None].expand(TOPK, TOPK, -1).reshape(TOPK * TOPK, -1)
    values = model(
        left.to(device), right.to(device), relation.reshape(TOPK * TOPK, -1).to(device),
        relation.transpose(0, 1).reshape(TOPK * TOPK, -1).to(device),
        interaction[:, None].expand(TOPK, TOPK, -1).reshape(TOPK * TOPK, -1).to(device),
        interaction[None].expand(TOPK, TOPK, -1).reshape(TOPK * TOPK, -1).to(device),
    ).cpu().reshape(TOPK, TOPK)
    return 0.5 * (values - values.T)


@torch.no_grad()
def predict_group_evidence(model, groups, arm, group_indices, device, batch_size):
    """Evaluate complete top-5 graphs in batches to avoid tiny GPU launches."""
    output = {}
    for start in range(0, len(group_indices), batch_size):
        indices = group_indices[start : start + batch_size]
        base = torch.stack([groups[index]["base"] for index in indices])
        blocks = [arm_blocks(groups[index], arm) for index in indices]
        relation = torch.stack([block[0] for block in blocks])
        interaction = torch.stack([block[1] for block in blocks])
        count = len(indices)
        left = base[:, :, None].expand(count, TOPK, TOPK, -1).reshape(count * TOPK * TOPK, -1)
        right = base[:, None].expand(count, TOPK, TOPK, -1).reshape(count * TOPK * TOPK, -1)
        left_interaction = interaction[:, :, None].expand(count, TOPK, TOPK, -1).reshape(count * TOPK * TOPK, -1)
        right_interaction = interaction[:, None].expand(count, TOPK, TOPK, -1).reshape(count * TOPK * TOPK, -1)
        values = model(
            left.to(device), right.to(device),
            relation.reshape(count * TOPK * TOPK, -1).to(device),
            relation.transpose(1, 2).reshape(count * TOPK * TOPK, -1).to(device),
            left_interaction.to(device), right_interaction.to(device),
        ).cpu().reshape(count, TOPK, TOPK)
        values = 0.5 * (values - values.transpose(1, 2))
        output.update({index: values[local] for local, index in enumerate(indices)})
    return output


def fit_calibrator(matrices, groups, calibration_indices, device, seed):
    evidence, targets, weights = [], [], []
    for group_id in calibration_indices:
        group = groups[group_id]
        matrix = matrices[group_id]
        challenger = graph_challenger(matrix)
        if challenger == 0:
            continue
        delta = group["top_rewards"][challenger] - group["top_rewards"][0]
        if abs(float(delta)) <= 1e-6:
            continue
        evidence.append(matrix[challenger, 0])
        targets.append(float(delta > 0))
        weights.append(abs(float(delta)))
    if not evidence:
        return {"scale": 1.0, "bias": 0.0, "examples": 0, "fallback": True}
    x = torch.stack(evidence).to(device)
    y = torch.tensor(targets, device=device)
    w = torch.tensor(weights, device=device)
    w = w / w.mean().clamp_min(1e-6)
    torch.manual_seed(seed)
    raw_scale = nn.Parameter(torch.tensor(0.5413, device=device))
    bias = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam((raw_scale, bias), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        scale = F.softplus(raw_scale) + 1e-4
        loss = (F.binary_cross_entropy_with_logits(scale * x + bias, y, reduction="none") * w).mean()
        loss.backward()
        optimizer.step()
    return {"scale": float((F.softplus(raw_scale) + 1e-4).detach().cpu()), "bias": float(bias.detach().cpu()), "examples": len(evidence), "fallback": False}


def selected_metrics(groups, selected, repetitions, seed):
    rows = []
    per_log = defaultdict(list)
    by_token = defaultdict(list)
    v3_by_token = defaultdict(list)
    for group, rank in zip(groups, selected):
        reward = float(group["top_rewards"][rank])
        baseline = float(group["top_rewards"][0])
        safety = float(group["top_components"][rank, :2].prod())
        baseline_safety = float(group["top_components"][0, :2].prod())
        actual = int(group["top_indices"][rank])
        row = {
            "reward": reward, "baseline": baseline,
            "hard_safety": safety, "baseline_hard_safety": baseline_safety,
            "stratum": group["stratum"], "log": group["log_name"],
        }
        rows.append(row)
        per_log[group["log_name"]].append(reward - baseline)
        by_token[group["token"]].append(actual)
        v3_by_token[group["token"]].append(group["v3_index"])
    def mean_for(key, value, field="reward"):
        values = [row[field] for row in rows if row[key] == value]
        return float(np.mean(values))
    common = mean_for("stratum", "common")
    rare = mean_for("stratum", "rare")
    base_common = mean_for("stratum", "common", "baseline")
    base_rare = mean_for("stratum", "rare", "baseline")
    common_safety = mean_for("stratum", "common", "hard_safety")
    rare_safety = mean_for("stratum", "rare", "hard_safety")
    base_common_safety = mean_for("stratum", "common", "baseline_hard_safety")
    base_rare_safety = mean_for("stratum", "rare", "baseline_hard_safety")
    log_deltas = [float(np.mean(values)) for values in per_log.values()]
    def agreement(mapping):
        hits, count = 0, 0
        for values in mapping.values():
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    hits += int(values[i] == values[j]); count += 1
        return float(hits / count)
    common_rows = [row for row in rows if row["stratum"] == "common"]
    return {
        "common_pdm": common, "rare_pdm": rare, "equal_stratum_pdm": 0.5 * (common + rare),
        "v3_common_pdm": base_common, "v3_rare_pdm": base_rare,
        "v3_equal_stratum_pdm": 0.5 * (base_common + base_rare),
        "gain_equal_stratum_pdm": 0.5 * ((common - base_common) + (rare - base_rare)),
        "common_hard_safety": common_safety, "rare_hard_safety": rare_safety,
        "v3_common_hard_safety": base_common_safety,
        "v3_rare_hard_safety": base_rare_safety,
        "common_degraded_fraction": float(np.mean([row["reward"] < row["baseline"] - 1e-6 for row in common_rows])),
        "selected_noise_view_agreement": agreement(by_token), "v3_noise_view_agreement": agreement(v3_by_token),
        "log_bootstrap_gain": bootstrap_mean(log_deltas, repetitions, seed),
        "per_log_gain": {name: float(np.mean(values)) for name, values in per_log.items()},
    }


def eligibility_gates(metrics):
    return {
        "equal_stratum_gain_at_least_0005": metrics["gain_equal_stratum_pdm"] >= 0.005,
        "common_floor": metrics["common_pdm"] >= metrics["v3_common_pdm"] - 0.005,
        "rare_floor": metrics["rare_pdm"] >= metrics["v3_rare_pdm"],
        "common_hard_safety_floor": metrics["common_hard_safety"] >= metrics["v3_common_hard_safety"] - 0.002,
        "rare_hard_safety_floor": metrics["rare_hard_safety"] >= metrics["v3_rare_hard_safety"] - 0.002,
        "common_degraded_fraction_at_most_010": metrics["common_degraded_fraction"] <= 0.10,
        "log_bootstrap_gain_lower_positive": metrics["log_bootstrap_gain"]["lower_95"] > 0.0,
        "noise_agreement_floor": metrics["selected_noise_view_agreement"] >= metrics["v3_noise_view_agreement"] - 0.02,
    }


def choose_decision(reports, signals, comparisons):
    eligible = {name: all(reports[name]["gates"].values()) for name in ("H0_token", "H1_relation", "H2_interaction")}
    selected = None
    if eligible["H0_token"]:
        selected = "H0_token"
    if eligible["H1_relation"] and signals.get("H1_relation", False):
        if selected is None or comparisons.get("H1_over_H0", False):
            selected = "H1_relation"
    if eligible["H2_interaction"] and signals.get("H2_interaction", False):
        if selected is None or (selected == "H1_relation" and comparisons.get("H2_over_H1", False)):
            selected = "H2_interaction"
    if selected == "H0_token":
        return "AUTHORIZE_TOKEN_TOP5_EVALUATOR", selected
    if selected == "H1_relation":
        return "AUTHORIZE_RELATIONAL_TOP5_EVALUATOR", selected
    if selected == "H2_interaction":
        return "AUTHORIZE_INTERACTION_TOP5_EVALUATOR", selected
    h0 = reports["H0_token"]["pair_auc"]
    if min(h0["common"], h0["rare"]) >= 0.75 or any(signals.values()):
        return "AUTHORIZE_LOCKED_TOP1_OBJECTIVE", None
    return "AUTHORIZE_HISTORY_AUDIT", None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--prior-interaction-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def per_arm_pair_metrics(groups, predictions):
    result = {}
    for stratum in (None, "common", "rare"):
        scores, targets = [], []
        for group_id, group in enumerate(groups):
            if stratum is not None and group["stratum"] != stratum:
                continue
            matrix = predictions[group_id]
            delta = group["top_rewards"][PAIR_I] - group["top_rewards"][PAIR_J]
            active = delta.abs().gt(1e-6)
            scores.append(matrix[PAIR_I[active], PAIR_J[active]])
            targets.append(delta[active].gt(0).float())
        result["overall" if stratum is None else stratum] = binary_auc(torch.cat(scores), torch.cat(targets))
    return result


def representation_signal(reports, target, references, repetitions, seed):
    comparisons = {}
    passed = True
    for offset, reference in enumerate(references):
        target_auc = reports[target]["pair_auc"]
        reference_auc = reports[reference]["pair_auc"]
        common_delta = target_auc["common"] - reference_auc["common"]
        rare_delta = target_auc["rare"] - reference_auc["rare"]
        shared = sorted(set(reports[target]["per_log_auc"]) & set(reports[reference]["per_log_auc"]))
        deltas = [reports[target]["per_log_auc"][name] - reports[reference]["per_log_auc"][name] for name in shared]
        bootstrap = bootstrap_mean(deltas, repetitions, seed + offset) if deltas else {"lower_95": float("-inf"), "num_logs": 0}
        gates = {
            "overall_auc_gain_at_least_001": target_auc["overall"] - reference_auc["overall"] >= 0.01,
            "paired_log_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
            "common_delta_floor": common_delta >= -0.01,
            "rare_delta_floor": rare_delta >= -0.01,
        }
        comparisons[reference] = {"delta_overall": target_auc["overall"] - reference_auc["overall"], "delta_common": common_delta, "delta_rare": rare_delta, "log_bootstrap": bootstrap, "gates": gates}
        passed = passed and all(gates.values())
    return passed, comparisons


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.feature_batch_size, args.hidden_dim, args.bootstrap_repetitions) <= 0:
        raise ValueError("all hard-pair budgets must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaded = [common.load_cache(path, "train") for path in args.train_cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    if len(caches) != 3 or [int(row["noise_seed"]) for row in manifests] != [0, 1, 2]:
        raise RuntimeError("hard-pair audit requires train noise seeds 0,1,2")
    if any(cache.get("schema_version") != 4 for cache in caches):
        raise RuntimeError("hard-pair audit requires frozen-track schema-v4 caches")
    pair_rows, rare_audit, pair_path, audit_path = trainer.load_pair_contract(args.pair_manifest, args.rare_data_audit)
    trainer.validate_cache_pair_alignment(caches, manifests, pair_rows, rare_audit)
    token_meta = {}
    for row in pair_rows:
        for key, stratum in (("rare_token", "rare"), ("common_token", "common")):
            token = str(row[key]); value = {"stratum": stratum, "log_name": str(row["log_name"])}
            if token in token_meta and token_meta[token] != value:
                raise RuntimeError(f"ambiguous token provenance: {token}")
            token_meta[token] = value

    prior_path = args.prior_interaction_gate.expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    if prior.get("status") != "PASS" or prior.get("decision") != "STOP_INTERACTION_AND_TEST_HISTORY" or prior.get("all_gates_passed") is not False:
        raise RuntimeError("prior interaction probe does not authorize the hard-pair audit")
    model, checkpoint_payload, checkpoint_path, checkpoint_sha = load_checkpoint(args.v3_checkpoint, args.v3_checkpoint_sha256, caches[0], device)
    groups = build_groups(caches, manifests, model, token_meta, device, args.feature_batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fold_assignment = assign_log_folds(groups)
    reports = {arm: {"folds": [], "predictions": {}, "selected": {}} for arm in ARMS}

    for test_fold in range(NUM_FOLDS):
        calibration_fold = (test_fold + 1) % NUM_FOLDS
        train_folds = set(range(NUM_FOLDS)) - {test_fold, calibration_fold}
        for arm_index, arm in enumerate(ARMS):
            model = fit_comparator(groups, arm, train_folds, fold_assignment, args, args.seed + 1000 * test_fold + arm_index)
            calibration_indices = [index for index, group in enumerate(groups) if fold_assignment[group["log_name"]] == calibration_fold]
            test_indices = [index for index, group in enumerate(groups) if fold_assignment[group["log_name"]] == test_fold]
            matrices = predict_group_evidence(
                model, groups, arm, calibration_indices + test_indices, device,
                args.feature_batch_size,
            )
            calibrator = fit_calibrator(matrices, groups, calibration_indices, device, args.seed + 20000 + 1000 * test_fold + arm_index)
            fold_selected = []
            for group_id in test_indices:
                group = groups[group_id]
                matrix = matrices[group_id]
                challenger = graph_challenger(matrix)
                calibrated = calibrator["scale"] * float(matrix[challenger, 0]) + calibrator["bias"] if challenger != 0 else float("-inf")
                selected = challenger if calibrated > 0.0 else 0
                reports[arm]["predictions"][group_id] = matrix
                reports[arm]["selected"][group_id] = selected
                fold_selected.append(selected)
            reports[arm]["folds"].append({"test_fold": test_fold, "calibration_fold": calibration_fold, "train_folds": sorted(train_folds), "test_groups": len(fold_selected), "calibrator": calibrator})
            del model

    for arm_index, arm in enumerate(ARMS):
        if len(reports[arm]["predictions"]) != len(groups):
            raise RuntimeError(f"{arm} did not produce complete OOF predictions")
        predictions = [reports[arm]["predictions"][index] for index in range(len(groups))]
        selected = [reports[arm]["selected"][index] for index in range(len(groups))]
        reports[arm]["pair_auc"] = per_arm_pair_metrics(groups, predictions)
        reports[arm]["selection"] = selected_metrics(groups, selected, args.bootstrap_repetitions, args.seed + 30000 + arm_index)
        reports[arm]["gates"] = eligibility_gates(reports[arm]["selection"])
        per_log_auc = {}
        for name in sorted(fold_assignment):
            indices = [i for i, group in enumerate(groups) if group["log_name"] == name]
            scores, targets = [], []
            for i in indices:
                delta = groups[i]["top_rewards"][PAIR_I] - groups[i]["top_rewards"][PAIR_J]
                active = delta.abs().gt(1e-6)
                scores.append(predictions[i][PAIR_I[active], PAIR_J[active]])
                targets.append(delta[active].gt(0).float())
            auc = binary_auc(torch.cat(scores), torch.cat(targets))
            if auc is not None:
                per_log_auc[name] = auc
        reports[arm]["per_log_auc"] = per_log_auc
        reports[arm].pop("predictions"); reports[arm].pop("selected")

    h1_signal, h1_comparisons = representation_signal(reports, "H1_relation", ("H0_token", "S1_relation_shuffle"), args.bootstrap_repetitions, args.seed + 40000)
    h2_signal, h2_comparisons = representation_signal(reports, "H2_interaction", ("H1_relation", "S2_interaction_shuffle"), args.bootstrap_repetitions, args.seed + 50000)
    signals = {"H1_relation": h1_signal, "H2_interaction": h2_signal}
    promotion_comparisons = {}
    for target, reference, key, offset in (("H1_relation", "H0_token", "H1_over_H0", 0), ("H2_interaction", "H1_relation", "H2_over_H1", 1)):
        deltas = []
        for name in sorted(fold_assignment):
            deltas.append(reports[target]["selection"]["per_log_gain"][name] - reports[reference]["selection"]["per_log_gain"][name])
        boot = bootstrap_mean(deltas, args.bootstrap_repetitions, args.seed + 60000 + offset)
        pdm_gain = reports[target]["selection"]["equal_stratum_pdm"] - reports[reference]["selection"]["equal_stratum_pdm"]
        promotion_comparisons[key] = {"equal_stratum_pdm_gain": pdm_gain, "paired_log_bootstrap": boot, "passed": pdm_gain >= 0.002 and boot["lower_95"] > 0.0}
    decision, selected_arm = choose_decision(reports, signals, {key: value["passed"] for key, value in promotion_comparisons.items()})

    baseline_groups = [group["top_rewards"] for group in groups]
    full_rewards = [group["all_rewards"] for group in groups]
    v3 = float(np.mean([float(value[0]) for value in baseline_groups]))
    oracle20 = float(np.mean([float(value.max()) for value in full_rewards]))
    oracle5 = float(np.mean([float(value.max()) for value in baseline_groups]))
    oracle8 = float(np.mean([float(group["top8_rewards"].max()) for group in groups]))
    headroom = oracle20 - v3
    report = {
        "schema_version": 1, "status": "PASS", "method": "rare_v3_locked_top5_hard_pair_train_only_audit_v1",
        "development_or_certification_consumed": False, "decision": decision, "selected_arm": selected_arm,
        "question": "Does candidate information support a log-disjoint selective reranker inside rare-tuned V3's fixed top-5?",
        "locked_contract": {"incumbent": "rare_tuned_V3", "topk": TOPK, "top8_role": "ceiling_only", "reward_or_components_as_input": False, "shortlist_defined_before_reward": True, "primary_metric": "PDM", "cached_hard_safety_definition": "no_at_fault_collisions * drivable_area_compliance", "closed_loop_success_rate_role": "required only at a later development gate; unavailable and not imputed in this train-cache audit", "promotion_floors": ["common PDM", "rare PDM", "common/rare cached hard safety", "common degraded fraction", "noise-view agreement"]},
        "headroom": {"v3_selected_pdm": v3, "oracle20_pdm": oracle20, "oracle5_pdm": oracle5, "oracle8_pdm": oracle8, "top5_headroom_fraction": (oracle5 - v3) / headroom if headroom > 0 else 0.0, "top8_headroom_fraction": (oracle8 - v3) / headroom if headroom > 0 else 0.0},
        "budgets": {"folds": NUM_FOLDS, "outer_test_folds": NUM_FOLDS, "separate_calibration_fold": True, "train_folds_per_outer_fold": 3, "topk": TOPK, "pairs_per_group": int(len(PAIR_I)), "epochs": args.epochs, "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "bootstrap_repetitions": args.bootstrap_repetitions, "groups": len(groups)},
        "arms": reports,
        "representation_signals": {"H1_relation": {"passed": h1_signal, "comparisons": h1_comparisons}, "H2_interaction": {"passed": h2_signal, "comparisons": h2_comparisons}},
        "complexity_promotion": promotion_comparisons,
        "decision_semantics": {"AUTHORIZE_TOKEN_TOP5_EVALUATOR": "train a simple frozen-V3-token selective top5 evaluator", "AUTHORIZE_RELATIONAL_TOP5_EVALUATOR": "add only candidate-trajectory relations", "AUTHORIZE_INTERACTION_TOP5_EVALUATOR": "add frozen-track interactions after both matched controls pass", "AUTHORIZE_LOCKED_TOP1_OBJECTIVE": "information exists but top5 graph/override does not convert it; repair the incumbent-vs-challenger objective", "AUTHORIZE_HISTORY_AUDIT": "current-frame hard-pair information is insufficient; audit temporal/history inputs before any new selector"},
        "provenance": {"v3_checkpoint": str(checkpoint_path), "v3_checkpoint_sha256": checkpoint_sha, "v3_checkpoint_method": checkpoint_payload.get("method"), "prior_interaction_gate": str(prior_path), "prior_interaction_gate_sha256": common.sha256_file(prior_path), "pair_manifest": str(pair_path), "pair_manifest_sha256": common.sha256_file(pair_path), "rare_data_audit": str(audit_path), "rare_data_audit_sha256": common.sha256_file(audit_path), "train_manifests": list(manifests), "implementation_files": {str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()), str(Path(common.__file__).resolve()): common.sha256_file(Path(common.__file__).resolve())}},
    }
    output = args.output.expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": decision, "selected_arm": selected_arm, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
