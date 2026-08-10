#!/usr/bin/env python3
"""Offline diagnostic matrix for the frozen DiffusionDrive selector features.

The canonical clipped GRPO objective is compared with controlled probes on the
same cached action sets.  Oracle CE and soft-listwise are diagnostics only;
this script never changes the formal method or writes a deployable checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = ("canonical", "unclipped", "tempered2", "tempered4", "oracle_ce", "soft_listwise")


class Selector(nn.Sequential):
    """Exact final DiffusionDrive plan_cls_branch architecture."""

    def __init__(self):
        super().__init__(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path):
    path = path.expanduser().resolve()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or manifest.get("cache_sha256") != sha256_file(path):
        raise RuntimeError(f"cache provenance failed: {path}")
    cache = torch.load(path, map_location="cpu")
    n = len(cache["tokens"])
    shapes = {
        "candidate_features": (n, 20, 256),
        "candidate_rewards": (n, 20),
        "reference_logits": (n, 20),
    }
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"{path}: {key} shape drift")
    if len(set(cache["tokens"])) != n or len(cache["scenes"]) != n:
        raise RuntimeError(f"{path}: token/scene provenance drift")
    return cache, manifest


def build_selector(state, device):
    model = Selector().to(device)
    model.load_state_dict(state, strict=True)
    return model


def stable_scene_split(scenes, fraction=0.9):
    unique = sorted(set(scenes))
    train_scenes = {
        scene for scene in unique
        if int(hashlib.sha256(scene.encode()).hexdigest()[:16], 16) / 16**16 < fraction
    }
    # Deterministic nonempty fallback for extremely small synthetic tests.
    if not train_scenes or len(train_scenes) == len(unique):
        cut = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
        train_scenes = set(unique[:cut])
    train = torch.tensor([scene in train_scenes for scene in scenes], dtype=torch.bool)
    return train, ~train


def normalized_advantage(reward):
    centered = reward - reward.mean(dim=-1, keepdim=True)
    return centered / centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def objective_loss(method, logits, reference_logits, reward, kl_weight=1e-3):
    if method.startswith("tempered"):
        temperature = float(method.replace("tempered", ""))
    else:
        temperature = 1.0
    if method in ("canonical", "unclipped", "tempered2", "tempered4"):
        current_logp = F.log_softmax(logits / temperature, dim=-1)
        reference_logp = F.log_softmax(reference_logits / temperature, dim=-1)
        ratio = torch.exp(current_logp - reference_logp)
        advantage = normalized_advantage(reward)
        if method == "unclipped":
            surrogate = ratio * advantage
        else:
            clipped = ratio.clamp(0.8, 1.2)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        policy = -(reference_logp.exp() * surrogate).sum(dim=-1).mean()
        reverse_kl = (
            current_logp.exp() * (current_logp - reference_logp)
        ).sum(dim=-1).mean()
        return policy + kl_weight * reverse_kl
    if method == "oracle_ce":
        target = reward.eq(reward.max(dim=-1, keepdim=True).values).float()
        target = target / target.sum(dim=-1, keepdim=True)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    if method == "soft_listwise":
        target = F.softmax(normalized_advantage(reward) / 0.5, dim=-1)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    raise ValueError(method)


def metrics(logits, reference_logits, rewards):
    probability = logits.softmax(dim=-1)
    reference_probability = reference_logits.softmax(dim=-1)
    selected = logits.argmax(dim=-1)
    reference = reference_logits.argmax(dim=-1)
    oracle = rewards.argmax(dim=-1)
    gather = lambda value, index: value.gather(1, index[:, None]).squeeze(1)
    selected_reward = gather(rewards, selected)
    reference_reward = gather(rewards, reference)
    oracle_reward = gather(rewards, oracle)
    top2 = probability.topk(2, dim=-1).values
    reward_order = rewards.argsort(dim=-1).argsort(dim=-1).float()
    logit_order = logits.argsort(dim=-1).argsort(dim=-1).float()
    reward_order = reward_order - reward_order.mean(dim=-1, keepdim=True)
    logit_order = logit_order - logit_order.mean(dim=-1, keepdim=True)
    spearman = (reward_order * logit_order).sum(dim=-1) / (
        reward_order.square().sum(dim=-1).sqrt()
        * logit_order.square().sum(dim=-1).sqrt()
    ).clamp_min(1e-8)
    return {
        "top1_reward": selected_reward,
        "top1_gain": selected_reward - reference_reward,
        "oracle_reward": oracle_reward,
        "oracle_regret": oracle_reward - selected_reward,
        "oracle_match": selected.eq(oracle).float(),
        "oracle_probability": gather(probability, oracle),
        "expected_reward": (probability * rewards).sum(dim=-1),
        "reference_expected_reward": (reference_probability * rewards).sum(dim=-1),
        "entropy": -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1),
        "top1_margin": top2[:, 0] - top2[:, 1],
        "selection_disagreement": selected.ne(reference).float(),
        "spearman": spearman,
    }


def evaluate(model, cache, indices, device, batch_size=512):
    outputs = {}
    model.eval()
    chosen = indices.nonzero(as_tuple=False).flatten()
    with torch.no_grad():
        for start in range(0, len(chosen), batch_size):
            local = chosen[start : start + batch_size]
            feature = cache["candidate_features"][local].to(device=device, dtype=torch.float32)
            reward = cache["candidate_rewards"][local].to(device=device)
            reference = cache["reference_logits"][local].to(device=device)
            values = metrics(model(feature).squeeze(-1), reference, reward)
            for key, value in values.items():
                outputs.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(value) for key, value in outputs.items()}


def train_one(method, lr, seed, cache, train_mask, dev_mask, device):
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_selector(cache["baseline_selector_state"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs = 8 if method in ("canonical", "unclipped", "tempered2", "tempered4") else 100
    patience = epochs if epochs == 8 else 10
    train_indices = train_mask.nonzero(as_tuple=False).flatten()
    best_state = copy.deepcopy(model.state_dict())
    best_gain = -math.inf
    stale = 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(order), 256):
            local = order[start : start + 256]
            feature = cache["candidate_features"][local].to(device=device, dtype=torch.float32)
            reward = cache["candidate_rewards"][local].to(device=device)
            reference = cache["reference_logits"][local].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature).squeeze(-1)
            loss = objective_loss(method, logits, reference, reward)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {method} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        dev = evaluate(model, cache, dev_mask, device)
        gain = float(dev["top1_gain"].mean())
        if gain > best_gain + 1e-8:
            best_gain = gain
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_gain


def summarize(values):
    return {key: float(value.mean()) for key, value in values.items()}


def grouped_bootstrap(delta, scenes, replicates, seed=20260810):
    grouped = {}
    for value, scene in zip(delta.tolist(), scenes):
        grouped.setdefault(scene, []).append(value)
    names = sorted(grouped)
    arrays = [np.asarray(grouped[name], dtype=np.float64) for name in names]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = np.concatenate([arrays[i] for i in selected]).mean()
    return {
        "mean": float(np.mean(delta.numpy())),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_unit": "scene/log",
        "replicates": replicates,
    }


def method_grid(method):
    if method in ("canonical", "unclipped", "tempered2", "tempered4"):
        return (1e-6, 3e-6, 1e-5)
    return (1e-5, 1e-4, 1e-3)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_cache, train_manifest = load_cache(args.train_cache)
    calibration = [load_cache(path) for path in args.calibration_cache]
    if train_manifest.get("split") != "train":
        raise RuntimeError("train cache has wrong split")
    if len(calibration) != 3 or sorted(m["noise_seed"] for _, m in calibration) != [0, 1, 2]:
        raise RuntimeError("exactly calibration noise seeds 0,1,2 are required")
    train_tokens = set(train_cache["tokens"])
    if any(train_tokens.intersection(cache["tokens"]) for cache, _ in calibration):
        raise RuntimeError("train/calibration token leakage")
    state_keys = set(train_cache["baseline_selector_state"])
    for cache, _ in calibration:
        if set(cache["baseline_selector_state"]) != state_keys:
            raise RuntimeError("baseline selector state schema drift")
    train_mask, dev_mask = stable_scene_split(train_cache["scenes"])
    trials = []
    selected_models = {}
    for method in METHODS:
        method_trials = []
        for lr in method_grid(method):
            seed_models = []
            seed_gains = []
            for seed in (0, 1, 2):
                model, dev_gain = train_one(
                    method, lr, seed, train_cache, train_mask, dev_mask, device
                )
                seed_models.append({key: value.cpu() for key, value in model.state_dict().items()})
                seed_gains.append(dev_gain)
                trials.append({"method": method, "lr": lr, "seed": seed, "inner_dev_top1_gain": dev_gain})
            method_trials.append((float(np.mean(seed_gains)), lr, seed_models))
        _, best_lr, states = max(method_trials, key=lambda row: row[0])
        selected_models[method] = (best_lr, states)

    reports = {}
    for method, (lr, states) in selected_models.items():
        pooled = {}
        pooled_scenes = []
        per_run = []
        for train_seed, state in enumerate(states):
            model = build_selector(state, device)
            for cache, manifest in calibration:
                mask = torch.ones(len(cache["tokens"]), dtype=torch.bool)
                values = evaluate(model, cache, mask, device)
                per_run.append({
                    "train_seed": train_seed,
                    "noise_seed": manifest["noise_seed"],
                    **summarize(values),
                })
                for key, value in values.items():
                    pooled.setdefault(key, []).append(value)
                pooled_scenes.extend(cache["scenes"])
        pooled = {key: torch.cat(value) for key, value in pooled.items()}
        reports[method] = {
            "selected_lr": lr,
            "mean": summarize(pooled),
            "top1_gain_bootstrap": grouped_bootstrap(
                pooled["top1_gain"], pooled_scenes, args.bootstrap_replicates
            ),
            "runs": per_run,
        }

    best_grpo = max(reports[name]["mean"]["top1_gain"] for name in METHODS[:4])
    best_supervised = max(reports[name]["mean"]["top1_gain"] for name in METHODS[4:])
    oracle_gap = float(np.mean([
        (cache["candidate_rewards"].max(dim=-1).values - cache["candidate_rewards"].gather(
            1, cache["reference_logits"].argmax(dim=-1, keepdim=True)
        ).squeeze(1)).mean().item()
        for cache, _ in calibration
    ]))
    best_method = max(METHODS, key=lambda name: reports[name]["mean"]["top1_gain"])
    best_ci = reports[best_method]["top1_gain_bootstrap"]
    if best_supervised > best_grpo + 0.002:
        diagnosis = "objective_limited"
    elif oracle_gap > 0.02 and best_supervised <= 0.002:
        diagnosis = "feature_representation_limited"
    elif best_ci["ci95_low"] > 0 and best_ci["mean"] >= 0.005:
        diagnosis = "selector_signal_learnable"
    else:
        diagnosis = "weak_or_inconclusive_selector_signal"
    result = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "diagnostic_only_no_formal_method_change",
        "train_manifest": train_manifest,
        "calibration_manifests": [manifest for _, manifest in calibration],
        "inner_split": {
            "unit": "scene/log",
            "train_tokens": int(train_mask.sum()),
            "dev_tokens": int(dev_mask.sum()),
        },
        "oracle_gap": oracle_gap,
        "best_grpo_top1_gain": best_grpo,
        "best_supervised_top1_gain": best_supervised,
        "best_method": best_method,
        "diagnosis": diagnosis,
        "decision_thresholds": {
            "supervised_over_grpo": 0.002,
            "representation_gain_floor": 0.002,
            "oracle_gap_material": 0.02,
            "learnable_gain": 0.005,
        },
        "methods": reports,
        "trials": trials,
    }
    result_path = output_dir / "diagnostic_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# DiffusionDrive GRPO selector diagnostic report", "",
        "> Diagnostic only: this run does not replace the canonical selector-GRPO formula.", "",
        f"- Diagnosis: `{diagnosis}`", f"- Frozen-generator oracle gap: `{oracle_gap:.6f}`",
        f"- Best method: `{best_method}`", "", "| Probe | LR | top-1 gain | 95% scene-bootstrap CI | oracle match | regret | disagreement |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        report = reports[method]
        mean = report["mean"]
        ci = report["top1_gain_bootstrap"]
        lines.append(
            f"| {method} | {report['selected_lr']:.1e} | {mean['top1_gain']:.6f} | "
            f"[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}] | {mean['oracle_match']:.4f} | "
            f"{mean['oracle_regret']:.6f} | {mean['selection_disagreement']:.4f} |"
        )
    lines += ["", "Raw machine-readable output: `diagnostic_results.json`.", ""]
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "diagnosis": diagnosis, "best_method": best_method}, sort_keys=True))


if __name__ == "__main__":
    main()
