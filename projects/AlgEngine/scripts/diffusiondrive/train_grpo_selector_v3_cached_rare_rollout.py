#!/usr/bin/env python3
"""Train a fresh zero-residual V3 selector on 50/50 common/hard rollout data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare
from build_grpo_selector_v3_rare_rollout_data import (
    BASELINE_SHA256,
    EXPECTED_REAL_CACHE_SEEDS,
    METHOD as DATA_METHOD,
    parse_seed_path,
)


METHOD = "scene_conditioned_exact_group_grpo_v3_rare_rollout_v1"
FORMAL_EPOCHS = 16
FORMAL_EXAMPLES_PER_CACHE_EPOCH = 6339
FORMAL_BATCH_SIZE = 64
FORMAL_TOTAL_EXAMPLES = 304272
FORMAL_OPTIMIZER_STEPS = 4800


def implementation_provenance() -> dict[str, str]:
    algengine_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(standard.__file__).resolve(),
        Path(rare.__file__).resolve(),
        Path(__file__).with_name("build_grpo_selector_v3_rare_rollout_data.py"),
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py",
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py",
    )
    return {str(path): common.sha256_file(path) for path in paths}


def load_hard_pool(path: Path, expected_sha: str) -> list[dict]:
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError("hard-pool SHA256 drifted")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["hard_id"] for row in rows}) != len(rows):
        raise RuntimeError("hard pool is empty or repeats identities")
    for index, row in enumerate(rows):
        if row.get("hard_kind") not in ("real_rare", "synthetic_rollout"):
            raise RuntimeError(f"hard row {index} has an invalid source kind")
        if row.get("split") not in ("train", "development", "certification"):
            raise RuntimeError(f"hard row {index} has an invalid log split")
    return rows


def validate_synthetic_cache(cache: dict, manifest: dict) -> None:
    count = int(manifest["filtered_synthetic_records"])
    if cache.get("schema_version") != 3 or cache.get("source_kind") != (
        "base_policy_rollout"
    ):
        raise RuntimeError("synthetic cache schema drifted")
    required = {
        "candidate_features": (count, 20, 256),
        "candidate_trajectories_8": (count, 20, 8, 3),
        "route_bev_features": (count, 20, 8, 256),
        "status_tokens": (count, 1, 256),
        "ego_queries": (count, 1, 256),
        "agents_queries": (count, 30, 256),
        "candidate_rewards": (count, 20),
        "candidate_reward_components": (count, 20, 6),
        "candidate_reward_valid_mask": (count, 20),
        "reference_logits": (count, 20),
    }
    for key, shape in required.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"synthetic cache {key} shape drifted")
    if len(cache["tokens"]) != count or len(set(cache["tokens"])) != count:
        raise RuntimeError("synthetic cache identities drifted")
    if count and not bool(cache["candidate_reward_valid_mask"].all()):
        raise RuntimeError("synthetic cache contains invalid candidates")


def load_contract(args):
    manifest_path = args.data_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("method") != DATA_METHOD
        or manifest.get("baseline_checkpoint_sha256") != BASELINE_SHA256
        or manifest.get("behavior_policy_is_trained_v3") is not False
        or manifest.get("mixture_contract", {}).get("overall_common_fraction") != 0.5
        or manifest.get("mixture_contract", {}).get("overall_hard_fraction") != 0.5
    ):
        raise RuntimeError("rare-rollout data manifest did not pass")
    hard_pool_path = args.hard_pool.expanduser().resolve()
    if str(hard_pool_path) != str(Path(manifest["hard_pool"]).resolve()):
        raise RuntimeError("hard-pool path disagrees with data manifest")
    hard_rows = load_hard_pool(hard_pool_path, manifest["hard_pool_sha256"])
    if len(hard_rows) != int(manifest["hard_pool_rows"]):
        raise RuntimeError("hard-pool row count drifted")

    cache_paths = dict(args.real_cache)
    if set(cache_paths) != set(EXPECTED_REAL_CACHE_SEEDS):
        raise RuntimeError("exact real cache seeds 0,1,2 are required")
    pairs = [
        common.load_cache(cache_paths[seed], "train")
        for seed in EXPECTED_REAL_CACHE_SEEDS
    ]
    real_caches, real_manifests = zip(*pairs)
    standard.assert_cache_group(
        real_caches, real_manifests, "train", EXPECTED_REAL_CACHE_SEEDS
    )
    for seed, path in cache_paths.items():
        expected = manifest["real_caches"][str(seed)]
        if common.sha256_file(path) != expected["cache_sha256"]:
            raise RuntimeError(f"real cache seed {seed} drifted")

    synthetic_path = args.synthetic_cache.expanduser().resolve()
    if (
        str(synthetic_path) != str(Path(manifest["synthetic_cache"]).resolve())
        or common.sha256_file(synthetic_path) != manifest["synthetic_cache_sha256"]
    ):
        raise RuntimeError("synthetic cache provenance drifted")
    synthetic = torch.load(synthetic_path, map_location="cpu")
    validate_synthetic_cache(synthetic, manifest)
    if synthetic["scene_selector_config"] != real_caches[0]["scene_selector_config"]:
        raise RuntimeError("synthetic/real selector architecture drifted")

    token_maps = [
        {str(token): index for index, token in enumerate(cache["tokens"])}
        for cache in real_caches
    ]
    synthetic_count = len(synthetic["tokens"])
    for position, row in enumerate(hard_rows):
        for token_map in token_maps:
            if row["paired_common_token"] not in token_map:
                raise RuntimeError(f"hard row {position} has no paired common cache row")
            if row["hard_kind"] == "real_rare" and row["rare_token"] not in token_map:
                raise RuntimeError(f"hard row {position} has no real-rare cache row")
        if row["hard_kind"] == "synthetic_rollout":
            index = int(row.get("synthetic_index", -1))
            if not 0 <= index < synthetic_count:
                raise RuntimeError(f"hard row {position} has invalid synthetic index")
            if synthetic["origin_rare_tokens"][index] != row["rare_token"]:
                raise RuntimeError(f"hard row {position} synthetic origin drifted")
    return (
        manifest_path,
        manifest,
        hard_pool_path,
        hard_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    )


def gather_source(cache: dict, indices: list[int], device: torch.device):
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    inputs = common.batch_inputs(cache, tensor_indices, device)
    reference = cache["reference_logits"][tensor_indices].to(
        device=device, dtype=torch.float32
    )
    rewards = cache["candidate_rewards"][tensor_indices].to(
        device=device, dtype=torch.float32
    )
    valid = cache["candidate_reward_valid_mask"][tensor_indices].to(device=device)
    return inputs, reference, rewards, valid


def mixed_batch(
    rows: list[tuple[str, int]],
    hard_rows: list[dict],
    real_cache: dict,
    real_token_map: dict[str, int],
    synthetic: dict,
    device: torch.device,
):
    sources = {"real": [], "synthetic": []}
    source_kinds = Counter()
    for class_name, position in rows:
        row = hard_rows[position]
        if class_name == "common":
            sources["real"].append(real_token_map[row["paired_common_token"]])
            source_kinds["common"] += 1
        elif row["hard_kind"] == "real_rare":
            sources["real"].append(real_token_map[row["rare_token"]])
            source_kinds["hard_real_rare"] += 1
        else:
            sources["synthetic"].append(int(row["synthetic_index"]))
            source_kinds["hard_synthetic"] += 1

    batches = []
    if sources["real"]:
        batches.append(gather_source(real_cache, sources["real"], device))
    if sources["synthetic"]:
        batches.append(gather_source(synthetic, sources["synthetic"], device))
    if not batches:
        raise RuntimeError("empty mixed batch")
    input_keys = tuple(batches[0][0])
    inputs = {
        key: torch.cat([batch[0][key] for batch in batches], dim=0)
        for key in input_keys
    }
    reference = torch.cat([batch[1] for batch in batches], dim=0)
    rewards = torch.cat([batch[2] for batch in batches], dim=0)
    valid = torch.cat([batch[3] for batch in batches], dim=0)
    return inputs, reference, rewards, valid, source_kinds


def validate_formal_args(args) -> None:
    expected = {
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "epochs": FORMAL_EPOCHS,
        "examples_per_cache_epoch": FORMAL_EXAMPLES_PER_CACHE_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "method_name": METHOD,
    }
    drift = {
        key: {"actual": getattr(args, key), "expected": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if drift:
        raise RuntimeError("formal rare-rollout training contract drifted: " + repr(drift))


def save_selector_state(path, model, model_config, args, epoch, manifest_path, pool_path):
    payload = {
        "schema_version": 3,
        "method": args.method_name,
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "implementation_files": implementation_provenance(),
        "scene_selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scene_selector_config": model_config,
        "ablation": "full",
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "sampling_mode": "common_hard_balanced",
        "train_seed": args.seed,
        "epoch": epoch,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(pool_path),
        "hard_pool_sha256": common.sha256_file(pool_path),
    }
    torch.save(payload, path)


def train(args) -> dict:
    if args.method_name is None:
        args.method_name = METHOD
    if args.formal_contract:
        validate_formal_args(args)
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid training hyperparameters")
    if min(args.epochs, args.examples_per_cache_epoch, args.batch_size) <= 0:
        raise ValueError("training budgets must be positive")

    (
        manifest_path,
        manifest,
        hard_pool_path,
        hard_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = load_contract(args)
    if args.smoke_limit_hard_pool is not None:
        if args.formal_contract:
            raise RuntimeError("smoke hard-pool limiting is forbidden in formal mode")
        if args.smoke_limit_hard_pool < 1:
            raise ValueError("--smoke-limit-hard-pool must be positive")
        real_rows = [row for row in hard_rows if row["hard_kind"] == "real_rare"]
        synthetic_rows = [
            row for row in hard_rows if row["hard_kind"] == "synthetic_rollout"
        ]
        hard_rows = real_rows[: args.smoke_limit_hard_pool]
        if synthetic_rows and args.smoke_limit_hard_pool > 1:
            hard_rows[-1] = synthetic_rows[0]
    minimum_hard_draws = args.epochs * (args.examples_per_cache_epoch // 2)
    if len(hard_rows) > minimum_hard_draws:
        raise RuntimeError(
            "fixed compute cannot cover every hard row before reuse: "
            f"hard_rows={len(hard_rows)} hard_draws_per_cache={minimum_hard_draws}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model, model_config = common.model_from_cache(real_caches[0], "full")
    model = model.to(device)
    with torch.no_grad():
        initial_inputs, reference, _, _, _ = mixed_batch(
            [("hard", 0)],
            hard_rows,
            real_caches[0],
            token_maps[0],
            synthetic,
            device,
        )
        initial_delta = model(**initial_inputs)
        initial_max_abs_delta = float(initial_delta.abs().max().cpu())
    if initial_max_abs_delta != 0.0:
        raise RuntimeError("fresh V3 selector is not exact-zero at initialization")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, foreach=False
    )
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    samplers = {
        (cache_index, class_name): rare.CyclingPairSampler(
            len(hard_rows),
            args.seed * 100000
            + cache_index * 1000
            + (17 if class_name == "hard" else 37),
        )
        for cache_index in range(3)
        for class_name in ("hard", "common")
    }
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if not checkpoint_epochs or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must end at --epochs")

    total_examples = 0
    optimizer_steps = 0
    class_examples = Counter()
    source_examples = Counter()
    accumulated = Counter()
    checkpoint_reports = []
    global_block = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        cache_order = torch.randperm(3, generator=schedule).tolist()
        for cache_index in cache_order:
            hard_count, common_count = rare.block_class_counts(
                args.examples_per_cache_epoch, global_block
            )
            selected = [
                ("hard", position)
                for position in samplers[(cache_index, "hard")].take(hard_count)
            ]
            selected.extend(
                ("common", position)
                for position in samplers[(cache_index, "common")].take(common_count)
            )
            permutation = torch.randperm(len(selected), generator=schedule).tolist()
            selected = [selected[index] for index in permutation]
            for start in range(0, len(selected), args.batch_size):
                batch_rows = selected[start : start + args.batch_size]
                inputs, reference, rewards, valid, kinds = mixed_batch(
                    batch_rows,
                    hard_rows,
                    real_caches[cache_index],
                    token_maps[cache_index],
                    synthetic,
                    device,
                )
                logits = reference + model(**inputs)
                optimizer.zero_grad(set_to_none=True)
                loss, policy, kl = common.exact_group_loss(
                    logits,
                    reference,
                    rewards,
                    valid,
                    args.temperature,
                    args.kl_weight,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite rare-rollout loss at epoch {epoch}")
                loss.backward()
                common.clip_grad_norm_cpu_(model.parameters(), 10.0)
                optimizer.step()
                optimizer_steps += 1
                accumulated["loss"] += float(loss.detach().cpu())
                accumulated["policy"] += float(policy.detach().cpu())
                accumulated["kl"] += float(kl.detach().cpu())
                source_examples.update(kinds)
            class_examples["hard"] += hard_count
            class_examples["common"] += common_count
            total_examples += len(selected)
            global_block += 1

        if epoch in checkpoint_epochs:
            state_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
            save_selector_state(
                state_path,
                model,
                model_config,
                args,
                epoch,
                manifest_path,
                hard_pool_path,
            )
            checkpoint_reports.append(
                {
                    "epoch": epoch,
                    "scene_selector_state": str(state_path),
                    "scene_selector_state_sha256": common.sha256_file(state_path),
                }
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "examples": total_examples,
                        "optimizer_steps": optimizer_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if class_examples["hard"] != class_examples["common"]:
        raise RuntimeError("global 50/50 common/hard balance drifted")
    if args.formal_contract:
        if total_examples != FORMAL_TOTAL_EXAMPLES:
            raise RuntimeError("formal total example budget drifted")
        if optimizer_steps != FORMAL_OPTIMIZER_STEPS:
            raise RuntimeError("formal optimizer-step budget drifted")
    coverage = {}
    for cache_index in range(3):
        coverage[str(cache_index)] = {}
        for class_name in ("hard", "common"):
            sampler = samplers[(cache_index, class_name)]
            if len(sampler.visited) != len(hard_rows):
                raise RuntimeError(
                    f"cache {cache_index} {class_name} did not cover the hard pool"
                )
            coverage[str(cache_index)][class_name] = {
                "unique_hard_positions_visited": len(sampler.visited),
                "hard_pool_rows": len(hard_rows),
                "completed_cycles": sampler.completed_cycles,
            }

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.method_name,
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "source_policy": "immutable_epoch100_diffusiondrive",
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "fresh_selector_initialization": "exact_zero",
        "initial_max_abs_delta": initial_max_abs_delta,
        "sampling_mode": "common_hard_balanced",
        "ablation": "full",
        "scene_selector_config": model_config,
        "implementation_files": implementation_provenance(),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "fixed_budget_selection": "final_epoch",
        "early_stopping": False,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "real_cache_manifests": list(real_manifests),
        "sampling": {
            "contract": "50% common; 50% hard=(real rare + senior-v1 filtered synthetic)",
            "examples_per_cache_epoch": args.examples_per_cache_epoch,
            "total_examples": total_examples,
            "total_optimizer_steps": optimizer_steps,
            "class_examples": dict(class_examples),
            "source_examples": dict(source_examples),
            "coverage": coverage,
        },
        "mean_training_loss": accumulated["loss"] / optimizer_steps,
        "mean_training_policy": accumulated["policy"] / optimizer_steps,
        "mean_training_kl": accumulated["kl"] / optimizer_steps,
        "checkpoints": checkpoint_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V3 rare-rollout cached selector training: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument(
        "--examples-per-cache-epoch",
        type=int,
        default=FORMAL_EXAMPLES_PER_CACHE_EPOCH,
    )
    parser.add_argument("--checkpoint-epochs", default=str(FORMAL_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method-name")
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--smoke-limit-hard-pool", type=int)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
