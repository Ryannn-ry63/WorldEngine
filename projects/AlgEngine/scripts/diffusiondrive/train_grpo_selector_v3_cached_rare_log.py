#!/usr/bin/env python3
"""Train DiffusionDrive selector V3 on audited rare/same-log-common pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard


FORMAL_METHOD = "scene_conditioned_exact_group_grpo_v3_rare_log_v1"
FORMAL_NOISE_SEEDS = (0, 1, 2)
FORMAL_EPOCHS = 16
FORMAL_EXAMPLES_PER_CACHE_EPOCH = 6339
FORMAL_BATCH_SIZE = 64
FORMAL_TOTAL_EXAMPLES = 304272
FORMAL_OPTIMIZER_STEPS = 4800


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
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
    parser.add_argument("--method-name", default=FORMAL_METHOD)
    parser.add_argument(
        "--ablation",
        choices=("full", "feature_only", "feature_geometry", "feature_geometry_route"),
        default="full",
    )
    parser.add_argument(
        "--formal-contract",
        action="store_true",
        help="fail unless all V3 rare-log budget and hyperparameter constants match",
    )
    return parser.parse_args()


def implementation_provenance():
    algengine_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(standard.__file__).resolve(),
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py",
    )
    return {str(path): common.sha256_file(path) for path in paths}


def load_pair_contract(pair_manifest, rare_data_audit):
    pair_manifest = pair_manifest.expanduser().resolve()
    rare_data_audit = rare_data_audit.expanduser().resolve()
    for path in (pair_manifest, rare_data_audit):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(rare_data_audit.read_text())
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "PASS"
        or audit.get("method") != "diffusiondrive_v3_full_navtrain_rare_log_v1"
    ):
        raise RuntimeError("rare-data audit contract did not pass")
    expected_pair_sha = audit.get("outputs", {}).get("pairs_sha256")
    actual_pair_sha = common.sha256_file(pair_manifest)
    if expected_pair_sha != actual_pair_sha:
        raise RuntimeError("rare/common pair manifest SHA256 drifted")

    rows = []
    for line_number, line in enumerate(pair_manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "rare_token",
            "common_token",
            "log_name",
            "scene",
            "rare_votes",
            "failure_modes_by_seed",
        }
        if not required.issubset(row):
            raise RuntimeError(
                f"pair row {line_number} missing keys {sorted(required - set(row))}"
            )
        rare = str(row["rare_token"])
        paired_common = str(row["common_token"])
        if rare == paired_common:
            raise RuntimeError(f"pair row {line_number} pairs a token with itself")
        if int(row["rare_votes"]) not in (1, 2, 3):
            raise RuntimeError(f"pair row {line_number} has invalid rare vote count")
        rows.append({**row, "rare_token": rare, "common_token": paired_common})
    if not rows:
        raise RuntimeError("empty rare/common pair manifest")

    rare_tokens = [row["rare_token"] for row in rows]
    common_tokens = [row["common_token"] for row in rows]
    if len(set(rare_tokens)) != len(rare_tokens):
        raise RuntimeError("rare/common pair manifest repeats a rare token")
    if set(rare_tokens).intersection(common_tokens):
        raise RuntimeError("rare and common token populations overlap")
    if len(rows) != int(audit.get("rare_count", -1)):
        raise RuntimeError("pair row count disagrees with rare-data audit")
    if len(set(common_tokens)) != int(audit.get("paired_common_unique", -1)):
        raise RuntimeError("paired common population disagrees with rare-data audit")
    return rows, audit, pair_manifest, rare_data_audit


def validate_formal_contract(args):
    expected = {
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 0.0,
        "epochs": FORMAL_EPOCHS,
        "examples_per_cache_epoch": FORMAL_EXAMPLES_PER_CACHE_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "method_name": FORMAL_METHOD,
        "ablation": "full",
        "num_train_caches": 3,
    }
    actual = {
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "epochs": args.epochs,
        "examples_per_cache_epoch": args.examples_per_cache_epoch,
        "batch_size": args.batch_size,
        "method_name": args.method_name,
        "ablation": args.ablation,
        "num_train_caches": len(args.train_cache),
    }
    drift = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if drift:
        raise RuntimeError("formal V3 rare-log contract drifted: " + repr(drift))


class CyclingPairSampler:
    """Shuffle pair positions cycle-by-cycle without replacement within a cycle."""

    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError("sampler size must be positive")
        self.size = int(size)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order = []
        self.cursor = 0
        self.completed_cycles = 0
        self.visited = set()

    def take(self, count):
        output = []
        while len(output) < count:
            if self.cursor == len(self.order):
                self.order = torch.randperm(
                    self.size, generator=self.generator
                ).tolist()
                self.cursor = 0
                if self.visited:
                    self.completed_cycles += 1
            amount = min(count - len(output), len(self.order) - self.cursor)
            part = self.order[self.cursor : self.cursor + amount]
            output.extend(part)
            self.visited.update(part)
            self.cursor += amount
        return output


def block_class_counts(examples_per_cache_epoch, global_block):
    rare_count = examples_per_cache_epoch // 2
    common_count = examples_per_cache_epoch // 2
    if examples_per_cache_epoch % 2:
        if global_block % 2 == 0:
            rare_count += 1
        else:
            common_count += 1
    return rare_count, common_count


def load_caches(paths):
    if len(paths) != 3:
        raise RuntimeError("V3 rare-log training requires exactly three train caches")
    pairs = [common.load_cache(path, "train") for path in paths]
    pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*pairs)
    standard.assert_cache_group(
        caches, manifests, "train", FORMAL_NOISE_SEEDS
    )
    return caches, manifests


def validate_cache_pair_alignment(caches, manifests, pair_rows, rare_audit):
    token_to_index = {
        str(token): index for index, token in enumerate(caches[0]["tokens"])
    }
    rare_tokens = {row["rare_token"] for row in pair_rows}
    common_tokens = {row["common_token"] for row in pair_rows}
    expected_union = rare_tokens | common_tokens
    actual_union = set(token_to_index)
    if actual_union != expected_union:
        raise RuntimeError(
            "context-cache token set disagrees with audited rare/common union: "
            f"missing={len(expected_union - actual_union)} "
            f"extra={len(actual_union - expected_union)}"
        )
    if len(actual_union) != int(rare_audit.get("union_count", -1)):
        raise RuntimeError("context-cache coverage disagrees with rare-data audit")
    expected_filter_sha = rare_audit.get("outputs", {}).get(
        "union_filter_sha256"
    )
    for manifest in manifests:
        if manifest.get("nav_filter_sha256") != expected_filter_sha:
            raise RuntimeError("context cache was not extracted from the audited union")
        if int(manifest.get("num_tokens", -1)) != len(actual_union):
            raise RuntimeError("context cache manifest token count drifted")

    rare_indices = [token_to_index[row["rare_token"]] for row in pair_rows]
    common_indices = [token_to_index[row["common_token"]] for row in pair_rows]
    return rare_indices, common_indices


def save_selector_state(
    path,
    model,
    model_config,
    args,
    epoch,
    pair_manifest,
    pair_audit,
):
    payload = {
        "schema_version": 3,
        "method": args.method_name,
        "source_kind": "full_navtrain_rare_log_v1",
        "implementation_files": implementation_provenance(),
        "scene_selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scene_selector_config": model_config,
        "ablation": args.ablation,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epoch": epoch,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": common.sha256_file(pair_manifest),
        "rare_data_audit": str(pair_audit),
        "rare_data_audit_sha256": common.sha256_file(pair_audit),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid temperature/learning-rate/KL weight")
    if args.epochs <= 0 or args.examples_per_cache_epoch <= 0:
        raise ValueError("epochs and examples per cache epoch must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.formal_contract:
        validate_formal_contract(args)

    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if (
        not checkpoint_epochs
        or checkpoint_epochs[0] <= 0
        or checkpoint_epochs[-1] != args.epochs
    ):
        raise ValueError("checkpoint epochs must be positive and end at --epochs")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    pair_rows, rare_audit, pair_manifest, pair_audit = load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    caches, manifests = load_caches(args.train_cache)
    rare_indices, common_indices = validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )

    minimum_per_class = args.epochs * (args.examples_per_cache_epoch // 2)
    if len(pair_rows) > minimum_per_class:
        raise RuntimeError(
            "each fixed-noise cache cannot cover every rare/common pair: "
            f"pairs={len(pair_rows)} minimum_class_draws={minimum_per_class}"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, model_config = common.model_from_cache(caches[0], args.ablation)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    schedule_generator = torch.Generator().manual_seed(args.seed + 910000)
    samplers = {
        (cache_index, class_name): CyclingPairSampler(
            len(pair_rows),
            args.seed * 100000 + cache_index * 1000 + offset,
        )
        for cache_index in range(len(caches))
        for class_name, offset in (("rare", 17), ("common", 37))
    }

    total_examples = 0
    total_optimizer_steps = 0
    class_examples = {"rare": 0, "common": 0}
    class_examples_by_cache = {
        str(index): {"rare": 0, "common": 0}
        for index in range(len(caches))
    }
    accumulated_loss = 0.0
    accumulated_policy = 0.0
    accumulated_kl = 0.0
    checkpoint_reports = []
    global_block = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        cache_order = torch.randperm(
            len(caches), generator=schedule_generator
        ).tolist()
        for cache_index in cache_order:
            rare_count, common_count = block_class_counts(
                args.examples_per_cache_epoch, global_block
            )
            rare_positions = samplers[(cache_index, "rare")].take(rare_count)
            common_positions = samplers[(cache_index, "common")].take(common_count)
            selected = [
                rare_indices[position] for position in rare_positions
            ] + [
                common_indices[position] for position in common_positions
            ]
            permutation = torch.randperm(
                len(selected), generator=schedule_generator
            ).tolist()
            order = torch.tensor(
                [selected[index] for index in permutation], dtype=torch.long
            )
            cache = caches[cache_index]
            for start in range(0, len(order), args.batch_size):
                indices = order[start : start + args.batch_size]
                logits, reference = common.current_logits(
                    model, cache, indices, device
                )
                rewards = cache["candidate_rewards"][indices].to(device)
                valid = cache["candidate_reward_valid_mask"][indices].to(device)
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
                    raise RuntimeError(
                        f"non-finite V3 rare-log loss at epoch {epoch}"
                    )
                loss.backward()
                common.clip_grad_norm_cpu_(model.parameters(), 10.0)
                optimizer.step()
                total_optimizer_steps += 1
                accumulated_loss += float(loss.detach().cpu())
                accumulated_policy += float(policy.detach().cpu())
                accumulated_kl += float(kl.detach().cpu())

            class_examples["rare"] += rare_count
            class_examples["common"] += common_count
            class_examples_by_cache[str(cache_index)]["rare"] += rare_count
            class_examples_by_cache[str(cache_index)]["common"] += common_count
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
                pair_manifest,
                pair_audit,
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
                        "optimizer_steps": total_optimizer_steps,
                        "examples": total_examples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if class_examples["rare"] != class_examples["common"]:
        raise RuntimeError("global rare/common sampling balance drifted")
    if args.formal_contract:
        if total_examples != FORMAL_TOTAL_EXAMPLES:
            raise RuntimeError("formal total example budget drifted")
        if total_optimizer_steps != FORMAL_OPTIMIZER_STEPS:
            raise RuntimeError("formal optimizer-step budget drifted")

    sampling_coverage = {}
    for cache_index in range(len(caches)):
        sampling_coverage[str(cache_index)] = {}
        for class_name in ("rare", "common"):
            sampler = samplers[(cache_index, class_name)]
            if len(sampler.visited) != len(pair_rows):
                raise RuntimeError(
                    f"cache {cache_index} {class_name} did not cover every pair row"
                )
            sampling_coverage[str(cache_index)][class_name] = {
                "unique_pair_rows_visited": len(sampler.visited),
                "pair_rows": len(pair_rows),
                "completed_cycles": sampler.completed_cycles,
            }

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.method_name,
        "source_kind": "full_navtrain_rare_log_v1",
        "ablation": args.ablation,
        "scene_selector_config": model_config,
        "implementation_files": implementation_provenance(),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "early_stopping": False,
        "fixed_budget_selection": "final_epoch",
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": common.sha256_file(pair_manifest),
        "rare_data_audit": str(pair_audit),
        "rare_data_audit_sha256": common.sha256_file(pair_audit),
        "rare_pair_rows": len(pair_rows),
        "unique_common_tokens": len(
            {row["common_token"] for row in pair_rows}
        ),
        "train_manifests": list(manifests),
        "sampling": {
            "contract": (
                "three fixed noise caches; each cache contributes the original "
                "V3 6339-example epoch budget; rare/common are globally 1:1"
            ),
            "examples_per_cache_epoch": args.examples_per_cache_epoch,
            "total_examples": total_examples,
            "total_optimizer_steps": total_optimizer_steps,
            "class_examples": class_examples,
            "class_examples_by_cache": class_examples_by_cache,
            "coverage": sampling_coverage,
        },
        "mean_training_loss": accumulated_loss / total_optimizer_steps,
        "mean_training_policy": accumulated_policy / total_optimizer_steps,
        "mean_training_kl": accumulated_kl / total_optimizer_steps,
        "checkpoints": checkpoint_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V3 rare-log cached selector training: {report_path}")


if __name__ == "__main__":
    main()
