#!/usr/bin/env python3
"""Rescore immutable V2 candidate caches with the corrected PDM reward.

The expensive perception and frozen DiffusionDrive candidate generation are
not repeated.  Each H100 rank scores a disjoint subset of the exact cached
trajectories.  The merge step fails closed unless every non-reward tensor is
bit-identical to the pinned source cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch


REWARD_CONTRACT = "navsim_pairwise_raw_progress_then_candidate_gate_v1"
SOURCE_METHOD = "frozen_diffusiondrive_selector_diagnostic_cache"
OUTPUT_METHOD = "frozen_diffusiondrive_selector_v2_progress_fixed_cache"
BASELINE_SHA256 = (
    "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
)
IMMUTABLE_TENSOR_KEYS = (
    "candidate_features",
    "candidate_trajectories_8",
    "reference_logits",
    "current_logits",
)
REWARD_KEYS = (
    "candidate_rewards",
    "candidate_reward_components",
    "candidate_reward_valid_mask",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reward_implementation_path():
    return (
        Path(__file__).resolve().parents[2]
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    )


def _update_tensor_digest(digest, name, tensor):
    value = tensor.detach().contiguous().cpu()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def immutable_digest(cache):
    digest = hashlib.sha256()
    for key in ("tokens", "scenes"):
        digest.update(key.encode())
        digest.update(json.dumps([str(value) for value in cache[key]]).encode())
    for key in IMMUTABLE_TENSOR_KEYS:
        _update_tensor_digest(digest, key, cache[key])
    selector = cache["baseline_selector_state"]
    if not isinstance(selector, dict):
        raise RuntimeError("baseline_selector_state is not a tensor mapping")
    for key in sorted(selector):
        _update_tensor_digest(digest, f"baseline_selector_state.{key}", selector[key])
    return digest.hexdigest()


def validate_cache_shapes(cache, count):
    expected = {
        "candidate_features": (count, 20, 256),
        "candidate_trajectories_8": (count, 20, 8, 3),
        "candidate_rewards": (count, 20),
        "candidate_reward_components": (count, 20, 6),
        "candidate_reward_valid_mask": (count, 20),
        "reference_logits": (count, 20),
        "current_logits": (count, 20),
    }
    for key, shape in expected.items():
        value = cache.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            actual = None if value is None else tuple(value.shape)
            raise RuntimeError(f"{key} shape drifted: {actual} != {shape}")
    if len(cache.get("tokens", ())) != count or len(cache.get("scenes", ())) != count:
        raise RuntimeError("token/scene count drifted")
    if len(set(str(token) for token in cache["tokens"])) != count:
        raise RuntimeError("source cache contains duplicate tokens")
    if not isinstance(cache.get("baseline_selector_state"), dict):
        raise RuntimeError("source cache omitted baseline selector state")


def load_source(args):
    cache_path = args.source_cache.expanduser().resolve()
    manifest_path = args.source_manifest.expanduser().resolve()
    for path in (cache_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(cache_path) != args.expected_source_cache_sha256:
        raise RuntimeError("source cache SHA256 mismatch")
    if sha256_file(manifest_path) != args.expected_source_manifest_sha256:
        raise RuntimeError("source manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "status": "PASS",
        "schema_version": 1,
        "method": SOURCE_METHOD,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "nav_filter_sha256": args.expected_nav_filter_sha256,
        "split": args.expected_split,
        "noise_seed": args.expected_noise_seed,
        "num_tokens": args.expected_num_tokens,
        "cache_sha256": args.expected_source_cache_sha256,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"source manifest {key} drifted: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    cache = torch.load(cache_path, map_location="cpu")
    if cache.get("schema_version") != 1:
        raise RuntimeError("source cache schema is not the pinned V2 schema")
    validate_cache_shapes(cache, args.expected_num_tokens)
    limit = args.limit or args.expected_num_tokens
    if limit < 1 or limit > args.expected_num_tokens:
        raise ValueError("limit must be in [1, expected-num-tokens]")
    return cache_path, manifest_path, cache, manifest, limit


def subset_immutable_cache(source, count):
    output = {
        "schema_version": 2,
        "tokens": [str(token) for token in source["tokens"][:count]],
        "scenes": [str(scene) for scene in source["scenes"][:count]],
        "baseline_selector_state": {
            key: value.detach().cpu().clone()
            for key, value in source["baseline_selector_state"].items()
        },
        "reward_contract": REWARD_CONTRACT,
    }
    for key in IMMUTABLE_TENSOR_KEYS:
        output[key] = source[key][:count].detach().cpu().clone()
    return output


def assert_immutable_equal(source, corrected, count):
    if [str(value) for value in source["tokens"][:count]] != corrected["tokens"]:
        raise RuntimeError("corrected token order drifted")
    if [str(value) for value in source["scenes"][:count]] != corrected["scenes"]:
        raise RuntimeError("corrected scene order drifted")
    for key in IMMUTABLE_TENSOR_KEYS:
        if not torch.equal(source[key][:count], corrected[key]):
            raise RuntimeError(f"corrected cache mutated immutable tensor {key}")
    source_selector = source["baseline_selector_state"]
    corrected_selector = corrected["baseline_selector_state"]
    if set(source_selector) != set(corrected_selector):
        raise RuntimeError("corrected baseline selector keys drifted")
    for key in source_selector:
        if not torch.equal(source_selector[key], corrected_selector[key]):
            raise RuntimeError(f"corrected baseline selector tensor drifted: {key}")


def expected_rank_indices(count, rank, world_size):
    return list(range(rank, count, world_size))


def valid_part(path, args, indices, reward_sha):
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        return (
            payload.get("status") == "PASS"
            and payload.get("source_cache_sha256")
            == args.expected_source_cache_sha256
            and payload.get("reward_implementation_sha256") == reward_sha
            and payload.get("world_size") == args.world_size
            and payload.get("rank") == args.rank
            and payload.get("indices") == indices
            and tuple(payload["rewards"].shape) == (len(indices), 20)
            and tuple(payload["components"].shape) == (len(indices), 20, 6)
            and tuple(payload["valid"].shape) == (len(indices), 20)
            and bool(payload["valid"].all())
        )
    except Exception:
        return False


def run_shard(args):
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    _, _, source, _, count = load_source(args)
    metric_cache = args.metric_cache_path.expanduser().resolve()
    if not metric_cache.is_dir():
        raise FileNotFoundError(metric_cache)
    reward_path = reward_implementation_path()
    reward_sha = sha256_file(reward_path)
    indices = expected_rank_indices(count, args.rank, args.world_size)
    output_dir = args.output_dir.expanduser().resolve()
    part_path = output_dir / "parts" / f"rank{args.rank}.pt"
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if valid_part(part_path, args, indices, reward_sha):
        print(f"REUSE PASS corrected reward shard rank={args.rank}")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache rescoring requested but unavailable")
    from mmdet3d_plugin.navformer.dense_heads.diffusiondrive_online_pdm_reward import (
        OnlineDiffusionDrivePDMReward,
    )

    device = torch.device(args.device)
    reward = OnlineDiffusionDrivePDMReward(
        metric_cache_path=str(metric_cache),
        num_candidates=20,
        fail_on_missing_cache=True,
        use_cuda_simulator=device.type == "cuda",
    )
    rewards = []
    components = []
    valid = []
    with torch.no_grad():
        for local_index, source_index in enumerate(indices, start=1):
            candidates = source["candidate_trajectories_8"][
                source_index : source_index + 1
            ].to(device=device, dtype=torch.float32)
            payload = reward(candidates, [str(source["tokens"][source_index])])
            local_valid = payload["valid_mask"].detach().bool().cpu()[0]
            if not bool(local_valid.all()):
                raise RuntimeError(
                    f"corrected reward invalid for token {source['tokens'][source_index]}"
                )
            rewards.append(payload["score"].detach().float().cpu()[0])
            components.append(payload["components"].detach().float().cpu()[0])
            valid.append(local_valid)
            if local_index % 25 == 0 or local_index == len(indices):
                print(
                    f"rank={args.rank} corrected={local_index}/{len(indices)}",
                    flush=True,
                )
    if indices:
        reward_tensor = torch.stack(rewards)
        component_tensor = torch.stack(components)
        valid_tensor = torch.stack(valid)
    else:
        reward_tensor = torch.empty((0, 20), dtype=torch.float32)
        component_tensor = torch.empty((0, 20, 6), dtype=torch.float32)
        valid_tensor = torch.empty((0, 20), dtype=torch.bool)
    part = {
        "schema_version": 1,
        "status": "PASS",
        "rank": args.rank,
        "world_size": args.world_size,
        "indices": indices,
        "source_cache_sha256": args.expected_source_cache_sha256,
        "reward_contract": REWARD_CONTRACT,
        "reward_implementation_sha256": reward_sha,
        "rewards": reward_tensor,
        "components": component_tensor,
        "valid": valid_tensor,
    }
    temporary = part_path.with_name(f"{part_path.name}.tmp.{os.getpid()}")
    torch.save(part, temporary)
    temporary.replace(part_path)
    print(f"PASS corrected reward shard rank={args.rank} rows={len(indices)}")


def build_corrected_cache(source, parts, count):
    corrected = subset_immutable_cache(source, count)
    rewards = torch.full((count, 20), torch.nan, dtype=torch.float32)
    components = torch.full((count, 20, 6), torch.nan, dtype=torch.float32)
    valid = torch.zeros((count, 20), dtype=torch.bool)
    seen = torch.zeros(count, dtype=torch.bool)
    for part in parts:
        for local_index, source_index in enumerate(part["indices"]):
            if source_index < 0 or source_index >= count or bool(seen[source_index]):
                raise RuntimeError(f"duplicate/out-of-range rescore index {source_index}")
            rewards[source_index] = part["rewards"][local_index]
            components[source_index] = part["components"][local_index]
            valid[source_index] = part["valid"][local_index]
            seen[source_index] = True
    if not bool(seen.all()):
        raise RuntimeError(f"rescore coverage incomplete: {int(seen.sum())}/{count}")
    if not bool(valid.all()):
        raise RuntimeError("merged corrected cache contains invalid rewards")
    if not torch.isfinite(rewards).all() or not torch.isfinite(components).all():
        raise RuntimeError("merged corrected cache contains non-finite rewards")
    corrected["candidate_rewards"] = rewards
    corrected["candidate_reward_components"] = components
    corrected["candidate_reward_valid_mask"] = valid
    assert_immutable_equal(source, corrected, count)
    return corrected


def run_merge(args):
    cache_path, source_manifest_path, source, source_manifest, count = load_source(
        args
    )
    reward_path = reward_implementation_path()
    validator = args.validator.expanduser().resolve()
    metric_cache = args.metric_cache_path.expanduser().resolve()
    for path in (reward_path, validator):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not metric_cache.is_dir():
        raise FileNotFoundError(metric_cache)
    reward_sha = sha256_file(reward_path)
    output_dir = args.output_dir.expanduser().resolve()
    parts = []
    for rank in range(args.world_size):
        part_path = output_dir / "parts" / f"rank{rank}.pt"
        if not part_path.is_file():
            raise FileNotFoundError(part_path)
        part = torch.load(part_path, map_location="cpu")
        expected_indices = expected_rank_indices(count, rank, args.world_size)
        checks = {
            "status": "PASS",
            "rank": rank,
            "world_size": args.world_size,
            "indices": expected_indices,
            "source_cache_sha256": args.expected_source_cache_sha256,
            "reward_contract": REWARD_CONTRACT,
            "reward_implementation_sha256": reward_sha,
        }
        for key, expected in checks.items():
            if part.get(key) != expected:
                raise RuntimeError(f"rank {rank} part {key} drifted")
        parts.append(part)
    corrected = build_corrected_cache(source, parts, count)
    source_subset = subset_immutable_cache(source, count)
    source_immutable_sha = immutable_digest(source_subset)
    corrected_immutable_sha = immutable_digest(corrected)
    if source_immutable_sha != corrected_immutable_sha:
        raise RuntimeError("immutable cache digest changed during reward-only rescore")

    corrected["source_cache_sha256"] = args.expected_source_cache_sha256
    corrected["reward_implementation_sha256"] = reward_sha
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cache = output_dir / "cache.pt"
    temporary = output_cache.with_name(f"{output_cache.name}.tmp.{os.getpid()}")
    torch.save(corrected, temporary)
    temporary.replace(output_cache)
    reloaded = torch.load(output_cache, map_location="cpu")
    validate_cache_shapes(reloaded, count)
    assert_immutable_equal(source, reloaded, count)

    old_rewards = source["candidate_rewards"][:count].float()
    old_components = source["candidate_reward_components"][:count].float()
    reward_delta = corrected["candidate_rewards"] - old_rewards
    component_delta = corrected["candidate_reward_components"] - old_components
    output_manifest = {
        **copy.deepcopy(source_manifest),
        "schema_version": 2,
        "status": "PASS",
        "method": OUTPUT_METHOD,
        "reward_contract": REWARD_CONTRACT,
        "num_tokens": count,
        "num_scenes": len(set(corrected["scenes"])),
        "source_total_tokens": args.expected_num_tokens,
        "source_cache": str(cache_path),
        "source_cache_sha256": args.expected_source_cache_sha256,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "cache": str(output_cache),
        "cache_sha256": sha256_file(output_cache),
        "metric_cache_path": str(metric_cache),
        "reward_implementation": str(reward_path),
        "reward_implementation_sha256": reward_sha,
        "validator": str(validator),
        "validator_sha256": sha256_file(validator),
        "immutable_source_sha256": source_immutable_sha,
        "immutable_output_sha256": corrected_immutable_sha,
        "world_size": args.world_size,
        "reward_delta": {
            "changed_value_count": int((reward_delta != 0).sum()),
            "max_abs": float(reward_delta.abs().max()),
            "mean": float(reward_delta.mean()),
            "component_changed_value_count": int((component_delta != 0).sum()),
            "component_max_abs": float(component_delta.abs().max()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output_manifest, sort_keys=True))
    print(
        f"PASS V2 corrected cache split={args.expected_split} "
        f"seed={args.expected_noise_seed} rows={count}"
    )


def add_common_args(parser):
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=BASELINE_SHA256
    )
    parser.add_argument("--expected-nav-filter-sha256", required=True)
    parser.add_argument("--expected-split", choices=("train", "calibration"), required=True)
    parser.add_argument("--expected-noise-seed", type=int, required=True)
    parser.add_argument("--expected-num-tokens", type=int, required=True)
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    add_common_args(shard)
    shard.add_argument("--rank", type=int, required=True)
    shard.add_argument("--device", default="cuda:0")
    merge = subparsers.add_parser("merge")
    add_common_args(merge)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("world-size must be positive")
    if args.expected_num_tokens < 1:
        raise ValueError("expected-num-tokens must be positive")
    if args.command == "shard":
        run_shard(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
