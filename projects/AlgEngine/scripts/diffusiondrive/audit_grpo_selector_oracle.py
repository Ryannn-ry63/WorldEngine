#!/usr/bin/env python3
"""Audit the full best-of-20 ceiling of a frozen DiffusionDrive generator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d_plugin.datasets.builder import build_dataloader
from navsim.common.dataclasses import Trajectory

from grpo_selector_oracle_common import (
    COMPONENT_NAMES,
    candidate_diversity,
    summarize_records,
    write_records,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--nav-filter", type=Path, required=True)
    parser.add_argument("--failures-filter", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--expected-num-tokens", type=int, default=12146)
    parser.add_argument("--expected-num-failures", type=int, default=288)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-pdm-simulator", action="store_true")
    parser.add_argument("--launcher", default="pytorch")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exponent = np.exp(logits - logits.max())
    return exponent / exponent.sum()


def component_dict(values: np.ndarray) -> Dict[str, float]:
    return {
        name: float(values[index])
        for index, name in enumerate(COMPONENT_NAMES)
    }


def load_filter_tokens(path: Path) -> Sequence[str]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("tokens"), list
    ):
        raise ValueError(f"filter has no token list: {path}")
    tokens = [str(token) for token in payload["tokens"]]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"filter contains duplicate tokens: {path}")
    return tokens


def configure_audit_dataset(cfg, args):
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(args.annotation_file)
    dataset_cfg.nav_filter_path = str(args.nav_filter)
    dataset_cfg.metric_cache_path = str(args.metric_cache)
    dataset_cfg.test_mode = True
    loader_found = False
    for transform in dataset_cfg.pipeline:
        if transform.get("type") == "LoadMultiViewImageFromFilesWithDownsample":
            transform.img_root = str(args.image_root)
            loader_found = True
    if not loader_found:
        raise RuntimeError("audit pipeline has no multi-view image loader")
    return dataset_cfg


def record_batch(result) -> Sequence[dict]:
    tokens = result.get("sample_tokens")
    if tokens is None:
        raise RuntimeError("oracle audit result omitted sample tokens")
    candidates = (
        result["candidate_trajectories_8"].detach().float().cpu().numpy()
    )
    rewards = result["candidate_rewards"].detach().float().cpu().numpy()
    components = (
        result["candidate_reward_components"].detach().float().cpu().numpy()
    )
    valid = (
        result["candidate_reward_valid_mask"].detach().bool().cpu().numpy()
    )
    current_logits = result["selector_logits"].detach().float().cpu().numpy()
    reference_logits = (
        result["reference_selector_logits"].detach().float().cpu().numpy()
    )
    if candidates.shape[:2] != rewards.shape or rewards.shape != valid.shape:
        raise RuntimeError("candidate/reward alignment drifted")
    if components.shape[:2] != rewards.shape or components.shape[-1] != len(
        COMPONENT_NAMES
    ):
        raise RuntimeError("candidate reward component alignment drifted")
    if len(tokens) != rewards.shape[0]:
        raise RuntimeError("sample-token and candidate batch sizes drifted")

    records = []
    for batch_index, token in enumerate(tokens):
        local_valid = valid[batch_index]
        if local_valid.shape != (20,) or not bool(local_valid.all()):
            raise RuntimeError(
                f"token {token} did not produce exactly 20 valid rewards"
            )
        local_candidates = candidates[batch_index]
        local_rewards = rewards[batch_index]
        local_components = components[batch_index]
        current_probability = stable_softmax(current_logits[batch_index])
        reference_probability = stable_softmax(reference_logits[batch_index])
        current_index = int(np.argmax(current_logits[batch_index]))
        reference_index = int(np.argmax(reference_logits[batch_index]))
        oracle_index = int(np.argmax(local_rewards))
        indices = {
            "current": current_index,
            "reference": reference_index,
            "oracle": oracle_index,
        }
        diversity = candidate_diversity(local_candidates)
        current_top2 = np.sort(current_probability)[-2:]
        reference_top2 = np.sort(reference_probability)[-2:]
        record = {
            "token": str(token),
            "valid_candidate_count": int(local_valid.sum()),
            "candidate_trajectories_sha256": tensor_sha256(local_candidates),
            "candidate_rewards": local_rewards.astype(float).tolist(),
            "candidate_reward_components": local_components.astype(float).tolist(),
            "reward_spread": float(local_rewards.max() - local_rewards.min()),
            "reward_std": float(local_rewards.std()),
            "current_expected_reward": float(
                np.dot(current_probability, local_rewards)
            ),
            "reference_expected_reward": float(
                np.dot(reference_probability, local_rewards)
            ),
            "current_oracle_probability": float(
                current_probability[oracle_index]
            ),
            "reference_oracle_probability": float(
                reference_probability[oracle_index]
            ),
            "current_top1_margin": float(current_top2[-1] - current_top2[-2]),
            "reference_top1_margin": float(
                reference_top2[-1] - reference_top2[-2]
            ),
            "selection_disagreement": current_index != reference_index,
            **diversity,
        }
        for selection, index in indices.items():
            record[f"{selection}_index"] = index
            record[f"{selection}_reward"] = float(local_rewards[index])
            record[f"{selection}_components"] = component_dict(
                local_components[index]
            )
            record[f"{selection}_trajectory_8"] = (
                local_candidates[index].astype(float).tolist()
            )
            record[f"{selection}_oracle_match"] = index == oracle_index
        records.append(record)
    return records


def write_submission(path: Path, records: Sequence[Mapping], selection: str):
    predictions = {}
    trajectory_key = f"{selection}_trajectory_8"
    for record in records:
        predictions[str(record["token"])] = Trajectory(
            np.asarray(record[trajectory_key], dtype=np.float32)
        )
    payload = {
        "team_name": "DiffusionDrive GRPO oracle audit",
        "authors": ["DiffusionDrive GRPO oracle audit"],
        "email": "PLACEHOLDER@gmail.com",
        "institution": "PLACEHOLDER",
        "country / region": "PLACEHOLDER",
        "predictions": [predictions],
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def main():
    args = parse_args()
    paths = {
        "config": args.config.expanduser().resolve(),
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "baseline_checkpoint": args.baseline_checkpoint.expanduser().resolve(),
        "annotation_file": args.annotation_file.expanduser().resolve(),
        "image_root": args.image_root.expanduser().resolve(),
        "nav_filter": args.nav_filter.expanduser().resolve(),
        "failures_filter": args.failures_filter.expanduser().resolve(),
        "metric_cache": args.metric_cache.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    for name, path in paths.items():
        if name in ("image_root", "metric_cache"):
            exists = path.is_dir()
        else:
            exists = path.is_file()
        if not exists:
            raise FileNotFoundError(f"{name}: {path}")
    checkpoint_sha = sha256_file(paths["checkpoint"])
    baseline_sha = sha256_file(paths["baseline_checkpoint"])
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("oracle audit checkpoint SHA256 mismatch")
    if baseline_sha != args.expected_baseline_sha256:
        raise RuntimeError("oracle audit baseline SHA256 mismatch")

    cfg = Config.fromfile(str(paths["config"]))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.model.planning_head.reference_checkpoint = str(
        paths["baseline_checkpoint"]
    )
    cfg.model.planning_head.reference_checkpoint_sha256 = baseline_sha
    cfg.model.planning_head.online_reward.metric_cache_path = str(
        paths["metric_cache"]
    )
    if args.cpu_pdm_simulator:
        cfg.model.planning_head.online_reward.use_cuda_simulator = False
    dataset_cfg = configure_audit_dataset(cfg, args)

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    set_random_seed(args.noise_seed, deterministic=False)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != args.expected_num_tokens:
        raise RuntimeError(
            f"navtest coverage mismatch: {len(dataset)} != "
            f"{args.expected_num_tokens}"
        )
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=True,
        shuffle=False,
        seed=0,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    load_checkpoint(model, str(paths["checkpoint"]), map_location="cpu")
    head = model.planning_head
    if not hasattr(head, "selector_diagnostics_per_sample"):
        raise TypeError("checkpoint is not a DiffusionDrive selector-GRPO model")
    head.initialize_reference_selector()
    namespace = f"formal_navtest_seed{args.noise_seed}"
    head.set_candidate_noise_namespace(namespace)
    model.eval()
    captured = []
    original_forward_test = head._forward_test

    def audit_forward_test(
        bev_feature, ego_query, agents_query, status_token
    ):
        candidates_8, candidate_feature = head._generate_frozen_candidates(
            bev_feature, ego_query, agents_query, status_token
        )
        current_logits, reference_logits = head._selector_outputs(
            candidate_feature
        )
        if head._active_sample_tokens is None:
            raise RuntimeError("oracle audit inference omitted sample tokens")
        reward_payload = head.online_reward(
            candidates_8, head._active_sample_tokens
        )
        result = head._build_result(
            candidates_8,
            current_logits,
            reference_logits,
            reward_payload=reward_payload,
        )
        captured.append(result)
        return result

    # Preserve the exact formal model.forward_test / forward_track_test path.
    # Only the planning head inference leaf is replaced in this process so
    # the already generated 20 candidates are scored and captured.
    head._forward_test = audit_forward_test
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "parts").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    part_path = output_dir / "parts" / f"rank{rank}.jsonl"
    with part_path.open("w", encoding="utf-8") as stream, torch.no_grad():
        for data in data_loader:
            if captured:
                raise RuntimeError("stale oracle-audit inference capture")
            model(return_loss=False, rescale=True, **data)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one oracle-audit result, captured {len(captured)}"
                )
            for record in record_batch(captured.pop()):
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    head._forward_test = original_forward_test
    dist.barrier()

    if rank != 0:
        return

    by_token = {}
    for part_rank in range(world_size):
        path = output_dir / "parts" / f"rank{part_rank}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                token = record["token"]
                previous = by_token.get(token)
                if previous is not None:
                    if (
                        previous["candidate_trajectories_sha256"]
                        != record["candidate_trajectories_sha256"]
                        or previous["candidate_rewards"]
                        != record["candidate_rewards"]
                    ):
                        raise RuntimeError(
                            "distributed duplicate token produced different "
                            f"candidate results: {token}"
                        )
                    continue
                by_token[token] = record
    if len(by_token) != args.expected_num_tokens:
        raise RuntimeError(
            f"merged navtest coverage mismatch: {len(by_token)} != "
            f"{args.expected_num_tokens}"
        )
    records = [by_token[token] for token in sorted(by_token)]
    failure_tokens = set(load_filter_tokens(paths["failures_filter"]))
    if len(failure_tokens) != args.expected_num_failures:
        raise RuntimeError(
            f"failures filter count mismatch: {len(failure_tokens)} != "
            f"{args.expected_num_failures}"
        )
    missing_failures = failure_tokens - set(by_token)
    if missing_failures:
        raise RuntimeError(
            f"navtest audit omitted {len(missing_failures)} failure tokens"
        )
    failure_records = [
        record for record in records if record["token"] in failure_tokens
    ]
    records_path = output_dir / "records.jsonl.gz"
    write_records(records_path, records)
    for selection in ("reference", "current", "oracle"):
        write_submission(
            output_dir / f"{selection}_navsim_submission.pkl",
            records,
            selection,
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_best_of_20_oracle_audit",
        "train_seed": args.train_seed,
        "noise_seed": args.noise_seed,
        "noise_namespace": namespace,
        "world_size": world_size,
        "num_candidates": 20,
        "paths": {name: str(path) for name, path in paths.items()},
        "config_sha256": sha256_file(paths["config"]),
        "checkpoint_sha256": checkpoint_sha,
        "baseline_checkpoint_sha256": baseline_sha,
        "nav_filter_sha256": sha256_file(paths["nav_filter"]),
        "failures_filter_sha256": sha256_file(paths["failures_filter"]),
        "records": str(records_path),
        "submissions": {
            selection: str(
                output_dir / f"{selection}_navsim_submission.pkl"
            )
            for selection in ("reference", "current", "oracle")
        },
        "navtest": summarize_records(records),
        "navtest_failures": summarize_records(failure_records),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    for part_rank in range(world_size):
        (output_dir / "parts" / f"rank{part_rank}.jsonl").unlink()
    print(json.dumps(report, sort_keys=True))
    print(f"PASS oracle audit seed={args.noise_seed}")


if __name__ == "__main__":
    main()
