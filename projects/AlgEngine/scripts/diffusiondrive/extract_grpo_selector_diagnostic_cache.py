#!/usr/bin/env python3
"""Cache frozen DiffusionDrive candidates/features/online PDM rewards.

This is a diagnostic data extractor, not a new training method.  It executes
the canonical selector-GRPO forward path once and stores the exact action set
needed to compare selector objectives without repeatedly running perception,
the frozen DiT generator, or NAVSIM PDM simulation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mmcv
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--nav-filter", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--expected-num-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--launcher", default="pytorch")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_filter_tokens(path):
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
        raise ValueError(f"filter has no tokens list: {path}")
    tokens = [str(token) for token in payload["tokens"]]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"filter contains duplicate tokens: {path}")
    return tokens


def load_scene_map(annotation_file, eligible_tokens):
    payload = mmcv.load(str(annotation_file), file_format="pkl")
    infos = payload["infos"] if isinstance(payload, dict) else payload
    eligible = set(eligible_tokens)
    mapping = {}
    for info in infos:
        token = str(info["token"])
        if token in eligible:
            mapping[token] = str(info.get("scene_token") or info["log_token"])
    missing = eligible - set(mapping)
    if missing:
        raise RuntimeError(f"annotation omitted {len(missing)} filtered tokens")
    return mapping


def configure_dataset(cfg, nav_filter):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    dataset_cfg.nav_filter_path = str(nav_filter)
    dataset_cfg.test_mode = False
    return dataset_cfg


def cpu_record(token, scene, feature, result, index):
    candidates = result["candidate_trajectories_8"][index].detach().float().cpu()
    rewards = result["candidate_rewards"][index].detach().float().cpu()
    components = result["candidate_reward_components"][index].detach().float().cpu()
    valid = result["candidate_reward_valid_mask"][index].detach().bool().cpu()
    reference_logits = result["reference_selector_logits"][index].detach().float().cpu()
    current_logits = result["selector_logits"][index].detach().float().cpu()
    local_feature = feature[index].detach().float().cpu()
    expected = {
        "feature": (20, 256),
        "candidates": (20, 8, 3),
        "rewards": (20,),
        "components": (20, 6),
        "valid": (20,),
        "reference_logits": (20,),
        "current_logits": (20,),
    }
    values = {
        "feature": local_feature,
        "candidates": candidates,
        "rewards": rewards,
        "components": components,
        "valid": valid,
        "reference_logits": reference_logits,
        "current_logits": current_logits,
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise RuntimeError(f"{token}: {name} shape {tuple(values[name].shape)} != {shape}")
    if not bool(valid.all()) or not all(torch.isfinite(value).all() for value in values.values()):
        raise RuntimeError(f"{token}: invalid/non-finite diagnostic cache row")
    return {
        "token": str(token),
        "scene": str(scene),
        **values,
        "digest": tensor_digest(*values.values()),
    }


def merge_parts(output_dir, world_size, expected_tokens, selector_state):
    by_token = {}
    for rank in range(world_size):
        part = torch.load(output_dir / "parts" / f"rank{rank}.pt", map_location="cpu")
        for record in part:
            previous = by_token.get(record["token"])
            if previous is not None:
                if previous["digest"] != record["digest"] or previous["scene"] != record["scene"]:
                    raise RuntimeError(
                        "distributed duplicate token produced different diagnostic data: "
                        + record["token"]
                    )
                continue
            by_token[record["token"]] = record
    if len(by_token) != expected_tokens:
        raise RuntimeError(f"merged coverage {len(by_token)} != {expected_tokens}")
    rows = [by_token[token] for token in sorted(by_token)]
    cache = {
        "schema_version": 1,
        "tokens": [row["token"] for row in rows],
        "scenes": [row["scene"] for row in rows],
        "candidate_features": torch.stack([row["feature"] for row in rows]).half(),
        "candidate_trajectories_8": torch.stack([row["candidates"] for row in rows]),
        "candidate_rewards": torch.stack([row["rewards"] for row in rows]),
        "candidate_reward_components": torch.stack([row["components"] for row in rows]),
        "candidate_reward_valid_mask": torch.stack([row["valid"] for row in rows]),
        "reference_logits": torch.stack([row["reference_logits"] for row in rows]),
        "current_logits": torch.stack([row["current_logits"] for row in rows]),
        "baseline_selector_state": selector_state,
    }
    cache_path = output_dir / "cache.pt"
    torch.save(cache, cache_path)
    return cache_path, cache


def main():
    args = parse_args()
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    nav_filter = args.nav_filter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (config, checkpoint, nav_filter):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("diagnostic baseline checkpoint SHA256 mismatch")
    filter_tokens = load_filter_tokens(nav_filter)
    if len(filter_tokens) != args.expected_num_tokens:
        raise RuntimeError("filter token count disagrees with expected count")

    cfg = Config.fromfile(str(config))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.model.planning_head.reference_checkpoint = str(checkpoint)
    cfg.model.planning_head.reference_checkpoint_sha256 = checkpoint_sha
    dataset_cfg = configure_dataset(cfg, nav_filter)
    annotation_file = Path(dataset_cfg.ann_file).expanduser().resolve()
    scene_map = load_scene_map(annotation_file, filter_tokens)

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    set_random_seed(args.noise_seed, deterministic=False)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != args.expected_num_tokens:
        raise RuntimeError(f"dataset coverage {len(dataset)} != {args.expected_num_tokens}")
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=True,
        shuffle=False,
        seed=0,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=False)
    head = model.planning_head
    head.initialize_reference_selector()
    namespace = f"selector_diagnostic_{args.split}_seed{args.noise_seed}"
    head.set_candidate_noise_namespace(namespace)
    selector_state = {
        key: value.detach().cpu()
        for key, value in head.reference_selector.state_dict().items()
    }
    model.eval()
    head.train(True)
    feature_capture = []
    result_capture = []
    original_generate = head._generate_frozen_candidates
    original_loss = head.loss

    def capture_generate(*call_args, **call_kwargs):
        candidates, feature = original_generate(*call_args, **call_kwargs)
        feature_capture.append(feature)
        return candidates, feature

    def capture_loss(result=None, *loss_args, **loss_kwargs):
        result_capture.append(result)
        return original_loss(result, *loss_args, **loss_kwargs)

    head._generate_frozen_candidates = capture_generate
    head.loss = capture_loss
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
    )
    if rank == 0:
        (output_dir / "parts").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    records = []
    with torch.no_grad():
        for data in loader:
            if feature_capture or result_capture:
                raise RuntimeError("stale diagnostic capture")
            model(return_loss=True, **data)
            if len(feature_capture) != 1 or len(result_capture) != 1:
                raise RuntimeError("expected exactly one feature/result capture")
            feature = feature_capture.pop()
            result = result_capture.pop()
            tokens = result.get("sample_tokens")
            if tokens is None or len(tokens) != feature.shape[0]:
                raise RuntimeError("sample token provenance missing from diagnostic result")
            for index, token in enumerate(tokens):
                records.append(cpu_record(token, scene_map[str(token)], feature, result, index))
    torch.save(records, output_dir / "parts" / f"rank{rank}.pt")
    dist.barrier()
    if rank != 0:
        return
    cache_path, cache = merge_parts(
        output_dir, world_size, args.expected_num_tokens, selector_state
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": "frozen_diffusiondrive_selector_diagnostic_cache",
        "split": args.split,
        "noise_seed": args.noise_seed,
        "noise_namespace": namespace,
        "num_tokens": len(cache["tokens"]),
        "num_scenes": len(set(cache["scenes"])),
        "num_candidates": 20,
        "feature_shape": list(cache["candidate_features"].shape),
        "trajectory_shape": list(cache["candidate_trajectories_8"].shape),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "annotation_file": str(annotation_file),
        "annotation_sha256": sha256_file(annotation_file),
        "nav_filter": str(nav_filter),
        "nav_filter_sha256": sha256_file(nav_filter),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "world_size": world_size,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    print(f"PASS selector diagnostic cache split={args.split} seed={args.noise_seed}")


if __name__ == "__main__":
    main()
