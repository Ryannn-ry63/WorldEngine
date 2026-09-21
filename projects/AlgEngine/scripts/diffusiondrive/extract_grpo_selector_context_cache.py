#!/usr/bin/env python3
"""Extract schema-v2 frozen candidate/context caches for selector GRPO V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d_plugin.datasets.builder import build_dataloader

import extract_grpo_selector_diagnostic_cache as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--nav-filter", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "certification"), required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--expected-num-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--launcher", default="pytorch")
    return parser.parse_args()


def cpu_record(token, scene, feature, context, result, index):
    values = {
        "feature": feature[index].detach().half().cpu(),
        "candidates": result["candidate_trajectories_8"][index].detach().float().cpu(),
        "route_bev": context["route_bev_features"][index].detach().half().cpu(),
        "status": context["status_token"][index].detach().half().cpu(),
        "ego": context["ego_query"][index].detach().half().cpu(),
        "agents": context["agents_query"][index].detach().half().cpu(),
        "rewards": result["candidate_rewards"][index].detach().float().cpu(),
        "components": result["candidate_reward_components"][index].detach().float().cpu(),
        "valid": result["candidate_reward_valid_mask"][index].detach().bool().cpu(),
        "reference_logits": result["reference_selector_logits"][index].detach().float().cpu(),
    }
    expected = {
        "feature": (20, 256),
        "candidates": (20, 8, 3),
        "route_bev": (20, 8, 256),
        "status": (1, 256),
        "ego": (1, 256),
        "agents": (30, 256),
        "rewards": (20,),
        "components": (20, 6),
        "valid": (20,),
        "reference_logits": (20,),
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise RuntimeError(f"{token}: {name} shape {tuple(values[name].shape)} != {shape}")
    if not bool(values["valid"].all()):
        raise RuntimeError(f"{token}: invalid candidates in context cache")
    for name, value in values.items():
        if name != "valid" and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{token}: non-finite {name}")
    return {
        "token": str(token),
        "scene": str(scene),
        **values,
        "digest": base.tensor_digest(*values.values()),
    }


def merge_parts(output_dir, world_size, expected_tokens, selector_state, selector_config):
    by_token = {}
    for rank in range(world_size):
        part = torch.load(output_dir / "parts" / f"rank{rank}.pt", map_location="cpu")
        for record in part:
            previous = by_token.get(record["token"])
            if previous is not None:
                if previous["digest"] != record["digest"] or previous["scene"] != record["scene"]:
                    raise RuntimeError("duplicate token produced different context: " + record["token"])
                continue
            by_token[record["token"]] = record
    if len(by_token) != expected_tokens:
        raise RuntimeError(f"merged coverage {len(by_token)} != {expected_tokens}")
    rows = [by_token[token] for token in sorted(by_token)]
    cache = {
        "schema_version": 2,
        "tokens": [row["token"] for row in rows],
        "scenes": [row["scene"] for row in rows],
        "candidate_features": torch.stack([row["feature"] for row in rows]),
        "candidate_trajectories_8": torch.stack([row["candidates"] for row in rows]),
        "route_bev_features": torch.stack([row["route_bev"] for row in rows]),
        "status_tokens": torch.stack([row["status"] for row in rows]),
        "ego_queries": torch.stack([row["ego"] for row in rows]),
        "agents_queries": torch.stack([row["agents"] for row in rows]),
        "candidate_rewards": torch.stack([row["rewards"] for row in rows]),
        "candidate_reward_components": torch.stack([row["components"] for row in rows]),
        "candidate_reward_valid_mask": torch.stack([row["valid"] for row in rows]),
        "reference_logits": torch.stack([row["reference_logits"] for row in rows]),
        "baseline_selector_state": selector_state,
        "scene_selector_config": selector_config,
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
    checkpoint_sha = base.sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("context-cache baseline checkpoint SHA256 mismatch")
    filter_tokens = base.load_filter_tokens(nav_filter)
    if len(filter_tokens) != args.expected_num_tokens:
        raise RuntimeError("filter token count disagrees with expected count")

    cfg = Config.fromfile(str(config))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.model.planning_head.reference_checkpoint = str(checkpoint)
    cfg.model.planning_head.reference_checkpoint_sha256 = checkpoint_sha
    dataset_cfg = base.configure_dataset(cfg, nav_filter)
    annotation_file = Path(dataset_cfg.ann_file).expanduser().resolve()
    scene_map = base.load_scene_map(annotation_file, filter_tokens)

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
    if head.scene_selector is None:
        raise RuntimeError("schema-v2 extraction requires a V3 scene selector")
    head.initialize_reference_selector()
    namespace = f"selector_context_{args.split}_seed{args.noise_seed}"
    head.set_candidate_noise_namespace(namespace)
    selector_state = {
        key: value.detach().cpu() for key, value in head.reference_selector.state_dict().items()
    }
    selector_config = head.scene_selector.config_dict()
    model.eval()
    head.train(True)

    features = []
    contexts = []
    results = []
    original_generate = head._generate_frozen_candidates
    original_context = head._scene_selector_context
    original_loss = head.loss

    def capture_generate(*call_args, **call_kwargs):
        candidates, feature = original_generate(*call_args, **call_kwargs)
        features.append(feature)
        return candidates, feature

    def capture_context(*call_args, **call_kwargs):
        context = original_context(*call_args, **call_kwargs)
        contexts.append(context)
        return context

    def capture_loss(result=None, *loss_args, **loss_kwargs):
        results.append(result)
        return original_loss(result, *loss_args, **loss_kwargs)

    head._generate_frozen_candidates = capture_generate
    head._scene_selector_context = capture_context
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
            if features or contexts or results:
                raise RuntimeError("stale context-cache capture")
            model(return_loss=True, **data)
            if len(features) != 1 or len(contexts) != 1 or len(results) != 1:
                raise RuntimeError("expected one feature/context/result capture")
            feature, context, result = features.pop(), contexts.pop(), results.pop()
            tokens = result.get("sample_tokens")
            if tokens is None or len(tokens) != feature.shape[0]:
                raise RuntimeError("sample-token provenance missing")
            for index, token in enumerate(tokens):
                records.append(
                    cpu_record(token, scene_map[str(token)], feature, context, result, index)
                )
    torch.save(records, output_dir / "parts" / f"rank{rank}.pt")
    dist.barrier()
    if rank != 0:
        return
    cache_path, cache = merge_parts(
        output_dir, world_size, args.expected_num_tokens, selector_state, selector_config
    )
    algengine_root = Path(__file__).resolve().parents[2]
    implementation_paths = (
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py",
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py",
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py",
        algengine_root / "mmdet3d_plugin/navformer/detectors/navformer.py",
    )
    manifest = {
        "schema_version": 2,
        "status": "PASS",
        "method": "frozen_diffusiondrive_scene_selector_context_cache",
        "implementation_files": {
            str(path): base.sha256_file(path) for path in implementation_paths
        },
        "split": args.split,
        "noise_seed": args.noise_seed,
        "noise_namespace": namespace,
        "num_tokens": len(cache["tokens"]),
        "num_scenes": len(set(cache["scenes"])),
        "num_candidates": 20,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config": str(config),
        "config_sha256": base.sha256_file(config),
        "annotation_file": str(annotation_file),
        "annotation_sha256": base.sha256_file(annotation_file),
        "nav_filter": str(nav_filter),
        "nav_filter_sha256": base.sha256_file(nav_filter),
        "cache": str(cache_path),
        "cache_sha256": base.sha256_file(cache_path),
        "world_size": world_size,
        "tensor_shapes": {
            key: list(value.shape) for key, value in cache.items() if torch.is_tensor(value)
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    print(f"PASS selector context cache split={args.split} seed={args.noise_seed}")


if __name__ == "__main__":
    main()
