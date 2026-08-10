#!/usr/bin/env python3
"""Paired held-out navtrain calibration for DiffusionDrive selector GRPO."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
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


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def scalar(value):
    return float(value.detach().cpu().item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--calibration-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--launcher", default="pytorch")
    return parser.parse_args()


def record_from_result(head, result):
    diagnostics = head.selector_diagnostics_per_sample(result)
    tokens = result.get("sample_tokens")
    if tokens is None:
        raise RuntimeError("selector result omitted sample-token provenance")
    if len(tokens) != diagnostics["current_reward"].shape[0]:
        raise RuntimeError("sample-token and diagnostic batch sizes drifted")

    records = []
    for index, token in enumerate(tokens):
        current_components = diagnostics["current_components"][index]
        reference_components = diagnostics["reference_components"][index]
        oracle_components = diagnostics["oracle_components"][index]
        records.append(
            {
                "token": str(token),
                "valid_candidate_count": int(
                    diagnostics["valid_candidate_count"][index].cpu().item()
                ),
                "current_index": int(
                    diagnostics["current_index"][index].cpu().item()
                ),
                "reference_index": int(
                    diagnostics["reference_index"][index].cpu().item()
                ),
                "oracle_index": int(
                    diagnostics["oracle_index"][index].cpu().item()
                ),
                "current_reward": scalar(
                    diagnostics["current_reward"][index]
                ),
                "reference_reward": scalar(
                    diagnostics["reference_reward"][index]
                ),
                "oracle_reward": scalar(
                    diagnostics["oracle_reward"][index]
                ),
                "top1_reward_gain": scalar(
                    diagnostics["top1_reward_gain"][index]
                ),
                "current_expected_reward": scalar(
                    diagnostics["current_expected_reward"][index]
                ),
                "reference_expected_reward": scalar(
                    diagnostics["reference_expected_reward"][index]
                ),
                "current_oracle_match": bool(
                    diagnostics["current_oracle_match"][index].cpu().item()
                ),
                "reference_oracle_match": bool(
                    diagnostics["reference_oracle_match"][index].cpu().item()
                ),
                "selection_disagreement": bool(
                    diagnostics["selection_disagreement"][index].cpu().item()
                ),
                "current_components": {
                    name: scalar(current_components[component_index])
                    for component_index, name in enumerate(COMPONENT_NAMES)
                },
                "reference_components": {
                    name: scalar(reference_components[component_index])
                    for component_index, name in enumerate(COMPONENT_NAMES)
                },
                "oracle_components": {
                    name: scalar(oracle_components[component_index])
                    for component_index, name in enumerate(COMPONENT_NAMES)
                },
                "candidate_trajectories_sha256": tensor_sha256(
                    diagnostics["candidate_trajectories_8"][index]
                ),
            }
        )
    return records


def mean(records, getter):
    return float(np.mean([getter(record) for record in records]))


def summarize(records):
    summary = {
        "num_tokens": len(records),
        "current_reward": mean(records, lambda row: row["current_reward"]),
        "reference_reward": mean(
            records, lambda row: row["reference_reward"]
        ),
        "top1_reward_gain": mean(
            records, lambda row: row["top1_reward_gain"]
        ),
        "current_expected_reward": mean(
            records, lambda row: row["current_expected_reward"]
        ),
        "reference_expected_reward": mean(
            records, lambda row: row["reference_expected_reward"]
        ),
        "expected_reward_gain": mean(
            records,
            lambda row: row["current_expected_reward"]
            - row["reference_expected_reward"],
        ),
        "current_oracle_match": mean(
            records, lambda row: row["current_oracle_match"]
        ),
        "reference_oracle_match": mean(
            records, lambda row: row["reference_oracle_match"]
        ),
        "selection_disagreement": mean(
            records, lambda row: row["selection_disagreement"]
        ),
    }
    for name in COMPONENT_NAMES:
        current = mean(
            records, lambda row, key=name: row["current_components"][key]
        )
        reference = mean(
            records, lambda row, key=name: row["reference_components"][key]
        )
        summary[f"current_{name}"] = current
        summary[f"reference_{name}"] = reference
        summary[f"delta_{name}"] = current - reference
    return summary


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    calibration_filter = args.calibration_filter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for required in (config_path, checkpoint_path, calibration_filter):
        if not required.is_file():
            raise FileNotFoundError(required)

    cfg = Config.fromfile(str(config_path))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.data.train.nav_filter_path = str(calibration_filter)
    cfg.data.train.test_mode = False
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = args.workers_per_gpu

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    set_random_seed(args.noise_seed, deterministic=False)

    dataset = build_dataset(cfg.data.train)
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
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    head = model.planning_head
    if not hasattr(head, "selector_diagnostics_per_sample"):
        raise TypeError("checkpoint config is not the formal selector-GRPO model")
    head.initialize_reference_selector()
    namespace = f"formal_calibration_seed{args.noise_seed}"
    head.set_candidate_noise_namespace(namespace)
    # Keep perception and the frozen generator in deterministic evaluation
    # mode, while dispatching only the GRPO planning head through
    # _forward_train so it computes online candidate rewards. Setting the whole
    # model to train mode enables stochastic GridMask; DistributedSampler
    # padding can then evaluate the same token on two ranks with different
    # candidates, breaking paired calibration provenance.
    model.eval()
    head.train(True)
    captured = []
    original_loss = head.loss

    def capture_loss(result=None, *loss_args, **loss_kwargs):
        captured.append(result)
        return original_loss(result, *loss_args, **loss_kwargs)

    head.loss = capture_loss
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = Path(str(output) + f".rank{rank}.jsonl")
    with part_path.open("w") as stream, torch.no_grad():
        for data in data_loader:
            if captured:
                raise RuntimeError("stale selector loss-capture result")
            model(return_loss=True, **data)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one selector loss result, captured {len(captured)}"
                )
            for record in record_from_result(head, captured.pop()):
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    head.loss = original_loss
    dist.barrier()

    if rank == 0:
        by_token = {}
        for part_rank in range(world_size):
            path = Path(str(output) + f".rank{part_rank}.jsonl")
            with path.open("r") as stream:
                for line in stream:
                    record = json.loads(line)
                    previous = by_token.get(record["token"])
                    if previous is not None:
                        if (
                            previous["candidate_trajectories_sha256"]
                            != record["candidate_trajectories_sha256"]
                        ):
                            raise RuntimeError(
                                "distributed duplicate token candidate drift: "
                                + record["token"]
                            )
                        for key in (
                            "valid_candidate_count",
                            "current_index",
                            "reference_index",
                            "oracle_index",
                        ):
                            if previous[key] != record[key]:
                                raise RuntimeError(
                                    "distributed duplicate token selection drift: "
                                    + record["token"]
                                )
                        continue
                    by_token[record["token"]] = record
        if len(by_token) != len(dataset):
            raise RuntimeError(
                f"calibration coverage mismatch: {len(by_token)} != {len(dataset)}"
            )
        records = [by_token[token] for token in sorted(by_token)]
        records_path = output.with_suffix(output.suffix + ".records.jsonl")
        with records_path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        report = {
            "schema_version": 1,
            "status": "PASS",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "calibration_filter": str(calibration_filter),
            "calibration_filter_sha256": sha256_file(calibration_filter),
            "noise_seed": args.noise_seed,
            "learning_rate": args.learning_rate,
            "train_seed": args.train_seed,
            "epoch": args.epoch,
            "noise_namespace": namespace,
            "world_size": world_size,
            "records": str(records_path),
            "metrics": summarize(records),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        for part_rank in range(world_size):
            Path(str(output) + f".rank{part_rank}.jsonl").unlink()
    dist.barrier()


if __name__ == "__main__":
    main()
