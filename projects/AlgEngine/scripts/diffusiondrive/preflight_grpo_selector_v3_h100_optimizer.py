#!/usr/bin/env python3
"""Exercise the exact V3 optimizer path before launching a full H100 sweep."""

import argparse
import json

import torch

import grpo_selector_v3_cached_common as common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=("feature_only", "feature_geometry", "feature_geometry_route", "full"),
        default="full",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V3 optimizer preflight requires one visible CUDA device")
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"expected H100 sm90, found sm{capability[0]}{capability[1]}")

    torch.manual_seed(20260812)
    device = torch.device("cuda")
    model = common.SceneConditionedTrajectorySetSelector(
        use_trajectory_geometry=args.ablation != "feature_only",
        use_route_bev=args.ablation in {"feature_geometry_route", "full"},
        use_scene_context=args.ablation == "full",
        use_set_attention=args.ablation == "full",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, foreach=False
    )
    batch_size, num_candidates = 2, 20
    inputs = {
        "candidate_features": torch.randn(batch_size, num_candidates, 256, device=device),
        "candidate_trajectories": torch.randn(
            batch_size, num_candidates, 8, 3, device=device
        ),
        "route_bev_features": torch.randn(
            batch_size, num_candidates, 8, 256, device=device
        ),
        "status_token": torch.randn(batch_size, 1, 256, device=device),
        "ego_query": torch.randn(batch_size, 1, 256, device=device),
        "agents_query": torch.randn(batch_size, 30, 256, device=device),
    }
    reference = torch.randn(batch_size, num_candidates, device=device)
    rewards = torch.rand(batch_size, num_candidates, device=device)
    valid = torch.ones(batch_size, num_candidates, dtype=torch.bool, device=device)
    before = model.delta_head[-1].weight.detach().cpu().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = reference + model(**inputs)
    loss, policy, kl = common.exact_group_loss(
        logits, reference, rewards, valid, temperature=1.0, kl_weight=1e-3
    )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {"ablation": args.ablation, "stage": "forward", "status": "PASS"}
        ),
        flush=True,
    )

    loss.backward()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {"ablation": args.ablation, "stage": "backward", "status": "PASS"}
        ),
        flush=True,
    )

    grad_norm = common.clip_grad_norm_cpu_(model.parameters(), 10.0)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {"ablation": args.ablation, "stage": "cpu_global_grad_clip", "status": "PASS"}
        ),
        flush=True,
    )

    optimizer.step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {"ablation": args.ablation, "stage": "adamw_step", "status": "PASS"}
        ),
        flush=True,
    )

    after = model.delta_head[-1].weight.detach().cpu()
    parameter_delta = float((after - before).abs().max())
    if parameter_delta <= 0.0:
        raise RuntimeError("H100-safe V3 optimizer step did not update parameters")
    print(
        json.dumps(
            {
                "status": "PASS",
                "ablation": args.ablation,
                "device": torch.cuda.get_device_name(0),
                "capability": "sm90",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "loss": float(loss.detach()),
                "policy": float(policy.detach()),
                "kl": float(kl.detach()),
                "grad_norm": grad_norm,
                "parameter_delta": parameter_delta,
                "foreach": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
