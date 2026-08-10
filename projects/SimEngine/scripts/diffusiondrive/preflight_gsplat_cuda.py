#!/usr/bin/env python3
"""Execute gsplat spherical harmonics before loading a large scenario pickle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from gsplat_multiarch import load_isolated_gsplat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ray-workers", type=int, default=0)
    parser.add_argument("--expected-capability")
    return parser.parse_args()


def run_kernel(expected_extension: str) -> Dict[str, Any]:
    import torch
    import gsplat.cuda._backend as gsplat_backend
    from gsplat.cuda._wrapper import spherical_harmonics

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the simengine environment")

    actual_extension = str(Path(gsplat_backend._C.__file__).resolve())
    if actual_extension != expected_extension:
        raise RuntimeError(
            "gsplat extension mismatch: "
            f"expected {expected_extension}, got {actual_extension}"
        )

    device = torch.device("cuda:0")
    viewdirs = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    )
    coeffs = torch.zeros((2, 4, 3), device=device)
    coeffs[:, 0, :] = 1.0
    output = spherical_harmonics(1, viewdirs, coeffs)
    torch.cuda.synchronize(device)
    if output.shape != (2, 3) or not torch.isfinite(output).all():
        raise RuntimeError(f"unexpected spherical_harmonics output: {output}")

    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "device": name,
        "capability": f"sm_{capability[0]}{capability[1]}",
        "extension": actual_extension,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


class GsplatPreflightActor:
    def check(self, expected_extension: str) -> Dict[str, Any]:
        return run_kernel(expected_extension)


def main() -> int:
    args = parse_args()
    expected_extension = str(args.extension.expanduser().resolve())
    backend = load_isolated_gsplat(expected_extension)

    if args.ray_workers < 0:
        raise ValueError("--ray-workers must be non-negative")
    if args.ray_workers:
        import ray

        ray.init(include_dashboard=False)
        actors = []
        try:
            remote_actor = ray.remote(num_gpus=1)(GsplatPreflightActor)
            actors = [remote_actor.remote() for _ in range(args.ray_workers)]
            results = ray.get(
                [actor.check.remote(expected_extension) for actor in actors]
            )
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            ray.shutdown()

        visible_devices = {result["visible_devices"] for result in results}
        capabilities = {result["capability"] for result in results}
        extensions = {result["extension"] for result in results}
        if len(results) != args.ray_workers:
            raise RuntimeError(
                f"expected {args.ray_workers} Ray results, got {len(results)}"
            )
        if len(visible_devices) != args.ray_workers:
            raise RuntimeError(
                "Ray gsplat preflight did not cover distinct GPUs: "
                f"{sorted(visible_devices)}"
            )
        if extensions != {expected_extension}:
            raise RuntimeError(f"Ray workers loaded unexpected extensions: {extensions}")
        if args.expected_capability and capabilities != {args.expected_capability}:
            raise RuntimeError(
                "Ray workers use an unexpected CUDA capability: "
                f"expected {args.expected_capability}, got {sorted(capabilities)}"
            )
        print(
            "PASS isolated gsplat Ray CUDA preflight: "
            f"workers={len(results)}, devices={sorted(visible_devices)}, "
            f"capabilities={sorted(capabilities)}, extension={expected_extension}"
        )
        return 0

    result = run_kernel(expected_extension)
    print(
        "PASS isolated gsplat CUDA preflight: "
        f"device={result['device']}, capability={result['capability']}, "
        f"extension={backend.__file__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
