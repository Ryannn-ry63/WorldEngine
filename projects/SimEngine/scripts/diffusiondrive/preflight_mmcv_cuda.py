#!/usr/bin/env python3
"""Execute MMCV deformable attention before starting planner workers."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmcv_multiarch import load_isolated_mmcv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--expected-capability")
    parser.add_argument("--all-visible", action="store_true")
    return parser.parse_args()


def run_kernel(device_index: int, expected_extension: str) -> str:
    import torch
    import mmcv._ext as mmcv_ext
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
    )

    actual_extension = str(Path(mmcv_ext.__file__).resolve())
    if actual_extension != expected_extension:
        raise RuntimeError(
            "MMCV extension mismatch: "
            f"expected {expected_extension}, got {actual_extension}"
        )

    device = torch.device(f"cuda:{device_index}")
    value = torch.arange(16, dtype=torch.float32, device=device).reshape(
        1, 4, 1, 4
    )
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long, device=device)
    level_start_index = torch.tensor([0], dtype=torch.long, device=device)
    sampling_locations = torch.full(
        (1, 1, 1, 1, 1, 2), 0.5, dtype=torch.float32, device=device
    )
    attention_weights = torch.ones(
        (1, 1, 1, 1, 1), dtype=torch.float32, device=device
    )
    output = MultiScaleDeformableAttnFunction.apply(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        64,
    )
    torch.cuda.synchronize(device)
    if output.shape != (1, 1, 4) or not torch.isfinite(output).all():
        raise RuntimeError(f"unexpected deformable-attention output: {output}")
    capability = torch.cuda.get_device_capability(device)
    return f"sm_{capability[0]}{capability[1]}"


def main() -> int:
    args = parse_args()
    expected_extension = str(args.extension.expanduser().resolve())
    module = load_isolated_mmcv(expected_extension)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the algengine environment")
    device_indices = range(torch.cuda.device_count()) if args.all_visible else [0]
    capabilities = {
        run_kernel(index, expected_extension) for index in device_indices
    }
    if args.expected_capability and capabilities != {args.expected_capability}:
        raise RuntimeError(
            "AlgEngine GPUs use an unexpected CUDA capability: "
            f"expected {args.expected_capability}, got {sorted(capabilities)}"
        )
    print(
        "PASS isolated MMCV CUDA preflight: "
        f"devices={list(device_indices)}, capabilities={sorted(capabilities)}, "
        f"extension={module.__file__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
