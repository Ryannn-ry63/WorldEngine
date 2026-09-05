#!/usr/bin/env python3
"""Reward-free proposal/decision history features for selector audits."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch


def transform_trajectories_between_ego_frames(
    trajectories: torch.Tensor,
    source_ego2global: torch.Tensor,
    target_ego2global: torch.Tensor,
) -> torch.Tensor:
    """Express source-frame ``[x,y,yaw]`` trajectories in a target ego frame."""

    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must have shape [B, K, T, 3]")
    batch = trajectories.shape[0]
    if source_ego2global.shape != (batch, 4, 4):
        raise ValueError("source poses must have shape [B, 4, 4]")
    if target_ego2global.shape != (batch, 4, 4):
        raise ValueError("target poses must have shape [B, 4, 4]")
    dtype = trajectories.dtype
    device = trajectories.device
    source = source_ego2global.to(device=device, dtype=dtype)
    target = target_ego2global.to(device=device, dtype=dtype)
    relative = torch.linalg.solve(target, source)
    xy = trajectories[..., :2]
    zeros = torch.zeros_like(xy[..., :1])
    ones = torch.ones_like(xy[..., :1])
    homogeneous = torch.cat((xy, zeros, ones), dim=-1)
    transformed = torch.einsum("bij,bktj->bkti", relative, homogeneous)
    yaw_offset = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    yaw = trajectories[..., 2] + yaw_offset[:, None, None]
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.cat((transformed[..., :2], yaw[..., None]), dim=-1)


def proposal_history_candidate_features(
    current_trajectories: torch.Tensor,
    previous_trajectories_in_current_frame: torch.Tensor,
    previous_probability: torch.Tensor,
) -> torch.Tensor:
    """Relate every current proposal to the preceding proposal distribution.

    The feature set contains no outcome signal.  It uses the prior proposal
    bank, the frozen-V3 distribution, and its selected trajectory only.
    """

    if current_trajectories.shape != previous_trajectories_in_current_frame.shape:
        raise ValueError("current/previous trajectory banks must align")
    if current_trajectories.ndim != 4 or current_trajectories.shape[-1] != 3:
        raise ValueError("trajectory banks must have shape [B, K, T, 3]")
    if previous_probability.shape != current_trajectories.shape[:2]:
        raise ValueError("previous probability must have shape [B, K]")
    if not torch.isfinite(previous_probability).all():
        raise ValueError("previous probability must be finite")
    probability_sum = previous_probability.sum(dim=-1, keepdim=True)
    if not torch.allclose(
        probability_sum,
        torch.ones_like(probability_sum),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("previous probability must be normalized")

    current_xy = current_trajectories[..., :2]
    previous_xy = previous_trajectories_in_current_frame[..., :2]
    # [B, current K, previous K, T]
    point_distance = torch.linalg.vector_norm(
        current_xy[:, :, None] - previous_xy[:, None, :], dim=-1
    )
    ade = point_distance.mean(dim=-1)
    final = point_distance[..., -1]
    previous_index = previous_probability.argmax(dim=-1)
    selected_ade = ade.gather(
        -1,
        previous_index[:, None, None].expand(-1, ade.shape[1], 1),
    ).squeeze(-1)
    selected_final = final.gather(
        -1,
        previous_index[:, None, None].expand(-1, final.shape[1], 1),
    ).squeeze(-1)
    weighted_ade = (ade * previous_probability[:, None]).sum(dim=-1)
    weighted_final = (final * previous_probability[:, None]).sum(dim=-1)
    same_ade = ade.diagonal(dim1=1, dim2=2)
    same_final = final.diagonal(dim1=1, dim2=2)
    same_probability = previous_probability
    same_selected = torch.nn.functional.one_hot(
        previous_index, num_classes=current_trajectories.shape[1]
    ).to(current_trajectories.dtype)
    heading_delta = current_trajectories[..., -1, 2] - previous_trajectories_in_current_frame[..., -1, 2]
    heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta)).abs()
    return torch.stack(
        (
            same_ade,
            same_final,
            selected_ade,
            selected_final,
            ade.min(dim=-1).values,
            final.min(dim=-1).values,
            weighted_ade,
            weighted_final,
            same_probability,
            same_selected,
            heading_delta,
        ),
        dim=-1,
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def same_stratum_cross_log_derangement(
    strata: Sequence[str],
    logs: Sequence[str],
    tokens: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic marginal-matched history shuffle.

    Histories are reassigned only within common/rare strata and, whenever a
    valid derangement exists, across logs.  This avoids confusing temporal
    information with class balance or memorized log identity.
    """

    if not (len(strata) == len(logs) == len(tokens)) or not tokens:
        raise ValueError("shuffle metadata must be non-empty and aligned")
    output = torch.full((len(tokens),), -1, dtype=torch.long)
    for stratum in sorted(set(map(str, strata))):
        members = [index for index, value in enumerate(strata) if str(value) == stratum]
        if len(members) < 2:
            raise RuntimeError(f"cannot derange singleton stratum: {stratum}")
        generator = torch.Generator().manual_seed(stable_seed(seed, stratum))
        accepted = None
        for _ in range(512):
            order = torch.randperm(len(members), generator=generator).tolist()
            candidate = [members[position] for position in order]
            if all(
                source != target and str(logs[source]) != str(logs[target])
                for source, target in zip(members, candidate)
            ):
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(f"no same-stratum cross-log derangement: {stratum}")
        for source, target in zip(members, accepted):
            output[source] = target
    if bool(output.lt(0).any()) or len(set(output.tolist())) != len(tokens):
        raise RuntimeError("invalid history derangement")
    return output
