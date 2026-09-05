"""Scene-conditioned selector for DiffusionDrive exact-group GRPO.

The module only scores an already generated candidate set.  Perception, the
DiffusionDrive denoiser and trajectory regression stay frozen; observation
features are detached before entering this selector.  No reward, PDM component
or future label is accepted as an input.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAJECTORY_GEOMETRY_DIM = 58
TRAJECTORY_STEP_GEOMETRY_DIM = 11
PAIRWISE_TRAJECTORY_RELATION_DIM = 41
FROZEN_TRACK_STATE_DIM = 8
CANDIDATE_AGENT_INTERACTION_DIM = 16


def math_sdp_kernel(tensor: torch.Tensor):
    """Use the unfused SDPA backend required by torch-2.0.1/cu118 on H100."""
    if tensor.is_cuda:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    return nullcontext()


def candidate_trajectory_geometry(trajectories: torch.Tensor) -> torch.Tensor:
    """Encode eight ``(x, y, yaw)`` poses into ordered kinematics."""

    if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
        raise ValueError(
            "candidate trajectories must have shape (B, K, 8, 3), got "
            f"{tuple(trajectories.shape)}"
        )
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2]
    velocity = xy[..., 1:, :] - xy[..., :-1, :]
    acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
    encoded = torch.cat(
        (
            xy.flatten(start_dim=-2),
            torch.stack((yaw.sin(), yaw.cos()), dim=-1).flatten(start_dim=-2),
            velocity.flatten(start_dim=-2),
            acceleration.flatten(start_dim=-2),
        ),
        dim=-1,
    )
    if encoded.shape[-1] != TRAJECTORY_GEOMETRY_DIM:
        raise RuntimeError("trajectory geometry dimension drifted")
    return encoded


def trajectory_step_geometry(trajectories: torch.Tensor) -> torch.Tensor:
    """Encode ordered per-waypoint kinematics without flattening time."""

    if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
        raise ValueError(
            "candidate trajectories must have shape (B, K, 8, 3), got "
            f"{tuple(trajectories.shape)}"
        )
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2]
    velocity = torch.zeros_like(xy)
    velocity[..., 1:, :] = xy[..., 1:, :] - xy[..., :-1, :]
    acceleration = torch.zeros_like(xy)
    acceleration[..., 2:, :] = velocity[..., 2:, :] - velocity[..., 1:-1, :]
    yaw_delta = torch.zeros_like(yaw)
    yaw_delta[..., 1:] = yaw[..., 1:] - yaw[..., :-1]
    encoded = torch.cat(
        (
            xy,
            torch.stack((yaw.sin(), yaw.cos()), dim=-1),
            velocity,
            acceleration,
            velocity.norm(dim=-1, keepdim=True),
            torch.stack((yaw_delta.sin(), yaw_delta.cos()), dim=-1),
        ),
        dim=-1,
    )
    if encoded.shape[-1] != TRAJECTORY_STEP_GEOMETRY_DIM:
        raise RuntimeError("trajectory step geometry dimension drifted")
    return encoded


def pairwise_trajectory_relations(trajectories: torch.Tensor) -> torch.Tensor:
    """Build directed geometric relations for every ordered candidate pair."""

    if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
        raise ValueError(
            "candidate trajectories must have shape (B, K, 8, 3), got "
            f"{tuple(trajectories.shape)}"
        )
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2]
    relative_xy = xy[:, :, None] - xy[:, None, :]
    relative_yaw = yaw[:, :, None] - yaw[:, None, :]
    distance = relative_xy.norm(dim=-1)
    increments = xy[..., 1:, :] - xy[..., :-1, :]
    path_length = increments.norm(dim=-1).sum(dim=-1)
    path_length_delta = path_length[:, :, None] - path_length[:, None, :]
    encoded = torch.cat(
        (
            relative_xy.flatten(start_dim=-2),
            torch.stack((relative_yaw.sin(), relative_yaw.cos()), dim=-1).flatten(
                start_dim=-2
            ),
            distance,
            path_length_delta.unsqueeze(-1),
        ),
        dim=-1,
    )
    if encoded.shape[-1] != PAIRWISE_TRAJECTORY_RELATION_DIM:
        raise RuntimeError("pairwise trajectory relation dimension drifted")
    return encoded


def constant_velocity_track_rollout(
    track_states: torch.Tensor,
    num_steps: int = 8,
    seconds_per_step: float = 0.5,
) -> torch.Tensor:
    """Roll frozen tracker states forward without learned future labels.

    ``track_states`` stores ``x, y, length, width, yaw, vx, vy, confidence``
    in the current ego/LiDAR frame.  The output has shape ``(B, A, T, 2)``.
    """

    if track_states.ndim != 3 or track_states.shape[-1] != FROZEN_TRACK_STATE_DIM:
        raise ValueError("track states must have shape (B, A, 8)")
    if num_steps <= 0 or seconds_per_step <= 0.0:
        raise ValueError("track rollout horizon and step duration must be positive")
    times = torch.arange(
        1,
        num_steps + 1,
        dtype=track_states.dtype,
        device=track_states.device,
    ) * float(seconds_per_step)
    position = track_states[..., None, :2]
    velocity = track_states[..., None, 5:7]
    return position + velocity * times[None, None, :, None]


def candidate_frame_agent_interactions(
    candidate_trajectories: torch.Tensor,
    track_states: torch.Tensor,
    track_mask: torch.Tensor,
    seconds_per_step: float = 0.5,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
):
    """Construct causal candidate--agent relations in each candidate frame.

    This function is deterministic and label free. It uses only the frozen
    current-frame track state and a constant-velocity rollout.
    """

    if candidate_trajectories.ndim != 4 or candidate_trajectories.shape[-2:] != (8, 3):
        raise ValueError("candidate trajectories must have shape (B, K, 8, 3)")
    if track_states.ndim != 3 or track_states.shape[-1] != FROZEN_TRACK_STATE_DIM:
        raise ValueError("track states must have shape (B, A, 8)")
    if track_mask.shape != track_states.shape[:2]:
        raise ValueError("track mask must have shape (B, A)")
    if candidate_trajectories.shape[0] != track_states.shape[0]:
        raise ValueError("candidate and track batches are not aligned")
    if seconds_per_step <= 0.0 or min(ego_length, ego_width) <= 0.0:
        raise ValueError("interaction geometry constants must be positive")

    trajectories = candidate_trajectories.detach()
    states = track_states.detach()
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2]
    previous_xy = torch.cat(
        (torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), dim=-2
    )
    candidate_velocity = (xy - previous_xy) / float(seconds_per_step)
    agent_position = constant_velocity_track_rollout(
        states, num_steps=8, seconds_per_step=seconds_per_step
    )
    relative = agent_position[:, None] - xy[:, :, None]
    relative_velocity = (
        states[:, None, :, None, 5:7] - candidate_velocity[:, :, None]
    )

    cosine = yaw.cos()[:, :, None]
    sine = yaw.sin()[:, :, None]

    def rotate_to_candidate(values: torch.Tensor) -> torch.Tensor:
        local_x = cosine * values[..., 0] + sine * values[..., 1]
        local_y = -sine * values[..., 0] + cosine * values[..., 1]
        return torch.stack((local_x, local_y), dim=-1)

    local_relative = rotate_to_candidate(relative)
    local_relative_velocity = rotate_to_candidate(relative_velocity)
    distance = relative.norm(dim=-1, keepdim=True)
    relative_heading = states[:, None, :, None, 4] - yaw[:, :, None]
    length = states[:, None, :, None, 2:3].expand_as(distance)
    width = states[:, None, :, None, 3:4].expand_as(distance)
    confidence = states[:, None, :, None, 7:8].expand_as(distance)
    longitudinal_clearance = local_relative[..., 0:1].abs() - 0.5 * (
        length + float(ego_length)
    )
    lateral_clearance = local_relative[..., 1:2].abs() - 0.5 * (
        width + float(ego_width)
    )
    closing_speed = -(
        relative * relative_velocity
    ).sum(dim=-1, keepdim=True) / distance.clamp_min(1e-3)
    relative_speed_squared = relative_velocity.square().sum(dim=-1, keepdim=True)
    time_to_closest = -(
        relative * relative_velocity
    ).sum(dim=-1, keepdim=True) / relative_speed_squared.clamp_min(1e-3)
    time_to_closest = time_to_closest.clamp(0.0, 8.0) / 8.0
    normalized_time = torch.arange(
        1, 9, dtype=states.dtype, device=states.device
    ) / 8.0
    time_encoding = torch.stack(
        ((math.pi * normalized_time).sin(), (math.pi * normalized_time).cos()),
        dim=-1,
    )[None, None, None].expand(*distance.shape[:-1], 2)
    features = torch.cat(
        (
            local_relative,
            local_relative_velocity,
            distance,
            relative_heading.sin().unsqueeze(-1),
            relative_heading.cos().unsqueeze(-1),
            length,
            width,
            confidence,
            longitudinal_clearance,
            lateral_clearance,
            closing_speed,
            time_to_closest,
            time_encoding,
        ),
        dim=-1,
    )
    if features.shape[-1] != CANDIDATE_AGENT_INTERACTION_DIM:
        raise RuntimeError("candidate-agent interaction dimension drifted")
    relation_mask = track_mask[:, None, :, None].bool().expand(
        features.shape[0], features.shape[1], features.shape[2], features.shape[3]
    )
    features = torch.where(
        relation_mask[..., None], features, torch.zeros_like(features)
    )
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("non-finite candidate-agent interaction features")
    return features, relation_mask


def interaction_probe_features(
    candidate_trajectories: torch.Tensor,
    track_states: torch.Tensor,
    track_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact, non-learned summaries used only by the train-split probe."""

    features, mask = candidate_frame_agent_interactions(
        candidate_trajectories, track_states, track_mask
    )
    flat = features.flatten(start_dim=2, end_dim=3)
    flat_mask = mask.flatten(start_dim=2, end_dim=3)
    valid = flat_mask[..., None]
    count = valid.sum(dim=2).clamp_min(1)
    mean = (flat * valid).sum(dim=2) / count
    positive_fill = torch.finfo(flat.dtype).max
    negative_fill = torch.finfo(flat.dtype).min
    minimum = flat.masked_fill(~valid, positive_fill).min(dim=2).values
    maximum = flat.masked_fill(~valid, negative_fill).max(dim=2).values
    any_agent = flat_mask.any(dim=2, keepdim=True)
    minimum = torch.where(any_agent, minimum, torch.zeros_like(minimum))
    maximum = torch.where(any_agent, maximum, torch.zeros_like(maximum))
    return torch.cat((minimum, mean, maximum), dim=-1)


def sample_final_trajectory_bev_features(
    bev_feature: torch.Tensor,
    candidate_trajectories: torch.Tensor,
    bev_range_x: float,
    bev_range_y: float,
) -> torch.Tensor:
    """Sample frozen BEV features at the final candidate waypoints.

    Returns ``(B, K, 8, C)``.  This deliberately uses the final regression
    output rather than the noisy trajectory used inside the last DiT block.
    """

    if bev_feature.ndim != 4:
        raise ValueError("BEV feature must have shape (B, C, H, W)")
    if candidate_trajectories.ndim != 4 or candidate_trajectories.shape[-2:] != (8, 3):
        raise ValueError("candidate trajectories must have shape (B, K, 8, 3)")
    if bev_feature.shape[0] != candidate_trajectories.shape[0]:
        raise ValueError("BEV/candidate batch dimensions do not match")
    if bev_range_x <= 0.0 or bev_range_y <= 0.0:
        raise ValueError("BEV ranges must be positive")

    grid = candidate_trajectories[..., :2].detach().clone()
    grid[..., 0] = grid[..., 0] / float(bev_range_x)
    grid[..., 1] = grid[..., 1] / float(bev_range_y)
    sampled = F.grid_sample(
        bev_feature.detach(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1).contiguous().detach()


class SceneConditionedTrajectorySetSelector(nn.Module):
    """Permutation-equivariant residual scorer over dynamic trajectories."""

    def __init__(
        self,
        feature_dim: int = 256,
        model_dim: int = 256,
        route_bev_dim: int = 256,
        context_dim: int = 256,
        geometry_dim: int = TRAJECTORY_GEOMETRY_DIM,
        geometry_hidden_dim: int = 128,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        num_route_steps: int = 8,
        num_set_layers: int = 1,
        use_set_attention: bool = True,
        use_trajectory_geometry: bool = True,
        use_route_bev: bool = True,
        use_scene_context: bool = True,
    ) -> None:
        super().__init__()
        if min(feature_dim, model_dim, route_bev_dim, context_dim) <= 0:
            raise ValueError("selector dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if geometry_dim != TRAJECTORY_GEOMETRY_DIM:
            raise ValueError(
                f"geometry_dim must be {TRAJECTORY_GEOMETRY_DIM}, got {geometry_dim}"
            )
        if num_route_steps != 8:
            raise ValueError("DiffusionDrive V3 currently requires eight route steps")
        if num_set_layers < 0:
            raise ValueError("num_set_layers must be non-negative")

        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.route_bev_dim = int(route_bev_dim)
        self.context_dim = int(context_dim)
        self.geometry_dim = int(geometry_dim)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.num_route_steps = int(num_route_steps)
        self.num_set_layers = int(num_set_layers)
        self.use_set_attention = bool(use_set_attention)
        self.use_trajectory_geometry = bool(use_trajectory_geometry)
        self.use_route_bev = bool(use_route_bev)
        self.use_scene_context = bool(use_scene_context)

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, model_dim)
        )
        self.geometry_encoder = None
        if self.use_trajectory_geometry:
            self.geometry_encoder = nn.Sequential(
                nn.LayerNorm(geometry_dim),
                nn.Linear(geometry_dim, geometry_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(geometry_hidden_dim),
                nn.Linear(geometry_hidden_dim, model_dim),
            )

        self.route_projection = None
        self.route_step_embedding = None
        self.route_cross_attention = None
        self.route_norm = None
        if self.use_route_bev:
            self.route_projection = nn.Sequential(
                nn.LayerNorm(route_bev_dim), nn.Linear(route_bev_dim, model_dim)
            )
            self.route_step_embedding = nn.Parameter(
                torch.zeros(num_route_steps, model_dim)
            )
            nn.init.normal_(self.route_step_embedding, std=0.02)
            self.route_cross_attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=0.0, batch_first=True
            )
            self.route_norm = nn.LayerNorm(model_dim)

        self.context_projection = None
        self.context_type_embedding = None
        self.context_cross_attention = None
        self.context_norm = None
        if self.use_scene_context:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Linear(context_dim, model_dim)
            )
            self.context_type_embedding = nn.Parameter(torch.zeros(3, model_dim))
            nn.init.normal_(self.context_type_embedding, std=0.02)
            self.context_cross_attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=0.0, batch_first=True
            )
            self.context_norm = nn.LayerNorm(model_dim)

        if self.use_set_attention and num_set_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=num_set_layers)
        else:
            self.set_encoder = nn.Identity()
        self.output_norm = nn.LayerNorm(model_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "model_dim": self.model_dim,
            "route_bev_dim": self.route_bev_dim,
            "context_dim": self.context_dim,
            "geometry_dim": self.geometry_dim,
            "geometry_hidden_dim": self.geometry_hidden_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "num_route_steps": self.num_route_steps,
            "num_set_layers": self.num_set_layers,
            "use_set_attention": self.use_set_attention,
            "use_trajectory_geometry": self.use_trajectory_geometry,
            "use_route_bev": self.use_route_bev,
            "use_scene_context": self.use_scene_context,
        }

    @staticmethod
    def _check_candidate_shape(
        candidate_features: torch.Tensor, candidate_trajectories: torch.Tensor
    ) -> None:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must have shape (B, K, C)")
        if candidate_trajectories.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate feature/trajectory sets are not aligned")

    def encode_tokens(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._check_candidate_shape(candidate_features, candidate_trajectories)
        if candidate_features.shape[-1] != self.feature_dim:
            raise ValueError("candidate feature dimension drifted")
        batch_size, num_candidates, _ = candidate_features.shape
        tokens = self.feature_projection(candidate_features)

        if self.geometry_encoder is not None:
            tokens = tokens + self.geometry_encoder(
                candidate_trajectory_geometry(candidate_trajectories)
            )

        if self.use_route_bev:
            expected = (batch_size, num_candidates, 8, self.route_bev_dim)
            if route_bev_features is None or tuple(route_bev_features.shape) != expected:
                got = None if route_bev_features is None else tuple(route_bev_features.shape)
                raise ValueError(f"route BEV features must have shape {expected}, got {got}")
            route_tokens = self.route_projection(route_bev_features)
            route_tokens = route_tokens + self.route_step_embedding[None, None]
            flat_query = tokens.reshape(batch_size * num_candidates, 1, self.model_dim)
            flat_route = route_tokens.reshape(
                batch_size * num_candidates, 8, self.model_dim
            )
            with math_sdp_kernel(flat_query):
                route_update = self.route_cross_attention(
                    flat_query, flat_route, flat_route, need_weights=False
                )[0]
            tokens = self.route_norm(
                flat_query + route_update
            ).reshape(batch_size, num_candidates, self.model_dim)

        if self.use_scene_context:
            context_values = (status_token, ego_query, agents_query)
            if any(value is None for value in context_values):
                raise ValueError("status, ego and agent context are required")
            if any(value.ndim != 3 for value in context_values):
                raise ValueError("scene context tensors must have shape (B, N, C)")
            if any(
                value.shape[0] != batch_size or value.shape[-1] != self.context_dim
                for value in context_values
            ):
                raise ValueError("scene context dimensions drifted")
            projected = [self.context_projection(value) for value in context_values]
            projected[0] = projected[0] + self.context_type_embedding[0]
            projected[1] = projected[1] + self.context_type_embedding[1]
            projected[2] = projected[2] + self.context_type_embedding[2]
            scene = torch.cat(projected, dim=1)
            if scene.shape[1] != 2 + agents_query.shape[1]:
                raise RuntimeError("scene-context token accounting drifted")
            with math_sdp_kernel(tokens):
                context_update = self.context_cross_attention(
                    tokens, scene, scene, need_weights=False
                )[0]
            tokens = self.context_norm(tokens + context_update)

        with math_sdp_kernel(tokens):
            set_tokens = self.set_encoder(tokens)
        return self.output_norm(set_tokens)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        return self.delta_head(tokens).squeeze(-1)


class RelationAwareSetBlock(nn.Module):
    """Permutation-equivariant candidate attention with geometric edge bias."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        relation_dim: int = PAIRWISE_TRAJECTORY_RELATION_DIM,
        relation_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.relation_bias = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, num_heads),
        )
        self.output = nn.Linear(model_dim, model_dim)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)

    def forward(
        self, tokens: torch.Tensor, pairwise_relations: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("candidate tokens must have shape (B, K, D)")
        expected = (
            tokens.shape[0],
            tokens.shape[1],
            tokens.shape[1],
            PAIRWISE_TRAJECTORY_RELATION_DIM,
        )
        if tuple(pairwise_relations.shape) != expected:
            raise ValueError(
                f"pairwise relations must have shape {expected}, got "
                f"{tuple(pairwise_relations.shape)}"
            )
        batch_size, num_candidates, _ = tokens.shape

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch_size, num_candidates, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.query(tokens))
        key = split_heads(self.key(tokens))
        value = split_heads(self.value(tokens))
        attention_logits = torch.einsum("bhid,bhjd->bhij", query, key) * self.scale
        relation_bias = self.relation_bias(pairwise_relations).permute(0, 3, 1, 2)
        attention = (attention_logits + relation_bias).float().softmax(dim=-1)
        attention = attention.to(value.dtype)
        update = torch.einsum("bhij,bhjd->bhid", attention, value)
        update = update.transpose(1, 2).reshape(batch_size, num_candidates, -1)
        tokens = self.attention_norm(tokens + self.output(update))
        return self.feedforward_norm(tokens + self.feedforward(tokens))


class TrajectorySetReasoningResidualSelector(nn.Module):
    """Reason over trajectory time, scene context and candidate relations.

    The module predicts only a residual over the frozen DiffusionDrive logits.
    Its final layer is zero initialized, so materialization starts exactly at
    the frozen selector policy rather than at a randomly perturbed policy.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        model_dim: int = 256,
        route_bev_dim: int = 256,
        context_dim: int = 256,
        step_geometry_dim: int = TRAJECTORY_STEP_GEOMETRY_DIM,
        relation_dim: int = PAIRWISE_TRAJECTORY_RELATION_DIM,
        geometry_hidden_dim: int = 128,
        relation_hidden_dim: int = 128,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        num_route_steps: int = 8,
        num_temporal_layers: int = 2,
        num_relation_layers: int = 2,
        use_temporal_reasoning: bool = True,
        use_relational_reasoning: bool = True,
        use_route_bev: bool = True,
        use_scene_context: bool = True,
    ) -> None:
        super().__init__()
        if min(feature_dim, model_dim, route_bev_dim, context_dim) <= 0:
            raise ValueError("selector dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if step_geometry_dim != TRAJECTORY_STEP_GEOMETRY_DIM:
            raise ValueError("trajectory step geometry dimension drifted")
        if relation_dim != PAIRWISE_TRAJECTORY_RELATION_DIM:
            raise ValueError("pairwise relation dimension drifted")
        if num_route_steps != 8:
            raise ValueError("DiffusionDrive requires eight trajectory steps")
        if num_temporal_layers < 0 or num_relation_layers < 0:
            raise ValueError("reasoning layer counts must be non-negative")

        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.route_bev_dim = int(route_bev_dim)
        self.context_dim = int(context_dim)
        self.step_geometry_dim = int(step_geometry_dim)
        self.relation_dim = int(relation_dim)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.num_route_steps = int(num_route_steps)
        self.num_temporal_layers = int(num_temporal_layers)
        self.num_relation_layers = int(num_relation_layers)
        self.use_temporal_reasoning = bool(use_temporal_reasoning)
        self.use_relational_reasoning = bool(use_relational_reasoning)
        self.use_route_bev = bool(use_route_bev)
        self.use_scene_context = bool(use_scene_context)

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, model_dim)
        )
        self.step_geometry_encoder = nn.Sequential(
            nn.LayerNorm(step_geometry_dim),
            nn.Linear(step_geometry_dim, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, model_dim),
        )
        self.route_projection = None
        if self.use_route_bev:
            self.route_projection = nn.Sequential(
                nn.LayerNorm(route_bev_dim), nn.Linear(route_bev_dim, model_dim)
            )
        self.step_embedding = nn.Parameter(torch.zeros(num_route_steps, model_dim))
        nn.init.normal_(self.step_embedding, std=0.02)
        self.step_input_norm = nn.LayerNorm(model_dim)

        self.context_projection = None
        self.context_type_embedding = None
        self.context_cross_attention = None
        self.context_norm = None
        if self.use_scene_context:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Linear(context_dim, model_dim)
            )
            self.context_type_embedding = nn.Parameter(torch.zeros(3, model_dim))
            nn.init.normal_(self.context_type_embedding, std=0.02)
            self.context_cross_attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=0.0, batch_first=True
            )
            self.context_norm = nn.LayerNorm(model_dim)

        if self.use_temporal_reasoning and num_temporal_layers:
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                temporal_layer, num_layers=num_temporal_layers
            )
        else:
            self.temporal_encoder = nn.Identity()

        relation_layer_count = (
            num_relation_layers if self.use_relational_reasoning else 0
        )
        self.relation_blocks = nn.ModuleList(
            RelationAwareSetBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                relation_dim=relation_dim,
                relation_hidden_dim=relation_hidden_dim,
            )
            for _ in range(relation_layer_count)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        return {
            "architecture": "trajectory_set_reasoner",
            "feature_dim": self.feature_dim,
            "model_dim": self.model_dim,
            "route_bev_dim": self.route_bev_dim,
            "context_dim": self.context_dim,
            "step_geometry_dim": self.step_geometry_dim,
            "relation_dim": self.relation_dim,
            "geometry_hidden_dim": self.geometry_hidden_dim,
            "relation_hidden_dim": self.relation_hidden_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "num_route_steps": self.num_route_steps,
            "num_temporal_layers": self.num_temporal_layers,
            "num_relation_layers": self.num_relation_layers,
            "use_temporal_reasoning": self.use_temporal_reasoning,
            "use_relational_reasoning": self.use_relational_reasoning,
            "use_route_bev": self.use_route_bev,
            "use_scene_context": self.use_scene_context,
        }

    def _scene_tokens(
        self,
        batch_size: int,
        status_token: Optional[torch.Tensor],
        ego_query: Optional[torch.Tensor],
        agents_query: Optional[torch.Tensor],
    ) -> torch.Tensor:
        context_values = (status_token, ego_query, agents_query)
        if any(value is None for value in context_values):
            raise ValueError("status, ego and agent context are required")
        if any(value.ndim != 3 for value in context_values):
            raise ValueError("scene context tensors must have shape (B, N, C)")
        if any(
            value.shape[0] != batch_size or value.shape[-1] != self.context_dim
            for value in context_values
        ):
            raise ValueError("scene context dimensions drifted")
        projected = [self.context_projection(value.detach()) for value in context_values]
        projected[0] = projected[0] + self.context_type_embedding[0]
        projected[1] = projected[1] + self.context_type_embedding[1]
        projected[2] = projected[2] + self.context_type_embedding[2]
        return torch.cat(projected, dim=1)

    def encode_tokens(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        SceneConditionedTrajectorySetSelector._check_candidate_shape(
            candidate_features, candidate_trajectories
        )
        if candidate_features.shape[-1] != self.feature_dim:
            raise ValueError("candidate feature dimension drifted")
        batch_size, num_candidates, _ = candidate_features.shape
        trajectories = candidate_trajectories.detach()
        feature_tokens = self.feature_projection(candidate_features.detach())
        step_tokens = feature_tokens[:, :, None, :]
        step_tokens = step_tokens + self.step_geometry_encoder(
            trajectory_step_geometry(trajectories)
        )
        if self.use_route_bev:
            expected = (batch_size, num_candidates, 8, self.route_bev_dim)
            if route_bev_features is None or tuple(route_bev_features.shape) != expected:
                got = None if route_bev_features is None else tuple(route_bev_features.shape)
                raise ValueError(f"route BEV features must have shape {expected}, got {got}")
            step_tokens = step_tokens + self.route_projection(
                route_bev_features.detach()
            )
        step_tokens = self.step_input_norm(
            step_tokens + self.step_embedding[None, None, :, :]
        )

        flat_steps = step_tokens.reshape(
            batch_size * num_candidates, self.num_route_steps, self.model_dim
        )
        if self.use_scene_context:
            scene = self._scene_tokens(
                batch_size, status_token, ego_query, agents_query
            ).repeat_interleave(num_candidates, dim=0)
            with math_sdp_kernel(flat_steps):
                context_update = self.context_cross_attention(
                    flat_steps, scene, scene, need_weights=False
                )[0]
            flat_steps = self.context_norm(flat_steps + context_update)
        with math_sdp_kernel(flat_steps):
            flat_steps = self.temporal_encoder(flat_steps)
        candidate_tokens = flat_steps.mean(dim=1).reshape(
            batch_size, num_candidates, self.model_dim
        )

        if self.relation_blocks:
            relations = pairwise_trajectory_relations(trajectories)
            for block in self.relation_blocks:
                candidate_tokens = block(candidate_tokens, relations)
        return self.output_norm(candidate_tokens)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        return self.delta_head(tokens).squeeze(-1)


class ReferenceAnchoredPreferenceGraphSelector(
    TrajectorySetReasoningResidualSelector
):
    """Reference-anchored pairwise preference graph over a fixed candidate set.

    The frozen reference logits identify the incumbent trajectory.  A shared
    directed edge scorer is antisymmetrized before aggregation, so the graph
    represents relative preferences instead of a second unary score.  Both the
    unary and pairwise output layers are zero initialized; consequently the
    complete selector starts as an exact behavioral copy of the reference.
    """

    requires_reference_logits = True

    def __init__(
        self,
        *args,
        pairwise_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if pairwise_hidden_dim <= 0:
            raise ValueError("pairwise_hidden_dim must be positive")
        self.pairwise_hidden_dim = int(pairwise_hidden_dim)
        pair_input_dim = 2 * self.model_dim + self.relation_dim
        self.pairwise_preference_head = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, pairwise_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(pairwise_hidden_dim),
            nn.Linear(pairwise_hidden_dim, 1),
        )
        nn.init.zeros_(self.pairwise_preference_head[-1].weight)
        nn.init.zeros_(self.pairwise_preference_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture="reference_anchored_preference_graph",
            pairwise_hidden_dim=self.pairwise_hidden_dim,
        )
        return config

    @staticmethod
    def _candidate_mask(
        tokens: torch.Tensor, candidate_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        expected = tokens.shape[:2]
        if candidate_mask is None:
            return torch.ones(expected, dtype=torch.bool, device=tokens.device)
        if tuple(candidate_mask.shape) != expected:
            raise ValueError(
                f"candidate mask must have shape {tuple(expected)}, got "
                f"{tuple(candidate_mask.shape)}"
            )
        candidate_mask = candidate_mask.to(device=tokens.device, dtype=torch.bool)
        if not bool(candidate_mask.any(dim=-1).all()):
            raise ValueError("every selector group must contain a valid candidate")
        return candidate_mask

    def pairwise_preferences(
        self,
        tokens: torch.Tensor,
        trajectories: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return antisymmetric edge preferences and the valid-edge mask."""

        if tokens.ndim != 3:
            raise ValueError("candidate tokens must have shape (B, K, D)")
        batch_size, num_candidates, model_dim = tokens.shape
        if model_dim != self.model_dim:
            raise ValueError("candidate token dimension drifted")
        if trajectories.shape[:2] != tokens.shape[:2]:
            raise ValueError("candidate tokens and trajectories are not aligned")
        mask = self._candidate_mask(tokens, candidate_mask)
        relations = pairwise_trajectory_relations(trajectories.detach())
        source = tokens[:, :, None, :].expand(
            batch_size, num_candidates, num_candidates, model_dim
        )
        target = tokens[:, None, :, :].expand(
            batch_size, num_candidates, num_candidates, model_dim
        )
        directed = self.pairwise_preference_head(
            torch.cat((source, target, relations), dim=-1)
        ).squeeze(-1)
        preferences = 0.5 * (directed - directed.transpose(1, 2))
        diagonal = torch.eye(
            num_candidates, dtype=torch.bool, device=tokens.device
        )[None]
        edge_mask = mask[:, :, None] & mask[:, None, :] & ~diagonal
        preferences = preferences.masked_fill(~edge_mask, 0.0)
        return preferences, edge_mask

    @staticmethod
    def aggregate_preferences(
        preferences: torch.Tensor,
        edge_mask: torch.Tensor,
        reference_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate incumbent and all-opponent evidence for each candidate."""

        if preferences.ndim != 3 or preferences.shape[1] != preferences.shape[2]:
            raise ValueError("preferences must have shape (B, K, K)")
        if edge_mask.shape != preferences.shape:
            raise ValueError("preference and edge masks are not aligned")
        if reference_logits.shape != preferences.shape[:2]:
            raise ValueError("reference logits and preferences are not aligned")
        incumbent = reference_logits.detach().masked_fill(
            ~candidate_mask, -torch.inf
        ).argmax(dim=-1)
        gather_index = incumbent[:, None, None].expand(
            -1, preferences.shape[1], 1
        )
        incumbent_evidence = preferences.gather(2, gather_index).squeeze(-1)
        opponent_count = edge_mask.sum(dim=-1).clamp_min(1).to(preferences.dtype)
        opponent_evidence = preferences.sum(dim=-1) / opponent_count
        graph_delta = incumbent_evidence + opponent_evidence
        graph_delta = graph_delta.masked_fill(~candidate_mask, 0.0)
        return graph_delta, incumbent

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_pair_diagnostics: bool = False,
    ):
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        if reference_logits.shape != tokens.shape[:2]:
            raise ValueError(
                "reference logits must align with the candidate token set"
            )
        mask = self._candidate_mask(tokens, candidate_mask)
        unary_delta = self.delta_head(tokens).squeeze(-1).masked_fill(~mask, 0.0)
        preferences, edge_mask = self.pairwise_preferences(
            tokens, candidate_trajectories, mask
        )
        graph_delta, incumbent = self.aggregate_preferences(
            preferences, edge_mask, reference_logits, mask
        )
        residual = unary_delta + graph_delta
        if not return_pair_diagnostics:
            return residual
        return residual, {
            "unary_delta": unary_delta,
            "graph_delta": graph_delta,
            "pairwise_preferences": preferences,
            "edge_mask": edge_mask,
            "incumbent_index": incumbent,
        }


class IncumbentVerifiedPreferenceSelector(
    TrajectorySetReasoningResidualSelector
):
    """Propose a candidate, then verify it against the frozen incumbent.

    The proposal branch is an ordinary relational residual selector.  The
    verifier is asymmetric by design: it predicts evidence for replacing the
    incumbent selected by the immutable reference logits.  A strict positive
    verifier logit is required to expose the proposal at deployment, so a
    zero-initialized verifier exactly preserves the reference policy even when
    a non-zero pretrained proposal is loaded.
    """

    requires_reference_logits = True
    supports_override_diagnostics = True

    def __init__(
        self,
        *args,
        verifier_hidden_dim: int = 128,
        override_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if verifier_hidden_dim <= 0:
            raise ValueError("verifier_hidden_dim must be positive")
        if not math.isfinite(float(override_threshold)):
            raise ValueError("override_threshold must be finite")
        self.verifier_hidden_dim = int(verifier_hidden_dim)
        self.override_threshold = float(override_threshold)
        verifier_input_dim = 2 * self.model_dim + self.relation_dim + 2
        self.override_verifier_head = nn.Sequential(
            nn.LayerNorm(verifier_input_dim),
            nn.Linear(verifier_input_dim, verifier_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(verifier_hidden_dim),
            nn.Linear(verifier_hidden_dim, 1),
        )
        nn.init.zeros_(self.override_verifier_head[-1].weight)
        nn.init.zeros_(self.override_verifier_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture="incumbent_verified_preference",
            verifier_hidden_dim=self.verifier_hidden_dim,
            override_threshold=self.override_threshold,
        )
        return config

    @staticmethod
    def _candidate_mask(
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reference_logits.ndim != 2:
            raise ValueError("reference logits must have shape (B, K)")
        if candidate_mask is None:
            return torch.ones_like(reference_logits, dtype=torch.bool)
        if candidate_mask.shape != reference_logits.shape:
            raise ValueError("candidate mask and reference logits are not aligned")
        mask = candidate_mask.to(device=reference_logits.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every selector group must contain a valid candidate")
        return mask

    def proposal_and_verification(
        self,
        tokens: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[:2] != reference_logits.shape:
            raise ValueError("candidate tokens and reference logits are not aligned")
        if candidate_trajectories.shape[:2] != tokens.shape[:2]:
            raise ValueError("candidate tokens and trajectories are not aligned")
        batch_size, num_candidates, model_dim = tokens.shape
        mask = self._candidate_mask(reference_logits, candidate_mask)
        proposal_delta = self.delta_head(tokens).squeeze(-1).masked_fill(~mask, 0.0)
        masked_reference = reference_logits.detach().masked_fill(~mask, -torch.inf)
        incumbent = masked_reference.argmax(dim=-1)
        incumbent_token = tokens.gather(
            1, incumbent[:, None, None].expand(-1, 1, model_dim)
        ).expand(-1, num_candidates, -1)
        relations = pairwise_trajectory_relations(candidate_trajectories.detach())
        relation_to_incumbent = relations.gather(
            2,
            incumbent[:, None, None, None].expand(
                batch_size, num_candidates, 1, self.relation_dim
            ),
        ).squeeze(2)
        incumbent_reference = masked_reference.gather(1, incumbent[:, None])
        reference_gap = reference_logits.detach() - incumbent_reference
        proposal_logits = reference_logits.detach() + proposal_delta
        incumbent_proposal = proposal_logits.gather(1, incumbent[:, None])
        proposal_gap = proposal_logits - incumbent_proposal
        verifier_inputs = torch.cat(
            (
                tokens,
                incumbent_token,
                relation_to_incumbent,
                reference_gap[:, :, None],
                proposal_gap[:, :, None],
            ),
            dim=-1,
        )
        verification_logits = self.override_verifier_head(
            verifier_inputs
        ).squeeze(-1)
        verification_logits = verification_logits.masked_fill(~mask, 0.0)
        return proposal_delta, verification_logits, mask

    @staticmethod
    def apply_verified_override(
        proposal_delta: torch.Tensor,
        verification_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (
            proposal_delta.shape
            == verification_logits.shape
            == reference_logits.shape
            == candidate_mask.shape
        ):
            raise ValueError("verified-override tensors are not aligned")
        masked_reference = reference_logits.detach().masked_fill(
            ~candidate_mask, -torch.inf
        )
        proposal_logits = (reference_logits.detach() + proposal_delta).masked_fill(
            ~candidate_mask, -torch.inf
        )
        incumbent = masked_reference.argmax(dim=-1)
        proposal = proposal_logits.argmax(dim=-1)
        proposal_evidence = verification_logits.gather(
            1, proposal[:, None]
        ).squeeze(1)
        override = proposal.ne(incumbent) & proposal_evidence.gt(float(threshold))
        residual = torch.where(
            override[:, None], proposal_delta, torch.zeros_like(proposal_delta)
        )
        return residual, incumbent, proposal, override

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_override_diagnostics: bool = False,
    ):
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        proposal_delta, verification_logits, mask = self.proposal_and_verification(
            tokens, candidate_trajectories, reference_logits, candidate_mask
        )
        residual, incumbent, proposal, override = self.apply_verified_override(
            proposal_delta,
            verification_logits,
            reference_logits,
            mask,
            self.override_threshold,
        )
        if not return_override_diagnostics:
            return residual
        return residual, {
            "proposal_delta": proposal_delta,
            "verification_logits": verification_logits,
            "candidate_mask": mask,
            "incumbent_index": incumbent,
            "proposal_index": proposal,
            "override_mask": override,
        }


class ProposalConditionedRegretArbitrator(
    TrajectorySetReasoningResidualSelector
):
    """Arbitrate only the frozen proposal actually exposed at deployment.

    Unlike IVPS, this module does not predict an improvement label for every
    candidate. It obtains the deterministic top-1 proposal from the frozen
    residual selector, then emits one decision for that proposal/incumbent
    pair. The proposal encoder and delta head stay frozen during arbitration.
    """

    requires_reference_logits = True
    supports_override_diagnostics = True
    requires_candidate_mask = True

    def __init__(
        self,
        *args,
        arbiter_hidden_dim: int = 128,
        use_decision_context: bool = True,
        override_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if arbiter_hidden_dim <= 0:
            raise ValueError("arbiter_hidden_dim must be positive")
        if not math.isfinite(float(override_threshold)):
            raise ValueError("override_threshold must be finite")
        self.arbiter_hidden_dim = int(arbiter_hidden_dim)
        self.use_decision_context = bool(use_decision_context)
        self.override_threshold = float(override_threshold)
        scalar_dim = 2 + (4 if self.use_decision_context else 0)
        arbiter_input_dim = 2 * self.model_dim + self.relation_dim + scalar_dim
        self.regret_arbiter_head = nn.Sequential(
            nn.LayerNorm(arbiter_input_dim),
            nn.Linear(arbiter_input_dim, arbiter_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(arbiter_hidden_dim),
            nn.Linear(arbiter_hidden_dim, 1),
        )
        nn.init.zeros_(self.regret_arbiter_head[-1].weight)
        nn.init.zeros_(self.regret_arbiter_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture="proposal_conditioned_regret_arbitration",
            arbiter_hidden_dim=self.arbiter_hidden_dim,
            use_decision_context=self.use_decision_context,
            override_threshold=self.override_threshold,
        )
        return config

    @staticmethod
    def _candidate_mask(
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reference_logits.ndim != 2:
            raise ValueError("reference logits must have shape (B, K)")
        if candidate_mask is None:
            return torch.ones_like(reference_logits, dtype=torch.bool)
        if candidate_mask.shape != reference_logits.shape:
            raise ValueError("candidate mask and reference logits are not aligned")
        mask = candidate_mask.to(device=reference_logits.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every selector group must contain a valid candidate")
        return mask

    @staticmethod
    def _score_context(
        logits: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-1 margin and entropy normalized by valid set size."""

        masked = logits.detach().masked_fill(~candidate_mask, -torch.inf)
        valid_count = candidate_mask.sum(dim=-1)
        top_k = min(2, logits.shape[-1])
        top = masked.topk(top_k, dim=-1).values
        if top_k == 1:
            margin = torch.zeros_like(top[:, 0])
        else:
            margin = top[:, 0] - top[:, 1]
            margin = torch.where(valid_count >= 2, margin, torch.zeros_like(margin))
        probability = F.softmax(masked, dim=-1)
        log_probability = F.log_softmax(masked, dim=-1)
        entropy_terms = torch.where(
            candidate_mask,
            probability * log_probability,
            torch.zeros_like(probability),
        )
        entropy = -entropy_terms.sum(dim=-1)
        normalizer = valid_count.clamp_min(2).to(logits.dtype).log()
        return margin, entropy / normalizer

    def proposal_and_arbitration(
        self,
        tokens: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if tokens.ndim != 3 or tokens.shape[:2] != reference_logits.shape:
            raise ValueError("candidate tokens and reference logits are not aligned")
        if candidate_trajectories.shape[:2] != tokens.shape[:2]:
            raise ValueError("candidate tokens and trajectories are not aligned")
        batch_size, _, model_dim = tokens.shape
        mask = self._candidate_mask(reference_logits, candidate_mask)
        proposal_delta = self.delta_head(tokens).squeeze(-1).masked_fill(~mask, 0.0)
        masked_reference = reference_logits.detach().masked_fill(~mask, -torch.inf)
        proposal_logits = (reference_logits.detach() + proposal_delta).masked_fill(
            ~mask, -torch.inf
        )
        incumbent = masked_reference.argmax(dim=-1)
        proposal = proposal_logits.argmax(dim=-1)
        incumbent_token = tokens.gather(
            1, incumbent[:, None, None].expand(-1, 1, model_dim)
        ).squeeze(1)
        proposal_token = tokens.gather(
            1, proposal[:, None, None].expand(-1, 1, model_dim)
        ).squeeze(1)
        relations = pairwise_trajectory_relations(candidate_trajectories.detach())
        batch_index = torch.arange(batch_size, device=tokens.device)
        proposal_to_incumbent = relations[batch_index, proposal, incumbent]
        reference_gap = (
            masked_reference.gather(1, proposal[:, None])
            - masked_reference.gather(1, incumbent[:, None])
        ).squeeze(1)
        proposal_gap = (
            proposal_logits.gather(1, proposal[:, None])
            - proposal_logits.gather(1, incumbent[:, None])
        ).squeeze(1)
        scalar_context = [reference_gap, proposal_gap]
        if self.use_decision_context:
            reference_margin, reference_entropy = self._score_context(
                reference_logits, mask
            )
            proposal_margin, proposal_entropy = self._score_context(
                reference_logits + proposal_delta, mask
            )
            scalar_context.extend(
                (
                    reference_margin,
                    proposal_margin,
                    reference_entropy,
                    proposal_entropy,
                )
            )
        arbiter_inputs = torch.cat(
            (
                proposal_token,
                incumbent_token,
                proposal_to_incumbent,
                torch.stack(scalar_context, dim=-1),
            ),
            dim=-1,
        )
        proposal_evidence = self.regret_arbiter_head(arbiter_inputs).squeeze(-1)
        return proposal_delta, proposal_evidence, mask, incumbent, proposal

    @staticmethod
    def apply_regret_arbitration(
        proposal_delta: torch.Tensor,
        proposal_evidence: torch.Tensor,
        incumbent: torch.Tensor,
        proposal: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proposal_delta.ndim != 2:
            raise ValueError("proposal delta must have shape (B, K)")
        expected = proposal_delta.shape[0]
        if any(
            value.shape != (expected,)
            for value in (proposal_evidence, incumbent, proposal)
        ):
            raise ValueError("proposal arbitration group tensors are not aligned")
        override = proposal.ne(incumbent) & proposal_evidence.gt(float(threshold))
        residual = torch.where(
            override[:, None], proposal_delta, torch.zeros_like(proposal_delta)
        )
        return residual, override

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_override_diagnostics: bool = False,
    ):
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        (
            proposal_delta,
            proposal_evidence,
            mask,
            incumbent,
            proposal,
        ) = self.proposal_and_arbitration(
            tokens, candidate_trajectories, reference_logits, candidate_mask
        )
        residual, override = self.apply_regret_arbitration(
            proposal_delta,
            proposal_evidence,
            incumbent,
            proposal,
            self.override_threshold,
        )
        if not return_override_diagnostics:
            return residual
        return residual, {
            "proposal_delta": proposal_delta,
            "proposal_evidence": proposal_evidence,
            "candidate_mask": mask,
            "incumbent_index": incumbent,
            "proposal_index": proposal,
            "override_mask": override,
        }


class ProposalConditionedCounterfactualEvaluator(
    TrajectorySetReasoningResidualSelector
):
    """Evaluate a realized proposal with an independently trainable encoder."""

    requires_reference_logits = True
    supports_override_diagnostics = True
    requires_candidate_mask = True
    is_counterfactual_evaluator = True

    def __init__(
        self,
        *args,
        counterfactual_hidden_dim: int = 128,
        counterfactual_loss: str = "opportunity_risk",
        train_evaluator_encoder: bool = True,
        override_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if counterfactual_hidden_dim <= 0:
            raise ValueError("counterfactual_hidden_dim must be positive")
        if counterfactual_loss not in {
            "regret_bce", "signed_gain", "opportunity_risk"
        }:
            raise ValueError(f"unsupported counterfactual loss: {counterfactual_loss}")
        if not math.isfinite(float(override_threshold)):
            raise ValueError("override_threshold must be finite")
        self.counterfactual_hidden_dim = int(counterfactual_hidden_dim)
        self.counterfactual_loss = str(counterfactual_loss)
        self.train_evaluator_encoder = bool(train_evaluator_encoder)
        self.override_threshold = float(override_threshold)

        # This is a parameter-independent copy of the complete proposal
        # scene/set encoder. Its unused residual head is removed so checkpoint
        # and trainability audits cannot confuse it with another proposal.
        self.counterfactual_encoder = TrajectorySetReasoningResidualSelector(
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            route_bev_dim=self.route_bev_dim,
            context_dim=self.context_dim,
            step_geometry_dim=self.step_geometry_dim,
            relation_dim=self.relation_dim,
            geometry_hidden_dim=self.geometry_hidden_dim,
            relation_hidden_dim=self.relation_hidden_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            num_route_steps=self.num_route_steps,
            num_temporal_layers=self.num_temporal_layers,
            num_relation_layers=self.num_relation_layers,
            use_temporal_reasoning=self.use_temporal_reasoning,
            use_relational_reasoning=self.use_relational_reasoning,
            use_route_bev=self.use_route_bev,
            use_scene_context=self.use_scene_context,
        )
        del self.counterfactual_encoder.delta_head
        directed_input_dim = 2 * self.model_dim + self.relation_dim + 2
        self.counterfactual_value_head = nn.Sequential(
            nn.LayerNorm(directed_input_dim),
            nn.Linear(directed_input_dim, counterfactual_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(counterfactual_hidden_dim),
            nn.Linear(counterfactual_hidden_dim, 1),
        )
        nn.init.zeros_(self.counterfactual_value_head[-1].weight)
        nn.init.zeros_(self.counterfactual_value_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture="proposal_conditioned_counterfactual_evaluator",
            counterfactual_hidden_dim=self.counterfactual_hidden_dim,
            counterfactual_loss=self.counterfactual_loss,
            train_evaluator_encoder=self.train_evaluator_encoder,
            override_threshold=self.override_threshold,
        )
        return config

    def initialize_evaluator_from_proposal(self) -> float:
        """Copy the loaded proposal encoder into the independent evaluator."""
        complete_state = self.state_dict()
        copied = {}
        for name in self.counterfactual_encoder.state_dict():
            if name not in complete_state:
                raise RuntimeError(
                    f"proposal encoder has no evaluator initialization tensor: {name}"
                )
            copied[name] = complete_state[name].detach().clone()
        self.counterfactual_encoder.load_state_dict(copied, strict=True)
        initialized = self.counterfactual_encoder.state_dict()
        return max(
            (float((initialized[name] - copied[name]).abs().max()) for name in copied),
            default=0.0,
        )

    @staticmethod
    def _candidate_mask(
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return ProposalConditionedRegretArbitrator._candidate_mask(
            reference_logits, candidate_mask
        )

    def _directed_inputs(
        self,
        tokens: torch.Tensor,
        relations: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        reference_logits: torch.Tensor,
        proposal_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, model_dim = tokens.shape
        batch_index = torch.arange(batch_size, device=tokens.device)
        first_token = tokens.gather(
            1, first[:, None, None].expand(-1, 1, model_dim)
        ).squeeze(1)
        second_token = tokens.gather(
            1, second[:, None, None].expand(-1, 1, model_dim)
        ).squeeze(1)
        relation = relations[batch_index, first, second]
        reference_gap = (
            reference_logits.gather(1, first[:, None])
            - reference_logits.gather(1, second[:, None])
        ).squeeze(1)
        proposal_gap = (
            proposal_logits.gather(1, first[:, None])
            - proposal_logits.gather(1, second[:, None])
        ).squeeze(1)
        return torch.cat(
            (
                first_token,
                second_token,
                relation,
                torch.stack((reference_gap, proposal_gap), dim=-1),
            ),
            dim=-1,
        )

    def proposal_and_counterfactual_evaluation(
        self,
        proposal_tokens: torch.Tensor,
        evaluator_tokens: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ):
        if proposal_tokens.shape != evaluator_tokens.shape:
            raise ValueError("proposal and evaluator tokens are not aligned")
        if proposal_tokens.ndim != 3 or proposal_tokens.shape[:2] != reference_logits.shape:
            raise ValueError("candidate tokens and reference logits are not aligned")
        mask = self._candidate_mask(reference_logits, candidate_mask)
        proposal_delta = self.delta_head(proposal_tokens).squeeze(-1).masked_fill(
            ~mask, 0.0
        )
        masked_reference = reference_logits.detach().masked_fill(~mask, -torch.inf)
        proposal_logits = (reference_logits.detach() + proposal_delta).masked_fill(
            ~mask, -torch.inf
        )
        incumbent = masked_reference.argmax(dim=-1)
        proposal = proposal_logits.argmax(dim=-1)
        relations = pairwise_trajectory_relations(candidate_trajectories.detach())
        forward_inputs = self._directed_inputs(
            evaluator_tokens, relations, proposal, incumbent,
            masked_reference, proposal_logits,
        )
        reverse_inputs = self._directed_inputs(
            evaluator_tokens, relations, incumbent, proposal,
            masked_reference, proposal_logits,
        )
        forward_value = self.counterfactual_value_head(forward_inputs).squeeze(-1)
        reverse_value = self.counterfactual_value_head(reverse_inputs).squeeze(-1)
        if self.counterfactual_loss == "opportunity_risk":
            opportunity = F.softplus(forward_value)
            risk = F.softplus(reverse_value)
        else:
            opportunity = forward_value
            risk = reverse_value
        evidence = opportunity - risk
        return (
            proposal_delta, opportunity, risk, evidence, mask, incumbent, proposal
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_override_diagnostics: bool = False,
    ):
        encoder_inputs = dict(
            candidate_features=candidate_features,
            candidate_trajectories=candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        proposal_tokens = self.encode_tokens(**encoder_inputs)
        evaluator_tokens = self.counterfactual_encoder.encode_tokens(**encoder_inputs)
        (
            proposal_delta, opportunity, risk, evidence, mask, incumbent, proposal
        ) = self.proposal_and_counterfactual_evaluation(
            proposal_tokens, evaluator_tokens, candidate_trajectories,
            reference_logits, candidate_mask,
        )
        residual, override = ProposalConditionedRegretArbitrator.apply_regret_arbitration(
            proposal_delta, evidence, incumbent, proposal, self.override_threshold
        )
        if not return_override_diagnostics:
            return residual
        return residual, {
            "proposal_delta": proposal_delta,
            "opportunity": opportunity,
            "risk": risk,
            "proposal_evidence": evidence,
            "candidate_mask": mask,
            "incumbent_index": incumbent,
            "proposal_index": proposal,
            "override_mask": override,
        }


class CapacityMatchedUnaryResidualSelector(
    TrajectorySetReasoningResidualSelector
):
    """Unary control with nearly the same extra parameter count as RAPG."""

    requires_reference_logits = False

    def __init__(self, *args, capacity_hidden_dim: int = 277, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if capacity_hidden_dim <= 0:
            raise ValueError("capacity_hidden_dim must be positive")
        self.capacity_hidden_dim = int(capacity_hidden_dim)
        self.capacity_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, capacity_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(capacity_hidden_dim),
            nn.Linear(capacity_hidden_dim, 1),
        )
        nn.init.zeros_(self.capacity_head[-1].weight)
        nn.init.zeros_(self.capacity_head[-1].bias)

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture="capacity_matched_unary",
            capacity_hidden_dim=self.capacity_hidden_dim,
        )
        return config

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
        )
        return (
            self.delta_head(tokens) + self.capacity_head(tokens)
        ).squeeze(-1)


class InteractionAwareResidualSelector(SceneConditionedTrajectorySetSelector):
    """V3 residual selector with explicit frozen candidate--agent relations.

    ``interaction_generic`` (A1) pools unordered frozen tracks for each
    candidate with shared attention. ``interaction_relation`` (A2) adds the
    existing directed candidate-relation block after that interaction update.
    Neither arm consumes reward components, future labels, or a learned motion
    forecast. The inherited final residual layer is zero initialized.
    """

    requires_frozen_track_states = True

    def __init__(
        self,
        *args,
        interaction_dim: int = CANDIDATE_AGENT_INTERACTION_DIM,
        interaction_hidden_dim: int = 128,
        num_agent_classes: int = 10,
        max_agents: int = 30,
        seconds_per_step: float = 0.5,
        use_relation_aware_set_block: bool = False,
        num_interaction_heads: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if interaction_dim != CANDIDATE_AGENT_INTERACTION_DIM:
            raise ValueError("candidate-agent interaction dimension drifted")
        if min(interaction_hidden_dim, num_agent_classes, max_agents) <= 0:
            raise ValueError("interaction selector dimensions must be positive")
        if seconds_per_step <= 0.0:
            raise ValueError("seconds_per_step must be positive")
        interaction_heads = int(num_interaction_heads or self.num_heads)
        if self.model_dim % interaction_heads:
            raise ValueError("model_dim must be divisible by interaction heads")

        self.interaction_dim = int(interaction_dim)
        self.interaction_hidden_dim = int(interaction_hidden_dim)
        self.num_agent_classes = int(num_agent_classes)
        self.max_agents = int(max_agents)
        self.seconds_per_step = float(seconds_per_step)
        self.use_relation_aware_set_block = bool(use_relation_aware_set_block)
        self.num_interaction_heads = interaction_heads
        self.interaction_encoder = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, interaction_hidden_dim),
            nn.GELU(),
            nn.Linear(interaction_hidden_dim, self.model_dim),
        )
        self.agent_class_embedding = nn.Embedding(
            self.num_agent_classes + 1, self.model_dim, padding_idx=0
        )
        self.interaction_step_embedding = nn.Parameter(
            torch.zeros(self.num_route_steps, self.model_dim)
        )
        nn.init.normal_(self.interaction_step_embedding, std=0.02)
        self.null_agent_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        nn.init.normal_(self.null_agent_token, std=0.02)
        self.interaction_attention = nn.MultiheadAttention(
            self.model_dim,
            self.num_interaction_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.interaction_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.model_dim),
            nn.Linear(2 * self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.interaction_norm = nn.LayerNorm(self.model_dim)
        self.candidate_relation_block = None
        if self.use_relation_aware_set_block:
            self.candidate_relation_block = RelationAwareSetBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                relation_dim=PAIRWISE_TRAJECTORY_RELATION_DIM,
                relation_hidden_dim=self.interaction_hidden_dim,
            )

    def config_dict(self) -> Dict[str, object]:
        config = super().config_dict()
        config.update(
            architecture=(
                "interaction_relation"
                if self.use_relation_aware_set_block
                else "interaction_generic"
            ),
            interaction_dim=self.interaction_dim,
            interaction_hidden_dim=self.interaction_hidden_dim,
            num_agent_classes=self.num_agent_classes,
            max_agents=self.max_agents,
            seconds_per_step=self.seconds_per_step,
            use_relation_aware_set_block=self.use_relation_aware_set_block,
            num_interaction_heads=self.num_interaction_heads,
        )
        return config

    def _interaction_update(
        self,
        candidate_tokens: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        frozen_track_states: torch.Tensor,
        frozen_track_classes: torch.Tensor,
        frozen_track_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = candidate_tokens.shape
        expected_states = (batch_size, self.max_agents, FROZEN_TRACK_STATE_DIM)
        expected_agents = (batch_size, self.max_agents)
        if tuple(frozen_track_states.shape) != expected_states:
            raise ValueError(
                f"frozen track states must have shape {expected_states}, got "
                f"{tuple(frozen_track_states.shape)}"
            )
        if tuple(frozen_track_classes.shape) != expected_agents:
            raise ValueError("frozen track classes must have shape (B, A)")
        if tuple(frozen_track_mask.shape) != expected_agents:
            raise ValueError("frozen track mask must have shape (B, A)")
        if bool(
            (
                frozen_track_mask
                & ((frozen_track_classes < 0) | (frozen_track_classes >= self.num_agent_classes))
            ).any()
        ):
            raise ValueError("valid frozen track class is out of range")

        relations, relation_mask = candidate_frame_agent_interactions(
            candidate_trajectories,
            frozen_track_states,
            frozen_track_mask,
            seconds_per_step=self.seconds_per_step,
        )
        relations = relations.to(dtype=candidate_tokens.dtype)
        encoded = self.interaction_encoder(relations)
        class_index = torch.where(
            frozen_track_mask,
            frozen_track_classes + 1,
            torch.zeros_like(frozen_track_classes),
        ).long()
        encoded = encoded + self.agent_class_embedding(class_index)[
            :, None, :, None
        ]
        encoded = encoded + self.interaction_step_embedding[None, None, None]
        flat_relations = encoded.reshape(
            batch_size * num_candidates,
            self.max_agents * self.num_route_steps,
            self.model_dim,
        )
        flat_padding = ~relation_mask.reshape(
            batch_size * num_candidates,
            self.max_agents * self.num_route_steps,
        )
        null_token = self.null_agent_token.expand(
            batch_size * num_candidates, -1, -1
        )
        flat_relations = torch.cat((flat_relations, null_token), dim=1)
        flat_padding = torch.cat(
            (
                flat_padding,
                torch.zeros(
                    batch_size * num_candidates,
                    1,
                    dtype=torch.bool,
                    device=flat_padding.device,
                ),
            ),
            dim=1,
        )
        query = candidate_tokens.reshape(
            batch_size * num_candidates, 1, self.model_dim
        )
        with math_sdp_kernel(query):
            update = self.interaction_attention(
                query,
                flat_relations,
                flat_relations,
                key_padding_mask=flat_padding,
                need_weights=False,
            )[0]
        update = update.reshape(batch_size, num_candidates, self.model_dim)
        fused = self.interaction_fusion(torch.cat((candidate_tokens, update), dim=-1))
        return self.interaction_norm(candidate_tokens + fused)

    def encode_tokens(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        frozen_track_states: Optional[torch.Tensor] = None,
        frozen_track_classes: Optional[torch.Tensor] = None,
        frozen_track_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if any(
            value is None
            for value in (
                frozen_track_states,
                frozen_track_classes,
                frozen_track_mask,
            )
        ):
            raise ValueError("interaction selector requires frozen track states/classes/mask")
        tokens = super().encode_tokens(
            candidate_features.detach(),
            candidate_trajectories.detach(),
            route_bev_features=(
                None if route_bev_features is None else route_bev_features.detach()
            ),
            status_token=None if status_token is None else status_token.detach(),
            ego_query=None if ego_query is None else ego_query.detach(),
            agents_query=None if agents_query is None else agents_query.detach(),
        )
        tokens = self._interaction_update(
            tokens,
            candidate_trajectories.detach(),
            frozen_track_states.detach(),
            frozen_track_classes.detach(),
            frozen_track_mask.detach(),
        )
        if self.candidate_relation_block is not None:
            tokens = self.candidate_relation_block(
                tokens, pairwise_trajectory_relations(candidate_trajectories.detach())
            )
        return self.output_norm(tokens)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        route_bev_features: Optional[torch.Tensor] = None,
        status_token: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        agents_query: Optional[torch.Tensor] = None,
        frozen_track_states: Optional[torch.Tensor] = None,
        frozen_track_classes: Optional[torch.Tensor] = None,
        frozen_track_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(
            candidate_features,
            candidate_trajectories,
            route_bev_features=route_bev_features,
            status_token=status_token,
            ego_query=ego_query,
            agents_query=agents_query,
            frozen_track_states=frozen_track_states,
            frozen_track_classes=frozen_track_classes,
            frozen_track_mask=frozen_track_mask,
        )
        return self.delta_head(tokens).squeeze(-1)


def build_scene_selector(config: Dict[str, object]) -> nn.Module:
    """Build a selector while keeping all frozen V3 configs load-compatible."""

    selector_config = dict(config)
    architecture = selector_config.pop("architecture", "scene_conditioned_v3")
    if architecture in {"scene_conditioned_v3", "v3"}:
        return SceneConditionedTrajectorySetSelector(**selector_config)
    if architecture == "trajectory_set_reasoner":
        return TrajectorySetReasoningResidualSelector(**selector_config)
    if architecture == "reference_anchored_preference_graph":
        return ReferenceAnchoredPreferenceGraphSelector(**selector_config)
    if architecture == "incumbent_verified_preference":
        return IncumbentVerifiedPreferenceSelector(**selector_config)
    if architecture == "proposal_conditioned_regret_arbitration":
        return ProposalConditionedRegretArbitrator(**selector_config)
    if architecture == "proposal_conditioned_counterfactual_evaluator":
        return ProposalConditionedCounterfactualEvaluator(**selector_config)
    if architecture == "capacity_matched_unary":
        return CapacityMatchedUnaryResidualSelector(**selector_config)
    if architecture in {"interaction_generic", "interaction_relation"}:
        selector_config["use_relation_aware_set_block"] = (
            architecture == "interaction_relation"
        )
        return InteractionAwareResidualSelector(**selector_config)
    raise ValueError(f"unsupported selector architecture: {architecture}")
