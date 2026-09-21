"""Scene-conditioned selector for DiffusionDrive exact-group GRPO.

The module only scores an already generated candidate set.  Perception, the
DiffusionDrive denoiser and trajectory regression stay frozen; observation
features are detached before entering this selector.  No reward, PDM component
or future label is accepted as an input.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAJECTORY_GEOMETRY_DIM = 58


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
