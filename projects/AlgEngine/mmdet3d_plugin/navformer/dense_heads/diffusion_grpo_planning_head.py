"""Selector-only GRPO for the WorldEngine DiffusionDrive planning head.

The diffusion generator is intentionally kept identical to and initialized
from :class:`DiffusionPlanningHead`.  GRPO is applied only to the categorical
selector that chooses one of the generated trajectories.  This module does
not optimize denoising actions and does not use the GT-nearest trajectory as
the selector target.

Training data must provide one PDM reward per generated candidate through
``gt_pdm_score[reward_key]``.  With the default DiffusionDrive setup this is a
``(B, 20)`` tensor aligned with ``candidate_trajectories``.  The ordinary
NAVSIM ``(B, 8192)`` vocabulary cache is deliberately rejected because those
scores do not correspond to the trajectories generated in this forward pass.
"""

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.optimizer import (
    OPTIMIZER_BUILDERS,
    OPTIMIZERS,
    DefaultOptimizerConstructor,
)
from mmcv.utils import build_from_cfg
from mmdet.models.builder import HEADS

from .diffusion_planning_head import (
    DiffusionPlanningHead,
    gen_sineembed_for_position,
)


@OPTIMIZER_BUILDERS.register_module()
class DiffusionGRPOSelectorOptimizerConstructor(DefaultOptimizerConstructor):
    """Build an optimizer containing only the reopened selector parameters."""

    def __call__(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module

        selector_parameters = []
        for name, parameter in model.named_parameters():
            is_selector = (
                name.startswith("planning_head.diff_decoder.layers.")
                and ".task_decoder.plan_cls_branch." in name
                and parameter.requires_grad
            )
            parameter.requires_grad = is_selector
            if is_selector:
                selector_parameters.append(parameter)

        if not selector_parameters:
            raise RuntimeError(
                "DiffusionGRPOSelectorOptimizerConstructor found no trainable "
                "DiffusionDrive selector parameters"
            )

        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cfg["params"] = selector_parameters
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@HEADS.register_module()
class DiffusionGRPOSelectorPlanningHead(DiffusionPlanningHead):
    """Freeze DiffusionDrive generation and GRPO-tune its trajectory selector.

    The final DiffusionDrive decoder layer supplies ``num_anchors`` generated
    trajectories and one logit per trajectory.  Rewards are normalized within
    each candidate group, following the group-relative policy-gradient form
    used by ReCogDrive, but the diffusion chain itself is never treated as an
    action and receives no gradient.
    """

    requires_candidate_pdm_rewards = True

    def __init__(
        self,
        *args,
        selector_layer: int = -1,
        train_all_selector_layers: bool = False,
        reward_key: str = "score",
        policy_loss_weight: float = 1.0,
        entropy_weight: float = 0.01,
        advantage_epsilon: float = 1e-6,
        minimum_reward_std: float = 1e-6,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not 0.0 <= clip_advantage_lower_quantile <= 1.0:
            raise ValueError("clip_advantage_lower_quantile must be in [0, 1]")
        if not 0.0 <= clip_advantage_upper_quantile <= 1.0:
            raise ValueError("clip_advantage_upper_quantile must be in [0, 1]")
        if clip_advantage_lower_quantile > clip_advantage_upper_quantile:
            raise ValueError("advantage quantiles are reversed")

        num_layers = len(self.diff_decoder.layers)
        normalized_layer = selector_layer if selector_layer >= 0 else num_layers + selector_layer
        if not 0 <= normalized_layer < num_layers:
            raise ValueError(
                f"selector_layer={selector_layer} is invalid for {num_layers} decoder layers"
            )

        self.selector_layer = normalized_layer
        self.train_all_selector_layers = train_all_selector_layers
        self.reward_key = reward_key
        self.policy_loss_weight = policy_loss_weight
        self.entropy_weight = entropy_weight
        self.advantage_epsilon = advantage_epsilon
        self.minimum_reward_std = minimum_reward_std
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

        if train_all_selector_layers:
            self._trainable_selector_layers = tuple(range(num_layers))
        else:
            self._trainable_selector_layers = (normalized_layer,)
        self._freeze_generator_and_open_selector()
        self.train(self.training)

    def _selector_modules(self) -> Sequence[nn.Module]:
        return tuple(
            self.diff_decoder.layers[index].task_decoder.plan_cls_branch
            for index in self._trainable_selector_layers
        )

    def _freeze_generator_and_open_selector(self) -> None:
        """Make the trainable boundary explicit at parameter level."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for selector in self._selector_modules():
            for parameter in selector.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen generator deterministic while the selector trains."""
        nn.Module.train(self, False)
        # ``DiffusionPlanningHead.forward`` dispatches on the top-level flag.
        self.training = mode
        for selector in self._selector_modules():
            selector.train(mode)
        return self

    @property
    def trainable_parameter_names(self):
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    @auto_fp16(apply_to=("bev_embed",))
    def forward(
        self,
        bev_embed,
        command=None,
        sdc_planning_past=None,
        sdc_status=None,
        sdc_planning_mask_past=None,
        gt_pre_command_sdc=None,
        navigation_goal=None,
    ):
        # This cuts the gradient path into NAVFormer perception/tracking.  The
        # only leaves requiring gradients are the selector parameters reopened
        # in ``_freeze_generator_and_open_selector``.
        return super().forward(
            bev_embed.detach(),
            command,
            sdc_planning_past,
            sdc_status,
            sdc_planning_mask_past,
            gt_pre_command_sdc,
            navigation_goal=navigation_goal,
        )

    def _all_candidates_to_40(self, candidates_8):
        batch_size, num_candidates, num_poses, pose_dim = candidates_8.shape
        flattened = candidates_8.reshape(batch_size * num_candidates, num_poses, pose_dim)
        expanded = self._expand_to_40(flattened)
        return expanded.reshape(batch_size, num_candidates, expanded.shape[1], pose_dim)

    def _attach_selector_outputs(self, result: Dict[str, torch.Tensor]):
        candidates_8 = result["poses_reg_list"][-1]
        selector_logits = result["poses_cls_list"][-1]
        result["selector_logits"] = selector_logits
        # Rewards are a non-differentiable property of generated trajectories.
        result["candidate_trajectories_8"] = candidates_8.detach()
        result["candidate_trajectories"] = self._all_candidates_to_40(candidates_8).detach()
        return result

    def _forward_train(self, bev_feature, ego_query, agents_query, status_token):
        return self._attach_selector_outputs(
            super()._forward_train(bev_feature, ego_query, agents_query, status_token)
        )

    def _forward_test(self, bev_feature, ego_query, agents_query, status_token):
        """Official inference loop with all generated candidates exposed."""
        batch_size = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / self.inference_steps
        roll_timesteps = (np.arange(0, self.inference_steps) * step_ratio).round()[::-1].copy()
        roll_timesteps = torch.from_numpy(roll_timesteps.astype(np.int64)).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        image = self.norm_odo_xy(plan_anchor)
        noise = torch.randn(image.shape, device=device)
        truncation = torch.full(
            (batch_size,), self.trunc_timesteps, device=device, dtype=torch.long
        )
        image = self.diffusion_scheduler.add_noise(image, noise, truncation)

        num_modes = image.shape[1]
        bev_spatial_shape = (self.bev_h, self.bev_w)
        poses_reg = None
        poses_cls = None
        for timestep in roll_timesteps:
            noisy_traj_points = self.denorm_odo_xy(torch.clamp(image, min=-1, max=1))
            traj_pos_embed = gen_sineembed_for_position(
                noisy_traj_points, hidden_dim=64
            ).flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(
                batch_size, num_modes, -1
            )

            timesteps = timestep.reshape(1).expand(batch_size)
            time_embed = self.time_mlp(timesteps).view(batch_size, 1, -1)
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_token,
                None,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            image = self.diffusion_scheduler.step(
                model_output=self.norm_odo_xy(poses_reg[..., :2]),
                timestep=timestep,
                sample=image,
            ).prev_sample

        selected_indices = poses_cls.argmax(dim=-1)
        gather_index = selected_indices[..., None, None, None].repeat(
            1, 1, self._num_poses, 3
        )
        selected_8 = torch.gather(poses_reg, 1, gather_index).squeeze(1)
        return {
            "trajectory": self._expand_to_40(selected_8),
            "trajectory_8": selected_8,
            "selected_indices": selected_indices,
            "selector_logits": poses_cls,
            "candidate_trajectories_8": poses_reg.detach(),
            "candidate_trajectories": self._all_candidates_to_40(poses_reg).detach(),
        }

    def _extract_candidate_rewards(self, gt_pdm_score, selector_logits):
        if not isinstance(gt_pdm_score, dict) or self.reward_key not in gt_pdm_score:
            raise KeyError(
                f"selector GRPO requires gt_pdm_score[{self.reward_key!r}] with one "
                "PDM reward per generated candidate"
            )
        rewards = gt_pdm_score[self.reward_key]
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=selector_logits.device)
        rewards = rewards.to(device=selector_logits.device, dtype=torch.float32)
        while rewards.ndim > 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)
        if rewards.ndim == 1 and selector_logits.shape[0] == 1:
            rewards = rewards.unsqueeze(0)
        if rewards.shape != selector_logits.shape:
            raise ValueError(
                "selector rewards must align exactly with generated candidates: "
                f"expected {tuple(selector_logits.shape)}, got {tuple(rewards.shape)}. "
                "The ordinary 8192-trajectory NAVSIM cache is not valid for the "
                f"{self.num_anchors} DiffusionDrive candidates."
            )
        return rewards.detach()

    def _group_relative_advantages(self, rewards):
        valid = torch.isfinite(rewards)
        count = valid.sum(dim=-1)
        safe_count = count.clamp_min(1).to(rewards.dtype)
        safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
        reward_mean = safe_rewards.sum(dim=-1) / safe_count
        centered = torch.where(valid, rewards - reward_mean[:, None], torch.zeros_like(rewards))
        reward_var = centered.square().sum(dim=-1) / safe_count
        reward_std = reward_var.sqrt()
        active_group = (count >= 2) & (reward_std > self.minimum_reward_std)
        advantages = centered / reward_std.clamp_min(self.advantage_epsilon)[:, None]
        policy_mask = valid & active_group[:, None]
        advantages = torch.where(policy_mask, advantages, torch.zeros_like(advantages))

        valid_advantages = advantages[policy_mask]
        if valid_advantages.numel() and (
            self.clip_advantage_lower_quantile > 0.0
            or self.clip_advantage_upper_quantile < 1.0
        ):
            lower = torch.quantile(valid_advantages, self.clip_advantage_lower_quantile)
            upper = torch.quantile(valid_advantages, self.clip_advantage_upper_quantile)
            advantages = advantages.clamp(min=lower, max=upper)
        return advantages.detach(), valid, policy_mask, active_group, reward_mean, reward_std

    @force_fp32(apply_to=("result", "gt_pdm_score"))
    def loss(
        self,
        result=None,
        gt_pdm_score=None,
        sdc_planning=None,
        sdc_planning_mask=None,
        il_target=None,
        il_target_mask=None,
    ):
        """Group-relative policy gradient over the trajectory selector only."""
        selector_logits = result["selector_logits"].float()
        rewards = self._extract_candidate_rewards(gt_pdm_score, selector_logits)
        (
            advantages,
            valid,
            policy_mask,
            active_group,
            reward_mean,
            reward_std,
        ) = self._group_relative_advantages(rewards)

        masked_logits = selector_logits.masked_fill(~valid, -1e4)
        log_probabilities = F.log_softmax(masked_logits, dim=-1)
        probabilities = log_probabilities.exp()
        policy_denominator = policy_mask.sum().clamp_min(1).to(selector_logits.dtype)
        policy_loss = -(
            advantages * log_probabilities * policy_mask.to(selector_logits.dtype)
        ).sum() / policy_denominator

        entropy_per_group = -(
            probabilities * log_probabilities * valid.to(selector_logits.dtype)
        ).sum(dim=-1)
        if active_group.any():
            entropy = entropy_per_group[active_group].mean()
            advantage_abs_mean = advantages[policy_mask].abs().mean()
            selected_indices = masked_logits.argmax(dim=-1)
            oracle_indices = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)
            selector_oracle_match = (
                selected_indices[active_group] == oracle_indices[active_group]
            ).float().mean()
            selected_reward = rewards.gather(1, selected_indices[:, None]).squeeze(1)
            selected_reward = selected_reward[active_group].mean()
        else:
            zero = selector_logits.sum() * 0.0
            entropy = zero
            advantage_abs_mean = zero
            selector_oracle_match = zero
            selected_reward = zero

        return {
            "loss.grpo_selector_policy": self.policy_loss_weight * policy_loss,
            "loss.grpo_selector_entropy": -self.entropy_weight * entropy,
            "grpo.selector.reward_mean": reward_mean.mean(),
            "grpo.selector.reward_std": reward_std.mean(),
            "grpo.selector.selected_reward": selected_reward,
            "grpo.selector.oracle_match": selector_oracle_match,
            "grpo.selector.active_group_fraction": active_group.float().mean(),
            "grpo.selector.advantage_abs_mean": advantage_abs_mean,
        }
