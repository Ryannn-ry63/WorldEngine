"""Selector-only GRPO for WorldEngine DiffusionDrive.

The frozen DiffusionDrive denoiser dynamically generates 20 trajectories.
Two selectors evaluate the same final candidate features: a frozen baseline
(``pi_ref == pi_old``) and a trainable current policy.  NAVSIM-v1 PDM rewards
are computed online for those exact trajectories; no 8192-entry vocabulary or
GT-nearest imitation target is used by this training path.
"""

import copy
import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import HOOKS, Hook, auto_fp16, force_fp32
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
from .diffusion_grpo_scene_selector import (
    SceneConditionedTrajectorySetSelector,
    sample_final_trajectory_bev_features,
)
from .diffusiondrive_online_pdm_reward import OnlineDiffusionDrivePDMReward


@OPTIMIZER_BUILDERS.register_module()
class DiffusionGRPOOnlineSelectorOptimizerConstructor(DefaultOptimizerConstructor):
    """Build an optimizer containing only the current final selector."""

    def __call__(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module

        selector_parameters = []
        for name, parameter in model.named_parameters():
            is_original_selector = (
                name.startswith("planning_head.diff_decoder.layers.")
                and ".task_decoder.plan_cls_branch." in name
                and parameter.requires_grad
            )
            is_scene_selector = (
                name.startswith("planning_head.scene_selector.") and parameter.requires_grad
            )
            is_selector = is_original_selector or is_scene_selector
            parameter.requires_grad = is_selector
            if is_selector:
                selector_parameters.append(parameter)

        if not selector_parameters:
            raise RuntimeError(
                "DiffusionGRPOOnlineSelectorOptimizerConstructor found no trainable "
                "DiffusionDrive selector parameters"
            )

        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cfg["params"] = selector_parameters
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@HOOKS.register_module()
class DiffusionGRPOOnlineReferenceSelectorHook(Hook):
    """Initialize the immutable reference selector after checkpoint loading."""

    def __init__(self, out_dir=None):
        # WorldEngine's shared trainer injects work_dir into every custom hook.
        # Reference initialization itself does not write files, but accepting
        # this field keeps the DiffusionDrive-only hook compatible.
        self.out_dir = out_dir

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        head = getattr(model, "planning_head", None)
        if not isinstance(head, DiffusionGRPOOnlineSelectorPlanningHead):
            raise TypeError(
                "DiffusionGRPOOnlineReferenceSelectorHook requires "
                "DiffusionGRPOOnlineSelectorPlanningHead"
            )
        tensor_count = head.initialize_reference_selector()
        runner.logger.info(
            "Initialized frozen DiffusionDrive reference selector from %s (%d tensors)",
            head.reference_checkpoint or "current selector snapshot",
            tensor_count,
        )


@HEADS.register_module()
class DiffusionGRPOOnlineSelectorPlanningHead(DiffusionPlanningHead):
    """GRPO-tune only the selector over frozen dynamic trajectories."""

    requires_online_candidate_rewards = True
    requires_paired_inference_sample_tokens = True

    def __init__(
        self,
        *args,
        selector_layer: int = -1,
        reward_key: str = "score",
        policy_loss_weight: float = 1.0,
        policy_objective: str = "clipped_reference_grpo",
        policy_temperature: float = 1.0,
        clip_epsilon: float = 0.2,
        kl_weight: float = 1e-3,
        advantage_epsilon: float = 1e-6,
        minimum_reward_std: float = 1e-6,
        reference_checkpoint: Optional[str] = None,
        reference_checkpoint_sha256: Optional[str] = None,
        candidate_noise_namespace: Optional[str] = None,
        scene_selector: Optional[Dict] = None,
        online_reward: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in [0, 1)")
        supported_objectives = {
            "clipped_reference_grpo",
            "exact_group_grpo",
        }
        if policy_objective not in supported_objectives:
            raise ValueError(
                f"unsupported policy_objective={policy_objective!r}; "
                f"expected one of {sorted(supported_objectives)}"
            )
        if policy_temperature <= 0.0:
            raise ValueError("policy_temperature must be positive")
        if kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")

        num_layers = len(self.diff_decoder.layers)
        normalized_layer = selector_layer if selector_layer >= 0 else num_layers + selector_layer
        if not 0 <= normalized_layer < num_layers:
            raise ValueError(
                f"selector_layer={selector_layer} is invalid for {num_layers} decoder layers"
            )

        self.selector_layer = normalized_layer
        self.reward_key = reward_key
        self.policy_loss_weight = float(policy_loss_weight)
        self.policy_objective = str(policy_objective)
        self.policy_temperature = float(policy_temperature)
        self.clip_epsilon = float(clip_epsilon)
        self.kl_weight = float(kl_weight)
        self.advantage_epsilon = float(advantage_epsilon)
        self.minimum_reward_std = float(minimum_reward_std)
        self.reference_checkpoint = reference_checkpoint
        self.reference_checkpoint_sha256 = reference_checkpoint_sha256

        self.reference_selector = copy.deepcopy(self._current_selector())
        self.reference_selector.requires_grad_(False)
        self.reference_selector.eval()
        self._reference_selector_initialized = False
        self.scene_selector = (
            SceneConditionedTrajectorySetSelector(**scene_selector)
            if scene_selector is not None else None
        )
        self.online_reward = (
            OnlineDiffusionDrivePDMReward(**online_reward)
            if online_reward is not None
            else None
        )
        self._active_sample_tokens = None
        self._candidate_noise_namespace = candidate_noise_namespace

        self._freeze_generator_and_open_selector()
        self.train(self.training)

    def _current_selector(self) -> nn.Module:
        return self.diff_decoder.layers[
            self.selector_layer
        ].task_decoder.plan_cls_branch

    def _freeze_generator_and_open_selector(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        trainable_selector = (
            self._current_selector()
            if self.scene_selector is None
            else self.scene_selector
        )
        for parameter in trainable_selector.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        """Keep generator/reference deterministic and train only current selector."""
        nn.Module.train(self, False)
        self.training = mode
        self._current_selector().train(mode if self.scene_selector is None else False)
        if self.scene_selector is not None:
            self.scene_selector.train(mode)
        self.reference_selector.eval()
        return self

    @property
    def trainable_parameter_names(self):
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError("reference checkpoint must contain a state dict")
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    def initialize_reference_selector(self) -> int:
        """Load pi_ref from the immutable baseline, independently of pi_current."""
        if self.reference_checkpoint is None:
            state = self._current_selector().state_dict()
        else:
            checkpoint_path = Path(self.reference_checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"reference checkpoint does not exist: {checkpoint_path}"
                )
            if self.reference_checkpoint_sha256 is not None:
                actual_sha = self._sha256(checkpoint_path)
                if actual_sha != self.reference_checkpoint_sha256:
                    raise RuntimeError(
                        "reference checkpoint SHA256 mismatch: "
                        f"expected {self.reference_checkpoint_sha256}, got {actual_sha}"
                    )
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            source = self._checkpoint_state_dict(checkpoint)
            relative_state = self.reference_selector.state_dict()
            prefixes = (
                f"planning_head.diff_decoder.layers.{self.selector_layer}."
                "task_decoder.plan_cls_branch.",
                f"module.planning_head.diff_decoder.layers.{self.selector_layer}."
                "task_decoder.plan_cls_branch.",
                f"diff_decoder.layers.{self.selector_layer}."
                "task_decoder.plan_cls_branch.",
            )
            state = {}
            missing = []
            for relative_key in relative_state:
                source_key = next(
                    (
                        prefix + relative_key
                        for prefix in prefixes
                        if prefix + relative_key in source
                    ),
                    None,
                )
                if source_key is None:
                    missing.append(relative_key)
                else:
                    state[relative_key] = source[source_key]
            if missing:
                raise KeyError(
                    "reference checkpoint is missing selector tensors: "
                    + ", ".join(missing)
                )

        self.reference_selector.load_state_dict(state, strict=True)
        self.reference_selector.requires_grad_(False)
        self.reference_selector.eval()
        self._reference_selector_initialized = True
        return len(state)

    def _ensure_reference_selector(self) -> None:
        if not self._reference_selector_initialized:
            self.initialize_reference_selector()

    def set_candidate_noise_namespace(self, namespace: Optional[str]) -> None:
        """Pin diffusion noise by (namespace, sample token) for paired eval."""
        self._candidate_noise_namespace = (
            None if namespace is None else str(namespace)
        )

    def _candidate_noise(self, image: torch.Tensor) -> torch.Tensor:
        if self._candidate_noise_namespace is None:
            return torch.randn_like(image)
        if self._active_sample_tokens is None:
            raise RuntimeError(
                "deterministic candidate noise requires sample_tokens"
            )
        if len(self._active_sample_tokens) != image.shape[0]:
            raise ValueError(
                "sample token count does not match the candidate batch"
            )

        samples = []
        for batch_index, token in enumerate(self._active_sample_tokens):
            digest = hashlib.sha256(
                (
                    self._candidate_noise_namespace + ":" + str(token)
                ).encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            seed &= (1 << 63) - 1
            generator = torch.Generator(device=image.device)
            generator.manual_seed(seed)
            samples.append(
                torch.randn(
                    image[batch_index : batch_index + 1].shape,
                    dtype=image.dtype,
                    device=image.device,
                    generator=generator,
                )
            )
        return torch.cat(samples, dim=0)

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
        sample_tokens: Optional[Sequence[str]] = None,
    ):
        self._active_sample_tokens = sample_tokens
        try:
            return super().forward(
                bev_embed.detach(),
                command,
                sdc_planning_past,
                sdc_status,
                sdc_planning_mask_past,
                gt_pre_command_sdc,
                navigation_goal=navigation_goal,
            )
        finally:
            self._active_sample_tokens = None

    def _all_candidates_to_40(self, candidates_8):
        batch_size, num_candidates, num_poses, pose_dim = candidates_8.shape
        flattened = candidates_8.reshape(
            batch_size * num_candidates, num_poses, pose_dim
        )
        expanded = self._expand_to_40(flattened)
        return expanded.reshape(
            batch_size, num_candidates, expanded.shape[1], pose_dim
        )

    @torch.no_grad()
    def _generate_frozen_candidates(
        self, bev_feature, ego_query, agents_query, status_token
    ):
        """Run the same complete DDIM schedule used at inference."""
        batch_size = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / self.inference_steps
        roll_timesteps = (
            np.arange(0, self.inference_steps) * step_ratio
        ).round()[::-1].copy()
        roll_timesteps = torch.from_numpy(
            roll_timesteps.astype(np.int64)
        ).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        image = self.norm_odo_xy(plan_anchor)
        noise = self._candidate_noise(image)
        truncation = torch.full(
            (batch_size,), self.trunc_timesteps, device=device, dtype=torch.long
        )
        image = self.diffusion_scheduler.add_noise(image, noise, truncation)

        num_modes = image.shape[1]
        bev_spatial_shape = (self.bev_h, self.bev_w)
        final_reg = None
        final_feature = None
        for timestep in roll_timesteps:
            noisy_traj_points = self.denorm_odo_xy(
                torch.clamp(image, min=-1, max=1)
            )
            traj_pos_embed = gen_sineembed_for_position(
                noisy_traj_points, hidden_dim=64
            ).flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(
                batch_size, num_modes, -1
            )
            timesteps = timestep.reshape(1).expand(batch_size)
            time_embed = self.time_mlp(timesteps).view(batch_size, 1, -1)
            poses_reg_list, _, feature_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_token,
                None,
                return_traj_features=True,
            )
            final_reg = poses_reg_list[-1]
            final_feature = feature_list[self.selector_layer]
            image = self.diffusion_scheduler.step(
                model_output=self.norm_odo_xy(final_reg[..., :2]),
                timestep=timestep,
                sample=image,
            ).prev_sample

        if final_reg is None or final_feature is None:
            raise RuntimeError("DiffusionDrive produced no candidates")
        return final_reg.detach(), final_feature.detach()

    def _scene_selector_context(
        self,
        candidates_8,
        bev_feature,
        ego_query,
        agents_query,
        status_token,
    ):
        cross_bev_attention = self.diff_decoder.layers[
            self.selector_layer
        ].cross_bev_attention
        route_bev_features = sample_final_trajectory_bev_features(
            bev_feature,
            candidates_8,
            bev_range_x=cross_bev_attention.bev_range_x,
            bev_range_y=cross_bev_attention.bev_range_y,
        )
        return {
            "route_bev_features": route_bev_features,
            "status_token": status_token.detach(),
            "ego_query": ego_query.detach(),
            "agents_query": agents_query.detach(),
        }

    def _selector_outputs(
        self,
        candidate_feature,
        candidates_8,
        bev_feature,
        ego_query,
        agents_query,
        status_token,
    ):
        self._ensure_reference_selector()
        with torch.no_grad():
            reference_logits = self.reference_selector(
                candidate_feature.detach()
            ).squeeze(-1)
        if self.scene_selector is None:
            current_logits = self._current_selector()(candidate_feature).squeeze(-1)
        else:
            context = self._scene_selector_context(
                candidates_8, bev_feature, ego_query, agents_query, status_token
            )
            delta_logits = self.scene_selector(
                candidate_feature.detach(), candidates_8.detach(), **context
            )
            current_logits = reference_logits.detach() + delta_logits
        return current_logits, reference_logits.detach()

    def _build_result(
        self,
        candidates_8,
        current_logits,
        reference_logits,
        reward_payload=None,
    ):
        selected_indices = current_logits.argmax(dim=-1)
        gather_index = selected_indices[:, None, None, None].expand(
            -1, 1, self._num_poses, 3
        )
        selected_8 = torch.gather(candidates_8, 1, gather_index).squeeze(1)
        result = {
            "trajectory": self._expand_to_40(selected_8),
            "trajectory_8": selected_8,
            "selected_indices": selected_indices,
            "selector_logits": current_logits,
            "reference_selector_logits": reference_logits,
            "candidate_trajectories_8": candidates_8.detach(),
            "candidate_trajectories": self._all_candidates_to_40(
                candidates_8
            ).detach(),
        }
        if self._active_sample_tokens is not None:
            result["sample_tokens"] = tuple(
                str(token) for token in self._active_sample_tokens
            )
        if reward_payload is not None:
            result["candidate_rewards"] = reward_payload["score"]
            result["candidate_reward_components"] = reward_payload["components"]
            result["candidate_reward_valid_mask"] = reward_payload["valid_mask"]
        return result

    def _forward_train(self, bev_feature, ego_query, agents_query, status_token):
        candidates_8, candidate_feature = self._generate_frozen_candidates(
            bev_feature, ego_query, agents_query, status_token
        )
        current_logits, reference_logits = self._selector_outputs(
            candidate_feature,
            candidates_8,
            bev_feature,
            ego_query,
            agents_query,
            status_token,
        )
        reward_payload = None
        if self.online_reward is not None:
            if self._active_sample_tokens is None:
                raise RuntimeError(
                    "online DiffusionDrive PDM reward requires sample_tokens"
                )
            reward_payload = self.online_reward(
                candidates_8, self._active_sample_tokens
            )
        return self._build_result(
            candidates_8,
            current_logits,
            reference_logits,
            reward_payload=reward_payload,
        )

    def _forward_test(self, bev_feature, ego_query, agents_query, status_token):
        candidates_8, candidate_feature = self._generate_frozen_candidates(
            bev_feature, ego_query, agents_query, status_token
        )
        current_logits, reference_logits = self._selector_outputs(
            candidate_feature,
            candidates_8,
            bev_feature,
            ego_query,
            agents_query,
            status_token,
        )
        return self._build_result(
            candidates_8, current_logits, reference_logits
        )

    @torch.no_grad()
    def selector_diagnostics_per_sample(self, result):
        """Return paired deployment diagnostics without batch averaging."""
        current_logits = result["selector_logits"].float()
        reference_logits = result["reference_selector_logits"].float()
        rewards, valid = self._extract_candidate_rewards(
            result, None, current_logits
        )
        if "candidate_reward_components" not in result:
            raise KeyError("candidate reward components are required")
        components = result["candidate_reward_components"].float()
        if components.shape[:2] != rewards.shape:
            raise ValueError("candidate reward component alignment drifted")

        current_masked = (current_logits / self.policy_temperature).masked_fill(
            ~valid, -1e4
        )
        reference_masked = (
            reference_logits / self.policy_temperature
        ).masked_fill(~valid, -1e4)
        current_probability = F.softmax(current_masked, dim=-1)
        reference_probability = F.softmax(reference_masked, dim=-1)
        current_indices = current_masked.argmax(dim=-1)
        reference_indices = reference_masked.argmax(dim=-1)
        oracle_indices = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)

        def gather_2d(values, indices):
            return values.gather(1, indices[:, None]).squeeze(1)

        def gather_components(indices):
            gather_index = indices[:, None, None].expand(
                -1, 1, components.shape[-1]
            )
            return components.gather(1, gather_index).squeeze(1)

        safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
        current_reward = gather_2d(rewards, current_indices)
        reference_reward = gather_2d(rewards, reference_indices)
        oracle_reward = gather_2d(rewards, oracle_indices)
        return {
            "valid_candidate_count": valid.sum(dim=-1),
            "current_index": current_indices,
            "reference_index": reference_indices,
            "oracle_index": oracle_indices,
            "current_reward": current_reward,
            "reference_reward": reference_reward,
            "oracle_reward": oracle_reward,
            "top1_reward_gain": current_reward - reference_reward,
            "current_expected_reward": (
                current_probability * safe_rewards
            ).sum(dim=-1),
            "reference_expected_reward": (
                reference_probability * safe_rewards
            ).sum(dim=-1),
            "current_oracle_match": current_indices.eq(oracle_indices),
            "reference_oracle_match": reference_indices.eq(oracle_indices),
            "selection_disagreement": current_indices.ne(reference_indices),
            "current_components": gather_components(current_indices),
            "reference_components": gather_components(reference_indices),
            "oracle_components": gather_components(oracle_indices),
            "candidate_trajectories_8": result[
                "candidate_trajectories_8"
            ],
        }

    def _extract_candidate_rewards(self, result, gt_pdm_score, selector_logits):
        valid_mask = result.get("candidate_reward_valid_mask")
        rewards = result.get("candidate_rewards")
        if rewards is None and isinstance(gt_pdm_score, dict):
            fallback = gt_pdm_score.get(self.reward_key)
            if fallback is not None:
                rewards = fallback
        if rewards is None:
            raise KeyError(
                "selector GRPO requires online candidate rewards aligned with "
                "the 20 trajectories generated in this forward pass; got None. "
                "For real-candidate training/parity the model must be in train mode"
            )
        # Formal training supplies result.candidate_rewards from this forward.
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=selector_logits.device)
        rewards = rewards.to(device=selector_logits.device, dtype=torch.float32)
        while rewards.ndim > 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)
        if rewards.ndim == 1 and selector_logits.shape[0] == 1:
            rewards = rewards.unsqueeze(0)
        if rewards.shape != selector_logits.shape:
            raise ValueError(
                "candidate rewards must align with dynamic trajectories: "
                f"expected {tuple(selector_logits.shape)}, got {tuple(rewards.shape)}; "
                "the NAVSIM 8192-vocabulary cache is invalid here"
            )
        if valid_mask is None:
            valid_mask = torch.isfinite(rewards)
        else:
            valid_mask = valid_mask.to(device=selector_logits.device, dtype=torch.bool)
            if valid_mask.shape != rewards.shape:
                raise ValueError("candidate reward valid mask has the wrong shape")
            valid_mask = valid_mask & torch.isfinite(rewards)
        return rewards.detach(), valid_mask.detach()

    def _group_relative_advantages(self, rewards, valid):
        count = valid.sum(dim=-1)
        safe_count = count.clamp_min(1).to(rewards.dtype)
        safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
        reward_mean = safe_rewards.sum(dim=-1) / safe_count
        centered = torch.where(
            valid, rewards - reward_mean[:, None], torch.zeros_like(rewards)
        )
        reward_var = centered.square().sum(dim=-1) / safe_count
        reward_std = reward_var.sqrt()
        active_group = (count >= 2) & (reward_std > self.minimum_reward_std)
        policy_mask = valid & active_group[:, None]
        advantages = centered / reward_std.clamp_min(
            self.advantage_epsilon
        )[:, None]
        advantages = torch.where(
            policy_mask, advantages, torch.zeros_like(advantages)
        )
        return (
            advantages.detach(),
            policy_mask,
            active_group,
            reward_mean,
            reward_std,
        )

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
        """Selector-only GRPO with pi_old == pi_ref and no imitation loss."""
        current_logits = result["selector_logits"].float()
        reference_logits = result["reference_selector_logits"].float().detach()
        if current_logits.shape != reference_logits.shape:
            raise ValueError("current/reference selector logits are not aligned")
        rewards, valid = self._extract_candidate_rewards(
            result, gt_pdm_score, current_logits
        )
        (
            advantages,
            policy_mask,
            active_group,
            reward_mean,
            reward_std,
        ) = self._group_relative_advantages(rewards, valid)

        current_masked = (current_logits / self.policy_temperature).masked_fill(
            ~valid, -1e4
        )
        reference_masked = (
            reference_logits / self.policy_temperature
        ).masked_fill(~valid, -1e4)
        current_log_prob = F.log_softmax(current_masked, dim=-1)
        reference_log_prob = F.log_softmax(reference_masked, dim=-1)
        log_ratio = current_log_prob - reference_log_prob
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        clipped_ratio = ratio.clamp(
            1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        current_probability = current_log_prob.exp()
        reference_probability = reference_log_prob.exp().detach()
        policy_weight = (
            reference_probability * policy_mask.to(current_logits.dtype)
        )
        policy_weight = policy_weight / policy_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.advantage_epsilon)
        if self.policy_objective == "clipped_reference_grpo":
            surrogate = torch.minimum(
                ratio * advantages, clipped_ratio * advantages
            )
            # Legacy compatibility: evaluate the clipped surrogate exactly under
            # pi_old == pi_ref over the complete dynamic action set.
            policy_objective_per_group = (policy_weight * surrogate).sum(dim=-1)
        elif self.policy_objective == "exact_group_grpo":
            # All 20 candidates are available, so the importance-sampling
            # expectation is exact:
            #   sum_a pi_old(a) * pi(a)/pi_old(a) * A(a)
            # == sum_a pi(a) * A(a).
            # Avoiding the fixed clip lets probability cross a sharp frozen
            # reference ranking while preserving the same rewards/actions.
            policy_objective_per_group = (
                current_probability
                * advantages
                * policy_mask.to(current_logits.dtype)
            ).sum(dim=-1)
        else:  # guarded in __init__; keep failure local after deserialization.
            raise RuntimeError(
                f"unsupported policy objective: {self.policy_objective}"
            )
        if active_group.any():
            policy_loss = -policy_objective_per_group[active_group].mean()
        else:
            policy_loss = current_logits.sum() * 0.0
        kl_per_group = (
            current_probability
            * (current_log_prob - reference_log_prob)
            * valid.to(current_logits.dtype)
        ).sum(dim=-1)
        policy_tv_per_group = 0.5 * (
            (current_probability - reference_probability).abs()
            * valid.to(current_logits.dtype)
        ).sum(dim=-1)
        if active_group.any():
            kl = kl_per_group[active_group].mean()
            policy_tv = policy_tv_per_group[active_group].mean()
            ratio_mean = (
                policy_weight * ratio
            ).sum(dim=-1)[active_group].mean()
            clip_fraction = (
                policy_weight
                * ((ratio - 1.0).abs() > self.clip_epsilon).to(
                    current_logits.dtype
                )
            ).sum(dim=-1)[active_group].mean()
            advantage_weight = (
                policy_weight
                if self.policy_objective == "clipped_reference_grpo"
                else current_probability
            )
            advantage_abs_mean = (
                advantage_weight * advantages.abs()
            ).sum(dim=-1)[active_group].mean()

            safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
            current_expected_reward = (
                current_probability * safe_rewards
            ).sum(dim=-1)[active_group].mean()
            reference_expected_reward = (
                reference_probability * safe_rewards
            ).sum(dim=-1)[active_group].mean()
            expected_reward_gain = (
                current_expected_reward - reference_expected_reward
            )
            current_entropy = -(
                current_probability
                * current_log_prob
                * valid.to(current_logits.dtype)
            ).sum(dim=-1)[active_group].mean()
            reference_entropy = -(
                reference_probability
                * reference_log_prob
                * valid.to(current_logits.dtype)
            ).sum(dim=-1)[active_group].mean()
            current_top2_probability = current_probability.masked_fill(
                ~valid, -torch.inf
            ).topk(k=2, dim=-1).values
            reference_top2_probability = reference_probability.masked_fill(
                ~valid, -torch.inf
            ).topk(k=2, dim=-1).values
            current_top1_margin = (
                current_top2_probability[:, 0] - current_top2_probability[:, 1]
            )[active_group].mean()
            reference_top1_margin = (
                reference_top2_probability[:, 0]
                - reference_top2_probability[:, 1]
            )[active_group].mean()

            current_indices = current_masked.argmax(dim=-1)
            reference_indices = reference_masked.argmax(dim=-1)
            oracle_indices = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)
            current_reward = rewards.gather(
                1, current_indices[:, None]
            ).squeeze(1)[active_group].mean()
            reference_reward = rewards.gather(
                1, reference_indices[:, None]
            ).squeeze(1)[active_group].mean()
            oracle_reward = rewards.gather(
                1, oracle_indices[:, None]
            ).squeeze(1)[active_group].mean()
            oracle_match = (
                current_indices[active_group] == oracle_indices[active_group]
            ).float().mean()
            reference_oracle_match = (
                reference_indices[active_group] == oracle_indices[active_group]
            ).float().mean()
            selection_disagreement = (
                current_indices[active_group] != reference_indices[active_group]
            ).float().mean()
            reward_gain = current_reward - reference_reward
            current_oracle_probability = current_probability.gather(
                1, oracle_indices[:, None]
            ).squeeze(1)[active_group].mean()
            reference_oracle_probability = reference_probability.gather(
                1, oracle_indices[:, None]
            ).squeeze(1)[active_group].mean()
            oracle_probability_gain = (
                current_oracle_probability - reference_oracle_probability
            )
        else:
            zero = current_logits.sum() * 0.0
            kl = zero
            policy_tv = zero
            ratio_mean = zero
            clip_fraction = zero
            advantage_abs_mean = zero
            current_expected_reward = zero
            reference_expected_reward = zero
            expected_reward_gain = zero
            current_entropy = zero
            reference_entropy = zero
            current_top1_margin = zero
            reference_top1_margin = zero
            current_reward = zero
            reference_reward = zero
            oracle_reward = zero
            oracle_match = zero
            reference_oracle_match = zero
            selection_disagreement = zero
            reward_gain = zero
            current_oracle_probability = zero
            reference_oracle_probability = zero
            oracle_probability_gain = zero

        return {
            "loss.grpo_selector_policy": self.policy_loss_weight * policy_loss,
            "loss.grpo_selector_kl": self.kl_weight * kl,
            "grpo.selector.reward_mean": reward_mean.mean(),
            "grpo.selector.reward_std": reward_std.mean(),
            "grpo.selector.current_expected_reward": current_expected_reward,
            "grpo.selector.reference_expected_reward": reference_expected_reward,
            "grpo.selector.expected_reward_gain": expected_reward_gain,
            "grpo.selector.current_entropy": current_entropy,
            "grpo.selector.reference_entropy": reference_entropy,
            "grpo.selector.current_top1_margin": current_top1_margin,
            "grpo.selector.reference_top1_margin": reference_top1_margin,
            "grpo.selector.current_reward": current_reward,
            "grpo.selector.reference_reward": reference_reward,
            "grpo.selector.oracle_reward": oracle_reward,
            "grpo.selector.oracle_match": oracle_match,
            "grpo.selector.reference_oracle_match": reference_oracle_match,
            "grpo.selector.selection_disagreement": selection_disagreement,
            "grpo.selector.reward_gain": reward_gain,
            "grpo.selector.current_oracle_probability": current_oracle_probability,
            "grpo.selector.reference_oracle_probability": reference_oracle_probability,
            "grpo.selector.oracle_probability_gain": oracle_probability_gain,
            "grpo.selector.policy_tv": policy_tv,
            "grpo.selector.ratio_mean": ratio_mean,
            "grpo.selector.clip_fraction": clip_fraction,
            "grpo.selector.fixed_clip_applied": current_logits.new_tensor(
                float(self.policy_objective == "clipped_reference_grpo")
            ),
            "grpo.selector.policy_temperature": current_logits.new_tensor(
                self.policy_temperature
            ),
            "grpo.selector.kl": kl,
            "grpo.selector.active_group_fraction": active_group.float().mean(),
            "grpo.selector.advantage_abs_mean": advantage_abs_mean,
        }
