"""Generation-only GRPO planning head for WorldEngine DiffusionDrive."""

import copy
import os

import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS

from .diffusion_planning_head import DiffusionPlanningHead, gen_sineembed_for_position
from .diffusion_grpo_utils import (
    diagonal_gaussian_kl_same_std,
    diagonal_gaussian_log_prob,
    ddim_transition_with_log_prob,
    generation_grpo_objective,
    interpolate_trajectory_8_to_40,
)


@HEADS.register_module()
class DiffusionGRPOPlanningHead(DiffusionPlanningHead):
    """Replay stored stochastic diffusion actions and optimize PDM reward."""

    requires_grpo_context = True

    def __init__(
        self,
        reference_checkpoint=None,
        roll_timesteps=(8, 0),
        scheduler_num_inference_steps=125,
        generation_ddim_eta=1.0,
        generation_final_std=0.05,
        generation_sigma_min=1e-4,
        grpo_clip_ratio=0.2,
        grpo_advantage_eps=1e-3,
        generation_kl_weight=0.1,
        heading_min=-2.0,
        heading_range=3.9,
        freeze_non_diffusion=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reference_checkpoint = reference_checkpoint
        self.roll_timesteps = tuple(int(t) for t in roll_timesteps)
        self.scheduler_num_inference_steps = int(scheduler_num_inference_steps)
        self.generation_ddim_eta = float(generation_ddim_eta)
        self.generation_final_std = float(generation_final_std)
        self.generation_sigma_min = float(generation_sigma_min)
        self.grpo_clip_ratio = float(grpo_clip_ratio)
        self.grpo_advantage_eps = float(grpo_advantage_eps)
        self.generation_kl_weight = float(generation_kl_weight)
        self.heading_min = float(heading_min)
        self.heading_range = float(heading_range)
        self._validate_grpo_config()
        self.reference_decoder = copy.deepcopy(self.diff_decoder)
        if reference_checkpoint:
            self._load_reference_decoder(reference_checkpoint)
        self.reference_decoder.requires_grad_(False)
        self.reference_decoder.eval()
        if freeze_non_diffusion:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.startswith("diff_decoder.")

    def _validate_grpo_config(self):
        if len(self.roll_timesteps) != 2 or self.roll_timesteps[-1] != 0:
            raise ValueError("generation GRPO requires two timesteps ending at zero")
        stride = (
            self.diffusion_scheduler.config.num_train_timesteps
            // self.scheduler_num_inference_steps
        )
        if self.roll_timesteps[0] - stride != self.roll_timesteps[1]:
            raise ValueError("scheduler stride must connect the first timestep to zero")
        if self.generation_ddim_eta <= 0 or self.generation_final_std <= 0:
            raise ValueError("GRPO sampling standard deviations must be positive")

    def _load_reference_decoder(self, checkpoint_path):
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("GRPO reference checkpoint missing: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefixes = (
            "module.planning_head.diff_decoder.",
            "planning_head.diff_decoder.",
        )
        decoder_state = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    decoder_state[key[len(prefix):]] = value
                    break
        if not decoder_state:
            raise KeyError("checkpoint has no planning_head.diff_decoder weights")
        self.reference_decoder.load_state_dict(decoder_state, strict=True)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Keep the constructor-loaded reference fixed across policy rounds."""
        reference_prefix = prefix + "reference_decoder."
        for key in [
            key for key in state_dict if key.startswith(reference_prefix)
        ]:
            state_dict.pop(key)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def train(self, mode=True):
        super().train(mode)
        self.reference_decoder.eval()
        return self

    def norm_odo(self, trajectory):
        xy = self.norm_odo_xy(trajectory[..., :2])
        heading = 2 * (
            trajectory[..., 2:3] - self.heading_min
        ) / self.heading_range - 1
        return torch.cat((xy, heading), -1)

    def denorm_odo(self, trajectory):
        xy = self.denorm_odo_xy(trajectory[..., :2])
        heading = (
            (trajectory[..., 2:3] + 1) / 2 * self.heading_range
            + self.heading_min
        )
        return torch.cat((xy, heading), -1)

    def _decode(
        self, policy, sample, timestep, ego_query, agents_query, bev_feature, status_token
    ):
        batch_size = sample.shape[0]
        noisy_points = self.denorm_odo_xy(sample.clamp(-1, 1))
        position = gen_sineembed_for_position(noisy_points, hidden_dim=64).flatten(-2)
        feature = self.plan_anchor_encoder(position).view(
            batch_size, noisy_points.shape[1], -1
        )
        time_embedding = self.time_mlp(
            torch.full(
                (batch_size,), int(timestep), device=sample.device, dtype=torch.long
            )
        ).view(batch_size, 1, -1)
        regression, classification = policy(
            feature,
            noisy_points,
            bev_feature,
            (self.bev_h, self.bev_w),
            agents_query,
            ego_query,
            time_embedding,
            status_token,
            None,
        )
        return regression[-1], classification[-1]

    def _replay(
        self,
        policy,
        initial_sample,
        transition_action,
        final_action,
        ego_query,
        agents_query,
        bev_feature,
        status_token,
    ):
        first_t, final_t = self.roll_timesteps
        first_reg, _ = self._decode(
            policy, initial_sample, first_t, ego_query, agents_query, bev_feature, status_token
        )
        _, transition_log_prob, transition_mean, transition_std = (
            ddim_transition_with_log_prob(
                self.diffusion_scheduler,
                self.norm_odo_xy(first_reg[..., :2]),
                first_t,
                initial_sample,
                eta=self.generation_ddim_eta,
                action=transition_action,
                sigma_min=self.generation_sigma_min,
            )
        )
        final_reg, final_cls = self._decode(
            policy, transition_action, final_t, ego_query, agents_query, bev_feature, status_token
        )
        final_mean = self.norm_odo(final_reg)
        final_std = final_mean.new_tensor(self.generation_final_std)
        final_log_prob = diagonal_gaussian_log_prob(final_action, final_mean, final_std)
        return (
            torch.stack((transition_log_prob, final_log_prob), -1),
            transition_mean,
            transition_std,
            final_mean,
            final_std,
            final_cls,
        )

    def _format_output(self, final_action, reference_cls):
        trajectories_8 = self.denorm_odo(final_action)
        selected = reference_cls.argmax(-1)
        gather = selected[..., None, None, None].expand(
            -1, -1, self._num_poses, 3
        )
        selected_8 = torch.gather(trajectories_8, 1, gather).squeeze(1)
        return {
            "trajectory": interpolate_trajectory_8_to_40(selected_8),
            "trajectory_8": selected_8,
            "all_trajectories": interpolate_trajectory_8_to_40(trajectories_8),
            "all_trajectories_8": trajectories_8,
            "poses_cls": reference_cls,
            "selected_indices": selected,
        }

    def _collect_rollout(self, bev_feature, ego_query, agents_query, status_token):
        self.diffusion_scheduler.set_timesteps(
            self.scheduler_num_inference_steps, bev_feature.device
        )
        batch_size = ego_query.shape[0]
        anchor = self.plan_anchor.unsqueeze(0).expand(batch_size, -1, -1, -1)
        normalized_anchor = self.norm_odo_xy(anchor)
        initial_sample = self.diffusion_scheduler.add_noise(
            normalized_anchor,
            torch.randn_like(normalized_anchor),
            torch.full(
                (batch_size,),
                self.roll_timesteps[0],
                device=bev_feature.device,
                dtype=torch.long,
            ),
        )
        first_t, final_t = self.roll_timesteps
        first_reg, _ = self._decode(
            self.diff_decoder, initial_sample, first_t,
            ego_query, agents_query, bev_feature, status_token
        )
        transition_action, transition_log_prob, _, _ = (
            ddim_transition_with_log_prob(
                self.diffusion_scheduler,
                self.norm_odo_xy(first_reg[..., :2]),
                first_t,
                initial_sample,
                eta=self.generation_ddim_eta,
                sigma_min=self.generation_sigma_min,
            )
        )
        final_reg, _ = self._decode(
            self.diff_decoder, transition_action, final_t,
            ego_query, agents_query, bev_feature, status_token
        )
        final_mean = self.norm_odo(final_reg)
        final_std = final_mean.new_tensor(self.generation_final_std)
        final_action = final_mean + final_std * torch.randn_like(final_mean)
        final_log_prob = diagonal_gaussian_log_prob(final_action, final_mean, final_std)
        with torch.no_grad():
            reference_cls = self._replay(
                self.reference_decoder,
                initial_sample,
                transition_action,
                final_action,
                ego_query,
                agents_query,
                bev_feature,
                status_token,
            )[-1]
        output = self._format_output(final_action, reference_cls)
        output.update({
            "grpo_initial_sample": initial_sample,
            "grpo_transition_action": transition_action,
            "grpo_final_action": final_action,
            "grpo_old_log_probs": torch.stack(
                (transition_log_prob, final_log_prob), -1
            ),
        })
        return output

    def _forward_train_replay(
        self, bev_feature, ego_query, agents_query, status_token, grpo_data
    ):
        required = ("initial_sample", "transition_action", "final_action", "old_log_probs")
        missing = [key for key in required if grpo_data.get(key) is None]
        if missing:
            raise KeyError("missing GRPO replay tensors: " + ", ".join(missing))
        self.diffusion_scheduler.set_timesteps(
            self.scheduler_num_inference_steps, bev_feature.device
        )
        initial_sample = grpo_data["initial_sample"].float()
        transition_action = grpo_data["transition_action"].float()
        final_action = grpo_data["final_action"].float()
        current = self._replay(
            self.diff_decoder,
            initial_sample,
            transition_action,
            final_action,
            ego_query,
            agents_query,
            bev_feature,
            status_token,
        )
        with torch.no_grad():
            reference = self._replay(
                self.reference_decoder,
                initial_sample,
                transition_action,
                final_action,
                ego_query,
                agents_query,
                bev_feature,
                status_token,
            )
        current_logp, current_transition_mean, transition_std, current_final_mean, final_std, _ = current
        _, ref_transition_mean, _, ref_final_mean, _, reference_cls = reference
        generation_kl = torch.stack((
            diagonal_gaussian_kl_same_std(
                current_transition_mean, ref_transition_mean, transition_std
            ),
            diagonal_gaussian_kl_same_std(
                current_final_mean, ref_final_mean, final_std
            ),
        ), -1)
        output = self._format_output(final_action, reference_cls)
        output.update({
            "generation_current_log_probs": current_logp,
            "generation_old_log_probs": grpo_data["old_log_probs"].float(),
            "generation_kl": generation_kl,
        })
        return output

    @auto_fp16(apply_to=("bev_embed",))
    def forward(
        self,
        bev_embed,
        command=None,
        sdc_planning_past=None,
        sdc_status=None,
        sdc_planning_mask_past=None,
        gt_pre_command_sdc=None,
        grpo_data=None,
    ):
        if bev_embed.dim() != 3 or bev_embed.shape[0] != self.bev_h * self.bev_w:
            raise ValueError("unexpected bev_embed shape for GRPO head")
        _, batch_size, channels = bev_embed.shape
        bev_feature = bev_embed.permute(1, 2, 0).contiguous().view(
            batch_size, channels, self.bev_h, self.bev_w
        )
        status_token = self._build_status_token(
            command, sdc_planning_past, sdc_status,
            sdc_planning_mask_past, gt_pre_command_sdc
        )
        ego_query, agents_query = self._prepare_queries(bev_feature, status_token)
        if self.training:
            return self._forward_train_replay(
                bev_feature, ego_query, agents_query, status_token, grpo_data or {}
            )
        return self._collect_rollout(bev_feature, ego_query, agents_query, status_token)

    @force_fp32(apply_to=("result", "gt_pdm_score"))
    def loss(self, result=None, gt_pdm_score=None, **kwargs):
        rewards = gt_pdm_score.get("grpo_rewards")
        valid_mask = gt_pdm_score.get("grpo_valid_mask")
        if rewards is None or valid_mask is None:
            raise KeyError("GRPO training requires rewards and valid_mask")
        objective = generation_grpo_objective(
            result["generation_current_log_probs"],
            result["generation_old_log_probs"],
            result["generation_kl"],
            rewards.float(),
            valid_mask.bool(),
            clip_ratio=self.grpo_clip_ratio,
            advantage_eps=self.grpo_advantage_eps,
            selected_indices=gt_pdm_score.get("grpo_selected_index"),
        )
        return {
            "loss.grpo_policy": objective["policy_loss"],
            "loss.grpo_kl": objective["kl_loss"] * self.generation_kl_weight,
            "grpo.reward_mean": objective["reward_mean"],
            "grpo.reward_std": objective["reward_std"],
            "grpo.selected_reward": objective["selected_reward"],
            "grpo.oracle_reward": objective["oracle_reward"],
            "grpo.selection_regret": objective["selection_regret"],
            "grpo.ratio_mean": objective["ratio_mean"],
            "grpo.clip_fraction": objective["clip_fraction"],
            "grpo.valid_group_fraction": objective["valid_group_fraction"],
        }
