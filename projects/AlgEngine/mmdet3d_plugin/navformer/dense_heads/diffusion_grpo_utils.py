"""Probability and trajectory helpers for DiffusionDrive generation GRPO."""

import math
from typing import Optional, Tuple

import torch


def diagonal_gaussian_log_prob(value, mean, std):
    """Return one log-probability per batch/candidate."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if torch.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    log_prob = -0.5 * (
        ((value - mean) / std).square()
        + 2.0 * std.log()
        + math.log(2.0 * math.pi)
    )
    return log_prob.flatten(start_dim=2).sum(dim=-1)


def diagonal_gaussian_kl_same_std(mean, reference_mean, std):
    """Dimension-normalized exact KL for diagonal Gaussians with equal std."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if torch.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    return (0.5 * ((mean - reference_mean) / std).square()).flatten(
        start_dim=2
    ).mean(dim=-1)


def ddim_transition_with_log_prob(
    scheduler,
    model_output,
    timestep,
    sample,
    eta,
    action: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    sigma_min=1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample or replay one stochastic DDIM transition."""
    if eta <= 0:
        raise ValueError("eta must be positive for stochastic GRPO")
    if scheduler.num_inference_steps is None:
        raise RuntimeError("scheduler.set_timesteps must be called before GRPO")
    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = int(timestep) - stride
    variance = scheduler._get_variance(int(timestep), prev_timestep)
    std = (eta * variance.sqrt()).clamp_min(sigma_min).to(
        device=model_output.device, dtype=model_output.dtype
    )
    mean = scheduler.step(
        model_output=model_output,
        timestep=int(timestep),
        sample=sample,
        eta=eta,
        variance_noise=torch.zeros_like(model_output),
    ).prev_sample
    if action is None:
        action = mean + std * (
            torch.randn_like(mean) if noise is None else noise
        )
    return action, diagonal_gaussian_log_prob(action, mean, std), mean, std


def group_relative_advantages(rewards, valid_mask, eps=1e-3):
    if rewards.ndim != 2 or valid_mask.shape != rewards.shape:
        raise ValueError("rewards and valid_mask must have shape [batch, mode]")
    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    safe = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    count = valid_mask.sum(-1).clamp_min(1)
    mean = safe.sum(-1) / count
    centered = torch.where(
        valid_mask, rewards - mean[:, None], torch.zeros_like(rewards)
    )
    std = (centered.square().sum(-1) / count).sqrt()
    group_valid = (valid_mask.sum(-1) >= 2) & (std > eps)
    advantage = centered / std.clamp_min(eps)[:, None]
    advantage = torch.where(
        valid_mask & group_valid[:, None], advantage, torch.zeros_like(advantage)
    )
    return advantage, valid_mask, group_valid, mean, std


def generation_grpo_objective(
    current_log_probs,
    old_log_probs,
    generation_kl,
    rewards,
    valid_mask,
    clip_ratio=0.2,
    advantage_eps=1e-3,
    selected_indices=None,
):
    """Clipped group-relative objective over stored denoising actions."""
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log-probs must have identical shapes")
    if current_log_probs.ndim != 3 or current_log_probs.shape[:2] != rewards.shape:
        raise ValueError("generation log-probs must have shape [batch, mode, step]")
    if generation_kl.shape != current_log_probs.shape:
        raise ValueError("generation KL must match generation log-prob shape")
    advantage, valid_mask, group_valid, reward_mean, reward_std = (
        group_relative_advantages(rewards, valid_mask, advantage_eps)
    )
    log_ratio = (current_log_probs - old_log_probs.detach()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantage.unsqueeze(-1),
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
        * advantage.unsqueeze(-1),
    )
    optimize_mask = (
        valid_mask & group_valid[:, None]
    ).unsqueeze(-1).expand_as(surrogate)
    policy_loss = (
        -surrogate[optimize_mask].mean()
        if optimize_mask.any()
        else current_log_probs.sum() * 0.0
    )
    kl_mask = valid_mask.unsqueeze(-1).expand_as(generation_kl)
    kl_loss = (
        generation_kl[kl_mask].mean()
        if kl_mask.any()
        else generation_kl.sum() * 0.0
    )
    clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
    if selected_indices is None:
        selected_indices = torch.zeros(
            rewards.shape[0], device=rewards.device, dtype=torch.long
        )
    selected_indices = selected_indices.reshape(-1).long()
    selected = rewards.gather(1, selected_indices[:, None]).squeeze(1)
    oracle = torch.where(
        valid_mask, rewards, rewards.new_full((), -torch.inf)
    ).max(-1).values
    finite_oracle = torch.isfinite(oracle)
    def safe_metric(value):
        metric_mask = finite_oracle & torch.isfinite(value)
        return (
            value[metric_mask].mean().detach()
            if metric_mask.any()
            else rewards.new_zeros(())
        )
    ratio_mean = (
        ratio[kl_mask].mean().detach() if kl_mask.any() else rewards.new_zeros(())
    )
    clip_fraction = (
        clipped[kl_mask].float().mean().detach()
        if kl_mask.any() else rewards.new_zeros(())
    )

    return {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "reward_mean": reward_mean.mean().detach(),
        "reward_std": reward_std.mean().detach(),
        "selected_reward": safe_metric(selected),
        "oracle_reward": safe_metric(oracle),
        "selection_regret": safe_metric(oracle - selected),
        "ratio_mean": ratio_mean,
        "clip_fraction": clip_fraction,
        "valid_group_fraction": group_valid.float().mean().detach(),
    }


def interpolate_trajectory_8_to_40(trajectory):
    """Interpolate 2 Hz rear-axle poses to the 10 Hz PDM proposal grid."""
    if trajectory.shape[-2:] != (8, 3):
        raise ValueError("trajectory must end in [8, 3]")
    origin = torch.zeros_like(trajectory[..., :1, :])
    start_xy = torch.cat((origin[..., :1, :2], trajectory[..., :-1, :2]), -2)
    end_xy = trajectory[..., :, :2]
    fraction = torch.arange(
        1, 6, device=trajectory.device, dtype=trajectory.dtype
    ) / 5.0
    fraction_xy = fraction.view(
        *([1] * (trajectory.ndim - 1)), 5, 1
    )
    xy = start_xy.unsqueeze(-2) + (
        end_xy - start_xy
    ).unsqueeze(-2) * fraction_xy
    raw_heading = trajectory[..., :, 2]
    with_origin = torch.cat(
        (torch.zeros_like(raw_heading[..., :1]), raw_heading), -1
    )
    delta = with_origin[..., 1:] - with_origin[..., :-1]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    end_heading = torch.cumsum(delta, -1)
    start_heading = torch.cat(
        (torch.zeros_like(end_heading[..., :1]), end_heading[..., :-1]), -1
    )
    heading = start_heading.unsqueeze(-1) + (
        end_heading - start_heading
    ).unsqueeze(-1) * fraction
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    return torch.cat((xy, heading.unsqueeze(-1)), -1).reshape(
        *trajectory.shape[:-2], 40, 3
    )
