import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/diffusiondrive")
)
import grpo_selector_v3_cached_common as common


def synthetic_cache(count=3):
    return {
        "scene_selector_config": {
            "feature_dim": 256,
            "model_dim": 256,
            "route_bev_dim": 256,
            "context_dim": 256,
            "geometry_dim": 58,
            "geometry_hidden_dim": 128,
            "num_heads": 4,
            "feedforward_dim": 512,
            "num_route_steps": 8,
            "num_set_layers": 1,
            "use_set_attention": True,
            "use_trajectory_geometry": True,
            "use_route_bev": True,
            "use_scene_context": True,
        },
        "candidate_features": torch.randn(count, 20, 256).half(),
        "candidate_trajectories_8": torch.randn(count, 20, 8, 3),
        "route_bev_features": torch.randn(count, 20, 8, 256).half(),
        "status_tokens": torch.randn(count, 1, 256).half(),
        "ego_queries": torch.randn(count, 1, 256).half(),
        "agents_queries": torch.randn(count, 30, 256).half(),
        "reference_logits": torch.randn(count, 20),
        "candidate_rewards": torch.rand(count, 20),
        "candidate_reward_valid_mask": torch.ones(count, 20, dtype=torch.bool),
    }


def test_cached_v3_forward_and_exact_group_backward():
    cache = synthetic_cache()
    model, _ = common.model_from_cache(cache)
    logits, reference = common.current_logits(model, cache, slice(None), "cpu")
    assert torch.equal(logits, reference)
    loss, policy, kl = common.exact_group_loss(
        logits,
        reference,
        cache["candidate_rewards"],
        cache["candidate_reward_valid_mask"],
        temperature=1.0,
        kl_weight=1e-3,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(policy)
    assert torch.isfinite(kl)
    loss.backward()
    assert model.delta_head[-1].weight.grad.abs().sum() > 0


def test_cpu_global_grad_clip_preserves_norm_semantics():
    parameter_a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter_b = torch.nn.Parameter(torch.tensor([0.0, 12.0]))
    parameter_a.grad = parameter_a.detach().clone()
    parameter_b.grad = parameter_b.detach().clone()

    observed = common.clip_grad_norm_cpu_([parameter_a, parameter_b], 6.5)

    assert abs(observed - 13.0) < 1e-6
    clipped = torch.cat([parameter_a.grad, parameter_b.grad])
    assert abs(float(torch.linalg.vector_norm(clipped)) - 6.5) < 1e-5


def test_cpu_global_grad_clip_rejects_nonfinite_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite V3 gradient"):
        common.clip_grad_norm_cpu_([parameter], 10.0)


def test_cached_v3_ablation_configs_are_explicit():
    cache = synthetic_cache(count=1)
    _, feature_only = common.model_from_cache(cache, "feature_only")
    _, full = common.model_from_cache(cache, "full")
    assert not feature_only["use_trajectory_geometry"]
    assert not feature_only["use_route_bev"]
    assert not feature_only["use_scene_context"]
    assert full["use_trajectory_geometry"]
    assert full["use_route_bev"]
    assert full["use_scene_context"]
