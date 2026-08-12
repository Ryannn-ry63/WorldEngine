import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
SPEC = importlib.util.spec_from_file_location("diffusion_grpo_scene_selector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SceneConditionedTrajectorySetSelector = MODULE.SceneConditionedTrajectorySetSelector
candidate_trajectory_geometry = MODULE.candidate_trajectory_geometry
sample_final_trajectory_bev_features = MODULE.sample_final_trajectory_bev_features


def selector(feature_dim=16):
    return SceneConditionedTrajectorySetSelector(
        feature_dim=feature_dim,
        model_dim=16,
        route_bev_dim=16,
        context_dim=16,
        geometry_hidden_dim=8,
        num_heads=4,
        feedforward_dim=32,
        num_set_layers=1,
    )


def inputs(batch=2, candidates=5, dim=16):
    return dict(
        candidate_features=torch.randn(batch, candidates, dim),
        candidate_trajectories=torch.randn(batch, candidates, 8, 3),
        route_bev_features=torch.randn(batch, candidates, 8, dim),
        status_token=torch.randn(batch, 1, dim),
        ego_query=torch.randn(batch, 1, dim),
        agents_query=torch.randn(batch, 3, dim),
    )


def test_zero_residual_preserves_reference_and_has_gradient():
    module = selector()
    delta = module(**inputs())
    assert delta.shape == (2, 5)
    assert torch.equal(delta, torch.zeros_like(delta))
    (-delta[:, 0].mean()).backward()
    assert module.delta_head[-1].weight.grad.abs().sum() > 0


def test_candidate_permutation_equivariance():
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    values = inputs(batch=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = module(**values)
    permuted_values = dict(values)
    for key in ("candidate_features", "candidate_trajectories", "route_bev_features"):
        permuted_values[key] = values[key][:, permutation]
    permuted = module(**permuted_values)
    assert torch.allclose(permuted, original[:, permutation], atol=1e-5, rtol=1e-5)


def test_geometry_and_route_sampling_use_final_trajectories():
    trajectories = torch.zeros(1, 2, 8, 3)
    trajectories[:, 1, :, 0] = 1.0
    geometry = candidate_trajectory_geometry(trajectories)
    assert geometry.shape == (1, 2, 58)
    assert not torch.equal(geometry[:, 0], geometry[:, 1])
    bev = torch.randn(1, 16, 10, 10, requires_grad=True)
    sampled = sample_final_trajectory_bev_features(bev, trajectories, 51.2, 51.2)
    assert sampled.shape == (1, 2, 8, 16)
    assert not sampled.requires_grad


def test_reward_cannot_be_passed_as_selector_input():
    with pytest.raises(TypeError):
        selector()(**inputs(), candidate_rewards=torch.randn(2, 5))


def test_selector_config_round_trips_all_architecture_parameters():
    module = SceneConditionedTrajectorySetSelector(
        feature_dim=16,
        model_dim=24,
        route_bev_dim=12,
        context_dim=20,
        geometry_hidden_dim=10,
        num_heads=3,
        feedforward_dim=48,
        num_route_steps=8,
        num_set_layers=2,
    )
    config = module.config_dict()
    rebuilt = SceneConditionedTrajectorySetSelector(**config)
    assert config["geometry_hidden_dim"] == 10
    assert config["num_heads"] == 3
    assert config["feedforward_dim"] == 48
    assert config["num_set_layers"] == 2
    assert rebuilt.config_dict() == config
