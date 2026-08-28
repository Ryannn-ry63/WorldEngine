import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
SPEC = importlib.util.spec_from_file_location("trajectory_set_reasoner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def selector(**overrides):
    config = dict(
        architecture="trajectory_set_reasoner",
        feature_dim=16,
        model_dim=16,
        route_bev_dim=16,
        context_dim=16,
        geometry_hidden_dim=8,
        relation_hidden_dim=8,
        num_heads=4,
        feedforward_dim=32,
        num_temporal_layers=1,
        num_relation_layers=1,
    )
    config.update(overrides)
    return MODULE.build_scene_selector(config)


def inputs(batch=2, candidates=5, dim=16):
    return dict(
        candidate_features=torch.randn(batch, candidates, dim),
        candidate_trajectories=torch.randn(batch, candidates, 8, 3),
        route_bev_features=torch.randn(batch, candidates, 8, dim),
        status_token=torch.randn(batch, 1, dim),
        ego_query=torch.randn(batch, 1, dim),
        agents_query=torch.randn(batch, 3, dim),
    )


def test_step_and_pairwise_geometry_shapes_and_diagonal():
    trajectories = torch.randn(2, 5, 8, 3)
    step = MODULE.trajectory_step_geometry(trajectories)
    pairwise = MODULE.pairwise_trajectory_relations(trajectories)
    assert step.shape == (2, 5, 8, 11)
    assert pairwise.shape == (2, 5, 5, 41)
    diagonal = pairwise[:, torch.arange(5), torch.arange(5)]
    assert torch.allclose(diagonal[..., :16], torch.zeros_like(diagonal[..., :16]))
    assert torch.allclose(diagonal[..., -9:], torch.zeros_like(diagonal[..., -9:]))


def test_zero_residual_preserves_reference_and_trains_output_head():
    module = selector()
    delta = module(**inputs())
    assert delta.shape == (2, 5)
    assert torch.equal(delta, torch.zeros_like(delta))
    (-delta[:, 0].mean()).backward()
    assert module.delta_head[-1].weight.grad.abs().sum() > 0


def test_candidate_permutation_equivariance():
    torch.manual_seed(7)
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
    assert torch.allclose(permuted, original[:, permutation], atol=2e-5, rtol=2e-5)


def test_pairwise_reasoning_changes_other_candidate_score():
    torch.manual_seed(11)
    module = selector(use_temporal_reasoning=False).eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    values = inputs(batch=1)
    original = module(**values)
    changed = dict(values)
    changed["candidate_trajectories"] = values["candidate_trajectories"].clone()
    changed["candidate_trajectories"][:, 1, :, 0] += 20.0
    updated = module(**changed)
    assert not torch.allclose(original[:, 0], updated[:, 0])


def test_no_reward_input_and_frozen_input_boundary():
    module = selector()
    values = inputs()
    values["candidate_features"].requires_grad_(True)
    delta = module(**values)
    delta.sum().backward()
    assert values["candidate_features"].grad is None
    with pytest.raises(TypeError):
        module(**inputs(), candidate_rewards=torch.randn(2, 5))


def test_config_round_trip_and_legacy_factory_default():
    module = selector(num_temporal_layers=2, num_relation_layers=2)
    config = module.config_dict()
    rebuilt = MODULE.build_scene_selector(config)
    assert rebuilt.config_dict() == config
    legacy = MODULE.build_scene_selector(
        dict(
            feature_dim=16,
            model_dim=16,
            route_bev_dim=16,
            context_dim=16,
            geometry_hidden_dim=8,
            num_heads=4,
            feedforward_dim=32,
            num_set_layers=1,
        )
    )
    assert isinstance(legacy, MODULE.SceneConditionedTrajectorySetSelector)
