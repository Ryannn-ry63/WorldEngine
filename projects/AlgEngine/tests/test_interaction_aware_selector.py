import importlib.util
import math
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
SPEC = importlib.util.spec_from_file_location("interaction_selector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def selector(architecture="interaction_generic"):
    return MODULE.build_scene_selector(
        dict(
            architecture=architecture,
            feature_dim=16,
            model_dim=16,
            route_bev_dim=16,
            context_dim=16,
            geometry_hidden_dim=8,
            num_heads=4,
            feedforward_dim=32,
            num_route_steps=8,
            num_set_layers=1,
            interaction_hidden_dim=8,
            num_agent_classes=10,
            max_agents=4,
            num_interaction_heads=4,
        )
    )


def inputs(batch=2, candidates=5, agents=4, dim=16):
    states = torch.zeros(batch, agents, 8)
    states[..., 0:2] = torch.randn(batch, agents, 2) * 5.0
    states[..., 2:4] = torch.rand(batch, agents, 2) + 1.0
    states[..., 4] = torch.randn(batch, agents)
    states[..., 5:7] = torch.randn(batch, agents, 2)
    states[..., 7] = torch.rand(batch, agents) * 0.6 + 0.4
    return dict(
        candidate_features=torch.randn(batch, candidates, dim),
        candidate_trajectories=torch.randn(batch, candidates, 8, 3),
        route_bev_features=torch.randn(batch, candidates, 8, dim),
        status_token=torch.randn(batch, 1, dim),
        ego_query=torch.randn(batch, 1, dim),
        agents_query=torch.randn(batch, 3, dim),
        frozen_track_states=states,
        frozen_track_classes=torch.randint(0, 10, (batch, agents)),
        frozen_track_mask=torch.ones(batch, agents, dtype=torch.bool),
    )


def test_constant_velocity_rollout_uses_half_second_waypoints():
    states = torch.zeros(1, 1, 8)
    states[..., 0] = 2.0
    states[..., 1] = -1.0
    states[..., 5] = 4.0
    states[..., 6] = 2.0
    rollout = MODULE.constant_velocity_track_rollout(states)
    assert rollout.shape == (1, 1, 8, 2)
    assert torch.allclose(rollout[0, 0, 0], torch.tensor([4.0, 0.0]))
    assert torch.allclose(rollout[0, 0, -1], torch.tensor([18.0, 7.0]))


def test_candidate_frame_rotation_is_correct():
    trajectories = torch.zeros(1, 1, 8, 3)
    trajectories[..., 2] = math.pi / 2
    states = torch.zeros(1, 1, 8)
    states[..., 0] = 1.0
    states[..., 2:4] = 1.0
    states[..., 7] = 0.9
    features, mask = MODULE.candidate_frame_agent_interactions(
        trajectories, states, torch.ones(1, 1, dtype=torch.bool)
    )
    assert mask.all()
    assert torch.allclose(features[0, 0, 0, 0, :2], torch.tensor([0.0, -1.0]), atol=1e-5)


@pytest.mark.parametrize("architecture", ["interaction_generic", "interaction_relation"])
def test_zero_initialized_residual_and_no_agent_case_are_finite(architecture):
    module = selector(architecture)
    values = inputs()
    values["frozen_track_mask"].zero_()
    values["frozen_track_classes"].fill_(-1)
    delta = module(**values)
    assert torch.isfinite(delta).all()
    assert torch.equal(delta, torch.zeros_like(delta))
    (-delta[:, 0].mean()).backward()
    assert module.delta_head[-1].weight.grad.abs().sum() > 0


def test_agent_permutation_and_padding_invariance():
    torch.manual_seed(13)
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    values = inputs(batch=1)
    values["frozen_track_mask"][0, -1] = False
    values["frozen_track_classes"][0, -1] = -1
    original = module(**values)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(values)
    for key in ("frozen_track_states", "frozen_track_classes", "frozen_track_mask"):
        changed[key] = values[key][:, permutation]
    permuted = module(**changed)
    assert torch.allclose(original, permuted, atol=2e-5, rtol=2e-5)
    changed["frozen_track_states"] = changed["frozen_track_states"].clone()
    changed["frozen_track_states"][~changed["frozen_track_mask"]] = 1e6
    assert torch.allclose(original, module(**changed), atol=2e-5, rtol=2e-5)


def test_candidate_permutation_equivariance_for_both_arms():
    permutation = torch.tensor([3, 0, 4, 1, 2])
    for architecture in ("interaction_generic", "interaction_relation"):
        torch.manual_seed(17)
        module = selector(architecture).eval()
        with torch.no_grad():
            module.delta_head[-1].weight.normal_()
        values = inputs(batch=1)
        original = module(**values)
        changed = dict(values)
        for key in ("candidate_features", "candidate_trajectories", "route_bev_features"):
            changed[key] = values[key][:, permutation]
        assert torch.allclose(module(**changed), original[:, permutation], atol=3e-5, rtol=3e-5)


def test_frozen_inputs_are_detached_and_rewards_are_rejected():
    module = selector()
    values = inputs()
    values["candidate_features"].requires_grad_(True)
    values["frozen_track_states"].requires_grad_(True)
    module(**values).sum().backward()
    assert values["candidate_features"].grad is None
    assert values["frozen_track_states"].grad is None
    with pytest.raises(TypeError):
        module(**inputs(), candidate_rewards=torch.randn(2, 5))


def test_config_round_trip_preserves_causal_arm_identity():
    for architecture in ("interaction_generic", "interaction_relation"):
        module = selector(architecture)
        rebuilt = MODULE.build_scene_selector(module.config_dict())
        assert rebuilt.config_dict() == module.config_dict()
        assert rebuilt.requires_frozen_track_states

