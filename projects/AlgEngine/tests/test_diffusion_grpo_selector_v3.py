import numpy as np
import pytest
import torch

from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import (
    DiffusionGRPOOnlineSelectorPlanningHead,
)
from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_scene_selector import (
    SceneConditionedTrajectorySetSelector,
    candidate_trajectory_geometry,
    sample_final_trajectory_bev_features,
)


def head_kwargs(tmp_path):
    anchor_path = tmp_path / "v3_anchors.npy"
    np.save(anchor_path, np.zeros((4, 8, 2), dtype=np.float32))
    return dict(
        num_poses=8,
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_bounding_boxes=2,
        num_query_decoder_layers=1,
        query_keyval_size=2,
        num_anchors=4,
        num_diff_decoder_layers=2,
        plan_anchor_path=str(anchor_path),
        bev_h=4,
        bev_w=4,
        num_train_timesteps=1000,
        train_timestep_max=5,
        inference_steps=2,
        trunc_timesteps=2,
        use_nerf=False,
    )


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


def selector_inputs(batch=2, candidates=5, dim=16):
    return dict(
        candidate_features=torch.randn(batch, candidates, dim),
        candidate_trajectories=torch.randn(batch, candidates, 8, 3),
        route_bev_features=torch.randn(batch, candidates, 8, dim),
        status_token=torch.randn(batch, 1, dim),
        ego_query=torch.randn(batch, 1, dim),
        agents_query=torch.randn(batch, 3, dim),
    )


def test_v3_zero_residual_preserves_reference_and_has_gradient():
    module = selector()
    inputs = selector_inputs()
    delta = module(**inputs)
    assert delta.shape == (2, 5)
    assert torch.equal(delta, torch.zeros_like(delta))
    (-delta[:, 0].mean()).backward()
    assert module.delta_head[-1].weight.grad is not None
    assert module.delta_head[-1].weight.grad.abs().sum() > 0


def test_v3_selector_is_candidate_permutation_equivariant():
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    inputs = selector_inputs(batch=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = module(**inputs)
    permuted_inputs = dict(inputs)
    for key in ("candidate_features", "candidate_trajectories", "route_bev_features"):
        permuted_inputs[key] = inputs[key][:, permutation]
    permuted = module(**permuted_inputs)
    assert torch.allclose(permuted, original[:, permutation], atol=1e-5, rtol=1e-5)


def test_v3_geometry_and_route_sampling_use_final_trajectories():
    trajectories = torch.zeros(1, 2, 8, 3)
    trajectories[:, 1, :, 0] = 1.0
    geometry = candidate_trajectory_geometry(trajectories)
    assert geometry.shape == (1, 2, 58)
    assert not torch.equal(geometry[:, 0], geometry[:, 1])

    bev = torch.randn(1, 16, 10, 10, requires_grad=True)
    sampled = sample_final_trajectory_bev_features(bev, trajectories, 51.2, 51.2)
    assert sampled.shape == (1, 2, 8, 16)
    assert not sampled.requires_grad


def test_v3_head_opens_only_scene_selector(tmp_path):
    head = DiffusionGRPOOnlineSelectorPlanningHead(
        **head_kwargs(tmp_path),
        online_reward=None,
        policy_objective="exact_group_grpo",
        scene_selector=dict(
            feature_dim=32,
            model_dim=32,
            route_bev_dim=32,
            context_dim=32,
            geometry_hidden_dim=16,
            num_heads=4,
            feedforward_dim=64,
            num_set_layers=1,
        ),
    )
    names = head.trainable_parameter_names
    assert names
    assert all(name.startswith("scene_selector.") for name in names)
    assert not any("plan_cls_branch" in name for name in names)
    assert not any(parameter.requires_grad for parameter in head.reference_selector.parameters())


def test_v3_selector_rejects_reward_as_an_input():
    module = selector()
    inputs = selector_inputs()
    with pytest.raises(TypeError):
        module(**inputs, candidate_rewards=torch.randn(2, 5))
