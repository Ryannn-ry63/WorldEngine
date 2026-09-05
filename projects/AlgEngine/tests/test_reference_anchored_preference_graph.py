import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
SPEC = importlib.util.spec_from_file_location("rapg_selector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def selector(architecture="reference_anchored_preference_graph", **overrides):
    config = dict(
        architecture=architecture,
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
        use_temporal_reasoning=False,
        use_relational_reasoning=True,
        pairwise_hidden_dim=8,
    )
    if architecture == "capacity_matched_unary":
        config.pop("pairwise_hidden_dim")
        config["capacity_hidden_dim"] = 17
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
        reference_logits=torch.randn(batch, candidates),
    )


def test_zero_initialization_exactly_preserves_reference_policy():
    module = selector().eval()
    values = inputs()
    residual, diagnostics = module(**values, return_pair_diagnostics=True)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(
        diagnostics["pairwise_preferences"],
        torch.zeros_like(diagnostics["pairwise_preferences"]),
    )
    assert torch.equal(
        (values["reference_logits"] + residual).argmax(dim=-1),
        values["reference_logits"].argmax(dim=-1),
    )
    (-residual[:, 0].mean()).backward()
    assert module.delta_head[-1].weight.grad.abs().sum() > 0
    assert module.pairwise_preference_head[-1].weight.grad.abs().sum() > 0


def test_pair_preferences_are_antisymmetric_diagonal_zero_and_masked():
    torch.manual_seed(3)
    module = selector().eval()
    with torch.no_grad():
        module.pairwise_preference_head[-1].weight.normal_()
    values = inputs(batch=1)
    mask = torch.tensor([[True, True, False, True, True]])
    _, diagnostics = module(
        **values, candidate_mask=mask, return_pair_diagnostics=True
    )
    pair = diagnostics["pairwise_preferences"]
    assert torch.allclose(pair, -pair.transpose(1, 2), atol=1e-7, rtol=0.0)
    assert torch.equal(pair.diagonal(dim1=1, dim2=2), torch.zeros(1, 5))
    assert torch.equal(pair[:, 2], torch.zeros_like(pair[:, 2]))
    assert torch.equal(pair[:, :, 2], torch.zeros_like(pair[:, :, 2]))


def test_candidate_permutation_equivariance_including_incumbent():
    torch.manual_seed(7)
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
        module.pairwise_preference_head[-1].weight.normal_()
    values = inputs(batch=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = module(**values)
    permuted = dict(values)
    for key in (
        "candidate_features",
        "candidate_trajectories",
        "route_bev_features",
        "reference_logits",
    ):
        permuted[key] = values[key][:, permutation]
    updated = module(**permuted)
    assert torch.allclose(updated, original[:, permutation], atol=3e-5, rtol=3e-5)


def test_reference_logits_only_change_the_incumbent_aggregation():
    torch.manual_seed(11)
    module = selector().eval()
    with torch.no_grad():
        module.pairwise_preference_head[-1].weight.normal_()
    values = inputs(batch=1)
    first = dict(values)
    second = dict(values)
    first["reference_logits"] = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
    second["reference_logits"] = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])
    residual_first = module(**first)
    residual_second = module(**second)
    assert not torch.allclose(residual_first, residual_second)


def test_no_reward_input_and_frozen_observation_boundary():
    module = selector()
    values = inputs()
    values["candidate_features"].requires_grad_(True)
    module(**values).sum().backward()
    assert values["candidate_features"].grad is None
    with pytest.raises(TypeError):
        module(**inputs(), candidate_rewards=torch.randn(2, 5))


def test_config_round_trip_and_capacity_control_zero():
    module = selector(num_relation_layers=2)
    rebuilt = MODULE.build_scene_selector(module.config_dict())
    assert rebuilt.config_dict() == module.config_dict()
    control = selector(architecture="capacity_matched_unary")
    values = inputs()
    values.pop("reference_logits")
    residual = control(**values)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_full_scale_capacity_control_matches_rapg_extra_parameters_within_one_percent():
    shared = dict(
        feature_dim=256,
        model_dim=256,
        route_bev_dim=256,
        context_dim=256,
        num_heads=4,
        feedforward_dim=512,
        num_temporal_layers=2,
        num_relation_layers=2,
        use_temporal_reasoning=False,
        use_relational_reasoning=True,
    )
    baseline = MODULE.build_scene_selector(
        dict(architecture="trajectory_set_reasoner", **shared)
    )
    rapg = MODULE.build_scene_selector(
        dict(architecture="reference_anchored_preference_graph", **shared)
    )
    control = MODULE.build_scene_selector(
        dict(architecture="capacity_matched_unary", **shared)
    )
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    rapg_extra = count(rapg) - count(baseline)
    control_extra = count(control) - count(baseline)
    assert abs(rapg_extra - control_extra) / rapg_extra < 0.01
