import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
SPEC = importlib.util.spec_from_file_location("ivps_selector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def selector(**overrides):
    config = dict(
        architecture="incumbent_verified_preference",
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
        verifier_hidden_dim=8,
        override_threshold=0.0,
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
        reference_logits=torch.randn(batch, candidates),
    )


def test_zero_verifier_preserves_reference_even_with_nonzero_proposal():
    torch.manual_seed(4)
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    residual, diagnostics = module(**inputs(), return_override_diagnostics=True)
    assert diagnostics["proposal_delta"].abs().max() > 0
    assert torch.equal(
        diagnostics["verification_logits"],
        torch.zeros_like(diagnostics["verification_logits"]),
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    assert not diagnostics["override_mask"].any()


def test_verified_override_is_one_sided_and_strict():
    reference = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    proposal_delta = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    verification = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.ones_like(reference, dtype=torch.bool)
    residual, incumbent, proposal, override = MODULE.IncumbentVerifiedPreferenceSelector.apply_verified_override(
        proposal_delta, verification, reference, mask, 0.0
    )
    assert incumbent.tolist() == [0, 0]
    assert proposal.tolist() == [1, 1]
    assert override.tolist() == [True, False]
    assert torch.equal(residual[0], proposal_delta[0])
    assert torch.equal(residual[1], torch.zeros_like(residual[1]))


def test_candidate_permutation_equivariance():
    torch.manual_seed(8)
    module = selector(override_threshold=-100.0).eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
        module.override_verifier_head[-1].weight.normal_()
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


def test_config_round_trip_and_no_reward_input():
    module = selector()
    rebuilt = MODULE.build_scene_selector(module.config_dict())
    assert rebuilt.config_dict() == module.config_dict()
    with pytest.raises(TypeError):
        module(**inputs(), candidate_rewards=torch.rand(2, 5))
