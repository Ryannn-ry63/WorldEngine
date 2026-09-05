import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
)
COMMON_PATH = ROOT / "scripts/diffusiondrive/grpo_selector_v3_cached_common.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load("pcra_selector", MODEL_PATH)
COMMON = load("pcra_common", COMMON_PATH)


def selector(**overrides):
    config = dict(
        architecture="proposal_conditioned_regret_arbitration",
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
        arbiter_hidden_dim=8,
        use_decision_context=False,
        override_threshold=0.0,
    )
    config.update(overrides)
    return MODEL.build_scene_selector(config)


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


def test_zero_arbiter_preserves_incumbent_with_nonzero_frozen_proposal():
    torch.manual_seed(13)
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    values = inputs()
    mask = torch.ones(2, 5, dtype=torch.bool)
    residual, diagnostics = module(
        **values, candidate_mask=mask, return_override_diagnostics=True
    )
    assert diagnostics["proposal_delta"].abs().max() > 0
    assert torch.equal(
        diagnostics["proposal_evidence"],
        torch.zeros_like(diagnostics["proposal_evidence"]),
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    assert not diagnostics["override_mask"].any()


def test_arbitration_is_strict_one_sided_and_ignores_unchanged_proposals():
    proposal_delta = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    evidence = torch.tensor([1.0, 0.0, 1.0])
    incumbent = torch.tensor([0, 0, 1])
    proposal = torch.tensor([1, 1, 1])
    residual, override = (
        MODEL.ProposalConditionedRegretArbitrator.apply_regret_arbitration(
            proposal_delta, evidence, incumbent, proposal, 0.0
        )
    )
    assert override.tolist() == [True, False, False]
    assert torch.equal(residual[0], proposal_delta[0])
    assert torch.equal(residual[1:], torch.zeros_like(residual[1:]))


def test_candidate_permutation_equivariance_with_mask_and_context():
    torch.manual_seed(21)
    module = selector(use_decision_context=True, override_threshold=-100.0).eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
        module.regret_arbiter_head[-1].weight.normal_()
    values = inputs(batch=1)
    mask = torch.tensor([[True, True, False, True, True]])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original, original_diagnostics = module(
        **values, candidate_mask=mask, return_override_diagnostics=True
    )
    permuted = dict(values)
    for key in (
        "candidate_features",
        "candidate_trajectories",
        "route_bev_features",
        "reference_logits",
    ):
        permuted[key] = values[key][:, permutation]
    updated, updated_diagnostics = module(
        **permuted,
        candidate_mask=mask[:, permutation],
        return_override_diagnostics=True,
    )
    assert torch.allclose(updated, original[:, permutation], atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        updated_diagnostics["proposal_evidence"],
        original_diagnostics["proposal_evidence"],
        atol=3e-5,
        rtol=3e-5,
    )
    assert original_diagnostics["proposal_index"].item() != 2


def test_regret_loss_pushes_actual_pair_in_correct_direction_and_scales_by_gain():
    evidence = torch.zeros(2, requires_grad=True)
    incumbent = torch.tensor([0, 0])
    proposal = torch.tensor([1, 1])
    rewards = torch.tensor([[0.2, 0.8], [0.7, 0.2]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, diagnostics = COMMON.proposal_conditioned_arbitration_loss(
        evidence, incumbent, proposal, rewards, valid, "regret"
    )
    loss.backward()
    assert diagnostics["gain"].tolist() == pytest.approx([0.6, -0.5])
    assert evidence.grad[0] < 0.0
    assert evidence.grad[1] > 0.0
    assert abs(evidence.grad[0] / evidence.grad[1]) == pytest.approx(1.2)

    sign_evidence = torch.zeros(2, requires_grad=True)
    sign_loss, _ = COMMON.proposal_conditioned_arbitration_loss(
        sign_evidence, incumbent, proposal, rewards, valid, "sign"
    )
    sign_loss.backward()
    assert abs(sign_evidence.grad[0]) == pytest.approx(abs(sign_evidence.grad[1]))


def test_ties_and_unchanged_proposals_have_exactly_zero_gradient():
    evidence = torch.randn(2, requires_grad=True)
    incumbent = torch.tensor([0, 0])
    proposal = torch.tensor([0, 1])
    rewards = torch.tensor([[0.2, 0.9], [0.4, 0.4]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, diagnostics = COMMON.proposal_conditioned_arbitration_loss(
        evidence, incumbent, proposal, rewards, valid, "regret"
    )
    loss.backward()
    assert loss == 0.0
    assert not diagnostics["active"].any()
    assert torch.equal(evidence.grad, torch.zeros_like(evidence.grad))


def test_invalid_selected_candidate_is_rejected_and_rewards_are_not_model_inputs():
    evidence = torch.zeros(1)
    incumbent = torch.tensor([0])
    proposal = torch.tensor([1])
    rewards = torch.tensor([[0.2, 0.8]])
    valid = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="invalid candidate"):
        COMMON.proposal_conditioned_arbitration_loss(
            evidence, incumbent, proposal, rewards, valid, "regret"
        )

    module = selector(use_decision_context=True)
    rebuilt = MODEL.build_scene_selector(module.config_dict())
    assert rebuilt.config_dict() == module.config_dict()
    values = inputs()
    with pytest.raises(TypeError):
        module(
            **values,
            candidate_mask=torch.ones(2, 5, dtype=torch.bool),
            candidate_rewards=torch.rand(2, 5),
        )
    with pytest.raises(TypeError):
        module(
            **values,
            candidate_mask=torch.ones(2, 5, dtype=torch.bool),
            source_labels=["common", "hard_real_rare"],
        )
