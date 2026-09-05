import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
COMMON_PATH = ROOT / "scripts/diffusiondrive/grpo_selector_v3_cached_common.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load("pcra_v2_selector", MODEL_PATH)
COMMON = load("pcra_v2_common", COMMON_PATH)


def selector(**overrides):
    config = dict(
        architecture="proposal_conditioned_counterfactual_evaluator",
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
        counterfactual_hidden_dim=8,
        counterfactual_loss="opportunity_risk",
        train_evaluator_encoder=True,
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


def test_zero_evidence_preserves_incumbent_with_nonzero_frozen_proposal():
    torch.manual_seed(3)
    module = selector().eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
    assert module.initialize_evaluator_from_proposal() == 0.0
    values = inputs()
    residual, diagnostics = module(
        **values,
        candidate_mask=torch.ones(2, 5, dtype=torch.bool),
        return_override_diagnostics=True,
    )
    assert diagnostics["proposal_delta"].abs().max() > 0
    assert torch.equal(
        diagnostics["proposal_evidence"],
        torch.zeros_like(diagnostics["proposal_evidence"]),
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    assert not diagnostics["override_mask"].any()
    assert (diagnostics["opportunity"] >= 0).all()
    assert (diagnostics["risk"] >= 0).all()


def test_directed_evidence_is_antisymmetric():
    torch.manual_seed(7)
    module = selector().eval()
    with torch.no_grad():
        module.counterfactual_value_head[-1].weight.normal_()
    batch, candidates = 3, 5
    tokens = torch.randn(batch, candidates, 16)
    trajectories = torch.randn(batch, candidates, 8, 3)
    relations = MODEL.pairwise_trajectory_relations(trajectories)
    first = torch.tensor([0, 1, 2])
    second = torch.tensor([3, 4, 0])
    reference = torch.randn(batch, candidates)
    proposal = torch.randn(batch, candidates)
    forward = module.counterfactual_value_head(
        module._directed_inputs(
            tokens, relations, first, second, reference, proposal
        )
    ).squeeze(-1)
    reverse = module.counterfactual_value_head(
        module._directed_inputs(
            tokens, relations, second, first, reference, proposal
        )
    ).squeeze(-1)
    evidence = F.softplus(forward) - F.softplus(reverse)
    swapped = F.softplus(reverse) - F.softplus(forward)
    assert torch.equal(swapped, -evidence)


def test_opportunity_risk_loss_trains_positive_negative_and_tie_pairs():
    opportunity = torch.full((3,), 0.4, requires_grad=True)
    risk = torch.full((3,), 0.4, requires_grad=True)
    evidence = opportunity - risk
    incumbent = torch.tensor([0, 0, 0])
    proposal = torch.tensor([1, 1, 1])
    rewards = torch.tensor([[0.2, 0.8], [0.8, 0.3], [0.5, 0.5]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, diagnostics = COMMON.proposal_conditioned_counterfactual_loss(
        opportunity,
        risk,
        evidence,
        incumbent,
        proposal,
        rewards,
        valid,
        "opportunity_risk",
    )
    loss.backward()
    assert diagnostics["active"].tolist() == [True, True, True]
    assert diagnostics["beneficial"].tolist() == [True, False, False]
    assert diagnostics["degrading"].tolist() == [False, True, False]
    assert diagnostics["ties"].tolist() == [False, False, True]
    assert opportunity.grad[0] < 0
    assert risk.grad[1] < 0
    assert opportunity.grad[2] > 0 and risk.grad[2] > 0


def test_equal_source_risk_averages_source_means_not_sample_counts():
    opportunity = torch.tensor([0.0, 0.0, 0.0, 0.0])
    risk = torch.tensor([0.0, 0.0, 0.0, 0.0])
    evidence = torch.tensor([0.0, 0.0, 0.0, 0.0])
    incumbent = torch.zeros(4, dtype=torch.long)
    proposal = torch.ones(4, dtype=torch.long)
    rewards = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 0.5], [1.0, 0.0]]
    )
    valid = torch.ones_like(rewards, dtype=torch.bool)
    labels = ["common", "common", "hard_real_rare", "hard_synthetic"]
    loss, diagnostics = COMMON.proposal_conditioned_counterfactual_loss(
        opportunity,
        risk,
        evidence,
        incumbent,
        proposal,
        rewards,
        valid,
        "signed_gain",
        source_labels=labels,
        source_risk="equal_strata",
    )
    expected = torch.tensor((1.0 + 0.25 + 1.0) / 3.0)
    assert torch.allclose(loss, expected)
    assert set(diagnostics["source_losses"]) == set(labels)


def test_permutation_equivariance_and_config_roundtrip():
    torch.manual_seed(11)
    module = selector(override_threshold=-100.0).eval()
    with torch.no_grad():
        module.delta_head[-1].weight.normal_()
        module.counterfactual_value_head[-1].weight.normal_()
    module.initialize_evaluator_from_proposal()
    values = inputs(batch=1)
    mask = torch.tensor([[True, True, False, True, True]])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original, original_diagnostics = module(
        **values, candidate_mask=mask, return_override_diagnostics=True
    )
    permuted = dict(values)
    for key in (
        "candidate_features", "candidate_trajectories",
        "route_bev_features", "reference_logits",
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
    rebuilt = MODEL.build_scene_selector(module.config_dict())
    assert rebuilt.config_dict() == module.config_dict()
    with pytest.raises(TypeError):
        module(
            **values,
            candidate_mask=torch.ones(1, 5, dtype=torch.bool),
            candidate_rewards=torch.rand(1, 5),
        )
