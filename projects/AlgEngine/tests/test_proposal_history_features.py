from pathlib import Path
import importlib.util

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/proposal_history_features.py"
)
SPEC = importlib.util.spec_from_file_location("proposal_history_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pose(x=0.0, y=0.0, yaw=0.0):
    result = torch.eye(4)
    cosine, sine = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    result[:2, :2] = torch.tensor([[cosine, -sine], [sine, cosine]])
    result[0, 3] = x
    result[1, 3] = y
    return result


def test_ego_transform_handles_translation_and_rotation():
    trajectories = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    transformed = MODULE.transform_trajectories_between_ego_frames(
        trajectories,
        pose(x=1.0, y=0.0, yaw=torch.pi / 2)[None],
        pose()[None],
    )
    assert torch.allclose(transformed[0, 0, 0, :2], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert transformed[0, 0, 0, 2].item() == pytest.approx(torch.pi / 2, abs=1e-6)


def test_history_features_recover_same_lineage_and_selected_trajectory():
    current = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[7.0, 0.0, 0.0]]]]
    )
    previous = current.clone()
    probability = torch.tensor([[0.1, 0.8, 0.1]])
    features = MODULE.proposal_history_candidate_features(
        current, previous, probability
    )
    assert torch.equal(features[0, :, 0], torch.zeros(3))
    assert torch.equal(features[0, :, 1], torch.zeros(3))
    assert torch.equal(features[0, :, 8], probability[0])
    assert torch.equal(features[0, :, 9], torch.tensor([0.0, 1.0, 0.0]))
    assert features[0, 1, 2].item() == pytest.approx(0.0)


def test_shuffle_is_within_stratum_cross_log_and_a_derangement():
    strata = ["rare"] * 4 + ["common"] * 4
    logs = [f"log-{index}" for index in range(4)] * 2
    tokens = [f"token-{index}" for index in range(8)]
    mapping = MODULE.same_stratum_cross_log_derangement(
        strata, logs, tokens, seed=17
    )
    for source, target in enumerate(mapping.tolist()):
        assert source != target
        assert strata[source] == strata[target]
        assert logs[source] != logs[target]
    assert mapping.tolist() == MODULE.same_stratum_cross_log_derangement(
        strata, logs, tokens, seed=17
    ).tolist()


def test_feature_module_has_no_outcome_argument_or_field_name():
    source = SCRIPT.read_text()
    forbidden = "candidate_reward_" + "components"
    assert forbidden not in source
    assert "rewards" not in source
