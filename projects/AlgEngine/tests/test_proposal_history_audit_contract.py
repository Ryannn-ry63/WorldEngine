from pathlib import Path
import hashlib
import importlib.util
import pickle
import sys

import numpy as np
import pytest
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "audit_proposal_history_information",
    SCRIPT_ROOT / "audit_proposal_history_information.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def info(token, previous, following, frame):
    return {
        "token": token,
        "sample_prev": previous,
        "sample_next": following,
        "log_name": "log-a",
        "scene_token": "scene-a",
        "timestamp": frame * 500_000,
        "frame_idx": frame,
        "ego2global": np.eye(4, dtype=np.float32),
    }


def test_annotation_reader_hashes_complete_pickle_and_extracts_only_targets(tmp_path):
    path = tmp_path / "annotations.pkl"
    payload = {
        "infos": [
            info("a", "", "b", 0),
            info("b", "a", "", 1),
            info("unused", "", "", 0),
        ]
    }
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    metadata, resolved, actual_sha = MODULE.load_annotation_metadata(
        path, {"a", "b"}, expected_sha
    )

    assert resolved == path.resolve()
    assert actual_sha == expected_sha
    assert set(metadata) == {"a", "b"}
    assert MODULE.immediate_predecessor_pairs(["a", "b"], metadata) == [(1, 0)]


def test_history_probabilities_do_not_consume_reward_validity(monkeypatch):
    logits = torch.tensor([[[3.0, 1.0]]])
    cache = {
        "candidate_reward_valid_mask": torch.tensor([[False, True]]),
        "candidate_features": torch.zeros(1, 2, 32),
        "candidate_trajectories_8": torch.zeros(1, 2, 8, 3),
        "route_bev_features": torch.zeros(1, 2, 8, 32),
    }
    captured = {}

    monkeypatch.setattr(
        MODULE,
        "current_candidate_features",
        lambda cache, indices, logits, probability: captured.setdefault(
            "probability", probability[indices].clone()
        ).unsqueeze(-1),
    )
    monkeypatch.setattr(
        MODULE.history,
        "transform_trajectories_between_ego_frames",
        lambda trajectories, source, target: trajectories,
    )
    monkeypatch.setattr(
        MODULE.history,
        "proposal_history_candidate_features",
        lambda current, previous, probability: torch.zeros(
            current.shape[0], current.shape[1], 1
        ),
    )
    poses = torch.eye(4).unsqueeze(0)

    MODULE.build_feature_banks(
        [cache],
        torch.tensor([0]),
        torch.tensor([0]),
        poses,
        poses,
        [logits.squeeze(0)],
        1,
    )

    expected = torch.softmax(logits.squeeze(0), dim=-1)
    assert torch.allclose(captured["probability"], expected)
    assert captured["probability"][0, 0] > captured["probability"][0, 1]


def test_incumbent_probe_locks_v3_topk_before_using_reward():
    cache = {
        "candidate_rewards": torch.tensor(
            [[0.5, 0.8, 0.3, 0.9], [0.6, 0.2, 0.7, 0.1]]
        ),
        "candidate_reward_valid_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    base = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    real_history = base + 10.0
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])

    examples = MODULE.build_incumbent_probe_examples(
        [cache],
        [base],
        [real_history],
        [logits],
        torch.tensor([1, 0]),
        torch.tensor([0, 1]),
        ["token-a", "token-b"],
        ["log-a", "log-b"],
        ["common", "rare"],
        3,
    )

    # Candidate 3 has the highest first reward but is excluded by frozen V3
    # top-3.  The four examples are challengers 1/2 against incumbent 0.
    assert examples["target"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert examples["current"].shape == (4, 2)
    assert examples["real"].shape == (4, 2)
    assert examples["logs"] == ["log-a", "log-a", "log-b", "log-b"]
    assert examples["strata"] == ["common", "common", "rare", "rare"]
