from pathlib import Path
import importlib
import json
import sys

import numpy as np
import pytest
import torch


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
)
sys.path.insert(0, str(SCRIPT_DIR))
trainer = importlib.import_module("train_diffusion_robust_group_grpo")
evaluator = importlib.import_module("evaluate_diffusion_robust_group_grpo")
mechanism = importlib.import_module("audit_diffusion_robust_group_mechanism")
gradient_audit = importlib.import_module("audit_diffusion_set_gradient_conflict")
development_gate = importlib.import_module(
    "select_diffusion_robust_group_development"
)


def test_arm_factorization_is_explicit_and_complete():
    repeat = trainer.arm_contract("repeat_grpo")
    mean_soft = trainer.arm_contract("mean_soft")
    mean_top1 = trainer.arm_contract("mean_top1")
    pure = trainer.arm_contract("pure_softmin_top1")
    bounded_soft = trainer.arm_contract("bounded_relative_soft")
    bounded_top1 = trainer.arm_contract("bounded_relative_top1")
    shuffled = trainer.arm_contract("bounded_relative_top1_shuffled")
    absolute = trainer.arm_contract("bounded_absolute_top1")
    assert repeat["objective"] == "original_per_draw_zscore_exact_group_grpo"
    assert mean_soft["aggregation"] == "mean"
    assert mean_soft["selection_mode"] == "soft"
    assert mean_top1["selection_mode"] == "straight_through_top1"
    assert pure["aggregation"] == "softmin"
    assert bounded_soft["aggregation"] == "bounded_mean_risk"
    assert bounded_soft["selection_mode"] == "soft"
    assert bounded_top1["aggregation"] == "bounded_mean_risk"
    assert bounded_top1["same_token_grouping"] is True
    assert shuffled["same_token_grouping"] is False
    assert absolute["objective_mode"] == "absolute_value"


def test_shuffled_control_preserves_stratum_and_marginal_indices():
    indices = torch.tensor([10, 11, 12, 20, 21, 22])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    generator = torch.Generator().manual_seed(7)
    shuffled = trainer.stratified_shuffle(indices, labels, generator)
    assert sorted(shuffled[labels.eq(0)].tolist()) == [10, 11, 12]
    assert sorted(shuffled[labels.eq(1)].tolist()) == [20, 21, 22]
    assert not bool(shuffled.eq(indices).any())


def test_grouping_control_recovers_true_outer_group_structure():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(200, 1))
    values = latent + 0.01 * rng.normal(size=(200, 3))
    report = mechanism.grouping_control(values, repetitions=200, seed=9)
    assert report["shuffled_minus_true"] > 0.5
    assert report["one_sided_permutation_p"] < 0.01


def test_log_bootstrap_uses_log_aggregates():
    gains = np.asarray([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    report = evaluator.bootstrap_log_gain(
        gains,
        logs=["long", "long", "short"],
        repetitions=200,
        seed=11,
    )
    assert report["num_logs"] == 2
    assert report["mean"] == 0.0
    assert report["bootstrap_unit"] == "log"


def test_gradient_cosine_detects_alignment_and_conflict():
    first = (torch.tensor([1.0, 2.0]), torch.tensor([-1.0]))
    aligned = (torch.tensor([2.0, 4.0]), torch.tensor([-2.0]))
    opposed = (torch.tensor([-2.0, -4.0]), torch.tensor([2.0]))
    assert gradient_audit.gradient_cosine(first, aligned) == pytest.approx(1.0)
    assert gradient_audit.gradient_cosine(first, opposed) == pytest.approx(-1.0)


def test_cache_contract_separates_scene_context_from_candidates():
    def cache(offset):
        return {
            "tokens": ["a", "b"],
            "status_tokens": torch.zeros(2, 1, 2),
            "ego_queries": torch.zeros(2, 1, 2),
            "agents_queries": torch.zeros(2, 2, 2),
            "candidate_reward_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "candidate_features": torch.zeros(2, 2, 2) + offset,
            "candidate_trajectories_8": torch.zeros(2, 2, 1, 3) + offset,
            "route_bev_features": torch.zeros(2, 2, 1, 2) + offset,
            "candidate_rewards": torch.zeros(2, 2) + offset,
            "reference_logits": torch.zeros(2, 2) + offset,
        }

    report = gradient_audit.tensor_contract((cache(0.0), cache(1.0), cache(2.0)))
    assert report["scene_context_exactly_invariant"] is True
    assert report["candidate_trajectories_change"] is True


def test_paired_log_bootstrap_equal_weights_common_and_rare(tmp_path):
    target_path = tmp_path / "target.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    target_rows = []
    reference_rows = []
    for log_name, common_gain, rare_gain in (
        ("log_a", 0.2, 0.0),
        ("log_b", 0.0, 0.2),
    ):
        for stratum, gain in (("common", common_gain), ("rare", rare_gain)):
            token = f"{log_name}_{stratum}"
            base = {
                "token": token,
                "scene": log_name,
                "log": log_name,
                "stratum": stratum,
                "noise_seed": 3,
                "anchor_reward": 0.5,
            }
            reference_rows.append({**base, "current_reward": 0.5})
            target_rows.append({**base, "current_reward": 0.5 + gain})
    target_path.write_text(
        "\n".join(json.dumps(row) for row in target_rows) + "\n"
    )
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in reference_rows) + "\n"
    )
    report = development_gate.paired_log_bootstrap(
        {"records_path": target_path},
        {"records_path": reference_path},
        "equal",
        repetitions=200,
        seed=17,
    )
    assert report["bootstrap_unit"] == "log"
    assert report["num_logs"] == 2
    assert report["mean"] == pytest.approx(0.1)
