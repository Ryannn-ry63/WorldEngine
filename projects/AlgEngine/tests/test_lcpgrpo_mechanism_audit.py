from pathlib import Path
import importlib.util
import sys

import pytest
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "audit_lineage_consistent_mechanism",
    SCRIPT_ROOT / "audit_lineage_consistent_mechanism.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_heldout_prediction_names_both_top4_directions_and_handles_ties():
    # Candidate 1 wins the other-draw lineage estimate.  On draw 0 it ties
    # candidate 0 for the oracle reward, but torch.argmax chooses index 0.
    # A scientific top-1 quality statistic must not call that a failure merely
    # because of the arbitrary first-argmax tie break.
    draw_rewards = (
        [1.0, 1.0, 0.3, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.2, 0.1],
    )
    caches = []
    for rewards in draw_rewards:
        tensor = torch.tensor([rewards], dtype=torch.float32)
        caches.append(
            {
                "candidate_rewards": tensor,
                "candidate_reward_valid_mask": torch.ones_like(
                    tensor, dtype=torch.bool
                ),
            }
        )

    result = MODULE.heldout_lineage_prediction(caches, [0])

    assert result["predicted_top1_is_first_heldout_argmax_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert result["predicted_top1_is_tie_aware_heldout_top1_fraction"] == 1.0
    assert result["predicted_top1_is_in_heldout_top4_fraction"] == 1.0
    assert result["first_heldout_argmax_is_in_predicted_top4_fraction"] == 1.0
    assert result["mean_oracle_gap"] == 0.0


def test_evaluator_allows_reused_common_tokens_but_rejects_stratum_overlap():
    source = (SCRIPT_ROOT / "evaluate_lineage_consistent_proximal_grpo.py").read_text()
    assert "selected_common_set = set(selected_common)" in source
    assert "rare/common evaluation strata overlap" in source
    assert "evaluation strata overlap or repeat tokens" not in source



def test_formal_mechanism_gate_locks_unambiguous_schema_v2_metric():
    audit = (SCRIPT_ROOT / "audit_lineage_consistent_mechanism.py").read_text()
    trainer = (
        SCRIPT_ROOT / "train_lineage_consistent_proximal_grpo.py"
    ).read_text()
    evaluator = (
        SCRIPT_ROOT / "evaluate_lineage_consistent_proximal_grpo.py"
    ).read_text()
    assert '"schema_version": 2' in audit
    for source in (audit, trainer, evaluator):
        assert "predicted_top1_is_in_heldout_top4_fraction" in source
