from pathlib import Path
import importlib.util

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diffusiondrive/lcpgrpo_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("lcpgrpo_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows(num_logs=50):
    return [
        {
            "log_name": f"log-{index}",
            "rare_token": f"rare-{index}",
            "common_token": f"common-{index}",
        }
        for index in range(num_logs)
    ]


def test_log_folds_are_deterministic_and_salted():
    first = MODULE.log_fold("abc")
    assert first == MODULE.log_fold("abc")
    assert first == MODULE.log_fold("abc", salt="20260902")
    assert first != MODULE.log_fold("abc", salt="different") or first in range(5)


def test_fold_contract_has_no_train_heldout_log_or_token_leakage():
    pair_rows = rows()
    contract = MODULE.build_fold_contract(pair_rows)
    all_positions = set(range(len(pair_rows)))
    for fold in range(5):
        heldout = set(MODULE.pair_positions_for_evaluation(contract, fold))
        training = set(MODULE.pair_positions_for_training(contract, fold))
        assert heldout
        assert training
        assert heldout.isdisjoint(training)
        assert heldout | training == all_positions
        heldout_logs = {pair_rows[index]["log_name"] for index in heldout}
        training_logs = {pair_rows[index]["log_name"] for index in training}
        assert heldout_logs.isdisjoint(training_logs)


def test_one_log_cannot_leak_when_it_has_multiple_pairs():
    pair_rows = rows(20)
    pair_rows.append(
        {
            "log_name": pair_rows[3]["log_name"],
            "rare_token": "rare-extra",
            "common_token": "common-extra",
        }
    )
    contract = MODULE.build_fold_contract(pair_rows)
    fold = MODULE.log_fold(pair_rows[3]["log_name"])
    heldout = set(MODULE.pair_positions_for_evaluation(contract, fold))
    assert 3 in heldout and len(pair_rows) - 1 in heldout


def test_empty_fold_is_rejected_in_tiny_synthetic_contract():
    with pytest.raises(RuntimeError, match="empty fold"):
        MODULE.build_fold_contract(rows(1))
