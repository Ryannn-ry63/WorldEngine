import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "diffusiondrive"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATERIALIZE = load_module(
    "materialize_grpo_selector_v2_replica",
    "materialize_grpo_selector_v2_replica.py",
)
FINALIZE = load_module(
    "finalize_grpo_selector_v2_formal",
    "finalize_grpo_selector_v2_formal.py",
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selector_payload(seed=1, epoch=32):
    return {
        "schema_version": 2,
        "method": "exact_group_grpo",
        "selector_state": {
            f"tensor_{index}": torch.tensor([float(index)])
            for index in range(10)
        },
        "temperature": 1.0,
        "learning_rate": 1e-3,
        "kl_weight": 1e-3,
        "train_seed": seed,
        "epoch": epoch,
    }


def test_replica_payload_requires_exact_formal_contract(tmp_path):
    state = tmp_path / "selector.pt"
    torch.save(selector_payload(), state)
    assert MATERIALIZE.validate_selector_payload(
        state, sha256_file(state), train_seed=1
    ) == sha256_file(state)

    torch.save(selector_payload(epoch=31), state)
    with pytest.raises(RuntimeError, match="epoch drifted"):
        MATERIALIZE.validate_selector_payload(
            state, sha256_file(state), train_seed=1
        )


def test_formal_hparams_are_seed_specific_and_fixed():
    payload = {**FINALIZE.FORMAL_HPARAMS, "train_seed": 2}
    FINALIZE.verify_hparams(payload, 2)
    payload["learning_rate"] = 3e-4
    with pytest.raises(RuntimeError, match="learning_rate drifted"):
        FINALIZE.verify_hparams(payload, 2)


def test_formal_flatten_preserves_all_four_pdm_blocks():
    pdm = {key: float(index) for index, key in enumerate(FINALIZE.PDM_KEYS)}
    metrics = {
        "openloop_navtest": {"ade": 0.1, "fde": 0.2, **pdm},
        "openloop_failures": pdm,
        "closedloop_nonreactive": pdm,
        "closedloop_reactive": pdm,
        "success_rate": 0.75,
    }
    flattened = FINALIZE.flatten(metrics)
    assert flattened["openloop_navtest.ade"] == 0.1
    assert flattened["closedloop_reactive.score"] == pdm["score"]
    assert flattened["success_rate"] == 0.75


def test_h100_seed_wrapper_is_three_seed_and_four_block():
    wrapper = (SCRIPT_DIR / "run_grpo_selector_v2_formal_seed_h100.sh").read_text()
    formal_table = (SCRIPT_DIR / "run_grpo_selector_formal_table_h100.sh").read_text()
    assert '[[ ! "${SEED}" =~ ^[012]$ ]]' in wrapper
    assert "run_grpo_selector_formal_table_h100.sh" in wrapper
    assert "DIFFUSIONDRIVE_GRPO_FORMAL_ROOT" in wrapper
    for stage in (
        "openloop_navtest",
        "openloop_failures",
        "closed_loop_nonreactive",
        "closed_loop_reactive",
    ):
        assert f"CURRENT_STAGE={stage}" in formal_table
