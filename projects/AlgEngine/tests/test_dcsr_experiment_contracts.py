from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "projects/AlgEngine/scripts/diffusiondrive/probe_dcsr_top1_objective.py"
RUNNER = ROOT / "projects/AlgEngine/scripts/diffusiondrive/run_dcsr_top1_audit_1gpu.sh"
WRAPPER = ROOT / "run_diffusiondrive_selector_dcsr_audit_1gpu.sh"


def test_runner_reuses_locked_train_assets_only():
    text = RUNNER.read_text()
    assert text.count("--train-cache") == 3
    assert "train_seed0/cache.pt" in text
    assert "train_seed1/cache.pt" in text
    assert "train_seed2/cache.pt" in text
    assert "audit_20260901T141121Z/hard_pair_gate.json" in text
    assert "expected exactly 1 visible GPU" in text
    assert 'PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"' in text
    assert "--development-cache" not in text
    assert "--certification-cache" not in text
    assert "d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133" in text


def test_probe_locks_the_nonduplicate_dcsr_contract():
    text = PROBE.read_text()
    assert "TOPK = 5" in text
    assert '"reward_or_components_as_input": False' in text
    assert '"interaction_features": False' in text
    assert '"old_rollout_or_synthetic_data_consumed": False' in text
    assert "one top5 evidence head both nominates and accepts" in text
    for arm in (
        "P0_pair_token",
        "P1_pair_relation",
        "T0_structured_token",
        "T1_structured_relation",
        "C1_structured_relation_shuffle",
    ):
        assert arm in text


def test_gate_has_exactly_the_final_current_frame_decisions():
    text = PROBE.read_text()
    for decision in (
        "AUTHORIZE_DCSR_RELATIONAL_V4",
        "AUTHORIZE_DCSR_TOKEN_V4",
        "STOP_CURRENT_FRAME_SELECTOR_AND_AUDIT_HISTORY",
    ):
        assert decision in text
    assert '"development_or_certification_consumed": False' in text


def test_root_wrapper_is_thin():
    text = WRAPPER.read_text()
    assert "run_dcsr_top1_audit_1gpu.sh" in text
    assert "exec " in text
