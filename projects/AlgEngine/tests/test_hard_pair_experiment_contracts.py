from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "projects/AlgEngine/scripts/diffusiondrive/probe_hard_pair_information.py"
RUNNER = ROOT / "projects/AlgEngine/scripts/diffusiondrive/run_hard_pair_audit_1gpu.sh"
WRAPPER = ROOT / "run_diffusiondrive_selector_hard_pair_audit_1gpu.sh"


def test_runner_reuses_only_three_train_caches_and_one_visible_gpu():
    text = RUNNER.read_text()
    assert text.count("--train-cache") == 3
    assert "train_seed0/cache.pt" in text
    assert "train_seed1/cache.pt" in text
    assert "train_seed2/cache.pt" in text
    assert "expected exactly 1 visible GPU" in text
    assert "--development-cache" not in text
    assert "--certification-cache" not in text
    assert "d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133" in text


def test_probe_locks_top5_and_keeps_rewards_out_of_inputs():
    text = PROBE.read_text()
    assert "TOPK = 5" in text
    assert '"reward_or_components_as_input": False' in text
    assert '"shortlist_defined_before_reward": True' in text
    assert '"development_or_certification_consumed": False' in text
    assert "S1_relation_shuffle" in text
    assert "S2_interaction_shuffle" in text


def test_gate_has_all_pre_registered_followup_decisions():
    text = PROBE.read_text()
    for decision in (
        "AUTHORIZE_TOKEN_TOP5_EVALUATOR",
        "AUTHORIZE_RELATIONAL_TOP5_EVALUATOR",
        "AUTHORIZE_INTERACTION_TOP5_EVALUATOR",
        "AUTHORIZE_LOCKED_TOP1_OBJECTIVE",
        "AUTHORIZE_HISTORY_AUDIT",
    ):
        assert decision in text


def test_root_wrapper_is_thin_and_points_to_the_locked_runner():
    text = WRAPPER.read_text()
    assert "run_hard_pair_audit_1gpu.sh" in text
    assert "exec " in text
