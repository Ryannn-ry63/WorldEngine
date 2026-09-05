from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = ROOT / "projects/AlgEngine/scripts/diffusiondrive"


def read(name):
    return (SCRIPT_ROOT / name).read_text()


def test_runner_has_two_mechanism_gates_before_search():
    source = read("run_proposal_aware_full_feedback_v1_8hopper.sh")
    assert "AUTHORIZE_FRESH_NOISE_CONFIRMATION" in source
    assert "AUTHORIZE_PROPOSAL_AWARE_FULL_FEEDBACK_GRPO" in source
    assert "verify_fresh_gate \"${fresh_gate}\"" in source
    assert source.index("run_existing_audit") < source.index("run_fresh_cache")
    assert source.index("run_fresh_audit") < source.index("run_search")


def test_runner_locks_four_arms_and_matched_budget():
    source = read("run_proposal_aware_full_feedback_v1_8hopper.sh")
    assert "local arms=(direct_grpo full_feedback opportunity_grpo paf_grpo)" in source
    for fragment in (
        "--learning-rate 3e-5",
        "--kl-weight 1e-3",
        "--epochs 8",
        "--examples-per-epoch 6339",
        "--batch-size 64",
        "--checkpoint-epochs 1,2,4,8",
        "--seed 0",
    ):
        assert fragment in source


def test_search_uses_fresh_noise_but_training_uses_original_caches():
    source = read("run_proposal_aware_full_feedback_v1_8hopper.sh")
    for seed in (0, 1, 2):
        assert f'--train-cache "${{OLD_CACHE_ROOT}}/train_seed{seed}/cache.pt"' in source
    for seed in (9, 10, 11):
        assert f'--cache "${{FRESH_CACHE_ROOT}}/train_seed{seed}/cache.pt"' in source
    assert "development_seed" not in source
    assert "certification_seed" not in source


def test_gate_contains_pre_registered_efficacy_and_attribution_thresholds():
    source = read("select_proposal_aware_full_feedback.py")
    for fragment in (
        'summary["equal_stratum_hard_gain"] >= 0.002',
        'summary["common_hard_gain"] >= 0.0',
        'summary["rare_hard_gain"] >= 0.0',
        "min(per_seed) >= -0.001",
        'summary["equal_suppressed_recoverable_oracle_regret_reduction"] >= 0.005',
        'summary["equal_solved_subset_hard_gain"] >= -0.001',
        "comparison[\"mean\"] >= margin",
    ):
        assert fragment in source


def test_method_consumes_no_reward_component_or_evaluator_network():
    objective = read("proposal_aware_full_feedback_grpo.py")
    trainer = read("train_proposal_aware_full_feedback_grpo.py")
    assert "candidate_reward_components" not in objective + trainer
    assert "evaluator" not in objective.lower()
    assert '"official_scalar_pdm_only": True' in trainer
    assert '"reward_components_consumed": False' in trainer
