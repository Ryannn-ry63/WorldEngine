from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = ROOT / "projects/AlgEngine/scripts/diffusiondrive"


def read(name):
    return (SCRIPT_ROOT / name).read_text()


def test_runner_locks_four_causal_arms_delta_grid_and_matched_budget():
    source = read("run_lcpgrpo_v1.sh")
    for fragment in (
        '"direct||a0_direct"',
        '"lineage_direct||a1_lineage_direct"',
        '"per_draw_proximal|0.01|a2_per_draw_proximal_d001"',
        '"per_draw_proximal|0.03|a2_per_draw_proximal_d003"',
        '"per_draw_proximal|0.10|a2_per_draw_proximal_d010"',
        '"lineage_proximal|0.01|a3_lineage_proximal_d001"',
        '"lineage_proximal|0.03|a3_lineage_proximal_d003"',
        '"lineage_proximal|0.10|a3_lineage_proximal_d010"',
        "--learning-rate 3e-5",
        "--kl-weight 1e-3",
        "--epochs 8",
        "--examples-per-epoch 6339",
        "--batch-size 64",
        "--checkpoint-epochs 1,2,4,8",
        "--seed 0",
    ):
        assert fragment in source


def test_protocol_uses_locked_sha256_log_folds_and_no_token_split():
    source = read("lcpgrpo_protocol.py")
    assert 'FOLD_SALT = "20260902"' in source
    assert "hashlib.sha256" in source
    assert "log_name" in source
    assert 'NUM_FOLDS = 5' in source
    assert 'raise RuntimeError("token leakage across log folds")' in source


def test_cv_gate_contains_every_preregistered_efficacy_guard():
    source = read("lcpgrpo_protocol.py")
    for fragment in (
        '"equal_stratum_hard_gain": 0.002',
        '"common_hard_gain": 0.0',
        '"rare_hard_gain": 0.0',
        '"per_noise_seed_floor": -0.001',
        '"solved_subset_floor": -0.001',
        '"suppressed_recoverable_regret_reduction": 0.005',
        '"positive_fold_count": 4',
        '"mean_current_to_v3_kl_ceiling": 0.9',
        '"method_margin": 0.001',
    ):
        assert fragment in source


def test_lineage_attribution_requires_a0_and_independent_shuffle_controls():
    cv = read("select_lcpgrpo_log_cv.py")
    control = read("select_lcpgrpo_lineage_control.py")
    assert 'comparisons["a1_over_matched_a0"]' in cv
    assert 'comparisons["a3_over_matched_a0"]' in cv
    assert 'comparison["mean"] >= protocol.THRESHOLDS["method_margin"]' in control
    assert 'comparison["lower_95"] > 0.0' in control
    assert '"independent_shuffle"' in control


def test_development_and_certification_are_sequential_single_winner_stages():
    runner = read("run_lcpgrpo_v1.sh")
    assert "AUTHORIZE_SINGLE_WINNER_DEVELOPMENT" in runner
    assert "AUTHORIZE_BLIND_CERTIFICATION" in runner
    assert runner.index("control_stage") < runner.index("development_stage")
    assert runner.index("development_stage") < runner.index("certification_stage")
    development = read("select_lcpgrpo_development.py")
    assert '"only_cv_selected_winner_evaluated": True' in development
    assert '"development_used_for_retuning": False' in development


def test_training_objective_consumes_only_one_scalar_reward_tensor():
    objective = read("lineage_consistent_proximal_grpo.py")
    trainer = read("train_lineage_consistent_proximal_grpo.py")
    forbidden = "candidate_reward_" + "components"
    assert forbidden not in objective
    assert forbidden not in trainer
    assert '"official_scalar_pdm_only": True' in objective
    assert '"official_scalar_pdm_only": True' in trainer


def test_cv_scheduler_uses_all_eight_hopper_lanes_without_changing_grid():
    runner = read("run_lcpgrpo_v1.sh")
    assert "num_tasks=$((5 * num_specs))" in runner
    assert "if (( lanes > 8 )); then lanes=8; fi" in runner
    assert "fold=$((task / num_specs))" in runner
    assert "spec_index=$((task % num_specs))" in runner
    assert "for epoch in 1 2 4 8" in runner
    assert "cached_hopper_preflight" in runner
