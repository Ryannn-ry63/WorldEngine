from pathlib import Path


ALG = Path(__file__).resolve().parents[1]


def text(path):
    return path.read_text()


def test_config_forbids_gt_and_new_label_inputs():
    config = text(
        ALG
        / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_interaction_v1.py"
    )
    assert "process_perception=False" in config
    assert 'tracker_execution="inference_only_without_gt_matching"' in config
    assert "new_annotations=False" in config
    assert "reward_components_as_inputs=False" in config
    assert "maximum_tracks=30" in config


def test_probe_is_train_only_and_preregisters_all_three_gates():
    probe = text(ALG / "scripts/diffusiondrive/probe_interaction_representation.py")
    assert 'common.load_cache(path, "train")' in probe
    assert "mean_auc_gain_at_least_001" in probe
    assert "paired_log_bootstrap_lower_positive" in probe
    assert "no_fold_worse_than_minus_001" in probe
    assert "development_or_certification_consumed" in probe


def test_development_runner_locks_equal_three_arm_budget():
    runner = text(
        ALG
        / "scripts/diffusiondrive/run_interaction_selector_development_8hopper.sh"
    )
    for architecture in (
        "scene_conditioned_v3",
        "interaction_generic",
        "interaction_relation",
    ):
        assert architecture in runner
    assert "--epochs 16" in runner
    assert "--examples-per-cache-epoch 6339" in runner
    assert "--batch-size 64" in runner
    assert "--learning-rate 1e-4" in runner
    assert "--kl-weight 1e-3" in runner
    assert 'total_optimizer_steps\"]==4800' in runner


def test_development_gate_encodes_stop_and_selection_rules():
    gate = text(
        ALG
        / "scripts/diffusiondrive/evaluate_interaction_selector_development.py"
    )
    assert "equal_weight_reward_gain_at_least_005" in gate
    assert "common_degraded_fraction_not_worse" in gate
    assert "pairwise_auc_ci_positive" in gate
    assert "noise_agreement_no_worse_than_minus_002" in gate
    assert "AUTHORIZE_A2_CLOSED_LOOP_DEVELOPMENT" in gate
    assert "AUTHORIZE_A1_CLOSED_LOOP_DEVELOPMENT" in gate
    assert "STOP_INTERACTION_AND_RETAIN_V3" in gate
