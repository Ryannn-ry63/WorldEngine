from pathlib import Path


WORLDENGINE_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = WORLDENGINE_ROOT / "projects/AlgEngine/scripts/diffusiondrive"


def test_formal_replicas_reuse_the_matching_development_optimizer_state():
    launcher = (
        SCRIPT_ROOT / "run_grpo_selector_rollout_v1_formal_seed_h100.sh"
    ).read_text()
    assert (
        'SELECTOR_STATE="${TRIAL_ROOT}/optimizer_seed${SEED}/'
        'epoch_${EPOCH}_scene_selector.pt"'
    ) in launcher
    assert 'if [[ "${SEED}" == "0" ]]' not in launcher
    assert "replica_train" not in launcher
    assert "train_grpo_selector_v3_cached.py" not in launcher
    assert "--stage certification" in launcher
    assert "--stage formal --seed" in launcher


def test_finish_pipeline_orders_all_stages_and_has_safe_stage_resume():
    launcher = (
        SCRIPT_ROOT / "run_grpo_selector_rollout_v1_finish_h100.sh"
    ).read_text()
    development = launcher.index(
        "run_grpo_selector_rollout_v1_development_h100.sh"
    )
    certification = launcher.index(
        "run_grpo_selector_rollout_v1_certify_h100.sh"
    )
    formal = launcher.index("run_grpo_selector_rollout_v1_formal_seed_h100.sh")
    assert development < certification < formal
    assert "for seed in 0 1 2" in launcher
    assert "SKIP completed and audited rollout-v1 development" in launcher
    assert "SKIP completed and audited one-shot rollout-v1 certification" in launcher
    assert "SKIP completed and audited rollout-v1 formal seed" in launcher
    assert "--stage complete" in launcher


def test_frozen_root_entry_point_supports_cpu_preflight_and_one_run_command():
    wrapper = (
        WORLDENGINE_ROOT / "run_diffusiondrive_rollout_v1_finish_8h100.sh"
    ).read_text()
    assert "diffusiondrive-selector-grpo-rollout-v1-finish-ready-20260814" in wrapper
    assert 'MODE="${1:-run}"' in wrapper
    assert "--stage inputs" in wrapper
    assert 'exec "${PIPELINE}"' in wrapper
