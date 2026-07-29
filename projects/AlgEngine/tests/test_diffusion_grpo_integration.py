"""Integration checks for the WorldEngine DiffusionDrive GRPO adapter."""

from pathlib import Path

import numpy as np
import torch
from mmcv import Config

from mmdet3d_plugin.navformer.detectors.navformer import NAVFormer


WORLDENGINE_ROOT = Path(__file__).resolve().parents[3]
GRPO_CONFIG = (
    WORLDENGINE_ROOT
    / "projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo.py"
)
OLD_GRPO_CONFIG = (
    WORLDENGINE_ROOT
    / "projects/AlgEngine/configs/navformer/e2e_diffusiondrive_grpo.py"
)


def test_grpo_config_follows_upstream_diffusiondrive_layout(monkeypatch, tmp_path):
    checkpoint = tmp_path / "policy.pth"
    checkpoint.touch()
    monkeypatch.setenv("WORLDENGINE_ROOT", str(WORLDENGINE_ROOT))
    monkeypatch.setenv("GRPO_REFERENCE_CKPT", str(checkpoint))
    monkeypatch.setenv("GRPO_POLICY_CKPT", str(checkpoint))
    monkeypatch.delenv("GRPO_ROLLOUT_MANIFEST", raising=False)

    config = Config.fromfile(str(GRPO_CONFIG))
    planning_head = config.model.planning_head

    assert GRPO_CONFIG.is_file()
    assert not OLD_GRPO_CONFIG.exists()
    assert planning_head.type == "DiffusionGRPOPlanningHead"
    assert planning_head.score_mode == "rollout"
    assert Path(planning_head.plan_anchor_path) == (
        WORLDENGINE_ROOT / "kmeans_navsim_traj_20.npy"
    )
    assert config.data.train.type == "NavSimOpenSceneE2EGRPORollout"


def test_rollout_result_keeps_diffusion_trace_without_local_recompute():
    detector = NAVFormer.__new__(NAVFormer)
    batch_size = 1
    num_modes = 3
    plan_results = {
        "trajectory": torch.zeros(batch_size, 40, 3),
        "trajectory_8": torch.zeros(batch_size, 8, 3),
        "all_trajectories": torch.zeros(batch_size, num_modes, 40, 3),
        "all_trajectories_8": torch.zeros(batch_size, num_modes, 8, 3),
        "poses_cls": torch.zeros(batch_size, num_modes),
        "selected_indices": torch.tensor([1]),
        "grpo_initial_sample": torch.zeros(batch_size, num_modes, 8, 2),
        "grpo_transition_action": torch.zeros(batch_size, num_modes, 8, 2),
        "grpo_final_action": torch.zeros(batch_size, num_modes, 8, 3),
        "grpo_old_log_probs": torch.zeros(batch_size, num_modes, 2),
    }
    img_metas = [[None, None, None, {"sample_idx": "token-1"}]]
    sdc_planning = [torch.zeros(batch_size, 1, 8, 3)]

    result = detector._forward_test_rollout(
        plan_results=plan_results,
        img_metas=img_metas,
        sdc_planning=sdc_planning,
        batch_size=batch_size,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample["token"] == "token-1"
    assert sample["chosen_ind"] == 1
    assert sample["trajectory"].shape == (40, 3)
    assert sample["all_trajectories"].shape == (num_modes, 40, 3)
    assert sample["grpo_initial_sample"].shape == (num_modes, 8, 2)
    assert sample["grpo_transition_action"].shape == (num_modes, 8, 2)
    assert sample["grpo_final_action"].shape == (num_modes, 8, 3)
    assert sample["grpo_old_log_probs"].shape == (num_modes, 2)
    assert np.isnan(sample["score"])
    assert not hasattr(NAVFormer, "_forward_test_recompute_pdm")
