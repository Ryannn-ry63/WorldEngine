from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config
from mmcv.runner import build_optimizer
from mmdet3d.models import build_detector

from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_planning_head import (
    DiffusionGRPOSelectorPlanningHead,
)


def build_head(tmp_path):
    anchor_path = tmp_path / "anchors.npy"
    np.save(anchor_path, np.zeros((4, 8, 2), dtype=np.float32))
    return DiffusionGRPOSelectorPlanningHead(
        num_poses=8,
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_bounding_boxes=2,
        num_query_decoder_layers=1,
        query_keyval_size=2,
        num_anchors=4,
        num_diff_decoder_layers=2,
        plan_anchor_path=str(anchor_path),
        bev_h=4,
        bev_w=4,
        num_train_timesteps=10,
        train_timestep_max=5,
        inference_steps=2,
        trunc_timesteps=2,
        use_nerf=False,
    )


def test_only_final_selector_is_trainable(tmp_path):
    head = build_head(tmp_path)
    trainable = head.trainable_parameter_names
    assert trainable
    assert all(
        name.startswith("diff_decoder.layers.1.task_decoder.plan_cls_branch.")
        for name in trainable
    )
    assert not any("plan_reg_branch" in name for name in trainable)

    head.train()
    assert head.training
    assert not head.diff_decoder.layers[0].training
    assert not head.diff_decoder.layers[1].task_decoder.plan_reg_branch.training
    assert head.diff_decoder.layers[1].task_decoder.plan_cls_branch.training


def test_group_relative_loss_increases_high_reward_probability(tmp_path):
    head = build_head(tmp_path)
    logits = torch.zeros((1, 4), requires_grad=True)
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    losses = head.loss(
        result={"selector_logits": logits},
        gt_pdm_score={"score": rewards},
    )
    total_loss = (
        losses["loss.grpo_selector_policy"]
        + losses["loss.grpo_selector_entropy"]
    )
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert logits.grad[0, -1] < 0
    assert logits.grad[0, 0] > 0
    assert losses["grpo.selector.active_group_fraction"].item() == 1.0


def test_navsim_8192_cache_is_rejected(tmp_path):
    head = build_head(tmp_path)
    with pytest.raises(ValueError, match="8192-trajectory NAVSIM cache"):
        head.loss(
            result={"selector_logits": torch.zeros((1, 4))},
            gt_pdm_score={"score": torch.zeros((1, 8192))},
        )


def test_config_inherits_official_diffusiondrive_and_selects_new_head():
    algengine_root = Path(__file__).resolve().parents[1]
    cfg = Config.fromfile(
        str(
            algengine_root
            / "configs"
            / "diffusiondrive"
            / "e2e_diffusiondrive_grpo_selector.py"
        )
    )
    assert cfg.model.planning_head.type == "DiffusionGRPOSelectorPlanningHead"
    assert cfg.model.planning_head.num_anchors == 20
    assert cfg.model.planning_head.train_all_selector_layers is False
    assert cfg.selector_reward_contract.reward_shape == "(B, 20)"
    assert cfg.load_from.endswith("diffusiondrive/e2e_diffusiondrive/epoch_100.pth")


def test_optimizer_freezes_complete_model_outside_selector():
    algengine_root = Path(__file__).resolve().parents[1]
    cfg = Config.fromfile(
        str(
            algengine_root
            / "configs"
            / "diffusiondrive"
            / "e2e_diffusiondrive_grpo_selector.py"
        )
    )
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    optimizer = build_optimizer(model, cfg.optimizer)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 10
    assert all(
        name.startswith("planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch.")
        for name in trainable
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
