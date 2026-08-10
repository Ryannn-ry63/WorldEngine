import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmcv import Config
from mmcv.runner import build_optimizer
from mmdet3d.models import build_detector
from mmdet3d_plugin.datasets.navsim_openscene_nuplan import NavSimOpenSceneE2E

from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import (
    DiffusionGRPOOnlineReferenceSelectorHook,
    DiffusionGRPOOnlineSelectorPlanningHead,
)
from mmdet3d_plugin.navformer.dense_heads.diffusiondrive_online_pdm_reward import (
    pairwise_official_scores,
)
from mmdet3d_plugin.navformer.diffusion_grpo_iter_runner import (
    DiffusionGRPOIterBasedRunner,
)


def build_head(tmp_path, **kwargs):
    anchor_path = tmp_path / "anchors.npy"
    np.save(anchor_path, np.zeros((4, 8, 2), dtype=np.float32))
    defaults = dict(
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
        num_train_timesteps=1000,
        train_timestep_max=5,
        inference_steps=2,
        trunc_timesteps=2,
        use_nerf=False,
        online_reward=None,
    )
    defaults.update(kwargs)
    return DiffusionGRPOOnlineSelectorPlanningHead(**defaults)


def test_only_current_final_selector_is_trainable(tmp_path):
    head = build_head(tmp_path)
    trainable = head.trainable_parameter_names
    assert len(trainable) == 10
    assert all(
        name.startswith("diff_decoder.layers.1.task_decoder.plan_cls_branch.")
        for name in trainable
    )
    assert not any(parameter.requires_grad for parameter in head.reference_selector.parameters())

    head.train()
    assert head.training
    assert head._current_selector().training
    assert not head.reference_selector.training
    assert not head.diff_decoder.layers[1].task_decoder.plan_reg_branch.training


def test_reference_selector_loads_immutable_checkpoint(tmp_path):
    head = build_head(tmp_path)
    prefix = (
        "planning_head.diff_decoder.layers.1."
        "task_decoder.plan_cls_branch."
    )
    reference_state = {
        prefix + key: torch.full_like(value, 0.125)
        for key, value in head.reference_selector.state_dict().items()
    }
    checkpoint_path = tmp_path / "baseline.pth"
    torch.save({"state_dict": reference_state}, checkpoint_path)
    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    head.reference_checkpoint = str(checkpoint_path)
    head.reference_checkpoint_sha256 = sha256
    assert head.initialize_reference_selector() == 10
    assert all(
        torch.equal(value, torch.full_like(value, 0.125))
        for value in head.reference_selector.state_dict().values()
    )
    assert not any(parameter.requires_grad for parameter in head.reference_selector.parameters())


def test_full_inference_schedule_feeds_both_selectors(tmp_path):
    torch.manual_seed(3)
    head = build_head(tmp_path)
    head.train()
    bev = torch.randn(1, 32, 4, 4)
    ego = torch.randn(1, 1, 32)
    agents = torch.randn(1, 2, 32)
    status = torch.randn(1, 1, 32)

    result = head._forward_train(bev, ego, agents, status)
    assert result["candidate_trajectories_8"].shape == (1, 4, 8, 3)
    assert result["candidate_trajectories"].shape == (1, 4, 40, 3)
    assert result["selector_logits"].shape == (1, 4)
    assert result["selector_logits"].requires_grad
    assert not result["reference_selector_logits"].requires_grad
    assert torch.allclose(
        result["selector_logits"], result["reference_selector_logits"]
    )


def test_clipped_grpo_improves_high_reward_probability(tmp_path):
    head = build_head(tmp_path, clip_epsilon=0.2, kl_weight=1e-3)
    logits = torch.zeros((1, 4), requires_grad=True)
    reference_logits = torch.zeros((1, 4))
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    result = {
        "selector_logits": logits,
        "reference_selector_logits": reference_logits,
        "candidate_rewards": rewards,
        "candidate_reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
    }

    losses = head.loss(result=result)
    total_loss = losses["loss.grpo_selector_policy"] + losses["loss.grpo_selector_kl"]
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert logits.grad[0, -1] < 0
    assert logits.grad[0, 0] > 0
    assert losses["grpo.selector.ratio_mean"].item() == pytest.approx(1.0)
    assert losses["grpo.selector.clip_fraction"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.kl"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.selection_disagreement"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.reward_gain"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.reference_oracle_match"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.current_oracle_probability"].item() == pytest.approx(0.25)
    assert losses["grpo.selector.reference_oracle_probability"].item() == pytest.approx(0.25)
    assert losses["grpo.selector.oracle_probability_gain"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.policy_tv"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.current_expected_reward"].item() == pytest.approx(1.5)
    assert losses["grpo.selector.reference_expected_reward"].item() == pytest.approx(1.5)
    assert losses["grpo.selector.expected_reward_gain"].item() == pytest.approx(0.0)
    assert "loss.grpo_selector_entropy" not in losses
    assert not any("imitation" in key for key in losses)


def test_full_action_grpo_uses_reference_policy_expectation(tmp_path):
    head = build_head(tmp_path, clip_epsilon=0.99, kl_weight=0.0)
    current_logits = torch.tensor([[1.9, 0.05, -0.95, -1.9]], requires_grad=True)
    reference_logits = torch.tensor([[2.0, 0.0, -1.0, -2.0]])
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    result = {
        "selector_logits": current_logits,
        "reference_selector_logits": reference_logits,
        "candidate_rewards": rewards,
        "candidate_reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
    }

    losses = head.loss(result=result)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    advantages = centered / centered.square().mean(dim=-1, keepdim=True).sqrt()
    current_probability = current_logits.softmax(dim=-1)
    expected_policy_loss = -(current_probability * advantages).sum(dim=-1).mean()

    assert losses["loss.grpo_selector_policy"].item() == pytest.approx(
        expected_policy_loss.item()
    )
    assert losses["grpo.selector.ratio_mean"].item() == pytest.approx(1.0)
    assert losses["grpo.selector.current_expected_reward"].item() == pytest.approx(
        (current_probability * rewards).sum().item()
    )

    losses["loss.grpo_selector_policy"].backward()
    updated_probability = (current_logits - 0.1 * current_logits.grad).softmax(dim=-1)
    assert (updated_probability * rewards).sum() > (current_probability * rewards).sum()


def test_selector_diagnostics_detect_argmax_reward_improvement(tmp_path):
    head = build_head(tmp_path)
    result = {
        "selector_logits": torch.tensor([[0.0, 0.0, 0.0, 6.0]]),
        "reference_selector_logits": torch.tensor([[6.0, 0.0, 0.0, 0.0]]),
        "candidate_rewards": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        "candidate_reward_valid_mask": torch.ones((1, 4), dtype=torch.bool),
    }
    losses = head.loss(result=result)
    assert losses["grpo.selector.selection_disagreement"].item() == pytest.approx(1.0)
    assert losses["grpo.selector.reward_gain"].item() == pytest.approx(3.0)
    assert losses["grpo.selector.oracle_match"].item() == pytest.approx(1.0)
    assert losses["grpo.selector.reference_oracle_match"].item() == pytest.approx(0.0)
    assert losses["grpo.selector.oracle_probability_gain"].item() > 0.9
    assert losses["grpo.selector.policy_tv"].item() > 0.9


def test_fixed_8192_reward_cache_is_rejected(tmp_path):
    head = build_head(tmp_path)
    with pytest.raises(ValueError, match="8192-vocabulary cache"):
        head.loss(
            result={
                "selector_logits": torch.zeros((1, 4)),
                "reference_selector_logits": torch.zeros((1, 4)),
            },
            gt_pdm_score={"score": torch.zeros((1, 8192))},
        )


def test_pairwise_score_reconstructs_independent_progress_normalization():
    scorer = SimpleNamespace(
        _multi_metrics=np.ones((2, 3), dtype=np.float64),
        _weighted_metrics=np.ones((4, 3), dtype=np.float64),
        _progress_raw=np.asarray([10.0, 5.0, 20.0], dtype=np.float64),
        _config=SimpleNamespace(
            progress_distance_threshold=0.1,
            weighted_metrics_array=np.asarray([5.0, 5.0, 2.0, 2.0]),
        ),
    )
    scores, components = pairwise_official_scores(scorer)
    assert scores.shape == (2,)
    assert components.shape == (2, 6)
    assert components[:, 2].tolist() == pytest.approx([0.5, 1.0])
    assert scores.tolist() == pytest.approx([11.5 / 14.0, 1.0])


def test_online_config_has_dynamic_reward_contract():
    algengine_root = Path(__file__).resolve().parents[1]
    cfg = Config.fromfile(
        str(
            algengine_root
            / "configs"
            / "diffusiondrive"
            / "e2e_diffusiondrive_grpo_selector.py"
        )
    )
    assert cfg.model.planning_head.type == "DiffusionGRPOOnlineSelectorPlanningHead"
    assert cfg.model.planning_head.num_anchors == 20
    assert cfg.model.planning_head.reference_checkpoint == cfg.load_from
    assert cfg.model.planning_head.online_reward.num_candidates == 20
    assert cfg.data.train.online_candidate_reward is True
    assert cfg.data.train.metric_cache_path.endswith("metric_cache_trainval")
    assert cfg.data.val.metric_cache_path.endswith("metric_cache_navtest")
    assert cfg.data.test.metric_cache_path.endswith("metric_cache_navtest")
    collected_keys = cfg.data.train.pipeline[-1]["keys"]
    assert "score" not in collected_keys
    assert "no_at_fault_collisions" not in collected_keys
    assert cfg.runner.type == "EpochBasedRunner"
    assert cfg.total_epochs == cfg.runner.max_epochs == 8
    assert cfg.selector_reward_contract.fixed_vocabulary_size is None


def test_online_iter_runner_rejects_mismatched_compatibility_limit():
    with pytest.raises(ValueError, match="must mirror max_iters"):
        DiffusionGRPOIterBasedRunner(
            model=torch.nn.Linear(1, 1),
            max_iters=2,
            max_epochs=3,
        )


def test_reference_hook_accepts_worldengine_out_dir(tmp_path):
    hook = DiffusionGRPOOnlineReferenceSelectorHook(out_dir=str(tmp_path))
    assert hook.out_dir == str(tmp_path)
    assert "priority" not in type(hook).__dict__


def test_online_dataset_filter_is_explicit_in_source():
    algengine_root = Path(__file__).resolve().parents[1]
    source = (
        algengine_root
        / "mmdet3d_plugin"
        / "datasets"
        / "navsim_openscene_nuplan.py"
    ).read_text()
    assert "online candidate reward filtered %d frames to %d" in source
    assert 'if str(self.data_infos[index]["token"]) in self.metric_cache_dict' in source
    assert "self._set_group_flag()" in source


def test_online_dataset_filter_rebuilds_sampler_flag():
    dataset = object.__new__(NavSimOpenSceneE2E)
    dataset.data_infos = [
        {"token": "missing"},
        {"token": "kept-a"},
        {"token": "kept-b"},
    ]
    dataset.index_map = [0, 1, 2]
    dataset.metric_cache_dict = {"kept-a": "a.pkl", "kept-b": "b.pkl"}
    dataset.flag = np.zeros(3, dtype=np.uint8)
    dataset._filter_online_candidate_reward_index()
    assert dataset.index_map == [1, 2]
    assert len(dataset.flag) == len(dataset) == 2
    assert dataset.online_candidate_reward_unfiltered_size == 3
    assert dataset.online_candidate_reward_filtered_size == 2


def test_online_optimizer_contains_exactly_current_selector():
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
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert len(trainable) == 10
    assert all(
        name.startswith(
            "planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch."
        )
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
