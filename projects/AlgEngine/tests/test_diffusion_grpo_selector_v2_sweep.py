import importlib.util
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "diffusiondrive"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load_module("train_grpo_selector_v2_cached", "train_grpo_selector_v2_cached.py")
SELECT = load_module("select_grpo_selector_v2", "select_grpo_selector_v2.py")


def test_cached_exact_loss_is_complete_action_expectation():
    logits = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    reference = torch.tensor([[4.0, 0.0, -2.0]])
    rewards = torch.tensor([[0.1, 0.7, 0.4]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    loss, policy, kl = TRAIN.exact_group_loss(
        logits, reference, rewards, valid, temperature=2.0, kl_weight=0.0
    )
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    advantage = centered / centered.square().mean(dim=-1, keepdim=True).sqrt()
    expected = -((logits / 2.0).softmax(dim=-1) * advantage).sum(dim=-1).mean()
    assert torch.allclose(loss, expected)
    assert torch.allclose(policy, expected)
    assert kl.item() >= 0.0


def test_scene_bootstrap_is_deterministic_and_scene_grouped():
    rows = [
        {"scene": "a", "top1_reward_gain": 0.1},
        {"scene": "a", "top1_reward_gain": 0.2},
        {"scene": "b", "top1_reward_gain": -0.1},
        {"scene": "c", "top1_reward_gain": 0.3},
    ]
    first = SELECT.scene_bootstrap(rows, replicates=200, seed=7)
    second = SELECT.scene_bootstrap(rows, replicates=200, seed=7)
    assert first == second
    assert first["bootstrap_unit"] == "scene/log"
    assert first["mean"] == 0.125


def test_materialized_checkpoint_changes_only_selector_and_adds_frozen_reference(
    tmp_path,
):
    baseline_state = {"unrelated.weight": torch.tensor([3.0])}
    selector_state = {}
    for index in range(10):
        relative = f"tensor_{index}"
        baseline_state[SELECT.CURRENT_PREFIX + relative] = torch.tensor(
            [float(index)]
        )
        selector_state[relative] = torch.tensor([float(index + 10)])
    baseline = tmp_path / "baseline.pth"
    selector = tmp_path / "selector.pt"
    output = tmp_path / "selected.pth"
    torch.save({"state_dict": baseline_state, "meta": {}}, baseline)
    torch.save({"selector_state": selector_state}, selector)

    SELECT.materialize_checkpoint(
        baseline, selector, output, {"temperature": 4.0}
    )
    payload = torch.load(output, map_location="cpu")
    state = payload["state_dict"]
    assert state["unrelated.weight"].item() == 3.0
    for index in range(10):
        relative = f"tensor_{index}"
        assert state[SELECT.CURRENT_PREFIX + relative].item() == index + 10
        assert state[SELECT.REFERENCE_PREFIX + relative].item() == index
