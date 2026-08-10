import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diffusiondrive" / "run_grpo_selector_diagnostic_probes.py"
SPEC = importlib.util.spec_from_file_location("selector_diagnostic_probes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scene_split_keeps_each_scene_whole():
    scenes = ["a", "a", "b", "b", "c", "c"]
    train, dev = MODULE.stable_scene_split(scenes, fraction=0.5)
    assert torch.equal(train, ~dev)
    for scene in set(scenes):
        values = train[torch.tensor([value == scene for value in scenes])]
        assert bool((values == values[0]).all())


def test_normalized_advantage_is_centered_and_rms_scaled():
    reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    advantage = MODULE.normalized_advantage(reward)
    assert torch.allclose(advantage.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(
        advantage.square().mean(dim=-1), torch.ones(1), atol=1e-6
    )


def test_canonical_loss_uses_reference_distribution_and_is_finite():
    logits = torch.tensor([[0.2, 0.0, 0.0, 0.0]], requires_grad=True)
    reference = torch.zeros_like(logits)
    reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    loss = MODULE.objective_loss("canonical", logits, reference, reward, 0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_oracle_ce_treats_reward_ties_uniformly():
    logits = torch.zeros((1, 4), requires_grad=True)
    reward = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
    loss = MODULE.objective_loss("oracle_ce", logits, logits.detach(), reward)
    loss.backward()
    assert logits.grad[0, 0].item() == logits.grad[0, 1].item()
    assert logits.grad[0, 0].item() < 0
    assert logits.grad[0, 2].item() > 0


def test_temperature_does_not_change_argmax():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert logits.argmax(dim=-1).item() == (logits / 4.0).argmax(dim=-1).item()
