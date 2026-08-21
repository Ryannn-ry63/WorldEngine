from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "diffusiondrive"
sys.path.insert(0, str(SCRIPT_DIR))
common = importlib.import_module("grpo_selector_v2_rare_common")
rare = importlib.import_module("train_grpo_selector_v2_cached_rare_original")
rollout = importlib.import_module("train_grpo_selector_v2_cached_rare_rollout")


def small_cache(count: int = 4):
    torch.manual_seed(17)
    source = common.Selector().eval()
    features32 = torch.randn(count, 20, 256)
    with torch.no_grad():
        reference = source(features32).squeeze(-1)
    return {
        "tokens": [f"token-{index}" for index in range(count)],
        "candidate_features": features32.half(),
        "reference_logits": reference,
        "baseline_selector_state": source.state_dict(),
    }


def test_v2_fp16_cache_is_exactly_delta_anchored():
    cache = small_cache()
    model = common.model_from_cache(cache, torch.device("cpu"))
    current, reference = common.current_logits(
        model, cache, slice(None), torch.device("cpu")
    )
    assert torch.equal(current, reference)
    assert len(model.state_dict()) == common.SELECTOR_TENSOR_COUNT == 10
    assert len(list(model.parameters())) == 10
    assert all(parameter.requires_grad for parameter in model.parameters())
    anchor = getattr(model, "_v2_frozen_anchor")
    assert not any(parameter.requires_grad for parameter in anchor.parameters())
    assert "_v2_frozen_anchor" not in model._modules

    audit = common.assert_reference_parity(
        model, cache, torch.device("cpu"), batch_size=2
    )
    assert audit["anchored_max_abs_error"] == 0.0
    assert audit["raw_max_abs_error"] > 0.0


def test_v2_anchor_preserves_trainable_mlp_delta_and_gradients():
    cache = small_cache(count=2)
    model = common.model_from_cache(cache, torch.device("cpu"))
    with torch.no_grad():
        model[6].bias.add_(0.125)
    current, reference = common.current_logits(
        model, cache, slice(None), torch.device("cpu")
    )
    assert torch.allclose(current - reference, torch.full_like(reference, 0.125))
    current.sum().backward()
    assert model[6].bias.grad is not None
    assert getattr(model, "_v2_frozen_anchor")[6].bias.grad is None


def test_v2_fixed_compute_contracts_match_v3_budget():
    rare_args = SimpleNamespace(
        temperature=1.0,
        learning_rate=1e-4,
        kl_weight=1e-3,
        epochs=16,
        examples_per_cache_epoch=6339,
        batch_size=64,
        method_name=rare.FORMAL_RARE_METHOD,
        sampling_mode="rare_balanced",
        train_cache=[1, 2, 3],
    )
    rare.validate_formal_contract(rare_args)
    rollout_args = SimpleNamespace(
        temperature=1.0,
        learning_rate=1e-4,
        kl_weight=1e-3,
        epochs=16,
        examples_per_cache_epoch=6339,
        batch_size=64,
        method_name=rollout.METHOD,
        hard_pool_split="all",
    )
    rollout.validate_formal_args(rollout_args)
    assert rare.FORMAL_TOTAL_EXAMPLES == rollout.FORMAL_TOTAL_EXAMPLES == 304272
    assert rare.FORMAL_OPTIMIZER_STEPS == rollout.FORMAL_OPTIMIZER_STEPS == 4800


def test_v2_rollout_entrypoint_reuses_v3_data_and_does_not_collect():
    root = SCRIPT_DIR.parents[3]
    script = (
        root / "run_diffusiondrive_grpo_selector_v2_rare_rollout_8h100.sh"
    ).read_text()
    assert "grpo_selector_v3_rare_rollout_v1" in script
    assert "data/manifest.json" not in script  # DATA_ROOT is joined at runtime.
    assert "run_grpo_selector_v3_rare_rollout_collect_h100.sh" not in script
    assert "run_simulation.py" not in script
    assert "--real-cache" in script and "--synthetic-cache" in script
