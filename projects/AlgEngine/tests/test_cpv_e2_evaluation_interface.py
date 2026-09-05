import importlib.util
import inspect
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/diffusiondrive"


def load(name, filename, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroSelector(torch.nn.Module):
    def forward(self, **_inputs):
        return torch.zeros(2, 2)


def test_offline_evaluation_explicitly_uses_paired_common_interface(monkeypatch):
    evaluation = load(
        "cpv_e2_evaluation_interface", "evaluate_rapg_offline.py", monkeypatch
    )
    calls = []

    def fake_mixed_batch(
        rows,
        hard_rows,
        real_cache,
        token_map,
        *,
        common_cache,
        common_token_map,
        common_tokens,
        synthetic,
        device,
    ):
        calls.append((common_cache, common_token_map, common_tokens, device))
        count = len(rows)
        reference = torch.tensor([[1.0, 0.0]]).expand(count, 2).clone()
        rewards = torch.tensor([[0.2, 0.8]]).expand(count, 2).clone()
        components = torch.zeros(count, 2, 6)
        valid = torch.ones(count, 2, dtype=torch.bool)
        return {}, reference, rewards, components, valid, {}

    monkeypatch.setattr(evaluation.trainer, "mixed_batch", fake_mixed_batch)
    values = evaluation.evaluate_rows(
        ZeroSelector(),
        [("common", 0), ("common", 1)],
        [{}, {}],
        {},
        {},
        {},
        torch.device("cpu"),
        1.0,
        64,
    )
    assert calls == [(None, None, None, torch.device("cpu"))]
    assert len(values["current_reward"]) == 2


def test_all_nontraining_mixed_batch_callers_name_the_e2_sidecar_arguments(
    monkeypatch,
):
    evaluation = load(
        "cpv_e2_evaluation_source", "evaluate_rapg_offline.py", monkeypatch
    )
    probe = load(
        "cpv_e2_probe_source", "probe_cpv_regime_predictability.py", monkeypatch
    )
    for function in (evaluation.evaluate_rows, probe.extract_features):
        source = inspect.getsource(function)
        assert "common_cache=None" in source
        assert "common_token_map=None" in source
        assert "common_tokens=None" in source
        assert "synthetic=synthetic" in source
        assert "device=device" in source
