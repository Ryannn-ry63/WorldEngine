"""Selector-visible inputs and explicit deployment scoring semantics."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

import cfpi_common as common
import grpo_selector_v3_cached_common as v3

INPUTS = {
    "candidate_features": ("candidate_features", (20, 256)),
    "candidate_trajectories": ("candidate_trajectories_8", (20, 8, 3)),
    "route_bev_features": ("route_bev_features", (20, 8, 256)),
    "status_token": ("status_tokens", (1, 256)),
    "ego_query": ("ego_queries", (1, 256)),
    "agents_query": ("agents_queries", (30, 256)),
}


def visible_batch(rows, indices, device):
    return {name: torch.as_tensor(np.stack([
        common.checked_array(rows[i][key], shape, key) for i in indices
    ]), dtype=torch.float32, device=device) for name, (key, shape) in INPUTS.items()}


def score(model, rows, indices, device, mode):
    # Explicit allowlist: neither labels nor failure/timing/fold metadata reach the model.
    output = model(**visible_batch(rows, indices, device))
    if mode == "direct_q":
        return output
    if mode != "residual":
        raise ValueError("Unsupported CFPI score mode")
    base = torch.as_tensor(np.stack([common.checked_array(rows[i]["reference_logits"], (20,), "base logits")
                                    for i in indices]),
                           dtype=torch.float32, device=device)
    return base + output


def load_incumbent(manifest_path):
    import json
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "PASS" or manifest["checkpoint_sha256"] != common.CHECKPOINT_SHA256:
        raise RuntimeError("Wrong incumbent checkpoint")
    path = Path(manifest["scene_selector_state"])
    if common.sha256_file(path) != manifest["scene_selector_state_sha256"]:
        raise RuntimeError("Incumbent selector state drifted")
    payload = torch.load(path, map_location="cpu")
    model = v3.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    return model.eval(), payload["scene_selector_config"], manifest


def initialize(incumbent, mode, training_values=None):
    model = copy.deepcopy(incumbent)
    if mode == "direct_q":
        if training_values is None:
            raise ValueError("Q head initialization requires training-fold labels")
        with torch.no_grad():
            model.delta_head[-1].weight.zero_()
            model.delta_head[-1].bias.fill_(float(training_values.mean()))
    elif mode != "residual":
        raise ValueError("Unknown score mode")
    return model


def save_selector(path, model, config, mode, provenance):
    path = Path(path)
    payload = dict(schema_version=1, method="selector_cfpi_v1", score_mode=mode,
                   scene_selector_config=config,
                   scene_selector_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   provenance=provenance, inference_uses_reward_or_q=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return common.sha256_file(path)


def load_selector(path):
    payload = torch.load(path, map_location="cpu")
    if (payload.get("method") != "selector_cfpi_v1" or payload.get("schema_version") != 1
            or payload.get("inference_uses_reward_or_q") is not False
            or payload.get("score_mode") not in {"residual", "direct_q"}):
        raise RuntimeError("Invalid CFPI selector checkpoint")
    model = v3.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    return model.eval(), payload
