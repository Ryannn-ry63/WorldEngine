"""Export selector weights without changing any generator/perception tensor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import cfpi_common as common
import selector_cfpi_model as models
from materialize_grpo_selector_v3 import state_dict, PREFIX


def replace_selector(checkpoint, selector):
    target = state_dict(checkpoint)
    old_keys = {key for key in target if key.startswith(PREFIX)}
    new_keys = {PREFIX + key for key in selector}
    if old_keys != new_keys:
        raise RuntimeError("Export selector tensor membership differs from incumbent")
    unchanged = {key: value.clone() for key, value in target.items() if not key.startswith(PREFIX)}
    for key, value in selector.items():
        if target[PREFIX + key].shape != value.shape:
            raise RuntimeError("Export selector tensor shape differs")
        target[PREFIX + key] = value.detach().cpu().clone()
    if any(not torch.equal(target[key], value) for key, value in unchanged.items()):
        raise RuntimeError("Export modified a non-selector tensor")
    return checkpoint, len(unchanged)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-manifest", type=Path, required=True)
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise RuntimeError("Refusing to overwrite an exported checkpoint")
    manifest = json.loads(args.checkpoint_manifest.read_text())
    baseline = Path(manifest["checkpoint"])
    if (manifest["checkpoint_sha256"] != common.CHECKPOINT_SHA256
            or common.sha256_file(baseline) != common.CHECKPOINT_SHA256):
        raise RuntimeError("Wrong incumbent checkpoint bytes")
    _, selector = models.load_selector(args.selector)
    if selector["provenance"]["incumbent_sha256"] != common.CHECKPOINT_SHA256:
        raise RuntimeError("Selector trained from a different incumbent")
    checkpoint, count = replace_selector(torch.load(baseline, map_location="cpu"),
                                         selector["scene_selector_state"])
    checkpoint.setdefault("meta", {})["selector_cfpi"] = dict(
        score_mode=selector["score_mode"], selector_sha256=common.sha256_file(args.selector))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    reloaded = torch.load(args.output, map_location="cpu")
    before, after = state_dict(checkpoint), state_dict(reloaded)
    if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
        raise RuntimeError("Serialized checkpoint tensor parity failed")
    common.atomic_json(args.output.with_suffix(".manifest.json"), dict(
        status="PASS", method="selector_cfpi_v1", checkpoint=str(args.output.resolve()),
        checkpoint_sha256=common.sha256_file(args.output),
        selector=str(args.selector.resolve()), selector_sha256=common.sha256_file(args.selector),
        unchanged_nonselector_tensors=count, incumbent_checkpoint_sha256=common.CHECKPOINT_SHA256,
        deployment_cfg_options={"model.planning_head.scene_selector_score_mode": selector["score_mode"],
                                "model.planning_head.online_reward": None},
        inference_uses_reward_or_q=False, continuous_deployment_evaluated=False))


if __name__ == "__main__":
    main()
