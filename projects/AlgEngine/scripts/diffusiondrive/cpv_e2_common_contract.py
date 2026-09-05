"""Validation helpers for CPV E2 common-only sidecar caches."""

from __future__ import annotations

import json
from pathlib import Path

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
from build_grpo_selector_v3_rare_rollout_data import (
    BASELINE_SHA256,
    EXPECTED_REAL_CACHE_SEEDS,
)


DATA_METHOD = "cpv_e2_existing_navtrain_common_coverage_v1"
ARMS = ("D1_diverse_matched", "D2_diverse_20k")


def load(args, real_caches):
    """Load an optional common sidecar while leaving hard-source caches frozen."""

    cache_arguments = getattr(args, "common_cache", None)
    manifest_argument = getattr(args, "common_data_manifest", None)
    arm = getattr(args, "common_arm", None)
    supplied = (cache_arguments is not None, manifest_argument is not None, arm is not None)
    if not any(supplied):
        return None, None, None, None, None, None
    if not all(supplied):
        raise RuntimeError("E2 common cache, data manifest and arm must be supplied together")
    if arm not in ARMS:
        raise RuntimeError(f"unsupported E2 common arm: {arm}")

    manifest_path = manifest_argument.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("method") != DATA_METHOD
        or manifest.get("existing_dataset_only") is not True
        or manifest.get("new_annotations") is not False
        or manifest.get("training_labels_changed") is not False
    ):
        raise RuntimeError("E2 common data manifest did not pass")
    arm_manifest = manifest.get("arms", {}).get(arm)
    if not isinstance(arm_manifest, dict) or arm_manifest.get("duplicate_rows") != 0:
        raise RuntimeError("E2 common arm is missing or duplicated")
    if arm_manifest.get("overlap_with_rare") or arm_manifest.get("heldout_log_overlap"):
        raise RuntimeError("E2 common arm leaked into rare or held-out data")
    records_path = Path(arm_manifest["token_records"]).expanduser().resolve()
    if common.sha256_file(records_path) != arm_manifest["token_records_sha256"]:
        raise RuntimeError("E2 common token-record SHA256 drifted")
    rows = [
        json.loads(line) for line in records_path.read_text().splitlines() if line.strip()
    ]
    common_tokens = [str(row["token"]) for row in rows]
    if (
        len(common_tokens) != int(arm_manifest["unique_tokens"])
        or len(common_tokens) != len(set(common_tokens))
    ):
        raise RuntimeError("E2 common token identities drifted")

    cache_paths = dict(cache_arguments)
    if set(cache_paths) != set(EXPECTED_REAL_CACHE_SEEDS):
        raise RuntimeError("E2 requires exact common-cache seeds 0,1,2")
    pairs = [
        common.load_cache(cache_paths[seed], "train")
        for seed in EXPECTED_REAL_CACHE_SEEDS
    ]
    caches, cache_manifests = zip(*pairs)
    standard.assert_cache_group(caches, cache_manifests, "train", EXPECTED_REAL_CACHE_SEEDS)
    expected_tokens = sorted(common_tokens)
    for seed, cache, cache_manifest in zip(
        EXPECTED_REAL_CACHE_SEEDS, caches, cache_manifests
    ):
        if cache["tokens"] != expected_tokens:
            raise RuntimeError(f"E2 common cache seed {seed} token set drifted")
        if cache["scene_selector_config"] != real_caches[0]["scene_selector_config"]:
            raise RuntimeError("E2 common/base selector architecture drifted")
        if cache_manifest.get("checkpoint_sha256") != BASELINE_SHA256:
            raise RuntimeError("E2 common cache baseline checkpoint drifted")
    token_maps = [
        {str(token): index for index, token in enumerate(cache["tokens"])}
        for cache in caches
    ]
    return (
        manifest_path,
        manifest,
        common_tokens,
        caches,
        cache_manifests,
        token_maps,
    )
