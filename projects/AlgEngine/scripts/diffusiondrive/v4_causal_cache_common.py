#!/usr/bin/env python3
"""Pure contracts for the method-neutral V4 one-step causal-value cache."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
DESIGN_VERSION = "selector_v4_causal_cache_v1"
SOURCE_METHOD = "diffusiondrive_selector_v4_causal_source_v1"
TARGET_METHOD = "diffusiondrive_selector_v4_causal_targets_v1"
TREATMENT_METHOD = "diffusiondrive_selector_v4_causal_treatment_v1"
MASTER_METHOD = "diffusiondrive_selector_v4_causal_master_v1"
COLLECTION_AUDIT_METHOD = "diffusiondrive_selector_v4_collection_audit_v1"
CACHE_METHOD = "diffusiondrive_selector_v4_causal_cache_v1"
PILOT_GATE_METHOD = "diffusiondrive_selector_v4_causal_information_gate_v1"

NUM_CANDIDATES = 20
DECISION_STEPS = tuple(range(4, 12))
ARRAY_TOLERANCE = 1e-5
ACTION_TOLERANCE = 1e-4
OUTCOME_TOLERANCE = 1e-3
Q_SPAN_THRESHOLD = 0.02
Q_POLICY_GAP_THRESHOLD = 0.02
TRAIN_POOL_PER_FAMILY = 512
VALIDATION_POOL_PER_FAMILY = 128
TRAIN_TARGETS = 256
PILOT_TARGETS = 64
DEV_TARGETS = 64
MAXIMUM_POOL_SCENES_PER_LOG = 4
MAXIMUM_TARGETS_PER_LOG = 2
ALLOWED_FAMILIES = ("rare_union", "matched_common")
ALLOWED_SPLITS = ("train", "validation")
SOURCE_REVISION = "diffusiondrive_grpo_v4_whole_log_70_15_15_membership_v1"
SOURCE_SEED = "diffusiondrive-grpo-v4-split0"
SOURCE_SELECTION_SALT = "diffusiondrive-selector-v4-causal-source-v1"
TARGET_SELECTION_SALT = "diffusiondrive-selector-v4-causal-target-v1"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path | str, payload: Mapping) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_pickle(path: Path | str):
    with Path(path).expanduser().resolve().open("rb") as stream:
        return pickle.load(stream)


def atomic_pickle(path: Path | str, payload) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def scene_token(scene_id: str) -> str:
    parts = str(scene_id).rsplit("-", 1)
    if len(parts) != 2 or len(parts[1]) != 16:
        raise ValueError(f"cannot recover 16-character token from scene id: {scene_id}")
    return parts[1]


def scene_origin_log(scene_id: str) -> str:
    parts = str(scene_id).rsplit("-", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"cannot recover origin log from scene id: {scene_id}")
    return parts[0]


def annotate_scenario(scenario: Mapping, family: str, split: str) -> dict:
    if family not in ALLOWED_FAMILIES or split not in ALLOWED_SPLITS:
        raise ValueError(f"forbidden V4 source: {family}/{split}")
    result = dict(scenario)
    scene_id = str(result.get("id", ""))
    token = scene_token(scene_id)
    if str(result.get("token")) not in {scene_id, token}:
        raise RuntimeError(f"scene identity token mismatch: {scene_id}")
    metadata = dict(result.get("metadata") or {})
    lidar_tokens = set(metadata.get("nuplan_lidar_pc_tokens", ()))
    infos = metadata.get("openscene_data_infos_dict", {})
    origin_log = scene_origin_log(scene_id)
    if token not in lidar_tokens or token not in infos:
        raise RuntimeError(f"scene center token missing from metadata: {scene_id}")
    if str(infos[token].get("log_name")) != origin_log:
        raise RuntimeError(f"scene origin-log metadata mismatch: {scene_id}")
    if "v4_causal_source" in metadata:
        raise RuntimeError(f"source scenario already has V4 annotation: {scene_id}")
    metadata["v4_causal_source"] = {
        "design_version": DESIGN_VERSION,
        "scenario_family": family,
        "scenario_variant": "original",
        "split": split,
        "origin_log": origin_log,
        "origin_token": token,
    }
    result["metadata"] = metadata
    return result


def source_metadata(scenario: Mapping) -> dict:
    row = (scenario.get("metadata") or {}).get("v4_causal_source")
    if not isinstance(row, dict):
        raise RuntimeError("scenario lacks frozen v4_causal_source metadata")
    required = {
        "design_version", "scenario_family", "scenario_variant", "split",
        "origin_log", "origin_token",
    }
    if set(row) != required or row["design_version"] != DESIGN_VERSION:
        raise RuntimeError("invalid V4 source annotation")
    return dict(row)


def deterministic_cap(
    scene_ids: Iterable[str], count: int, maximum_per_log: int, salt: str
) -> list[str]:
    """Select a hash-stable, origin-log-capped subset."""
    ranked = sorted(
        {str(scene_id) for scene_id in scene_ids},
        key=lambda scene_id: (stable_digest(salt, scene_id), scene_id),
    )
    selected: list[str] = []
    per_log: Counter = Counter()
    for scene_id in ranked:
        origin_log = scene_origin_log(scene_id)
        if per_log[origin_log] >= maximum_per_log:
            continue
        selected.append(scene_id)
        per_log[origin_log] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"insufficient log-capped coverage: selected={len(selected)} expected={count}"
        )
    return selected


def checked_array(value, shape: Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise RuntimeError(
            f"invalid {label}: expected finite {tuple(shape)}, got {array.shape}"
        )
    return array


def load_source_audit(path: Path | str) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != SOURCE_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != DESIGN_VERSION
    ):
        raise RuntimeError(f"invalid V4 source audit: {path}")
    for split in ALLOWED_SPLITS:
        key = f"{split}_scenario_file"
        if sha256_file(payload[key]) != payload[f"{key}_sha256"]:
            raise RuntimeError(f"V4 source artifact drifted: {key}")
    return path, payload


def validate_target(row: Mapping) -> None:
    required = {
        "scene_id", "split", "scenario_family", "origin_log", "origin_token",
        "outcome_stratum", "decision_step", "state_step", "policy_index",
        "candidate_trajectories_8", "current_logits", "candidate_rewards",
        "source_record_a", "source_record_a_sha256",
    }
    missing = required - set(row)
    if missing:
        raise RuntimeError(f"V4 target missing fields: {sorted(missing)}")
    if row["split"] not in ALLOWED_SPLITS or row["scenario_family"] not in ALLOWED_FAMILIES:
        raise RuntimeError("V4 target contains a forbidden source")
    if int(row["decision_step"]) not in DECISION_STEPS:
        raise RuntimeError("invalid V4 target decision step")
    if not 0 <= int(row["policy_index"]) < NUM_CANDIDATES:
        raise RuntimeError("invalid V4 target policy index")
    checked_array(row["candidate_trajectories_8"], (20, 8, 3), "target candidates")
    checked_array(row["current_logits"], (20,), "target logits")
    checked_array(row["candidate_rewards"], (20,), "target local rewards")


def load_target_manifest(path: Path | str) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != TARGET_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != DESIGN_VERSION
        or payload.get("intervention_outcomes_observed_before_freeze") is not False
    ):
        raise RuntimeError(f"invalid V4 target manifest: {path}")
    targets = payload.get("targets", [])
    if len(targets) != int(payload.get("target_count", -1)):
        raise RuntimeError("V4 target count drifted")
    for row in targets:
        validate_target(row)
    return path, payload


def load_treatment(path: Path | str, expected_sha: str | None = None) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError("V4 treatment manifest SHA256 drifted")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != TREATMENT_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != DESIGN_VERSION
        or payload.get("future_outcome_used_to_choose_treatment") is not False
        or payload.get("treatment_kind") not in {"fixed_candidate", "policy_sentinel"}
    ):
        raise RuntimeError(f"invalid V4 treatment manifest: {path}")
    fixed_index = payload.get("treatment_index")
    if payload["treatment_kind"] == "fixed_candidate":
        if (
            not isinstance(fixed_index, int)
            or fixed_index not in range(NUM_CANDIDATES)
        ):
            raise RuntimeError("invalid V4 fixed treatment index")
    elif fixed_index is not None:
        raise RuntimeError("policy sentinel must not freeze one global candidate index")
    target_path, target_payload = load_target_manifest(payload["target_manifest"])
    if sha256_file(target_path) != payload["target_manifest_sha256"]:
        raise RuntimeError("V4 treatment target provenance drifted")
    target_by_scene = {str(row["scene_id"]): row for row in target_payload["targets"]}
    rows = payload.get("targets", [])
    if len(rows) != int(payload.get("target_count", -1)):
        raise RuntimeError("V4 treatment target count drifted")
    hydrated = []
    for row in rows:
        target = target_by_scene.get(str(row.get("scene_id")))
        if target is None or int(row.get("decision_step", -1)) != int(target["decision_step"]):
            raise RuntimeError("V4 treatment/target membership drifted")
        row_index = int(row.get("treatment_index", -1))
        if row_index not in range(NUM_CANDIDATES):
            raise RuntimeError("invalid V4 treatment row index")
        if fixed_index is not None and row_index != int(fixed_index):
            raise RuntimeError("V4 treatment arm drifted")
        hydrated.append(dict(target, treatment_index=row_index))
    payload = dict(payload)
    payload["targets"] = hydrated
    return path, payload
