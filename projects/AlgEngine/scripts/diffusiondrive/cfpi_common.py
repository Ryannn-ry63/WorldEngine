#!/usr/bin/env python3
"""Pure contracts for the method-neutral CFPI one-step causal-value cache."""

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
DESIGN_VERSION = "selector_cfpi_causal_cache_v1"
SOURCE_METHOD = "diffusiondrive_selector_cfpi_causal_source_v1"
TARGET_METHOD = "diffusiondrive_selector_cfpi_causal_targets_v1"
TREATMENT_METHOD = "diffusiondrive_selector_cfpi_causal_treatment_v1"
MASTER_METHOD = "diffusiondrive_selector_cfpi_causal_master_v1"
COLLECTION_AUDIT_METHOD = "diffusiondrive_selector_cfpi_collection_audit_v1"
CACHE_METHOD = "diffusiondrive_selector_cfpi_causal_cache_v1"
PILOT_GATE_METHOD = "diffusiondrive_selector_cfpi_causal_information_gate_v1"

NUM_CANDIDATES = 20
DECISION_STEPS = tuple(range(4, 12))
ARRAY_TOLERANCE = 1e-5
ACTION_TOLERANCE = 1e-4
OUTCOME_TOLERANCE = 1e-3
Q_SPAN_THRESHOLD = 0.02
Q_POLICY_GAP_THRESHOLD = 0.02
TRAIN_POOL_PER_FAMILY = 512
VALIDATION_POOL_PER_FAMILY = 0
TRAIN_TARGETS = 64
PILOT_TARGETS = 64
DEV_TARGETS = 0
MAXIMUM_POOL_SCENES_PER_LOG = 4
MAXIMUM_TARGETS_PER_LOG = 2
ALLOWED_FAMILIES = ("rare_union", "matched_common")
ALLOWED_SPLITS = ("train",)
SOURCE_REVISION = "diffusiondrive_grpo_v4_whole_log_70_15_15_membership_v1"
SOURCE_SEED = "diffusiondrive-grpo-v4-split0"
SOURCE_SELECTION_SALT = "diffusiondrive-selector-cfpi-causal-source-v1"
TARGET_SELECTION_SALT = "diffusiondrive-selector-cfpi-causal-target-v1"


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
        raise ValueError(f"forbidden CFPI source: {family}/{split}")
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
    if "cfpi_causal_source" in metadata:
        raise RuntimeError(f"source scenario already has CFPI annotation: {scene_id}")
    metadata["cfpi_causal_source"] = {
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
    row = (scenario.get("metadata") or {}).get("cfpi_causal_source")
    if not isinstance(row, dict):
        raise RuntimeError("scenario lacks frozen cfpi_causal_source metadata")
    required = {
        "design_version", "scenario_family", "scenario_variant", "split",
        "origin_log", "origin_token",
    }
    if set(row) != required or row["design_version"] != DESIGN_VERSION:
        raise RuntimeError("invalid CFPI source annotation")
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
        raise RuntimeError(f"invalid CFPI source audit: {path}")
    for split in ALLOWED_SPLITS:
        key = f"{split}_scenario_file"
        if sha256_file(payload[key]) != payload[f"{key}_sha256"]:
            raise RuntimeError(f"CFPI source artifact drifted: {key}")
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
        raise RuntimeError(f"CFPI target missing fields: {sorted(missing)}")
    if row["split"] not in ALLOWED_SPLITS or row["scenario_family"] not in ALLOWED_FAMILIES:
        raise RuntimeError("CFPI target contains a forbidden source")
    if int(row["decision_step"]) not in DECISION_STEPS:
        raise RuntimeError("invalid CFPI target decision step")
    if not 0 <= int(row["policy_index"]) < NUM_CANDIDATES:
        raise RuntimeError("invalid CFPI target policy index")
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
        raise RuntimeError(f"invalid CFPI target manifest: {path}")
    targets = payload.get("targets", [])
    if len(targets) != int(payload.get("target_count", -1)):
        raise RuntimeError("CFPI target count drifted")
    for row in targets:
        validate_target(row)
    if len({row["scene_id"] for row in targets}) != len(targets):
        raise RuntimeError("Duplicate CFPI target scene")
    if sha256_file(payload["protocol_file"]) != payload["protocol_file_sha256"]:
        raise RuntimeError("CFPI target protocol drifted")
    return path, payload


def load_treatment(path: Path | str, expected_sha: str | None = None) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError("CFPI treatment manifest SHA256 drifted")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != TREATMENT_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != DESIGN_VERSION
        or payload.get("future_outcome_used_to_choose_treatment") is not False
        or payload.get("treatment_kind") not in {"fixed_candidate", "policy_sentinel"}
    ):
        raise RuntimeError(f"invalid CFPI treatment manifest: {path}")
    fixed_index = payload.get("treatment_index")
    if payload["treatment_kind"] == "fixed_candidate":
        if (
            not isinstance(fixed_index, int)
            or fixed_index not in range(NUM_CANDIDATES)
        ):
            raise RuntimeError("invalid CFPI fixed treatment index")
    elif fixed_index is not None:
        raise RuntimeError("policy sentinel must not freeze one global candidate index")
    target_path, target_payload = load_target_manifest(payload["target_manifest"])
    if sha256_file(target_path) != payload["target_manifest_sha256"]:
        raise RuntimeError("CFPI treatment target provenance drifted")
    if sha256_file(payload["scenario_file"]) != payload["scenario_file_sha256"]:
        raise RuntimeError("CFPI treatment scenario provenance drifted")
    if sha256_file(payload["protocol_file"]) != payload["protocol_file_sha256"]:
        raise RuntimeError("CFPI treatment protocol provenance drifted")
    target_by_scene = {str(row["scene_id"]): row for row in target_payload["targets"]}
    rows = payload.get("targets", [])
    if len(rows) != int(payload.get("target_count", -1)):
        raise RuntimeError("CFPI treatment target count drifted")
    if len({row["scene_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate CFPI treatment scene")
    hydrated = []
    for row in rows:
        target = target_by_scene.get(str(row.get("scene_id")))
        if target is None or int(row.get("decision_step", -1)) != int(target["decision_step"]):
            raise RuntimeError("CFPI treatment/target membership drifted")
        row_index = int(row.get("treatment_index", -1))
        if row_index not in range(NUM_CANDIDATES):
            raise RuntimeError("invalid CFPI treatment row index")
        if fixed_index is not None and row_index != int(fixed_index):
            raise RuntimeError("CFPI treatment arm drifted")
        hydrated.append(dict(target, treatment_index=row_index))
    payload = dict(payload)
    payload["targets"] = hydrated
    return path, payload


CHECKPOINT_SHA256 = "bd35f0293c878c6cddb985ac0b8aaae91d4f7f8e0bf2c3bf70ba49ba329c02a3"
NOISE_NAMESPACE = "selector_cfpi_v1_noise0"
REPEAT_TARGETS = 8
PILOT_MIN_IMPROVABLE = 16


def verified_json(path, hash_key=None):
    """Read an audit and verify the file it names, not just its PASS flag."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("status") != "PASS":
        raise RuntimeError(f"Unsuccessful audit: {path}")
    if hash_key and sha256_file(data[hash_key]) != data[hash_key + "_sha256"]:
        raise RuntimeError(f"Artifact changed after audit: {path}")
    return data


def locked_json(path, payload):
    """Frozen contracts cannot be silently replaced when resuming."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Frozen contract drift: {path}; use a new run ID")
    else:
        atomic_json(path, payload)


def validate_cache(payload, expected_count=64):
    if (payload.get("method") != CACHE_METHOD or payload.get("design_version") != DESIGN_VERSION
            or payload.get("schema_version") != SCHEMA_VERSION or payload.get("num_candidates") != 20
            or payload.get("stage") != "pilot64" or payload.get("status") != "PASS"
            or payload.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or payload.get("candidate_noise_namespace") != NOISE_NAMESPACE
            or payload.get("development_consumed") is not False
            or payload.get("test_consumed") is not False):
        raise RuntimeError("Not an authorized CFPI training cache")
    rows = payload["rows"]
    if len(rows) != expected_count or payload["num_rows"] != expected_count:
        raise RuntimeError("CFPI cache count mismatch")
    ids = [row["scene_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate CFPI cache scene")
    for row in rows:
        if row["split"] != "train" or scene_origin_log(row["scene_id"]) != row["origin_log"]:
            raise RuntimeError("CFPI cache split/log mismatch")
        q = checked_array(row["branch_returns"], (20,), "branch returns")
        checked_array(row["local_official_pdm"], (20,), "local reward")
        logits = checked_array(row["v3_logits"], (20,), "V3 logits")
        if row["policy_index"] != int(logits.argmax()) or np.any((q < 0) | (q > 1)):
            raise RuntimeError("Invalid incumbent or out-of-range branch return")
        if row.get("fold") not in range(4):
            raise RuntimeError("Missing/invalid preregistered fold")
    groups = {}
    for row in rows:
        if row["origin_log"] in groups and groups[row["origin_log"]] != row["fold"]:
            raise RuntimeError("Origin log crosses CV folds")
        groups[row["origin_log"]] = row["fold"]
    return rows
