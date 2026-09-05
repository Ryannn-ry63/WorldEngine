#!/usr/bin/env python3
"""Build immutable 20-arm treatment manifests from the frozen causal pilot."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

import ccv_sweep_common as common


SOURCE_TARGET_METHOD = "diffusiondrive_selector_causal_branch_pilot_targets_v1"
SOURCE_GATE_METHOD = "diffusiondrive_selector_causal_branch_pilot_gate_v2"


def checked_source(args):
    target_path = args.source_target_manifest.expanduser().resolve()
    gate_path = args.source_causal_gate.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    targets = json.loads(target_path.read_text())
    gate = json.loads(gate_path.read_text())
    if (
        targets.get("status") != "PASS"
        or targets.get("method") != SOURCE_TARGET_METHOD
        or targets.get("design_version") != "causal_branch_pilot_v2"
        or int(targets.get("target_count", -1)) != common.NUM_TARGETS
        or int(targets.get("selected", {}).get("failed", {}).get("count", -1))
        != common.NUM_FAILED_TARGETS
        or int(targets.get("selected", {}).get("solved", {}).get("count", -1))
        != common.NUM_SOLVED_TARGETS
    ):
        raise RuntimeError("invalid frozen causal-pilot target source")
    if (
        gate.get("status") != "PASS"
        or gate.get("method") != SOURCE_GATE_METHOD
        or gate.get("decision") != "REDIRECT_TO_REWARD_HORIZON_OR_ACTION_SENSITIVITY"
        or gate.get("target_manifest_sha256") != common.sha256_file(target_path)
    ):
        raise RuntimeError("CCV sweep requires the completed redirect causal gate")
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    return target_path, targets, gate_path, gate, protocol_path


def enrich_target(row):
    record_path = Path(row["source_record_a"]).expanduser().resolve()
    if common.sha256_file(record_path) != row["source_record_a_sha256"]:
        raise RuntimeError(f"source pre-action record drifted: {record_path}")
    with record_path.open("rb") as stream:
        record = pickle.load(stream)
    trajectories = common.checked_array(
        record["candidate_trajectories_8"], (20, 8, 3), "source trajectories"
    )
    logits = common.checked_array(record["current_logits"], (20,), "source logits")
    rewards = common.checked_array(record["candidate_rewards"], (20,), "source rewards")
    components = common.checked_array(
        record["candidate_reward_components"], (20, 6), "source reward components"
    )
    if (
        np.max(np.abs(trajectories - np.asarray(row["candidate_trajectories_8"])))
        > common.ARRAY_TOLERANCE
        or np.max(np.abs(logits - np.asarray(row["current_logits"])))
        > common.ARRAY_TOLERANCE
        or np.max(np.abs(rewards - np.asarray(row["candidate_rewards"])))
        > common.ARRAY_TOLERANCE
    ):
        raise RuntimeError(f"source target context drifted: {row['scene_id']}")
    return {
        "scene_id": str(row["scene_id"]),
        "train_seed": 0,
        "outcome_stratum": str(row["outcome_stratum"]),
        "origin_log": str(row["origin_log"]),
        "source_kind": str(row["source_kind"]),
        "pairing_method": str(row["pairing_method"]),
        "decision_step": int(row["decision_step"]),
        "state_step": int(row["state_step"]),
        "policy_index": int(row["policy_index"]),
        "oracle_index": int(row["oracle_index"]),
        "matched_index": int(row["matched_index"]),
        "candidate_trajectories_8": trajectories.astype(np.float32).tolist(),
        "current_logits": logits.astype(np.float32).tolist(),
        "candidate_rewards": rewards.astype(np.float32).tolist(),
        "candidate_reward_components": components.astype(np.float32).tolist(),
        "reward_component_names": list(record["reward_component_names"]),
        "source_record": str(record_path),
        "source_record_sha256": row["source_record_a_sha256"],
    }


def treatment_payload(
    *, treatment_id, treatment_kind, targets, scenario_file, source_target,
    source_target_sha, protocol, protocol_sha, fixed_index=None,
    expected_audit=None,
):
    treatment_rows = []
    for row in targets:
        if fixed_index is not None:
            index = int(fixed_index)
        else:
            key = {
                "sentinel_policy": "policy_index",
                "sentinel_oracle": "oracle_index",
                "sentinel_matched": "matched_index",
            }[treatment_kind]
            index = int(row[key])
        treatment_rows.append({
            key: row[key]
            for key in (
                "scene_id", "train_seed", "outcome_stratum", "origin_log",
                "decision_step", "state_step", "policy_index", "oracle_index",
                "matched_index", "candidate_trajectories_8", "current_logits",
                "candidate_rewards",
            )
        } | {"treatment_index": index})
    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "status": "PASS",
        "method": common.TREATMENT_METHOD,
        "design_version": "ccv_sweep_v1",
        "scientific_role": "one_step_frozen_candidate_intervention",
        "treatment_id": treatment_id,
        "treatment_kind": treatment_kind,
        "fixed_candidate_index": fixed_index,
        "source_target_manifest": str(source_target),
        "source_target_manifest_sha256": source_target_sha,
        "protocol_file": str(protocol),
        "protocol_file_sha256": protocol_sha,
        "scenario_file": str(scenario_file),
        "scenario_file_sha256": common.sha256_file(scenario_file),
        "target_count": len(treatment_rows),
        "targets": treatment_rows,
        "training_data_consumed": False,
        "future_outcome_used_to_choose_treatment": False,
    }
    if expected_audit is not None:
        payload["sentinel_expected_audit"] = str(expected_audit)
        payload["sentinel_expected_audit_sha256"] = common.sha256_file(expected_audit)
    return payload


def build(args):
    target_path, source, gate_path, gate, protocol_path = checked_source(args)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty CCV manifest root: {output_root}")
    source_sha = common.sha256_file(target_path)
    protocol_sha = common.sha256_file(protocol_path)
    enriched = [enrich_target(row) for row in source["targets"]]
    strata = Counter(row["outcome_stratum"] for row in enriched)
    if strata != {"failed": common.NUM_FAILED_TARGETS, "solved": common.NUM_SOLVED_TARGETS}:
        raise RuntimeError(f"CCV source strata drifted: {strata}")

    smoke_ids = set(map(str, source["smoke_scene_ids"]))
    smoke_targets = [row for row in enriched if row["scene_id"] in smoke_ids]
    if len(smoke_targets) != 8:
        raise RuntimeError("CCV sentinel requires exactly the prior smoke8 scenes")
    sentinel_expected = {
        "sentinel_policy": Path(source["baseline_audit_a"]),
        "sentinel_oracle": Path(gate["audits"]["intervention_smoke8:one_shot_oracle"]["path"]),
        "sentinel_matched": Path(gate["audits"]["intervention_smoke8:one_shot_matched"]["path"]),
    }

    treatment_files = {"sentinels": {}, "arms": {}}
    for treatment_id in ("sentinel_policy", "sentinel_oracle", "sentinel_matched"):
        path = output_root / f"{treatment_id}.json"
        payload = treatment_payload(
            treatment_id=treatment_id,
            treatment_kind=treatment_id,
            targets=smoke_targets,
            scenario_file=source["smoke_scenario_file"],
            source_target=target_path,
            source_target_sha=source_sha,
            protocol=protocol_path,
            protocol_sha=protocol_sha,
            expected_audit=sentinel_expected[treatment_id],
        )
        common.atomic_json(path, payload)
        treatment_files["sentinels"][treatment_id] = {
            "path": str(path), "sha256": common.sha256_file(path)
        }

    for index in range(common.NUM_CANDIDATES):
        treatment_id = f"arm_{index:02d}"
        path = output_root / f"{treatment_id}.json"
        payload = treatment_payload(
            treatment_id=treatment_id,
            treatment_kind="fixed_candidate_index",
            targets=enriched,
            scenario_file=source["formal_scenario_file"],
            source_target=target_path,
            source_target_sha=source_sha,
            protocol=protocol_path,
            protocol_sha=protocol_sha,
            fixed_index=index,
        )
        common.atomic_json(path, payload)
        treatment_files["arms"][str(index)] = {
            "path": str(path), "sha256": common.sha256_file(path)
        }

    master = {
        "schema_version": common.SCHEMA_VERSION,
        "status": "PASS",
        "method": common.MASTER_METHOD,
        "design_version": "ccv_sweep_v1",
        "scientific_question": "local_reward_horizon_or_diffusion_action_sensitivity",
        "source_target_manifest": str(target_path),
        "source_target_manifest_sha256": source_sha,
        "source_causal_gate": str(gate_path),
        "source_causal_gate_sha256": common.sha256_file(gate_path),
        "source_causal_decision": gate["decision"],
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": protocol_sha,
        "formal_scenario_file": str(Path(source["formal_scenario_file"]).resolve()),
        "formal_scenario_file_sha256": source["formal_scenario_file_sha256"],
        "smoke_scenario_file": str(Path(source["smoke_scenario_file"]).resolve()),
        "smoke_scenario_file_sha256": source["smoke_scenario_file_sha256"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "candidate_noise_namespace": source["candidate_noise_namespace"],
        "target_count": len(enriched),
        "failed_target_count": strata["failed"],
        "solved_target_count": strata["solved"],
        "num_candidates": common.NUM_CANDIDATES,
        "arm_indices": list(range(common.NUM_CANDIDATES)),
        "targets": enriched,
        "treatment_manifests": treatment_files,
        "training_data_consumed": False,
        "causal_values_authorized_for_training": False,
        "bootstrap_repetitions": common.BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": common.BOOTSTRAP_SEED,
        "gap_point_threshold": common.GAP_POINT_THRESHOLD,
        "minimum_positive_origin_logs": common.MINIMUM_POSITIVE_LOGS,
    }
    master_path = output_root / "master_manifest.json"
    common.atomic_json(master_path, master)
    common.load_master(master_path)
    return master_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-target-manifest", type=Path, required=True)
    parser.add_argument("--source-causal-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = build(args)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
