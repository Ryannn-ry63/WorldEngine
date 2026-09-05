#!/usr/bin/env python3
"""Analyze R1.5 oracle interventions and gate the selector mechanism direction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import selector_root_cause_metrics as metrics_common


AUDIT_METHOD = "diffusiondrive_selector_oracle_r15_collection_audit_v1"


def load_audit(path: Path, expected_mode: str, target_sha: str):
    path = path.expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("diagnostic_split") != "intervention_target8"
        or row.get("oracle_intervention_mode") != expected_mode
        or row.get("target_manifest_sha256") != target_sha
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid {expected_mode} R1.5 audit: {path}")
    return path, row


def load_baseline(path: Path, seed: int):
    path = path.expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("diagnostic_split") != "baseline_cl_dev58"
        or row.get("oracle_intervention_mode") != "observe_only"
        or int(row.get("behavior_policy_train_seed", -1)) != seed
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid baseline R1.5 audit: {path}")
    return path, row


def outcomes(audit):
    return metrics_common.load_outcomes(Path(audit["closed_loop_outcome"]["metrics_csv"]))


def bootstrap(values, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"lower95": None, "point": None, "upper95": None}
    if len(values) == 1 or repetitions <= 0:
        samples = values
    else:
        rng = np.random.default_rng(seed)
        chunks = []
        left = repetitions
        while left:
            count = min(left, 1024)
            indices = rng.integers(0, len(values), size=(count, len(values)))
            chunks.append(values[indices].mean(axis=1))
            left -= count
        samples = np.concatenate(chunks)
    return {
        "lower95": float(np.quantile(samples, 0.025)),
        "point": float(values.mean()),
        "upper95": float(np.quantile(samples, 0.975)),
    }


def summarize_arm(name, audits, baseline_by_seed, targets, repetitions, seed):
    audit_by_seed = {}
    repeatability_by_seed = {}
    paths = {}
    for path, audit in audits:
        train_seed = int(audit["behavior_policy_train_seed"])
        if train_seed in audit_by_seed:
            raise RuntimeError(f"duplicate {name} seed{train_seed} audit")
        audit_by_seed[train_seed] = audit
        repeatability_rows = audit.get("target_repeatability", [])
        repeatability_by_seed[train_seed] = {
            str(row["scene_id"]): row for row in repeatability_rows
        }
        if len(repeatability_by_seed[train_seed]) != len(repeatability_rows):
            raise RuntimeError(f"duplicate {name} repeatability scene")
        paths[str(train_seed)] = str(path)
    expected_seeds = {int(row["train_seed"]) for row in targets}
    if set(audit_by_seed) != expected_seeds:
        raise RuntimeError(f"{name} seed coverage {set(audit_by_seed)} != {expected_seeds}")
    baseline_outcomes = {value: outcomes(audit) for value, (_, audit) in baseline_by_seed.items()}
    arm_outcomes = {value: outcomes(audit) for value, audit in audit_by_seed.items()}
    units = []
    for target in targets:
        train_seed = int(target["train_seed"])
        scene = str(target["scene_id"])
        before = baseline_outcomes[train_seed].get(scene)
        after = arm_outcomes[train_seed].get(scene)
        repeatability = repeatability_by_seed[train_seed].get(scene)
        if repeatability is None:
            raise RuntimeError(f"missing target repeatability for seed{train_seed} {scene}")
        if before is None or after is None:
            raise RuntimeError(f"missing paired outcome for seed{train_seed} {scene}")
        if before["success"]:
            raise RuntimeError(f"frozen target was not a baseline failure: {scene}")
        units.append({
            "train_seed": train_seed,
            "scene_id": scene,
            "decision_step": int(target["decision_step"]),
            "headroom": float(target["headroom"]),
            "baseline_score": float(before["score"]),
            "intervention_score": float(after["score"]),
            "score_delta": float(after["score"] - before["score"]),
            "rescued": bool(after["success"]),
            "reward_stable": bool(repeatability["current_treatment_headroom_gt_0p02"]),
            "current_treatment_headroom": float(repeatability["current_treatment_headroom"]),
            "scored_oracle_agreement": bool(repeatability["scored_oracle_agreement"]),
            "reward_max_abs_error": float(repeatability["reward_max_abs_error"]),
            "baseline_no_at_fault_collisions": float(before["no_at_fault_collisions"]),
            "intervention_no_at_fault_collisions": float(after["no_at_fault_collisions"]),
            "baseline_drivable_area_compliance": float(before["drivable_area_compliance"]),
            "intervention_drivable_area_compliance": float(after["drivable_area_compliance"]),
        })
    all_grouped = defaultdict(list)
    stable_grouped = defaultdict(list)
    for row in units:
        all_grouped[row["scene_id"]].append(row)
        if row["reward_stable"]:
            stable_grouped[row["scene_id"]].append(row)
    unique = []
    for scene, rows in sorted(stable_grouped.items()):
        unique.append({
            "scene_id": scene,
            "train_seeds": sorted(row["train_seed"] for row in rows),
            "mean_score_delta": float(np.mean([row["score_delta"] for row in rows])),
            "rescued": bool(any(row["rescued"] for row in rows)),
            "unit_count": len(rows),
        })
    deltas = [row["mean_score_delta"] for row in unique]
    rescued = sum(row["rescued"] for row in unique)
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    passes = bool(rescued >= 2 and mean_delta > 0.0)
    return {
        "mode": name,
        "audits_by_seed": paths,
        "target_unit_count": len(units),
        "stable_target_unit_count": int(sum(row["reward_stable"] for row in units)),
        "all_unique_scene_count": len(all_grouped),
        "primary_gate_stratum": "current_treatment_headroom_gt_0p02",
        "unique_scene_count": len(unique),
        "rescued_unique_scenes": rescued,
        "mean_unique_scene_score_delta": mean_delta,
        "score_delta_unique_scene_bootstrap_95": bootstrap(deltas, repetitions, seed),
        "gate_pass": passes,
        "units": units,
        "unique_scenes": unique,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, action="append", default=[])
    parser.add_argument("--one-shot-audit", type=Path, action="append", default=[])
    parser.add_argument("--persistent-audit", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_path = args.target_manifest.expanduser().resolve()
    target_sha = r15.sha256_file(target_path)
    manifest = json.loads(target_path.read_text())
    if manifest.get("status") != "PASS" or manifest.get("method") != "diffusiondrive_selector_oracle_r15_targets_v1":
        raise RuntimeError("invalid target manifest")
    targets = list(manifest.get("targets", []))
    if not targets:
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "method": "diffusiondrive_selector_oracle_r15_causal_gate_v1",
            "decision": "STOP_R15_LOCAL_PDM_CAUSAL_ROUTE",
            "reason": "no_failed_scene_had_a_pre_violation_candidate_with_headroom_gt_0p02",
            "target_manifest": str(target_path),
            "target_manifest_sha256": target_sha,
            "one_shot_oracle": None,
            "persistent_oracle": None,
            "scientific_contract": {
                "causal_variable": "deployed_candidate_index_only",
                "new_training_performed": False,
                "certification_claim_allowed": False,
            },
        }
        output = args.output.expanduser().resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite immutable R1.5 result: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "decision": payload["decision"], "output": str(output)}, sort_keys=True))
        return
    target_seeds = {int(row["train_seed"]) for row in targets}
    baseline_by_seed = {}
    for path in args.baseline_audit:
        raw = json.loads(path.expanduser().resolve().read_text())
        train_seed = int(raw.get("behavior_policy_train_seed", -1))
        resolved, audit = load_baseline(path, train_seed)
        if train_seed in baseline_by_seed:
            raise RuntimeError(f"duplicate baseline seed{train_seed}")
        baseline_by_seed[train_seed] = (resolved, audit)
    if set(baseline_by_seed) != target_seeds:
        raise RuntimeError("baseline/target seed coverage drifted")
    one = [load_audit(path, "one_shot_oracle", target_sha) for path in args.one_shot_audit]
    persistent = [load_audit(path, "persistent_oracle", target_sha) for path in args.persistent_audit]
    one_result = summarize_arm("one_shot_oracle", one, baseline_by_seed, targets, args.bootstrap_repetitions, args.bootstrap_seed)
    persistent_result = summarize_arm("persistent_oracle", persistent, baseline_by_seed, targets, args.bootstrap_repetitions, args.bootstrap_seed + 1000)
    if one_result["gate_pass"]:
        decision = "AUTHORIZE_CURRENT_SET_CAUSAL_METHOD"
        reason = "single_preaction_frozen_treatment_rescued_at_least_two_reward_stable_unique_scenes_with_positive_mean_delta"
    elif persistent_result["gate_pass"]:
        decision = "AUTHORIZE_TEMPORAL_OVERRIDE_METHOD"
        reason = "only_persistent_oracle_sequence_met_the_causal_gate"
    elif max(one_result["rescued_unique_scenes"], persistent_result["rescued_unique_scenes"]) == 1:
        decision = "R15_SINGLE_CASE_ONLY"
        reason = "oracle_intervention_rescue_did_not_repeat_across_distinct_scenes"
    else:
        decision = "STOP_R15_LOCAL_PDM_CAUSAL_ROUTE"
        reason = "neither_local_oracle_intervention_established_repeatable_closed_loop_rescue"
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_selector_oracle_r15_causal_gate_v1",
        "decision": decision,
        "reason": reason,
        "target_manifest": str(target_path),
        "target_manifest_sha256": target_sha,
        "gate": {
            "minimum_rescued_distinct_scenes": 2,
            "required_target_stability": "current_treatment_headroom_gt_0p02",
            "mean_unique_scene_score_delta_strictly_greater_than": 0.0,
            "priority": ["one_shot_oracle", "persistent_oracle"],
            "bootstrap_interval_is_descriptive_not_a_gate": True,
        },
        "one_shot_oracle": one_result,
        "persistent_oracle": persistent_result,
        "scientific_contract": {
            "causal_variable": "deployed_candidate_index_only",
            "candidate_set_generator_perception_and_selector_frozen": True,
            "intervention_is_pre_action": True,
            "target_selection_uses_future_failure_information": True,
            "oracle_is_diagnostic_not_available_at_deployment": True,
            "frozen_treatment_is_not_reselected_from_rerun_scores": True,
            "rerun_reward_stability_is_a_required_gate_stratum": True,
            "new_training_performed": False,
            "new_loss_implemented": False,
            "certification_claim_allowed": False,
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable R1.5 result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": decision, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
