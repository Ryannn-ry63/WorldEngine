#!/usr/bin/env python3
"""Fail-closed audit for R1.5 pre-action oracle intervention rollouts."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import selector_root_cause_metrics as metrics_common
from audit_selector_root_cause_r1_collection import (
    completed_scenes,
    load_manifest,
    load_reports,
    load_scenario_contract,
)


AUDIT_METHOD = "diffusiondrive_selector_oracle_r15_collection_audit_v1"
R1_AUDIT_METHOD = "diffusiondrive_selector_root_cause_r1_collection_audit_v1"


def record_paths(root: Path, layout: str) -> list[Path]:
    suffix = "WE_output/openscene_format/diffusiondrive_preaction_records/*_preaction.pkl"
    paths = sorted((root / suffix).parent.glob("*_preaction.pkl")) if layout == "merged" else sorted(root.glob(f"split_*/{suffix}"))
    if not paths:
        raise RuntimeError(f"no {layout} pre-action records under {root}")
    return paths


def expected_contract(seed, checkpoint_sha, manifest_sha, phase, mode, target_sha, implementation_sha):
    return {
        "schema_version": 5,
        "experiment": "diffusiondrive_selector_oracle_r15",
        "source_policy": "trained_v3_selector",
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": seed,
        "expected_checkpoint_sha256": checkpoint_sha,
        "behavior_checkpoint_manifest_sha256": manifest_sha,
        "rollout_implementation_sha256": implementation_sha,
        "trained_v3_checkpoint_loaded": True,
        "deployed_action_parity_required": False,
        "num_dynamic_candidates": 20,
        "candidate_context_export": True,
        "reward_owner": "simengine_preaction_candidate_reward",
        "reward_scalar": "official_pairwise_pdm",
        "reward_components_role": "diagnostics_only",
        "generator_frozen": True,
        "perception_frozen": True,
        "base_selector_frozen": True,
        "diagnostic_split": phase,
        "react_type": "R",
        "training_data_consumed": False,
        "development_only": True,
        "action_score_timing": "pre_action",
        "oracle_intervention_mode": mode,
        "oracle_target_manifest_sha256": target_sha,
        "oracle_future_information_used": True,
        "oracle_intervention_treatment": "frozen_baseline_index_at_target_then_current_oracle",
        "target_reward_repeatability_role": "stability_sensitivity_not_treatment_identity",
        "causal_gate_requires_current_headroom_gt_0p02": True,
    }


def load_targets(path: Path | None, expected_sha: str, seed: int):
    if path is None:
        if expected_sha != "none":
            raise RuntimeError("missing target manifest")
        return {}, None
    path = path.expanduser().resolve()
    if r15.sha256_file(path) != expected_sha:
        raise RuntimeError("target manifest SHA256 drifted")
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS" or payload.get("method") != "diffusiondrive_selector_oracle_r15_targets_v1":
        raise RuntimeError("invalid R1.5 target manifest")
    rows = [row for row in payload.get("targets", []) if int(row["train_seed"]) == seed]
    targets = {str(row["scene_id"]): row for row in rows}
    if len(targets) != len(rows):
        raise RuntimeError("duplicate target scene")
    return targets, payload


def load_reference(path: Path | None, phase: str, seed: int):
    if path is None:
        return {}, None
    path = path.expanduser().resolve()
    audit = json.loads(path.read_text())
    expected_split = {"baseline_smoke8": "smoke8", "baseline_cl_dev58": "cl_dev58"}.get(phase)
    if (
        expected_split is None
        or audit.get("status") != "PASS"
        or audit.get("method") != R1_AUDIT_METHOD
        or audit.get("layout") != "merged"
        or audit.get("diagnostic_split") != expected_split
        or audit.get("behavior_policy_family") != "scalar_v3"
        or int(audit.get("behavior_policy_train_seed", -1)) != seed
    ):
        raise RuntimeError(f"invalid R1 reference audit: {path}")
    reference_root = Path(audit["rollout_root"])
    files = sorted((reference_root / "WE_output/openscene_format/diffusiondrive_rollout_records").glob("*_reward.pkl"))
    rows = {}
    for file in files:
        with file.open("rb") as stream:
            row = pickle.load(stream)
        key = (str(row["rollout_scene_id"]), int(row["planner_step"]))
        if key in rows:
            raise RuntimeError(f"duplicate R1 reference frame: {key}")
        rows[key] = row
    return rows, {"path": str(path), "sha256": r15.sha256_file(path)}


def array(path, row, key, shape):
    value = np.asarray(row.get(key))
    if value.shape != shape or not np.isfinite(value).all():
        raise RuntimeError(f"{path}: invalid {key} shape/value")
    return value


def validate_record(path, row, contract, namespace, checkpoint_sha, mode, targets):
    if row.get("schema_version") != 4 or row.get("record_type") != "diffusiondrive_preaction_oracle_record":
        raise RuntimeError(f"record schema drifted: {path}")
    if row.get("selector_rollout_contract") != contract:
        raise RuntimeError(f"rollout contract drifted: {path}")
    if row.get("checkpoint_sha256") != checkpoint_sha or row.get("candidate_noise_namespace") != namespace:
        raise RuntimeError(f"checkpoint/noise provenance drifted: {path}")
    flat = {
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": contract["behavior_policy_train_seed"],
        "behavior_checkpoint_manifest_sha256": contract["behavior_checkpoint_manifest_sha256"],
        "diagnostic_split": contract["diagnostic_split"],
        "react_type": "R",
        "intervention_mode": mode,
        "target_manifest_sha256": contract["oracle_target_manifest_sha256"],
    }
    for key, expected in flat.items():
        if row.get(key) != expected:
            raise RuntimeError(f"flat provenance {key} drifted: {path}")
    shapes = {
        "candidate_features": (20, 256),
        "candidate_trajectories_8": (20, 8, 3),
        "route_bev_features": (20, 8, 256),
        "status_tokens": (1, 256),
        "ego_queries": (1, 256),
        "agents_queries": (30, 256),
        "reference_logits": (20,),
        "current_logits": (20,),
        "candidate_rewards": (20,),
        "candidate_reward_components": (20, 6),
        "policy_deployed_trajectory": (40, 3),
        "deployed_trajectory": (40, 3),
    }
    values = {key: array(path, row, key, shape) for key, shape in shapes.items()}
    valid = np.asarray(row.get("candidate_reward_valid_mask"), dtype=np.bool_)
    if valid.shape != (20,) or not valid.all():
        raise RuntimeError(f"candidate-validity drifted: {path}")
    if tuple(row.get("reward_component_names", ())) != metrics_common.COMPONENT_NAMES:
        raise RuntimeError(f"reward component order drifted: {path}")
    scene = str(row["rollout_scene_id"])
    state = int(row["state_step"])
    decision = int(row["decision_step"])
    if int(row["worldengine_step"]) != state or decision != state + 1:
        raise RuntimeError(f"pre-action timing drifted: {path}")
    policy = int(row["policy_selected_index"])
    if int(row["selected_index"]) != policy or int(np.asarray(row["selected_indices"]).item()) != policy:
        raise RuntimeError(f"policy index export drifted: {path}")
    if policy != int(np.argmax(values["current_logits"])):
        raise RuntimeError(f"policy/logit argmax drifted: {path}")
    rewards = r15.checked_rewards(values["candidate_rewards"])
    raw_oracle = int(np.argmax(rewards))
    oracle = r15.stable_oracle_index(rewards, policy)
    if int(row["raw_oracle_index"]) != raw_oracle or int(row["oracle_index"]) != oracle:
        raise RuntimeError(f"oracle selection drifted: {path}")
    target = targets.get(scene)
    target_step = None if target is None else int(target["decision_step"])
    treatment = oracle
    repeatability = None
    if target is not None and decision == target_step:
        errors = r15.target_context_errors(
            target, values["candidate_trajectories_8"], values["current_logits"]
        )
        if any(value > r15.ARRAY_TOLERANCE for value in errors.values()):
            raise RuntimeError(f"frozen target context drifted: {path}: {errors}")
        treatment = int(target["oracle_index"])
        reward_error = r15.max_abs_error(
            target["candidate_rewards"], rewards
        )
        treatment_headroom = float(rewards[treatment] - rewards[policy])
        oracle_agreement = bool(treatment == oracle)
        repeatability = {
            "scene_id": scene,
            "decision_step": decision,
            "baseline_treatment_index": treatment,
            "current_scored_oracle_index": oracle,
            "scored_oracle_agreement": oracle_agreement,
            "reward_max_abs_error": reward_error,
            "current_treatment_headroom": treatment_headroom,
            "current_treatment_headroom_gt_0p02": bool(
                treatment_headroom > r15.HEADROOM_THRESHOLD
            ),
        }
        checks = {
            "target_reward_max_abs_error": reward_error,
            "target_treatment_current_headroom": treatment_headroom,
        }
        for key, expected in checks.items():
            if not np.isclose(float(row[key]), expected, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"target repeatability {key} drifted: {path}")
        if bool(row["target_scored_oracle_agreement"]) != oracle_agreement:
            raise RuntimeError(f"target oracle agreement drifted: {path}")
    elif any(row.get(key) is not None for key in (
        "target_reward_max_abs_error",
        "target_treatment_current_headroom",
        "target_scored_oracle_agreement",
    )):
        raise RuntimeError(f"target diagnostics leaked outside target frame: {path}")
    if int(row["treatment_index"]) != treatment:
        raise RuntimeError(f"treatment index drifted: {path}")
    expected_applied = r15.intervention_applies(mode, decision, target_step, False)
    if bool(row["intervention_applied"]) != expected_applied:
        raise RuntimeError(f"intervention schedule drifted: {path}")
    deployed = treatment if expected_applied else policy
    if int(row["deployed_index"]) != deployed:
        raise RuntimeError(f"deployed index drifted: {path}")
    if float(row["policy_action_parity_max_abs_error"]) > r15.ACTION_TOLERANCE or float(row["deployed_action_parity_max_abs_error"]) > r15.ACTION_TOLERANCE:
        raise RuntimeError(f"action conversion/replacement parity failed: {path}")
    expected_scalars = {
        "policy_candidate_reward": rewards[policy],
        "oracle_candidate_reward": rewards[oracle],
        "treatment_candidate_reward": rewards[treatment],
        "deployed_candidate_reward": rewards[deployed],
    }
    for key, expected in expected_scalars.items():
        if not np.isclose(float(row[key]), float(expected), rtol=0.0, atol=1e-6):
            raise RuntimeError(f"{key} drifted: {path}")
    for provenance in ("raw_observation_path", "sidecar_path"):
        if not Path(str(row.get(provenance, ""))).is_file():
            raise RuntimeError(f"missing source provenance {provenance}: {path}")
    return (
        scene, state, decision, policy, oracle, treatment, deployed,
        expected_applied, values, repeatability,
    )


def validate_reference(key, row, values, policy, reference):
    if key not in reference:
        raise RuntimeError(f"missing matching R1 reference frame: {key}")
    old = reference[key]
    if int(old["selected_index"]) != policy:
        raise RuntimeError(f"policy differs from R1 reference: {key}")
    pairs = {
        "candidate_trajectories_8": values["candidate_trajectories_8"],
        "current_logits": values["current_logits"],
        "reference_logits": values["reference_logits"],
        "policy_deployed_trajectory": values["policy_deployed_trajectory"],
    }
    for old_key, new_value in pairs.items():
        source_key = "deployed_trajectory" if old_key == "policy_deployed_trajectory" else old_key
        if r15.max_abs_error(old[source_key], new_value) > r15.ARRAY_TOLERANCE:
            raise RuntimeError(f"R1 baseline parity drifted for {old_key}: {key}")


def audit_collection(args):
    root = args.rollout_root.expanduser().resolve()
    manifest_path, manifest = load_manifest(args.checkpoint_manifest, "scalar_v3", args.train_seed)
    manifest_sha = r15.sha256_file(manifest_path)
    checkpoint_sha = str(manifest["checkpoint_sha256"])
    target_sha = "none" if args.target_manifest is None else r15.sha256_file(args.target_manifest.expanduser().resolve())
    if args.mode == "observe_only" and args.target_manifest is not None:
        raise RuntimeError("observe-only audit must not receive targets")
    if args.mode != "observe_only" and args.target_manifest is None:
        raise RuntimeError("intervention audit requires targets")
    contract = expected_contract(args.train_seed, checkpoint_sha, manifest_sha, args.phase, args.mode, target_sha, args.expected_implementation_sha256)
    targets, target_payload = load_targets(args.target_manifest, target_sha, args.train_seed)
    reference, reference_info = load_reference(args.reference_r1_audit, args.phase, args.train_seed)
    expected_scenes, excluded_short = load_scenario_contract(args.scenario_file, args.maximum_scenarios)
    report_paths, reports = load_reports(root)
    ledger_paths, ledgers = completed_scenes(root)
    unsuccessful = {scene for scene in expected_scenes if scene not in reports or not any(reports[scene])}
    if set(reports) != expected_scenes or ledgers != expected_scenes or unsuccessful:
        raise RuntimeError(f"scenario coverage drifted: reports={len(reports)} ledgers={len(ledgers)} expected={len(expected_scenes)} failed={sorted(unsuccessful)}")

    seen = set()
    steps = defaultdict(list)
    workers = set()
    config_shas = set()
    code_shas = set()
    resolved = defaultdict(set)
    interventions = []
    target_repeatability = []
    headrooms = []
    parity_max = 0.0
    for path in record_paths(root, args.layout):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        result = validate_record(path, row, contract, args.expected_noise_namespace, checkpoint_sha, args.mode, targets)
        scene, state, decision, policy, oracle, treatment, deployed, applied, values, repeatability = result
        if scene not in expected_scenes:
            raise RuntimeError(f"record outside scenario contract: {path}")
        key = (scene, decision)
        if key in seen:
            raise RuntimeError(f"duplicate logical pre-action frame: {key}")
        seen.add(key)
        steps[scene].append(decision)
        if reference:
            validate_reference(key, row, values, policy, reference)
        if repeatability is not None:
            target_repeatability.append(repeatability)
        if applied:
            interventions.append({"scene_id": scene, "decision_step": decision, "policy_index": policy, "scored_oracle_index": oracle, "treatment_index": treatment})
        headrooms.append(float(values["candidate_rewards"][oracle] - values["candidate_rewards"][policy]))
        parity_max = max(parity_max, float(row["policy_action_parity_max_abs_error"]), float(row["deployed_action_parity_max_abs_error"]))
        source = path if args.layout == "split" else Path(row["sidecar_path"])
        parts = [part for part in source.parts if part.startswith("split_")]
        if len(parts) != 1:
            raise RuntimeError(f"ambiguous worker provenance: {path}")
        worker = parts[0]
        workers.add(worker)
        config_shas.add(str(row.get("config_sha256")))
        code_shas.add(str(row.get("code_sha")))
        resolved[worker].add(str(row.get("resolved_config_sha256")))
    expected_decisions = list(range(4, 12))
    bad = {scene: sorted(value) for scene, value in steps.items() if sorted(value) != expected_decisions}
    if bad or set(steps) != expected_scenes:
        raise RuntimeError(f"record coverage failed: bad={bad} missing={sorted(expected_scenes-set(steps))}")
    if len(workers) != args.expected_workers:
        raise RuntimeError(f"expected {args.expected_workers} workers, got {sorted(workers)}")
    if len(config_shas) != 1 or "None" in config_shas or code_shas != {args.expected_code_sha}:
        raise RuntimeError("source code/config provenance drifted")
    if any(len(value) != 1 for value in resolved.values()):
        raise RuntimeError("resolved config drifted within a worker")
    if reference and len(reference) != len(seen):
        raise RuntimeError("R1 reference/new baseline frame counts differ")

    outcome = None
    if args.metrics_csv is not None:
        metrics_path = args.metrics_csv.expanduser().resolve()
        outcomes = metrics_common.load_outcomes(metrics_path)
        if set(outcomes) != expected_scenes:
            raise RuntimeError("metric scenario coverage drifted")
        outcome = {
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": r15.sha256_file(metrics_path),
            "mean_score": float(np.mean([row["score"] for row in outcomes.values()])),
            "success_rate": float(np.mean([row["success"] for row in outcomes.values()])),
            "failed_scenarios": int(sum(not row["success"] for row in outcomes.values())),
            "success_definition": "NC==1_and_DAC==1",
        }
    expected_target_scenes = set(targets) & expected_scenes
    intervened_scenes = {row["scene_id"] for row in interventions}
    if args.mode != "observe_only" and intervened_scenes != expected_target_scenes:
        raise RuntimeError(f"target/intervention scene coverage drifted: target={expected_target_scenes} actual={intervened_scenes}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "method": AUDIT_METHOD,
        "layout": args.layout,
        "rollout_root": str(root),
        "diagnostic_split": args.phase,
        "react_type": "R",
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": args.train_seed,
        "oracle_intervention_mode": args.mode,
        "target_manifest": None if args.target_manifest is None else str(args.target_manifest.expanduser().resolve()),
        "target_manifest_sha256": target_sha,
        "target_count_in_scenario_file": len(expected_target_scenes),
        "intervention_count": len(interventions),
        "intervened_scene_count": len(intervened_scenes),
        "interventions": interventions,
        "target_repeatability": target_repeatability,
        "target_scored_oracle_agreement_count": int(sum(row["scored_oracle_agreement"] for row in target_repeatability)),
        "target_current_headroom_gt_0p02_count": int(sum(row["current_treatment_headroom_gt_0p02"] for row in target_repeatability)),
        "target_reward_max_abs_error": max((row["reward_max_abs_error"] for row in target_repeatability), default=0.0),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "selector_state_sha256": manifest["scene_selector_state_sha256"],
        "candidate_noise_namespace": args.expected_noise_namespace,
        "rollout_implementation_sha256": args.expected_implementation_sha256,
        "code_sha": args.expected_code_sha,
        "selector_rollout_contract": contract,
        "scenario_file": str(args.scenario_file.expanduser().resolve()),
        "scenario_file_sha256": r15.sha256_file(args.scenario_file.expanduser().resolve()),
        "num_scenarios": len(expected_scenes),
        "num_records": len(seen),
        "records_per_scene": 8,
        "num_workers": len(workers),
        "workers": sorted(workers),
        "runner_report_files": [str(path) for path in report_paths],
        "completed_scenario_ledgers": [str(path) for path in ledger_paths],
        "excluded_short_scenarios": excluded_short,
        "maximum_action_parity_error": parity_max,
        "mean_candidate_headroom": float(np.mean(headrooms)),
        "resolved_config_sha256_by_worker": {key: next(iter(value)) for key, value in sorted(resolved.items())},
        "r1_reference_parity": reference_info,
        "closed_loop_outcome": outcome,
        "target_manifest_method": None if target_payload is None else target_payload["method"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase", choices=("baseline_smoke8", "baseline_cl_dev58", "intervention_smoke8", "intervention_target8"), required=True)
    parser.add_argument("--mode", choices=r15.MODES, required=True)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--reference-r1-audit", type=Path)
    parser.add_argument("--layout", choices=("split", "merged"), required=True)
    parser.add_argument("--expected-noise-namespace", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--maximum-scenarios", type=int)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_collection(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
