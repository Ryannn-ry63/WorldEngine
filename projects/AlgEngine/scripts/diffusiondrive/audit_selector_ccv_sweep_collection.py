#!/usr/bin/env python3
"""Fail-closed collection audit for one CCV treatment manifest."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import ccv_sweep_common as common
import oracle_r15_common as r15
import selector_root_cause_metrics as metrics_common
from audit_selector_root_cause_r1_collection import (
    completed_scenes,
    load_manifest,
    load_reports,
    load_scenario_contract,
)


def record_paths(root: Path, layout: str):
    suffix = "WE_output/openscene_format/diffusiondrive_ccv_records"
    if layout == "merged":
        paths = sorted((root / suffix).glob("*_ccv.pkl"))
    else:
        paths = sorted(root.glob(f"split_*/{suffix}/*_ccv.pkl"))
    if not paths:
        raise RuntimeError(f"no {layout} CCV records under {root}")
    return paths


def expected_contract(seed, checkpoint_sha, manifest_sha, treatment, implementation_sha):
    return {
        "schema_version": 8,
        "experiment": "diffusiondrive_selector_ccv_sweep_v1",
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
        "residual_selector_frozen": True,
        "diagnostic_split": treatment["treatment_id"],
        "react_type": "R",
        "training_data_consumed": False,
        "train_only": True,
        "development_only": False,
        "diagnostic_only": True,
        "action_score_timing": "pre_action",
        "intervention_mode": "one_shot_manifest",
        "treatment_id": treatment["treatment_id"],
        "treatment_manifest_sha256": treatment["manifest_sha256"],
        "treatment_index_source": "frozen_manifest",
        "treatment_future_outcome_used": False,
        "target_selection_uses_prior_baseline_outcome": True,
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "causal_value_authorized_for_training": False,
    }


def array(path, row, key, shape):
    value = np.asarray(row.get(key))
    if value.shape != shape or not np.isfinite(value).all():
        raise RuntimeError(f"{path}: invalid {key} shape/value")
    return value


def validate_record(path, row, contract, namespace, checkpoint_sha, targets):
    if (
        row.get("schema_version") != 5
        or row.get("record_type") != "diffusiondrive_candidate_causal_value_record"
    ):
        raise RuntimeError(f"CCV record schema drifted: {path}")
    if row.get("selector_rollout_contract") != contract:
        raise RuntimeError(f"CCV rollout contract drifted: {path}")
    if (
        row.get("checkpoint_sha256") != checkpoint_sha
        or row.get("candidate_noise_namespace") != namespace
    ):
        raise RuntimeError(f"CCV checkpoint/noise provenance drifted: {path}")
    expected_flat = {
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": 0,
        "behavior_checkpoint_manifest_sha256": contract[
            "behavior_checkpoint_manifest_sha256"
        ],
        "diagnostic_split": contract["diagnostic_split"],
        "react_type": "R",
        "intervention_mode": "one_shot_manifest",
        "target_manifest_sha256": contract["treatment_manifest_sha256"],
        "ccv_treatment_id": contract["treatment_id"],
    }
    for key, expected in expected_flat.items():
        if row.get(key) != expected:
            raise RuntimeError(f"CCV flat provenance {key} drifted: {path}")
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
        raise RuntimeError(f"CCV candidate validity drifted: {path}")
    if tuple(row.get("reward_component_names", ())) != metrics_common.COMPONENT_NAMES:
        raise RuntimeError(f"CCV reward component order drifted: {path}")

    scene = str(row["rollout_scene_id"])
    state = int(row["state_step"])
    decision = int(row["decision_step"])
    if int(row["worldengine_step"]) != state or decision != state + 1:
        raise RuntimeError(f"CCV pre-action timing drifted: {path}")
    policy = int(row["policy_selected_index"])
    if (
        int(row["selected_index"]) != policy
        or int(np.asarray(row["selected_indices"]).item()) != policy
        or policy != int(np.argmax(values["current_logits"]))
    ):
        raise RuntimeError(f"CCV policy index export drifted: {path}")
    rewards = r15.checked_rewards(values["candidate_rewards"])
    reward_oracle = r15.stable_oracle_index(rewards, policy)
    if int(row["local_reward_oracle_index"]) != reward_oracle:
        raise RuntimeError(f"CCV reward oracle drifted: {path}")

    target = targets.get(scene)
    target_step = None if target is None else int(target["decision_step"])
    treatment = policy
    at_target = target is not None and decision == target_step
    if at_target:
        errors = r15.target_context_errors(
            target, values["candidate_trajectories_8"], values["current_logits"]
        )
        if any(value > common.ARRAY_TOLERANCE for value in errors.values()):
            raise RuntimeError(f"CCV target context drifted: {path}: {errors}")
        treatment = int(target["treatment_index"])
        reward_error = r15.max_abs_error(target["candidate_rewards"], rewards)
        headroom = float(rewards[treatment] - rewards[policy])
        if not np.isclose(
            float(row["target_reward_max_abs_error"]), reward_error,
            rtol=0.0, atol=common.OUTCOME_TOLERANCE,
        ):
            raise RuntimeError(f"CCV target reward diagnostic drifted: {path}")
        if not np.isclose(
            float(row["target_treatment_current_headroom"]), headroom,
            rtol=0.0, atol=common.OUTCOME_TOLERANCE,
        ):
            raise RuntimeError(f"CCV target headroom diagnostic drifted: {path}")
    elif (
        row.get("target_reward_max_abs_error") is not None
        or row.get("target_treatment_current_headroom") is not None
    ):
        raise RuntimeError(f"CCV target diagnostics leaked outside target frame: {path}")

    if int(row["treatment_index"]) != treatment:
        raise RuntimeError(f"CCV treatment index drifted: {path}")
    applied = bool(at_target)
    if bool(row["intervention_applied"]) != applied:
        raise RuntimeError(f"CCV intervention schedule drifted: {path}")
    deployed = treatment if applied else policy
    if int(row["deployed_index"]) != deployed:
        raise RuntimeError(f"CCV deployed index drifted: {path}")
    parity = max(
        float(row["policy_action_parity_max_abs_error"]),
        float(row["deployed_action_parity_max_abs_error"]),
    )
    if parity > common.ACTION_TOLERANCE:
        raise RuntimeError(f"CCV action parity failed: {path}")
    expected_scalars = {
        "policy_candidate_reward": rewards[policy],
        "local_reward_oracle_candidate_reward": rewards[reward_oracle],
        "treatment_candidate_reward": rewards[treatment],
        "deployed_candidate_reward": rewards[deployed],
    }
    for key, expected in expected_scalars.items():
        if not np.isclose(float(row[key]), float(expected), rtol=0.0, atol=1e-6):
            raise RuntimeError(f"CCV {key} drifted: {path}")
    for provenance in ("raw_observation_path", "sidecar_path"):
        if not Path(str(row.get(provenance, ""))).is_file():
            raise RuntimeError(f"missing CCV provenance {provenance}: {path}")
    return scene, decision, policy, reward_oracle, treatment, deployed, applied, values, parity


def audit_collection(args):
    root = args.rollout_root.expanduser().resolve()
    manifest_path, manifest = load_manifest(
        args.checkpoint_manifest, "scalar_v3", args.train_seed
    )
    manifest_sha = common.sha256_file(manifest_path)
    checkpoint_sha = str(manifest["checkpoint_sha256"])
    treatment_path, treatment = common.load_treatment(args.treatment_manifest)
    treatment_sha = common.sha256_file(treatment_path)
    if treatment["treatment_id"] != args.treatment_id:
        raise RuntimeError("CCV treatment id/manifest drifted")
    treatment = dict(treatment, manifest_sha256=treatment_sha)
    contract = expected_contract(
        args.train_seed, checkpoint_sha, manifest_sha, treatment,
        args.expected_implementation_sha256,
    )
    targets = {str(row["scene_id"]): row for row in treatment["targets"]}
    expected_scenes, excluded_short = load_scenario_contract(
        args.scenario_file, args.maximum_scenarios
    )
    if set(targets) != expected_scenes:
        raise RuntimeError("CCV treatment/scenario coverage differs")
    if common.sha256_file(args.scenario_file) != treatment["scenario_file_sha256"]:
        raise RuntimeError("CCV scenario file drifted")
    report_paths, reports = load_reports(root)
    ledger_paths, ledgers = completed_scenes(root)
    unsuccessful = {
        scene for scene in expected_scenes
        if scene not in reports or not any(reports[scene])
    }
    if set(reports) != expected_scenes or ledgers != expected_scenes or unsuccessful:
        raise RuntimeError(
            f"CCV scenario coverage drifted: reports={len(reports)} "
            f"ledgers={len(ledgers)} expected={len(expected_scenes)} "
            f"failed={sorted(unsuccessful)}"
        )

    seen = set()
    steps = defaultdict(list)
    workers = set()
    config_shas = set()
    code_shas = set()
    resolved = defaultdict(set)
    interventions = []
    parity_max = 0.0
    target_reward_error = 0.0
    target_context_rows = []
    for path in record_paths(root, args.layout):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        result = validate_record(
            path, row, contract, args.expected_noise_namespace,
            checkpoint_sha, targets,
        )
        scene, decision, policy, reward_oracle, treatment_index, deployed, applied, values, parity = result
        if scene not in expected_scenes:
            raise RuntimeError(f"CCV record outside scenario contract: {path}")
        key = (scene, decision)
        if key in seen:
            raise RuntimeError(f"duplicate CCV logical frame: {key}")
        seen.add(key)
        steps[scene].append(decision)
        if applied:
            interventions.append({
                "scene_id": scene,
                "decision_step": decision,
                "policy_index": policy,
                "local_reward_oracle_index": reward_oracle,
                "treatment_index": treatment_index,
                "deployed_index": deployed,
            })
            target_reward_error = max(
                target_reward_error, float(row["target_reward_max_abs_error"])
            )
            target_context_rows.append({
                "scene_id": scene,
                "policy_index": policy,
                "treatment_index": treatment_index,
                "local_reward_oracle_index": reward_oracle,
                "treatment_reward": float(values["candidate_rewards"][treatment_index]),
                "policy_reward": float(values["candidate_rewards"][policy]),
            })
        parity_max = max(parity_max, parity)
        source = path if args.layout == "split" else Path(row["sidecar_path"])
        parts = [part for part in source.parts if part.startswith("split_")]
        if len(parts) != 1:
            raise RuntimeError(f"ambiguous CCV worker provenance: {path}")
        worker = parts[0]
        workers.add(worker)
        config_shas.add(str(row.get("config_sha256")))
        code_shas.add(str(row.get("code_sha")))
        resolved[worker].add(str(row.get("resolved_config_sha256")))

    expected_decisions = list(range(4, 12))
    bad = {
        scene: sorted(value) for scene, value in steps.items()
        if sorted(value) != expected_decisions
    }
    if bad or set(steps) != expected_scenes:
        raise RuntimeError(
            f"CCV record coverage failed: bad={bad} "
            f"missing={sorted(expected_scenes-set(steps))}"
        )
    if len(workers) != args.expected_workers:
        raise RuntimeError(f"expected {args.expected_workers} CCV workers, got {workers}")
    if len(config_shas) != 1 or "None" in config_shas:
        raise RuntimeError("CCV config provenance drifted")
    if code_shas != {args.expected_code_sha}:
        raise RuntimeError("CCV code provenance drifted")
    if any(len(value) != 1 for value in resolved.values()):
        raise RuntimeError("CCV resolved config drifted within worker")
    if len(interventions) != len(expected_scenes):
        raise RuntimeError("CCV requires exactly one target intervention per scene")

    outcome = None
    if args.metrics_csv is not None:
        metrics_path = args.metrics_csv.expanduser().resolve()
        outcomes = metrics_common.load_outcomes(metrics_path)
        if set(outcomes) != expected_scenes:
            raise RuntimeError("CCV metric scenario coverage drifted")
        outcome = {
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": common.sha256_file(metrics_path),
            "mean_score": float(np.mean([row["score"] for row in outcomes.values()])),
            "success_rate": float(np.mean([row["success"] for row in outcomes.values()])),
            "failed_scenarios": int(sum(not row["success"] for row in outcomes.values())),
            "success_definition": "NC==1_and_DAC==1",
        }
    return {
        "schema_version": common.SCHEMA_VERSION,
        "status": "PASS",
        "method": common.COLLECTION_AUDIT_METHOD,
        "layout": args.layout,
        "rollout_root": str(root),
        "treatment_id": args.treatment_id,
        "treatment_kind": treatment["treatment_kind"],
        "fixed_candidate_index": treatment["fixed_candidate_index"],
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": args.train_seed,
        "treatment_manifest": str(treatment_path),
        "treatment_manifest_sha256": treatment_sha,
        "intervention_count": len(interventions),
        "intervened_scene_count": len({row["scene_id"] for row in interventions}),
        "interventions": interventions,
        "target_context_rows": target_context_rows,
        "target_reward_max_abs_error": target_reward_error,
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
        "scenario_file_sha256": common.sha256_file(args.scenario_file),
        "num_scenarios": len(expected_scenes),
        "num_records": len(seen),
        "records_per_scene": 8,
        "num_workers": len(workers),
        "workers": sorted(workers),
        "runner_report_files": [str(path) for path in report_paths],
        "completed_scenario_ledgers": [str(path) for path in ledger_paths],
        "excluded_short_scenarios": excluded_short,
        "maximum_action_parity_error": parity_max,
        "resolved_config_sha256_by_worker": {
            key: next(iter(value)) for key, value in sorted(resolved.items())
        },
        "closed_loop_outcome": outcome,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(0,), required=True)
    parser.add_argument("--treatment-id", required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
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
    common.atomic_json(args.output, report)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
