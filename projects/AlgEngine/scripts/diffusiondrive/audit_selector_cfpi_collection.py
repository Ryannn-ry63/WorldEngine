#!/usr/bin/env python3
"""Fail-closed audit for one CFPI baseline or causal intervention collection."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import selector_root_cause_metrics as metrics_common
import cfpi_common as common
from audit_selector_root_cause_r1_collection import (
    completed_scenes,
    load_manifest,
    load_reports,
    load_scenario_contract,
)


def record_paths(root: Path, layout: str) -> list[Path]:
    suffix = "WE_output/openscene_format/diffusiondrive_cfpi_causal_records"
    if layout == "merged":
        paths = sorted((root / suffix).glob("*_cfpicausal.pkl"))
    else:
        paths = sorted(root.glob(f"split_*/{suffix}/*_cfpicausal.pkl"))
    if not paths:
        raise RuntimeError(f"no {layout} CFPI causal records under {root}")
    return paths


def expected_contract(args, checkpoint_sha: str, manifest_sha: str) -> dict:
    return {
        "schema_version": 9,
        "experiment": "diffusiondrive_selector_cfpi_causal_cache_v1",
        "source_policy": "trained_v3_selector",
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": 0,
        "expected_checkpoint_sha256": checkpoint_sha,
        "behavior_checkpoint_manifest_sha256": manifest_sha,
        "rollout_implementation_sha256": args.expected_implementation_sha256,
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
        "diagnostic_split": args.collection_id,
        "source_data_split": args.source_split,
        "react_type": "R",
        "training_data_consumed": False,
        "development_consumed": False,
        "test_consumed": False,
        "diagnostic_only": True,
        "action_score_timing": "pre_action",
        "intervention_mode": args.mode,
        "collection_id": args.collection_id,
        "treatment_manifest_sha256": args.treatment_sha256,
        "treatment_future_outcome_used": False,
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "method_training_performed": False,
        "inference_uses_reward_or_q": False,
    }


def array(path: Path, row: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(row.get(key))
    if value.shape != shape or not np.isfinite(value).all():
        raise RuntimeError(f"{path}: invalid {key}, expected finite {shape}")
    return value


def validate_record(
    path: Path,
    row: dict,
    contract: dict,
    namespace: str,
    checkpoint_sha: str,
    targets: dict[str, dict],
):
    if (
        row.get("schema_version") != 1
        or row.get("record_type") != "diffusiondrive_cfpi_causal_record"
    ):
        raise RuntimeError(f"CFPI causal record schema drifted: {path}")
    if row.get("selector_rollout_contract") != contract:
        raise RuntimeError(f"CFPI rollout contract drifted: {path}")
    if (
        row.get("checkpoint_sha256") != checkpoint_sha
        or row.get("candidate_noise_namespace") != namespace
    ):
        raise RuntimeError(f"CFPI checkpoint/noise provenance drifted: {path}")
    expected_flat = {
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": 0,
        "behavior_checkpoint_manifest_sha256": contract[
            "behavior_checkpoint_manifest_sha256"
        ],
        "diagnostic_split": contract["diagnostic_split"],
        "react_type": "R",
        "intervention_mode": contract["intervention_mode"],
        "target_manifest_sha256": contract["treatment_manifest_sha256"],
        "cfpi_collection_id": contract["collection_id"],
    }
    for key, expected in expected_flat.items():
        if row.get(key) != expected:
            raise RuntimeError(f"CFPI flat provenance {key} drifted: {path}")

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
    values = {
        key: array(path, row, key, shape) for key, shape in shapes.items()
    }
    valid = np.asarray(row.get("candidate_reward_valid_mask"), dtype=np.bool_)
    if valid.shape != (20,) or not valid.all():
        raise RuntimeError(f"CFPI candidate validity drifted: {path}")
    if tuple(row.get("reward_component_names", ())) != metrics_common.COMPONENT_NAMES:
        raise RuntimeError(f"CFPI reward component order drifted: {path}")

    scene = str(row["rollout_scene_id"])
    state = int(row["state_step"])
    decision = int(row["decision_step"])
    if int(row["worldengine_step"]) != state or decision != state + 1:
        raise RuntimeError(f"CFPI pre-action timing drifted: {path}")
    policy = int(row["policy_selected_index"])
    if (
        int(row["selected_index"]) != policy
        or int(np.asarray(row["selected_indices"]).item()) != policy
        or policy != int(np.argmax(values["current_logits"]))
    ):
        raise RuntimeError(f"CFPI policy index export drifted: {path}")
    rewards = r15.checked_rewards(values["candidate_rewards"])
    reward_oracle = r15.stable_oracle_index(rewards, policy)
    if int(row["local_reward_oracle_index"]) != reward_oracle:
        raise RuntimeError(f"CFPI local reward oracle drifted: {path}")

    target = targets.get(scene)
    target_step = None if target is None else int(target["decision_step"])
    if target is not None and decision <= target_step:
        baseline = target["prefix_records"][str(decision)]
        if common.sha256_file(baseline["path"]) != baseline["sha256"]:
            raise RuntimeError("CFPI baseline prefix artifact changed")
        with Path(baseline["path"]).open("rb") as stream:
            prefix = pickle.load(stream)
        for key in ("candidate_features", "candidate_trajectories_8", "route_bev_features",
                    "status_tokens", "ego_queries", "agents_queries", "reference_logits", "current_logits"):
            if r15.max_abs_error(values[key], prefix[key]) > common.ARRAY_TOLERANCE:
                raise RuntimeError(f"CFPI pre-intervention history/context drift: {key}")
        if decision < target_step and r15.max_abs_error(
                values["deployed_trajectory"], prefix["deployed_trajectory"]) > common.ACTION_TOLERANCE:
            raise RuntimeError("CFPI pre-intervention deployed-action drift")
    at_target = target is not None and decision == target_step
    expected_treatment = int(target["treatment_index"]) if at_target else policy
    expected_applied = bool(
        contract["intervention_mode"] == "one_shot_manifest" and at_target
    )
    if int(row["treatment_index"]) != expected_treatment:
        raise RuntimeError(f"CFPI treatment index drifted: {path}")
    if bool(row["intervention_applied"]) != expected_applied:
        raise RuntimeError(f"CFPI intervention schedule drifted: {path}")
    expected_deployed = expected_treatment if expected_applied else policy
    if int(row["deployed_index"]) != expected_deployed:
        raise RuntimeError(f"CFPI deployed index drifted: {path}")

    if at_target:
        errors = r15.target_context_errors(
            target,
            values["candidate_trajectories_8"],
            values["current_logits"],
        )
        if any(value > common.ARRAY_TOLERANCE for value in errors.values()):
            raise RuntimeError(f"CFPI target context drifted: {path}: {errors}")
        reward_error = r15.max_abs_error(target["candidate_rewards"], rewards)
        if not np.isclose(
            float(row["target_reward_max_abs_error"]),
            reward_error,
            rtol=0.0,
            atol=common.OUTCOME_TOLERANCE,
        ):
            raise RuntimeError(f"CFPI target local reward diagnostic drifted: {path}")
    elif row.get("target_reward_max_abs_error") is not None:
        raise RuntimeError(f"CFPI target diagnostic leaked outside target frame: {path}")

    parity = max(
        float(row["policy_action_parity_max_abs_error"]),
        float(row["deployed_action_parity_max_abs_error"]),
    )
    if parity > common.ACTION_TOLERANCE:
        raise RuntimeError(f"CFPI action parity failed: {path}")
    expected_scalars = {
        "policy_candidate_reward": rewards[policy],
        "local_reward_oracle_candidate_reward": rewards[reward_oracle],
        "treatment_candidate_reward": rewards[expected_treatment],
        "deployed_candidate_reward": rewards[expected_deployed],
    }
    for key, expected in expected_scalars.items():
        if not np.isclose(float(row[key]), float(expected), rtol=0.0, atol=1e-6):
            raise RuntimeError(f"CFPI {key} drifted: {path}")
    for provenance in ("raw_observation_path", "sidecar_path"):
        if not Path(str(row.get(provenance, ""))).is_file():
            raise RuntimeError(f"missing CFPI provenance {provenance}: {path}")
    return scene, decision, policy, expected_treatment, expected_deployed, expected_applied, parity


def audit_collection(args) -> dict:
    root = args.rollout_root.expanduser().resolve()
    manifest_path, manifest = load_manifest(
        args.checkpoint_manifest, "scalar_v3", 0
    )
    manifest_sha = common.sha256_file(manifest_path)
    checkpoint_sha = str(manifest["checkpoint_sha256"])
    treatment_path = None
    targets: dict[str, dict] = {}
    if args.mode == "observe_only":
        if args.treatment_manifest is not None or args.treatment_sha256 != "none":
            raise RuntimeError("CFPI baseline must not receive a treatment manifest")
    else:
        if args.treatment_manifest is None:
            raise RuntimeError("CFPI intervention is missing its treatment manifest")
        treatment_path, treatment = common.load_treatment(
            args.treatment_manifest, args.treatment_sha256
        )
        if treatment["collection_id"] != args.collection_id:
            raise RuntimeError("CFPI treatment collection id drifted")
        if treatment["source_split"] != args.source_split:
            raise RuntimeError("CFPI treatment source split drifted")
        targets = {str(row["scene_id"]): row for row in treatment["targets"]}
    contract = expected_contract(args, checkpoint_sha, manifest_sha)
    expected_scenes, excluded_short = load_scenario_contract(
        args.scenario_file, args.maximum_scenarios
    )
    if args.mode == "one_shot_manifest" and set(targets) != expected_scenes:
        raise RuntimeError("CFPI treatment/scenario coverage differs")

    report_paths, reports = load_reports(root)
    ledger_paths, ledgers = completed_scenes(root)
    unsuccessful = {
        scene for scene in expected_scenes
        if scene not in reports or not any(reports[scene])
    }
    if set(reports) != expected_scenes or ledgers != expected_scenes or unsuccessful:
        raise RuntimeError(
            "CFPI scenario coverage drifted: "
            f"reports={len(reports)} ledgers={len(ledgers)} "
            f"expected={len(expected_scenes)} failed={sorted(unsuccessful)}"
        )

    seen = set()
    steps = defaultdict(list)
    workers = set()
    config_shas = set()
    code_shas = set()
    resolved = defaultdict(set)
    interventions = []
    parity_max = 0.0
    for path in record_paths(root, args.layout):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        result = validate_record(
            path, row, contract, args.expected_noise_namespace,
            checkpoint_sha, targets,
        )
        scene, decision, policy, treatment, deployed, applied, parity = result
        if scene not in expected_scenes:
            raise RuntimeError(f"CFPI record outside scenario contract: {path}")
        key = (scene, decision)
        if key in seen:
            raise RuntimeError(f"duplicate CFPI logical frame: {key}")
        seen.add(key)
        steps[scene].append(decision)
        if applied:
            interventions.append({
                "scene_id": scene,
                "decision_step": decision,
                "policy_index": policy,
                "treatment_index": treatment,
                "deployed_index": deployed,
            })
        parity_max = max(parity_max, parity)
        provenance = path if args.layout == "split" else Path(row["sidecar_path"])
        parts = [part for part in provenance.parts if part.startswith("split_")]
        if len(parts) != 1:
            raise RuntimeError(f"ambiguous CFPI worker provenance: {path}")
        worker = parts[0]
        workers.add(worker)
        config_shas.add(str(row.get("config_sha256")))
        code_shas.add(str(row.get("code_sha")))
        resolved[worker].add(str(row.get("resolved_config_sha256")))

    bad = {
        scene: sorted(value) for scene, value in steps.items()
        if sorted(value) != list(common.DECISION_STEPS)
    }
    if bad or set(steps) != expected_scenes:
        raise RuntimeError(
            f"CFPI record coverage failed: bad={bad} "
            f"missing={sorted(expected_scenes-set(steps))}"
        )
    if len(workers) != args.expected_workers:
        raise RuntimeError(
            f"expected {args.expected_workers} CFPI workers, got {sorted(workers)}"
        )
    if len(config_shas) != 1 or "None" in config_shas:
        raise RuntimeError("CFPI config provenance drifted")
    if code_shas != {args.expected_code_sha}:
        raise RuntimeError("CFPI code provenance drifted")
    if any(len(value) != 1 for value in resolved.values()):
        raise RuntimeError("CFPI resolved config drifted within worker")
    expected_interventions = len(expected_scenes) if args.mode == "one_shot_manifest" else 0
    if len(interventions) != expected_interventions:
        raise RuntimeError(
            f"CFPI intervention count drifted: {len(interventions)} != {expected_interventions}"
        )

    outcome = None
    if args.metrics_csv is not None:
        metrics_path = args.metrics_csv.expanduser().resolve()
        outcomes = metrics_common.load_outcomes(metrics_path)
        if set(outcomes) != expected_scenes:
            raise RuntimeError("CFPI metric scenario coverage drifted")
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
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.COLLECTION_AUDIT_METHOD,
        "layout": args.layout,
        "rollout_root": str(root),
        "collection_id": args.collection_id,
        "source_split": args.source_split,
        "intervention_mode": args.mode,
        "treatment_manifest": None if treatment_path is None else str(treatment_path),
        "treatment_manifest_sha256": args.treatment_sha256,
        "intervention_count": len(interventions),
        "interventions": interventions,
        "behavior_policy_family": "scalar_v3",
        "behavior_policy_train_seed": 0,
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
        "records_per_scene": len(common.DECISION_STEPS),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--source-split", choices=common.ALLOWED_SPLITS, required=True)
    parser.add_argument(
        "--mode", choices=("observe_only", "one_shot_manifest"), required=True
    )
    parser.add_argument("--treatment-manifest", type=Path)
    parser.add_argument("--treatment-sha256", default="none")
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
