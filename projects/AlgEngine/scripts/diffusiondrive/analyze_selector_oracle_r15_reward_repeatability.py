#!/usr/bin/env python3
"""Audit pre-action scalar-PDM repeatability on identical initial candidate sets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import selector_root_cause_metrics as metrics_common


AUDIT_METHOD = "diffusiondrive_selector_oracle_r15_collection_audit_v1"
REPORT_METHOD = "diffusiondrive_selector_oracle_r15_reward_repeatability_v1"


def load_decision4_records(audit_path: Path) -> tuple[Path, dict, dict[str, dict]]:
    audit_path = audit_path.expanduser().resolve()
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("method") != AUDIT_METHOD
        or audit.get("layout") != "merged"
        or audit.get("diagnostic_split") != "baseline_cl_dev58"
        or audit.get("oracle_intervention_mode") != "observe_only"
        or not audit.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid R1.5 baseline audit: {audit_path}")
    root = Path(audit["rollout_root"]).expanduser().resolve()
    record_root = root / "WE_output/openscene_format/diffusiondrive_preaction_records"
    records = {}
    for path in sorted(record_root.glob("*_preaction.pkl")):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        if int(row.get("decision_step", -1)) != 4:
            continue
        scene = str(row["rollout_scene_id"])
        if scene in records:
            raise RuntimeError(f"duplicate decision-4 record: {scene}")
        records[scene] = row
    if len(records) != int(audit["num_scenarios"]):
        raise RuntimeError(
            f"decision-4 coverage drifted: {len(records)} != {audit['num_scenarios']}"
        )
    return audit_path, audit, records


def compare_records(left: dict, right: dict) -> dict:
    candidate_error = r15.max_abs_error(
        left["candidate_trajectories_8"], right["candidate_trajectories_8"]
    )
    reference_logit_error = r15.max_abs_error(
        left["reference_logits"], right["reference_logits"]
    )
    reward_error = r15.max_abs_error(
        left["candidate_rewards"], right["candidate_rewards"]
    )
    left_components = np.asarray(
        left["candidate_reward_components"], dtype=np.float64
    )
    right_components = np.asarray(
        right["candidate_reward_components"], dtype=np.float64
    )
    if left_components.shape != (20, len(metrics_common.COMPONENT_NAMES)):
        raise RuntimeError(f"invalid left component shape: {left_components.shape}")
    if right_components.shape != left_components.shape:
        raise RuntimeError(f"component shape mismatch: {right_components.shape}")
    component_errors = np.max(
        np.abs(left_components - right_components), axis=0
    )
    left_rewards = r15.checked_rewards(left["candidate_rewards"])
    right_rewards = r15.checked_rewards(right["candidate_rewards"])
    return {
        "candidate_trajectories_max_abs_error": candidate_error,
        "reference_logits_max_abs_error": reference_logit_error,
        "candidate_rewards_max_abs_error": reward_error,
        "raw_oracle_left": int(np.argmax(left_rewards)),
        "raw_oracle_right": int(np.argmax(right_rewards)),
        "raw_oracle_agreement": bool(
            int(np.argmax(left_rewards)) == int(np.argmax(right_rewards))
        ),
        "component_max_abs_error": {
            name: float(component_errors[index])
            for index, name in enumerate(metrics_common.COMPONENT_NAMES)
        },
    }


def analyze(audit_paths: list[Path]) -> dict:
    if len(audit_paths) != 2:
        raise RuntimeError("repeatability audit requires exactly two baseline audits")
    loaded = [load_decision4_records(path) for path in audit_paths]
    seeds = [int(item[1]["behavior_policy_train_seed"]) for item in loaded]
    if set(seeds) != {0, 1}:
        raise RuntimeError(f"expected train seeds 0 and 1, got {seeds}")
    by_seed = {seed: item for seed, item in zip(seeds, loaded)}
    left_path, left_audit, left_records = by_seed[0]
    right_path, right_audit, right_records = by_seed[1]
    if set(left_records) != set(right_records):
        raise RuntimeError("paired decision-4 scene coverage drifted")

    per_scene = []
    for scene in sorted(left_records):
        comparison = compare_records(left_records[scene], right_records[scene])
        comparison["scene_id"] = scene
        per_scene.append(comparison)

    tolerance = r15.ARRAY_TOLERANCE
    component_maxima = {
        name: max(row["component_max_abs_error"][name] for row in per_scene)
        for name in metrics_common.COMPONENT_NAMES
    }
    component_drift_counts = {
        name: sum(
            row["component_max_abs_error"][name] > tolerance
            for row in per_scene
        )
        for name in metrics_common.COMPONENT_NAMES
    }
    candidate_max = max(
        row["candidate_trajectories_max_abs_error"] for row in per_scene
    )
    reference_max = max(
        row["reference_logits_max_abs_error"] for row in per_scene
    )
    reward_errors = np.asarray(
        [row["candidate_rewards_max_abs_error"] for row in per_scene],
        dtype=np.float64,
    )
    drift_components = [
        name for name, value in component_drift_counts.items() if value
    ]
    if candidate_max > tolerance or reference_max > tolerance:
        diagnosis = "INPUT_CONTEXT_NOT_REPEATABLE"
    elif drift_components == ["ego_progress"]:
        diagnosis = "PAIRWISE_PROGRESS_NORMALIZATION_NOT_REPEATABLE"
    elif drift_components:
        diagnosis = "MULTIPLE_REWARD_COMPONENTS_NOT_REPEATABLE"
    else:
        diagnosis = "REWARD_REPEATABLE_AT_TOLERANCE"

    return {
        "schema_version": 1,
        "status": "PASS",
        "method": REPORT_METHOD,
        "diagnosis": diagnosis,
        "comparison_role": (
            "paired_initial_state_repeatability_probe_not_independent_policy_eval"
        ),
        "decision_step": 4,
        "array_tolerance": tolerance,
        "paired_scene_count": len(per_scene),
        "candidate_trajectories_exact_scene_count": int(
            sum(row["candidate_trajectories_max_abs_error"] <= tolerance for row in per_scene)
        ),
        "candidate_trajectories_global_max_abs_error": candidate_max,
        "reference_logits_exact_scene_count": int(
            sum(row["reference_logits_max_abs_error"] <= tolerance for row in per_scene)
        ),
        "reference_logits_global_max_abs_error": reference_max,
        "reward_drift_scene_count_gt_1e_5": int(sum(reward_errors > tolerance)),
        "reward_drift_scene_count_gt_0p02": int(
            sum(reward_errors > r15.HEADROOM_THRESHOLD)
        ),
        "reward_max_abs_error": float(np.max(reward_errors)),
        "reward_median_max_abs_error": float(np.median(reward_errors)),
        "raw_oracle_agreement_scene_count": int(
            sum(row["raw_oracle_agreement"] for row in per_scene)
        ),
        "raw_oracle_disagreement_scene_count": int(
            sum(not row["raw_oracle_agreement"] for row in per_scene)
        ),
        "component_global_max_abs_error": component_maxima,
        "component_drift_scene_count_gt_1e_5": component_drift_counts,
        "scientific_implication": (
            "freeze_the_preregistered_treatment_index; use_current_reward_only_as_"
            "a_stability_sensitivity_and_require_current_treatment_headroom_gt_0p02_"
            "for_the_primary_causal_gate"
        ),
        "baseline_audits": {
            "0": {"path": str(left_path), "sha256": r15.sha256_file(left_path)},
            "1": {"path": str(right_path), "sha256": r15.sha256_file(right_path)},
        },
        "baseline_code_sha": {
            "0": left_audit["code_sha"],
            "1": right_audit["code_sha"],
        },
        "per_scene": per_scene,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.baseline_audit)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "diagnosis": report["diagnosis"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
