#!/usr/bin/env python3
"""Freeze outcome-blind v2 causal targets from reproducible V3 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import causal_branch_pilot_common as common
import oracle_r15_common as r15


def load_outcomes(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            scene = str(row["token"])
            if scene == "overall_average":
                continue
            collision = float(row["no_at_fault_collisions"])
            drivable = float(row["drivable_area_compliance"])
            violation = row.get("first_violation_step", "").strip()
            rows[scene] = {
                "score": float(row["score"]),
                "success": bool(collision >= 1.0 and drivable >= 1.0),
                "no_at_fault_collisions": collision,
                "drivable_area_compliance": drivable,
                "first_violation_step": None if not violation else int(float(violation)),
            }
    return rows


def load_baseline_audit(path: Path, expected_phase: str) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != common.COLLECTION_AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("diagnostic_split") != expected_phase
        or row.get("oracle_intervention_mode") != "observe_only"
        or int(row.get("behavior_policy_train_seed", -1)) != 0
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid causal-pilot {expected_phase} baseline: {path}")
    return path, row


def load_records(root: Path) -> dict[str, dict[int, tuple[Path, dict]]]:
    paths = sorted(
        (root / "WE_output/openscene_format/diffusiondrive_preaction_records").glob(
            "*_preaction.pkl"
        )
    )
    if not paths:
        raise RuntimeError(f"no merged pre-action records under {root}")
    result = defaultdict(dict)
    for path in paths:
        with path.open("rb") as stream:
            row = pickle.load(stream)
        scene, step = str(row["rollout_scene_id"]), int(row["decision_step"])
        if step in result[scene]:
            raise RuntimeError(f"duplicate baseline frame: {(scene, step)}")
        result[scene][step] = (path, row)
    return result


def validate_repeatable_frame(row_a: dict, row_b: dict) -> tuple[np.ndarray, np.ndarray, int]:
    candidates_a = np.asarray(row_a["candidate_trajectories_8"], dtype=np.float64)
    candidates_b = np.asarray(row_b["candidate_trajectories_8"], dtype=np.float64)
    logits_a = np.asarray(row_a["current_logits"], dtype=np.float64)
    logits_b = np.asarray(row_b["current_logits"], dtype=np.float64)
    errors = {
        "candidate_trajectories": r15.max_abs_error(candidates_a, candidates_b),
        "current_logits": r15.max_abs_error(logits_a, logits_b),
    }
    if any(value > common.ARRAY_TOLERANCE for value in errors.values()):
        raise ValueError(f"candidate/logit rerun drift: {errors}")
    policy_a, policy_b = int(row_a["policy_selected_index"]), int(row_b["policy_selected_index"])
    if policy_a != policy_b or policy_a != int(np.argmax(logits_a)):
        raise ValueError("policy action rerun drift")
    return candidates_a, logits_a, policy_a


def target_for_scene(
    scene_id: str,
    scenario: dict,
    frames_a: dict[int, tuple[Path, dict]],
    frames_b: dict[int, tuple[Path, dict]],
    outcome_a: dict,
    outcome_b: dict,
    headroom_threshold: float = common.HEADROOM_THRESHOLD,
    matched_ceiling: float = common.MATCHED_REWARD_CEILING,
    matched_max_absolute_error_m: float = common.MATCHED_ADE_ABSOLUTE_ERROR_MAX_M,
    matched_max_relative_error: float = common.MATCHED_ADE_RELATIVE_ERROR_MAX,
    frame_rejections: Counter | None = None,
):
    frame_rejections = Counter() if frame_rejections is None else frame_rejections
    if outcome_a["success"] != outcome_b["success"]:
        return None, "unstable_success"
    if (
        abs(float(outcome_a["score"]) - float(outcome_b["score"])) > 1e-6
        or outcome_a["first_violation_step"] != outcome_b["first_violation_step"]
    ):
        return None, "unstable_closed_loop_outcome"
    stratum = "solved" if outcome_a["success"] else "failed"
    if set(frames_a) != set(range(4, 12)) or set(frames_b) != set(range(4, 12)):
        return None, "incomplete_frames"
    violation_limit = None
    if stratum == "failed":
        violations = [
            value
            for value in (
                outcome_a["first_violation_step"],
                outcome_b["first_violation_step"],
            )
            if value is not None
        ]
        if len(violations) != 2:
            return None, "missing_failure_boundary"
        violation_limit = min(violations)

    for decision in range(4, 12):
        if violation_limit is not None and decision >= violation_limit:
            frame_rejections["at_or_after_failure_boundary"] += 1
            continue
        path_a, row_a = frames_a[decision]
        path_b, row_b = frames_b[decision]
        try:
            candidates, logits, policy = validate_repeatable_frame(row_a, row_b)
        except ValueError:
            frame_rejections["candidate_or_logit_rerun_drift"] += 1
            continue
        rewards_a = r15.checked_rewards(row_a["candidate_rewards"])
        rewards_b = r15.checked_rewards(row_b["candidate_rewards"])
        reward_error = r15.max_abs_error(rewards_a, rewards_b)
        if reward_error > common.ARRAY_TOLERANCE:
            frame_rejections["reward_rerun_drift"] += 1
            continue
        oracle = r15.stable_oracle_index(rewards_a, policy)
        if r15.stable_oracle_index(rewards_b, policy) != oracle:
            frame_rejections["oracle_index_rerun_drift"] += 1
            continue
        headroom_a = float(rewards_a[oracle] - rewards_a[policy])
        headroom_b = float(rewards_b[oracle] - rewards_b[policy])
        if oracle == policy or min(headroom_a, headroom_b) <= headroom_threshold:
            frame_rejections["insufficient_oracle_headroom"] += 1
            continue
        try:
            (
                matched,
                matched_distance,
                oracle_distance,
                absolute_match_error,
                relative_match_error,
            ) = common.choose_matched_index(
                candidates,
                rewards_a,
                rewards_b,
                policy,
                oracle,
                matched_ceiling,
                matched_max_absolute_error_m,
                matched_max_relative_error,
            )
        except ValueError as error:
            reason = str(error)
            if reason not in {
                "no_reward_nonimproving_candidate",
                "matched_control_outside_magnitude_caliper",
            }:
                reason = "invalid_matched_control"
            frame_rejections[reason] += 1
            continue
        metadata = common.scenario_metadata(scenario)
        oracle_perturbation = common.trajectory_perturbation_diagnostics(
            candidates[oracle], candidates[policy]
        )
        matched_perturbation = common.trajectory_perturbation_diagnostics(
            candidates[matched], candidates[policy]
        )
        return {
            "scene_id": scene_id,
            "train_seed": 0,
            "outcome_stratum": stratum,
            "decision_step": decision,
            "state_step": int(row_a["state_step"]),
            "policy_index": policy,
            "oracle_index": oracle,
            "matched_index": matched,
            "oracle_headroom_a": headroom_a,
            "oracle_headroom_b": headroom_b,
            "matched_headroom_a": float(rewards_a[matched] - rewards_a[policy]),
            "matched_headroom_b": float(rewards_b[matched] - rewards_b[policy]),
            "reward_rerun_max_abs_error": reward_error,
            "match_metric": "policy_relative_xy_ade_magnitude",
            "oracle_policy_ade": oracle_distance,
            "matched_policy_ade": matched_distance,
            "ade_match_error": absolute_match_error,
            "relative_ade_match_error": relative_match_error,
            "oracle_perturbation_diagnostics": oracle_perturbation,
            "matched_perturbation_diagnostics": matched_perturbation,
            "oracle_is_top_in_baseline_b": True,
            "baseline_score_a": float(outcome_a["score"]),
            "baseline_score_b": float(outcome_b["score"]),
            "baseline_score_mean": float(
                (outcome_a["score"] + outcome_b["score"]) / 2.0
            ),
            "baseline_success": bool(outcome_a["success"]),
            "first_violation_step_a": outcome_a["first_violation_step"],
            "first_violation_step_b": outcome_b["first_violation_step"],
            "candidate_trajectories_8": candidates.astype(np.float32).tolist(),
            "candidate_rewards": rewards_a.astype(np.float32).tolist(),
            "baseline_b_candidate_rewards": rewards_b.astype(np.float32).tolist(),
            "current_logits": logits.astype(np.float32).tolist(),
            "origin_log": metadata["origin_log"],
            "origin_token": metadata["origin_token"],
            "source_kind": metadata["source_kind"],
            "pairing_method": metadata["pairing_method"],
            "source_record_a": str(path_a.resolve()),
            "source_record_a_sha256": r15.sha256_file(path_a),
            "source_record_b": str(path_b.resolve()),
            "source_record_b_sha256": r15.sha256_file(path_b),
        }, "eligible"
    return None, "no_eligible_preaction_frame"

def write_scenarios(path: Path, source: dict, targets: list[dict]):
    scene_ids = [row["scene_id"] for row in targets]
    common.write_pickle(path, {scene: source[scene] for scene in scene_ids})


def build(args) -> Path:
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root.parent
    observed_intervention_audits = [
        run_root / phase / mode / "collection_audit.json"
        for phase in ("intervention_smoke8", "intervention_target")
        for mode in ("one_shot_oracle", "one_shot_matched")
        if (run_root / phase / mode / "collection_audit.json").is_file()
    ]
    if observed_intervention_audits:
        raise RuntimeError(
            "refusing post-outcome target design: "
            + ", ".join(str(path) for path in observed_intervention_audits)
        )

    protocol_path = args.protocol.expanduser().resolve()
    if not protocol_path.is_file():
        raise RuntimeError(f"missing frozen causal protocol: {protocol_path}")
    baseline_a_path, baseline_a = load_baseline_audit(args.baseline_audit_a, "baseline_a")
    baseline_b_path, baseline_b = load_baseline_audit(args.baseline_audit_b, "baseline_b")
    source_path = args.source_scenarios.expanduser().resolve()
    source_sha = r15.sha256_file(source_path)
    if {
        baseline_a["scenario_file_sha256"],
        baseline_b["scenario_file_sha256"],
    } != {source_sha}:
        raise RuntimeError("baseline/source scenario SHA256 drifted")
    if (
        baseline_a["checkpoint_sha256"] != baseline_b["checkpoint_sha256"]
        or baseline_a["candidate_noise_namespace"]
        != baseline_b["candidate_noise_namespace"]
        or baseline_a["rollout_implementation_sha256"]
        != baseline_b["rollout_implementation_sha256"]
        or baseline_a["closed_loop_outcome"]["metrics_csv_sha256"]
        != baseline_b["closed_loop_outcome"]["metrics_csv_sha256"]
    ):
        raise RuntimeError("A/B baseline provenance or closed-loop outcome differs")

    source = common.load_pickle(source_path)
    if not isinstance(source, dict) or len(source) != 147:
        raise RuntimeError("pilot source must contain exactly 147 scenes")
    outcomes_a = load_outcomes(Path(baseline_a["closed_loop_outcome"]["metrics_csv"]))
    outcomes_b = load_outcomes(Path(baseline_b["closed_loop_outcome"]["metrics_csv"]))
    if set(outcomes_a) != set(source) or set(outcomes_b) != set(source):
        raise RuntimeError("baseline outcome coverage differs from source")
    records_a = load_records(Path(baseline_a["rollout_root"]))
    records_b = load_records(Path(baseline_b["rollout_root"]))

    reasons = Counter()
    frame_rejections = Counter()
    eligible = {"failed": [], "solved": []}
    for scene_id in sorted(source):
        target, reason = target_for_scene(
            scene_id,
            source[scene_id],
            records_a.get(scene_id, {}),
            records_b.get(scene_id, {}),
            outcomes_a[scene_id],
            outcomes_b[scene_id],
            args.headroom_threshold,
            args.matched_reward_ceiling,
            args.matched_max_absolute_error_m,
            args.matched_max_relative_error,
            frame_rejections,
        )
        reasons[reason] += 1
        if target is not None:
            eligible[target["outcome_stratum"]].append(target)

    selected = {}
    for stratum in ("failed", "solved"):
        selected[stratum] = common.stratified_scene_cap(
            eligible[stratum],
            args.maximum_per_stratum,
            args.maximum_per_origin_log,
            salt=f"formal-{stratum}",
        )
    formal_targets = selected["failed"] + selected["solved"]
    smoke_targets = (
        common.smoke_subset(selected["failed"], 4, "smoke-failed")
        + common.smoke_subset(selected["solved"], 4, "smoke-solved")
    )

    manifest_path = output_root / "target_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite immutable targets: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    formal_path = output_root / "formal_target_scenarios.pkl"
    smoke_path = output_root / "smoke8_target_scenarios.pkl"
    write_scenarios(formal_path, source, formal_targets)
    write_scenarios(smoke_path, source, smoke_targets)

    def numeric_summary(rows, key):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not values.size:
            return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }

    def stratum_summary(rows):
        return {
            "count": len(rows),
            "origin_log_count": len({row["origin_log"] for row in rows}),
            "source_counts": dict(sorted(Counter(row["source_kind"] for row in rows).items())),
            "pairing_counts": dict(sorted(Counter(row["pairing_method"] for row in rows).items())),
            "absolute_ade_match_error_m": numeric_summary(rows, "ade_match_error"),
            "relative_ade_match_error": numeric_summary(rows, "relative_ade_match_error"),
        }

    eligible_summary = {key: stratum_summary(value) for key, value in eligible.items()}
    selected_summary = {key: stratum_summary(value) for key, value in selected.items()}
    coverage_pass = all(
        selected_summary[key]["count"] >= 8
        and selected_summary[key]["origin_log_count"] >= 6
        for key in ("failed", "solved")
    )

    payload = {
        "schema_version": common.TARGET_SCHEMA_VERSION,
        "design_version": "causal_branch_pilot_v2",
        "status": "PASS",
        "method": common.TARGET_METHOD,
        "scientific_role": "frozen_train_only_local_action_causal_targets",
        "intervention_outcomes_observed_before_freeze": False,
        "primary_estimand": "one_step_local_oracle_minus_no_intervention_v3_baseline",
        "specificity_estimand": "one_step_local_oracle_minus_magnitude_matched_nonimproving_control",
        "selection_rule": "stable_A_B_context_earliest_pre_failure_or_solved_frame_after_all_calipers",
        "headroom_threshold_strictly_greater_than": args.headroom_threshold,
        "matched_reward_ceiling_at_most": args.matched_reward_ceiling,
        "matched_control_rule": "nonimproving_candidate_closest_in_policy_relative_xy_ADE_subject_to_fixed_calipers",
        "matching_contract": {
            "metric": "policy_relative_xy_ade_magnitude",
            "interpretation": "treatment_magnitude_match_not_directional_geometry_match",
            "maximum_absolute_ade_error_m": args.matched_max_absolute_error_m,
            "maximum_relative_ade_error": args.matched_max_relative_error,
            "relative_error_denominator": "max(abs(oracle_policy_ade),1e-6_m)",
            "trajectory_shape_and_yaw_role": "diagnostics_only",
        },
        "maximum_per_stratum": args.maximum_per_stratum,
        "maximum_per_origin_log_per_stratum": args.maximum_per_origin_log,
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": r15.sha256_file(protocol_path),
        "target_builder_file": str(Path(__file__).resolve()),
        "target_builder_sha256": r15.sha256_file(Path(__file__).resolve()),
        "common_helpers_file": str(Path(common.__file__).resolve()),
        "common_helpers_sha256": r15.sha256_file(Path(common.__file__).resolve()),
        "source_scenario_file": str(source_path),
        "source_scenario_file_sha256": source_sha,
        "baseline_audit_a": str(baseline_a_path),
        "baseline_audit_a_sha256": r15.sha256_file(baseline_a_path),
        "baseline_audit_b": str(baseline_b_path),
        "baseline_audit_b_sha256": r15.sha256_file(baseline_b_path),
        "baseline_metrics_sha256": baseline_a["closed_loop_outcome"]["metrics_csv_sha256"],
        "baseline_rollout_implementation_sha256": baseline_a["rollout_implementation_sha256"],
        "checkpoint_sha256": baseline_a["checkpoint_sha256"],
        "candidate_noise_namespace": baseline_a["candidate_noise_namespace"],
        "scene_eligibility_reasons": dict(sorted(reasons.items())),
        "frame_eligibility_rejections": dict(sorted(frame_rejections.items())),
        "eligible": eligible_summary,
        "selected": selected_summary,
        "pre_intervention_decision": "PROCEED_TO_INTERVENTIONS" if coverage_pass else "INSUFFICIENT_CAUSAL_BRANCH_COVERAGE",
        "coverage_gate": {
            "minimum_targets_per_stratum": 8,
            "minimum_origin_logs_per_stratum": 6,
        },
        "formal_scenario_file": str(formal_path),
        "formal_scenario_file_sha256": r15.sha256_file(formal_path),
        "formal_scenario_count": len(formal_targets),
        "smoke_scenario_file": str(smoke_path),
        "smoke_scenario_file_sha256": r15.sha256_file(smoke_path),
        "smoke_scenario_count": len(smoke_targets),
        "smoke_scene_ids": [row["scene_id"] for row in smoke_targets],
        "target_count": len(formal_targets),
        "targets": formal_targets,
    }
    common.write_json(manifest_path, payload)
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit-a", type=Path, required=True)
    parser.add_argument("--baseline-audit-b", type=Path, required=True)
    parser.add_argument("--source-scenarios", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "DIFFUSIONDRIVE_SELECTOR_CAUSAL_BRANCH_PILOT_PROTOCOL_20260903.md",
    )
    parser.add_argument("--headroom-threshold", type=float, default=common.HEADROOM_THRESHOLD)
    parser.add_argument(
        "--matched-reward-ceiling", type=float, default=common.MATCHED_REWARD_CEILING
    )
    parser.add_argument(
        "--matched-max-absolute-error-m",
        type=float,
        default=common.MATCHED_ADE_ABSOLUTE_ERROR_MAX_M,
    )
    parser.add_argument(
        "--matched-max-relative-error",
        type=float,
        default=common.MATCHED_ADE_RELATIVE_ERROR_MAX,
    )
    parser.add_argument("--maximum-per-stratum", type=int, default=24)
    parser.add_argument("--maximum-per-origin-log", type=int, default=2)
    args = parser.parse_args()
    output = build(args)
    payload = json.loads(output.read_text())
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "selected": payload["selected"],
        "smoke_scenario_count": payload["smoke_scenario_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()


