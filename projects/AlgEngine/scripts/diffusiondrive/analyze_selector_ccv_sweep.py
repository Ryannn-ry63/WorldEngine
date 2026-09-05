#!/usr/bin/env python3
"""Analyze the complete 33x20 candidate causal-value response surface."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import ccv_sweep_common as common
import selector_root_cause_metrics as metrics_common


FORMAL_POLICY_BASELINE_CONTINUOUS_EQUIVALENCE = 1e-3
ANALYSIS_NOTE_MARKER = (
    "FROZEN AFTER FORMAL COLLECTION AND BEFORE CAUSAL GAP ANALYSIS"
)


def load_arm(master, run_root, index, sentinel):
    reference = master["treatment_manifests"]["arms"][str(index)]
    treatment_path, treatment = common.load_treatment(
        reference["path"], reference["sha256"]
    )
    if (
        treatment["treatment_id"] != f"arm_{index:02d}"
        or treatment["treatment_kind"] != "fixed_candidate_index"
        or int(treatment["fixed_candidate_index"]) != index
        or any(int(row["treatment_index"]) != index for row in treatment["targets"])
    ):
        raise RuntimeError(f"CCV arm {index} treatment contract drifted")
    audit_path = run_root / "collections" / f"arm_{index:02d}" / "collection_audit.json"
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("method") != common.COLLECTION_AUDIT_METHOD
        or audit.get("layout") != "merged"
        or audit.get("treatment_id") != f"arm_{index:02d}"
        or audit.get("fixed_candidate_index") != index
        or audit.get("treatment_manifest_sha256") != reference["sha256"]
        or int(audit.get("num_scenarios", -1)) != common.NUM_TARGETS
        or int(audit.get("intervention_count", -1)) != common.NUM_TARGETS
        or float(audit.get("maximum_action_parity_error", float("inf")))
        > common.ACTION_TOLERANCE
        or not audit.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid CCV arm audit: {audit_path}")
    expected_provenance = {
        "rollout_implementation_sha256": sentinel["rollout_implementation_sha256"],
        "checkpoint_sha256": sentinel["checkpoint_sha256"],
        "checkpoint_manifest_sha256": sentinel["checkpoint_manifest_sha256"],
        "candidate_noise_namespace": sentinel["candidate_noise_namespace"],
    }
    drifted = {
        key: {"actual": audit.get(key), "expected": expected}
        for key, expected in expected_provenance.items()
        if audit.get(key) != expected
    }
    if drifted:
        raise RuntimeError(f"CCV arm {index} provenance differs from sentinel: {drifted}")
    metrics_path = Path(audit["closed_loop_outcome"]["metrics_csv"]).resolve()
    if common.sha256_file(metrics_path) != audit["closed_loop_outcome"]["metrics_csv_sha256"]:
        raise RuntimeError(f"CCV arm metrics drifted: {index}")
    return audit_path, audit, metrics_common.load_outcomes(metrics_path)


def reward_groups(rewards, tolerance=1e-8):
    groups = []
    for index in np.argsort(rewards, kind="mergesort"):
        value = float(rewards[index])
        if not groups or abs(value - groups[-1][0]) > tolerance:
            groups.append((value, [int(index)]))
        else:
            groups[-1][1].append(int(index))
    return groups


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def summarize_rows(rows, seed):
    keys = (
        "total_regret", "reward_ranking_horizon_gap", "reward_tie_gap",
        "selector_gap", "reward_q_spearman", "reward_q_kendall_tau_b",
        "maximum_same_reward_q_span", "policy_relative_ade_q_delta_pearson",
        "knn_spearman",
    )
    result = {
        "count": len(rows),
        "origin_log_count": len({row["origin_log"] for row in rows}),
    }
    for offset, key in enumerate(keys):
        result[f"mean_{key}"] = mean_or_none(row[key] for row in rows)
        result[f"{key}_cluster_bootstrap"] = common.cluster_bootstrap(
            rows, key, seed=seed + offset
        )
    for key in (
        "reward_top1_recall_causal_best", "reward_top3_recall_causal_best",
        "reward_top5_recall_causal_best", "reward_top1_recall_any_rescue",
        "reward_top3_recall_any_rescue", "reward_top5_recall_any_rescue",
        "knn_top3_recall_best",
    ):
        present = [row[key] for row in rows if row[key] is not None]
        result[key] = None if not present else float(np.mean(present))
    candidates = sum(row["nonimproving_candidate_count"] for row in rows)
    rescues = sum(row["nonimproving_rescue_count"] for row in rows)
    result["nonimproving_candidate_count"] = candidates
    result["nonimproving_rescue_count"] = rescues
    result["nonimproving_rescue_rate"] = None if not candidates else rescues / candidates
    result["scenes_with_nonimproving_rescue"] = sum(
        row["nonimproving_rescue_count"] > 0 for row in rows
    )
    result["candidate_harm_count"] = sum(row["candidate_harm_count"] for row in rows)
    result["scenes_with_candidate_harm"] = sum(row["candidate_harm_count"] > 0 for row in rows)
    return result


def decide(gates, geometry_gate):
    ranking = gates["reward_ranking_horizon_gap"]["pass"]
    tie = gates["reward_tie_gap"]["pass"]
    selector = gates["selector_gap"]["pass"]
    basin = geometry_gate["pass"]
    if ranking:
        decision = (
            "AUTHORIZE_BASIN_AWARE_LONG_HORIZON_CAUSAL_ADVANTAGE"
            if basin else "AUTHORIZE_LONG_HORIZON_CAUSAL_ADVANTAGE"
        )
        method = "long_horizon_branch_return_with_optional_reward_coarse_prior"
    elif tie:
        decision = "AUTHORIZE_SCALAR_COARSE_DIFFUSION_SET_TIE_BREAKER"
        method = "scalar_reward_coarse_ranking_plus_diffusion_candidate_tie_breaker"
    elif selector:
        decision = "RETAIN_SCALAR_REWARD_AND_IMPROVE_SELECTOR"
        method = "scalar_reward_selector_representation_or_optimization"
    elif any(gate["point_pass"] for gate in gates.values()):
        decision = "EXPAND_CCV_TARGETS_OR_FROZEN_NOISE"
        method = "no_training_until_causal_gap_has_log_cluster_support"
    else:
        decision = "STOP_CCV_METHOD_ROUTE"
        method = "no_actionable_candidate_causal_gap"
    return decision, method


def analyze(args):
    master_path, master = common.load_master(args.master_manifest)
    run_root = args.run_root.expanduser().resolve()
    sentinel_path = args.sentinel_gate.expanduser().resolve()
    sentinel = json.loads(sentinel_path.read_text())
    if (
        sentinel.get("status") != "PASS"
        or sentinel.get("method") != common.SENTINEL_METHOD
        or sentinel.get("decision") != "AUTHORIZE_CCV_20_ARM_SWEEP"
        or sentinel.get("master_manifest_sha256") != common.sha256_file(master_path)
    ):
        raise RuntimeError("CCV analysis requires a passing sentinel gate")

    amendment_path = Path(sentinel.get("sentinel_amendment", "")).resolve()
    if (
        not amendment_path.is_file()
        or common.sha256_file(amendment_path)
        != sentinel.get("sentinel_amendment_sha256")
        or "FROZEN AFTER SENTINEL AND BEFORE FORMAL ARMS"
        not in amendment_path.read_text()
    ):
        raise RuntimeError("CCV sentinel amendment provenance drifted")
    analysis_note_path = args.analysis_note.expanduser().resolve()
    if (
        not analysis_note_path.is_file()
        or ANALYSIS_NOTE_MARKER not in analysis_note_path.read_text()
    ):
        raise RuntimeError("CCV analysis note provenance drifted")
    source_target = json.loads(Path(master["source_target_manifest"]).read_text())
    baseline_audit_path = Path(source_target["baseline_audit_a"]).resolve()
    if common.sha256_file(baseline_audit_path) != source_target["baseline_audit_a_sha256"]:
        raise RuntimeError("CCV source baseline audit provenance drifted")
    baseline_audit = json.loads(baseline_audit_path.read_text())
    baseline_metrics_path = Path(
        baseline_audit["closed_loop_outcome"]["metrics_csv"]
    ).resolve()
    if common.sha256_file(baseline_metrics_path) != source_target["baseline_metrics_sha256"]:
        raise RuntimeError("CCV source baseline metrics provenance drifted")
    baseline_outcomes = metrics_common.load_outcomes(baseline_metrics_path)

    targets = {str(row["scene_id"]): row for row in master["targets"]}
    if not set(targets) <= set(baseline_outcomes):
        raise RuntimeError("CCV source baseline outcome coverage drifted")
    q_score = {scene: np.empty(20, dtype=np.float64) for scene in targets}
    q_progress = {scene: np.empty(20, dtype=np.float64) for scene in targets}
    q_nc = {scene: np.empty(20, dtype=np.float64) for scene in targets}
    q_dac = {scene: np.empty(20, dtype=np.float64) for scene in targets}
    q_success = {scene: np.empty(20, dtype=np.bool_) for scene in targets}
    arm_refs = {}
    arm_code_shas = set()
    for index in range(common.NUM_CANDIDATES):
        audit_path, audit, outcomes = load_arm(master, run_root, index, sentinel)
        arm_code_shas.add(str(audit["code_sha"]))
        if set(outcomes) != set(targets):
            raise RuntimeError(f"CCV arm {index} outcome coverage drifted")
        for scene, outcome in outcomes.items():
            q_score[scene][index] = float(outcome["score"])
            q_progress[scene][index] = float(outcome["ego_progress"])
            q_nc[scene][index] = float(outcome["no_at_fault_collisions"])
            q_dac[scene][index] = float(outcome["drivable_area_compliance"])
            q_success[scene][index] = bool(outcome["success"])
        arm_refs[str(index)] = {
            "audit": str(audit_path),
            "audit_sha256": common.sha256_file(audit_path),
            "metrics_csv": audit["closed_loop_outcome"]["metrics_csv"],
            "metrics_csv_sha256": audit["closed_loop_outcome"]["metrics_csv_sha256"],
        }

    if len(arm_code_shas) != 1:
        raise RuntimeError(f"CCV formal arm code provenance drifted: {arm_code_shas}")

    scene_rows = []
    matrix_rows = []
    for scene in sorted(targets):
        target = targets[scene]
        rewards = common.checked_array(target["candidate_rewards"], (20,), "local rewards")
        components = common.checked_array(
            target["candidate_reward_components"], (20, 6), "reward components"
        )
        trajectories = common.checked_array(
            target["candidate_trajectories_8"], (20, 8, 3), "candidate trajectories"
        )
        values = q_score[scene]
        successes = q_success[scene]
        policy = int(target["policy_index"])
        reward_oracle = int(target["oracle_index"])
        maximum_reward = float(rewards.max())
        reward_max_set = np.flatnonzero(
            np.isclose(rewards, maximum_reward, atol=1e-8, rtol=0.0)
        )
        q_policy = float(values[policy])
        baseline = baseline_outcomes[scene]
        baseline_score_error = abs(q_policy - float(baseline["score"]))
        baseline_progress_error = abs(
            float(q_progress[scene][policy]) - float(baseline["ego_progress"])
        )
        baseline_nc_equal = (
            float(q_nc[scene][policy])
            == float(baseline["no_at_fault_collisions"])
        )
        baseline_dac_equal = (
            float(q_dac[scene][policy])
            == float(baseline["drivable_area_compliance"])
        )
        baseline_success_equal = bool(successes[policy]) == bool(
            baseline["success"]
        )
        if (
            baseline_score_error
            > FORMAL_POLICY_BASELINE_CONTINUOUS_EQUIVALENCE
            or baseline_progress_error
            > FORMAL_POLICY_BASELINE_CONTINUOUS_EQUIVALENCE
            or not baseline_nc_equal
            or not baseline_dac_equal
            or not baseline_success_equal
        ):
            raise RuntimeError(f"CCV policy branch did not reproduce baseline: {scene}")
        q_reward_oracle = float(values[reward_oracle])
        q_reward_set = float(values[reward_max_set].max())
        q_star = float(values.max())
        total = q_star - q_policy
        ranking_gap = q_star - q_reward_set
        tie_gap = q_reward_set - q_reward_oracle
        selector_gap = q_reward_oracle - q_policy
        if not np.isclose(
            total, ranking_gap + tie_gap + selector_gap,
            rtol=0.0, atol=1e-8,
        ):
            raise RuntimeError(f"CCV regret decomposition failed: {scene}")

        best = set(np.flatnonzero(
            np.isclose(values, q_star, atol=common.OUTCOME_TOLERANCE, rtol=0.0)
        ).tolist())
        reward_order = np.argsort(-rewards, kind="mergesort")
        baseline_failed = not bool(successes[policy])
        rescue = set(np.flatnonzero(successes).tolist()) if baseline_failed else set()
        top_recall_best = {}
        top_recall_rescue = {}
        for k in (1, 3, 5):
            top = set(reward_order[:k].tolist())
            top_recall_best[k] = bool(top.intersection(best))
            top_recall_rescue[k] = None if not rescue else bool(top.intersection(rescue))

        groups = reward_groups(rewards)
        same_reward_spans = [
            float(values[indices].max() - values[indices].min())
            for _, indices in groups if len(indices) >= 2
        ]
        maximum_same_reward_q_span = max(same_reward_spans, default=0.0)
        policy_distances = np.asarray([
            common.trajectory_distance(trajectory, trajectories[policy])
            for trajectory in trajectories
        ])
        q_deltas = values - q_policy
        geometry = common.knn_geometry_prediction(trajectories, values, neighbors=3)
        nonimproving = rewards <= rewards[policy] + 0.005 + 1e-8
        nonimproving[policy] = False
        nonimproving_rescue = nonimproving & successes if baseline_failed else np.zeros(20, dtype=np.bool_)
        candidate_harm = (~successes) if bool(successes[policy]) else np.zeros(20, dtype=np.bool_)
        row = {
            "scene_id": scene,
            "outcome_stratum": target["outcome_stratum"],
            "origin_log": target["origin_log"],
            "source_kind": target["source_kind"],
            "pairing_method": target["pairing_method"],
            "decision_step": target["decision_step"],
            "policy_index": policy,
            "reward_oracle_index": reward_oracle,
            "causal_best_indices": sorted(best),
            "reward_max_tie_indices": reward_max_set.tolist(),
            "q_policy": q_policy,
            "baseline_score_absolute_error": baseline_score_error,
            "baseline_ego_progress_absolute_error": baseline_progress_error,
            "baseline_nc_equal": baseline_nc_equal,
            "baseline_dac_equal": baseline_dac_equal,
            "baseline_success_equal": baseline_success_equal,
            "q_reward_oracle": q_reward_oracle,
            "q_reward_set": q_reward_set,
            "q_star": q_star,
            "total_regret": total,
            "reward_ranking_horizon_gap": ranking_gap,
            "reward_tie_gap": tie_gap,
            "selector_gap": selector_gap,
            "reward_q_spearman": common.spearman(rewards, values),
            "reward_q_kendall_tau_b": common.kendall_tau_b(rewards, values),
            "reward_top1_recall_causal_best": top_recall_best[1],
            "reward_top3_recall_causal_best": top_recall_best[3],
            "reward_top5_recall_causal_best": top_recall_best[5],
            "reward_top1_recall_any_rescue": top_recall_rescue[1],
            "reward_top3_recall_any_rescue": top_recall_rescue[3],
            "reward_top5_recall_any_rescue": top_recall_rescue[5],
            "maximum_same_reward_q_span": maximum_same_reward_q_span,
            "policy_relative_ade_q_delta_pearson": common.pearson(
                policy_distances, q_deltas
            ),
            "knn_spearman": geometry["spearman"],
            "knn_top3_recall_best": geometry["top3_recall_best"],
            "nonimproving_candidate_count": int(nonimproving.sum()),
            "nonimproving_rescue_count": int(nonimproving_rescue.sum()),
            "candidate_harm_count": int(candidate_harm.sum()),
        }
        scene_rows.append(row)
        for index in range(20):
            matrix_row = {
                "scene_id": scene,
                "outcome_stratum": target["outcome_stratum"],
                "origin_log": target["origin_log"],
                "decision_step": target["decision_step"],
                "candidate_index": index,
                "is_policy": index == policy,
                "is_reward_oracle": index == reward_oracle,
                "is_reward_max_tie": index in set(reward_max_set.tolist()),
                "is_causal_best": index in best,
                "local_reward": float(rewards[index]),
                "closed_loop_q": float(values[index]),
                "closed_loop_success": bool(successes[index]),
                "policy_relative_xy_ade_m": float(policy_distances[index]),
                "knn_predicted_q": float(geometry["predictions"][index]),
            }
            for component_index, name in enumerate(target["reward_component_names"]):
                matrix_row[f"local_{name}"] = float(components[index, component_index])
            matrix_rows.append(matrix_row)

    failed = [row for row in scene_rows if row["outcome_stratum"] == "failed"]
    solved = [row for row in scene_rows if row["outcome_stratum"] == "solved"]
    if len(failed) != common.NUM_FAILED_TARGETS or len(solved) != common.NUM_SOLVED_TARGETS:
        raise RuntimeError("CCV analyzed stratum coverage drifted")
    gates = {
        "reward_ranking_horizon_gap": common.gap_gate(
            failed, "reward_ranking_horizon_gap", seed=common.BOOTSTRAP_SEED
        ),
        "reward_tie_gap": common.gap_gate(
            failed, "reward_tie_gap", seed=common.BOOTSTRAP_SEED + 1
        ),
        "selector_gap": common.gap_gate(
            failed, "selector_gap", seed=common.BOOTSTRAP_SEED + 2
        ),
    }
    knn_bootstrap = common.cluster_bootstrap(
        scene_rows, "knn_spearman", seed=common.BOOTSTRAP_SEED + 10
    )
    knn_top3 = mean_or_none(row["knn_top3_recall_best"] for row in scene_rows)
    geometry_gate = {
        "minimum_mean_scene_spearman": 0.25,
        "minimum_top3_recall_best": 0.50,
        "mean_scene_spearman": knn_bootstrap["estimate"],
        "top3_recall_best": knn_top3,
        "cluster_bootstrap": knn_bootstrap,
        "point_pass": bool(
            knn_bootstrap["estimate"] is not None
            and knn_bootstrap["estimate"] >= 0.25
            and knn_top3 is not None and knn_top3 >= 0.50
        ),
        "bootstrap_pass": bool(
            knn_bootstrap["lower_95"] is not None
            and knn_bootstrap["lower_95"] > 0.0
        ),
    }
    geometry_gate["pass"] = bool(
        geometry_gate["point_pass"] and geometry_gate["bootstrap_pass"]
    )
    decision, recommended_method = decide(gates, geometry_gate)

    matrix_path = args.matrix_csv.expanduser().resolve()
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    with temporary_matrix.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    temporary_matrix.replace(matrix_path)

    return {
        "schema_version": common.SCHEMA_VERSION,
        "status": "PASS",
        "method": common.ANALYSIS_METHOD,
        "decision": decision,
        "recommended_method_family": recommended_method,
        "master_manifest": str(master_path),
        "master_manifest_sha256": common.sha256_file(master_path),
        "sentinel_gate": str(sentinel_path),
        "sentinel_gate_sha256": common.sha256_file(sentinel_path),
        "analysis_note": str(analysis_note_path),
        "analysis_note_sha256": common.sha256_file(analysis_note_path),
        "candidate_causal_value_matrix": str(matrix_path),
        "candidate_causal_value_matrix_sha256": common.sha256_file(matrix_path),
        "matrix_row_count": len(matrix_rows),
        "expected_matrix_row_count": common.NUM_TARGETS * common.NUM_CANDIDATES,
        "complete_33_by_20_response_surface": len(matrix_rows) == 660,
        "causal_value_definition": "one_candidate_action_then_frozen_v3",
        "causal_values_authorized_for_training": False,
        "policy_baseline_reproduction": {
            "continuous_equivalence_tolerance": FORMAL_POLICY_BASELINE_CONTINUOUS_EQUIVALENCE,
            "continuous_equivalence_rationale": "0.1pct_unit_range_and_50x_below_0.05_primary_gap",
            "maximum_score_absolute_error": max(row["baseline_score_absolute_error"] for row in scene_rows),
            "maximum_ego_progress_absolute_error": max(
                row["baseline_ego_progress_absolute_error"] for row in scene_rows
            ),
            "score_exact_within_1e_6_count": sum(
                row["baseline_score_absolute_error"] <= common.OUTCOME_TOLERANCE
                for row in scene_rows
            ),
            "all_nc_equal": all(row["baseline_nc_equal"] for row in scene_rows),
            "all_dac_equal": all(row["baseline_dac_equal"] for row in scene_rows),
            "all_success_labels_equal": all(row["baseline_success_equal"] for row in scene_rows),
            "source_baseline_audit": str(baseline_audit_path),
            "source_baseline_audit_sha256": common.sha256_file(baseline_audit_path),
        },
        "collection_provenance": {
            "rollout_implementation_sha256": sentinel["rollout_implementation_sha256"],
            "sentinel_code_sha": sentinel["code_sha"],
            "formal_arm_code_sha": next(iter(arm_code_shas)),
            "checkpoint_sha256": sentinel["checkpoint_sha256"],
            "checkpoint_manifest_sha256": sentinel["checkpoint_manifest_sha256"],
            "candidate_noise_namespace": sentinel["candidate_noise_namespace"],
        },
        "primary_stratum": "failed",
        "gates": gates,
        "geometry_neighborhood_gate": geometry_gate,
        "strata": {
            "failed": summarize_rows(failed, common.BOOTSTRAP_SEED + 100),
            "solved": summarize_rows(solved, common.BOOTSTRAP_SEED + 200),
            "all": summarize_rows(scene_rows, common.BOOTSTRAP_SEED + 300),
        },
        "scene_rows": scene_rows,
        "arm_provenance": arm_refs,
        "decision_contract": {
            "ranking_horizon_gap": "long_horizon_causal_advantage",
            "ranking_horizon_plus_geometry": "basin_aware_long_horizon_causal_advantage",
            "tie_gap_only": "scalar_coarse_plus_diffusion_set_tie_breaker",
            "selector_gap_only": "retain_scalar_reward_improve_selector",
            "point_without_cluster_support": "expand_targets_or_frozen_noise",
            "no_actionable_gap": "stop_ccv_method_route",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sentinel-gate", type=Path, required=True)
    parser.add_argument("--analysis-note", type=Path, required=True)
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args)
    common.atomic_json(args.output, report)
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "output": str(args.output.resolve()),
        "matrix": str(args.matrix_csv.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
