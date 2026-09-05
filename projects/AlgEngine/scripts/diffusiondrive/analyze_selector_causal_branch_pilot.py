#!/usr/bin/env python3
"""Analyze causal pilot v2 primary/specificity estimands and decide the route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import causal_branch_pilot_common as common
import oracle_r15_common as r15


def load_audit(path: Path, phase: str, mode: str, target_sha: str) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    row = json.loads(path.read_text())
    if (
        row.get("status") != "PASS"
        or row.get("method") != common.COLLECTION_AUDIT_METHOD
        or row.get("layout") != "merged"
        or row.get("diagnostic_split") != phase
        or row.get("oracle_intervention_mode") != mode
        or int(row.get("behavior_policy_train_seed", -1)) != 0
        or row.get("target_manifest_sha256") != target_sha
        or not row.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid {phase}/{mode} audit: {path}")
    return path, row


def outcomes(audit: dict) -> dict[str, dict]:
    path = Path(audit["closed_loop_outcome"]["metrics_csv"]).expanduser().resolve()
    if r15.sha256_file(path) != audit["closed_loop_outcome"]["metrics_csv_sha256"]:
        raise RuntimeError(f"metrics CSV drifted: {path}")
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            scene = str(row["token"])
            if scene == "overall_average":
                continue
            collision = float(row["no_at_fault_collisions"])
            drivable = float(row["drivable_area_compliance"])
            result[scene] = {
                "score": float(row["score"]),
                "success": bool(collision >= 1.0 and drivable >= 1.0),
                "no_at_fault_collisions": collision,
                "drivable_area_compliance": drivable,
            }
    return result


def audit_provenance(audits: list[tuple[Path, dict]], checkpoint_sha: str):
    implementations = {row["rollout_implementation_sha256"] for _, row in audits}
    checkpoints = {row["checkpoint_sha256"] for _, row in audits}
    namespaces = {row["candidate_noise_namespace"] for _, row in audits}
    if checkpoints != {checkpoint_sha} or len(implementations) != 1 or len(namespaces) != 1:
        raise RuntimeError("checkpoint/implementation/noise provenance differs across arms")


def repeatability(smoke: dict, formal: dict, scene_ids: list[str]) -> dict:
    rows = []
    for scene in scene_ids:
        if scene not in smoke or scene not in formal:
            rows.append({"scene_id": scene, "repeatable": False, "reason": "missing"})
            continue
        score_error = abs(smoke[scene]["score"] - formal[scene]["score"])
        success_equal = smoke[scene]["success"] == formal[scene]["success"]
        rows.append({
            "scene_id": scene,
            "score_abs_error": score_error,
            "success_equal": success_equal,
            "repeatable": bool(success_equal and score_error <= 1e-6),
        })
    return {
        "all_repeatable": bool(rows and all(row["repeatable"] for row in rows)),
        "repeatable_count": int(sum(row["repeatable"] for row in rows)),
        "scene_count": len(rows),
        "rows": rows,
    }


def decide_causal_route(
    *,
    coverage_pass: bool,
    reproducible: bool,
    primary_rescue_rate: float | None,
    oracle_rescue_origin_logs: int,
    primary_lower_95: float | None,
    specificity_rescue_rate: float | None,
    differential_rescue_origin_logs: int,
    specificity_lower_95: float | None,
) -> dict:
    primary_point = bool(
        primary_rescue_rate is not None
        and primary_rescue_rate >= 0.15
        and oracle_rescue_origin_logs >= 4
    )
    specificity_point = bool(
        specificity_rescue_rate is not None
        and specificity_rescue_rate >= 0.15
        and differential_rescue_origin_logs >= 4
    )
    primary_bootstrap = bool(primary_lower_95 is not None and primary_lower_95 > 0.0)
    specificity_bootstrap = bool(
        specificity_lower_95 is not None and specificity_lower_95 > 0.0
    )
    if not coverage_pass:
        decision = "INSUFFICIENT_CAUSAL_BRANCH_COVERAGE"
    elif not reproducible:
        decision = "INVALID_CAUSAL_BRANCH_REPRODUCIBILITY"
    elif not primary_point:
        decision = "STOP_LOCAL_HEADROOM_CAUSAL_ROUTE"
    elif not specificity_point:
        decision = "REDIRECT_TO_REWARD_HORIZON_OR_ACTION_SENSITIVITY"
    elif not (primary_bootstrap and specificity_bootstrap):
        decision = "EXPAND_CAUSAL_BRANCH_SEED1"
    else:
        decision = "AUTHORIZE_DIFFUSION_SET_SELECTOR_METHOD"
    return {
        "decision": decision,
        "primary_point_pass": primary_point,
        "primary_bootstrap_pass": primary_bootstrap,
        "specificity_point_pass": specificity_point,
        "specificity_bootstrap_pass": specificity_bootstrap,
    }

def summarize_stratum(rows: list[dict], stratum: str, repetitions: int, seed: int):
    if not rows:
        return {
            "count": 0,
            "origin_log_count": 0,
            "mean_oracle_minus_baseline_score": None,
            "mean_matched_minus_baseline_score": None,
            "mean_oracle_minus_matched_score": None,
        }
    result = {
        "count": len(rows),
        "origin_log_count": len({row["origin_log"] for row in rows}),
        "mean_oracle_minus_baseline_score": float(
            np.mean([row["oracle_minus_baseline_score"] for row in rows])
        ),
        "mean_matched_minus_baseline_score": float(
            np.mean([row["matched_minus_baseline_score"] for row in rows])
        ),
        "mean_oracle_minus_matched_score": float(
            np.mean([row["oracle_minus_matched_score"] for row in rows])
        ),
        "oracle_minus_baseline_score_bootstrap": common.cluster_bootstrap(
            rows, "oracle_minus_baseline_score", repetitions, seed + 1
        ),
        "oracle_minus_matched_score_bootstrap": common.cluster_bootstrap(
            rows, "oracle_minus_matched_score", repetitions, seed + 2
        ),
        "local_headroom_primary_score_delta_pearson": common.pearson(
            [row["frozen_oracle_headroom"] for row in rows],
            [row["oracle_minus_baseline_score"] for row in rows],
        ),
        "local_headroom_specificity_score_delta_pearson": common.pearson(
            [row["frozen_oracle_headroom"] for row in rows],
            [row["oracle_minus_matched_score"] for row in rows],
        ),
    }
    if stratum == "failed":
        result.update({
            "baseline_rescue_rate": float(np.mean([row["baseline_rescue"] for row in rows])),
            "oracle_rescue_rate": float(np.mean([row["oracle_rescue"] for row in rows])),
            "matched_rescue_rate": float(np.mean([row["matched_rescue"] for row in rows])),
            "oracle_minus_baseline_rescue_rate": float(
                np.mean([row["primary_rescue_contrast"] for row in rows])
            ),
            "oracle_minus_matched_rescue_rate": float(
                np.mean([row["specificity_rescue_contrast"] for row in rows])
            ),
            "oracle_rescue_origin_log_count": len({
                row["origin_log"] for row in rows if row["oracle_rescue"]
            }),
            "differential_oracle_rescue_origin_log_count": len({
                row["origin_log"]
                for row in rows
                if row["specificity_rescue_contrast"] > 0
            }),
            "primary_rescue_contrast_bootstrap": common.cluster_bootstrap(
                rows, "primary_rescue_contrast", repetitions, seed
            ),
            "specificity_rescue_contrast_bootstrap": common.cluster_bootstrap(
                rows, "specificity_rescue_contrast", repetitions, seed + 3
            ),
        })
    else:
        result.update({
            "oracle_harm_rate": float(np.mean([row["oracle_harm"] for row in rows])),
            "matched_harm_rate": float(np.mean([row["matched_harm"] for row in rows])),
            "matched_minus_oracle_harm_rate": float(
                np.mean([row["harm_benefit"] for row in rows])
            ),
            "harm_benefit_bootstrap": common.cluster_bootstrap(
                rows, "harm_benefit", repetitions, seed + 4
            ),
        })
    return result

def analyze(args) -> dict:
    target_path = args.target_manifest.expanduser().resolve()
    target_sha = r15.sha256_file(target_path)
    target_payload = json.loads(target_path.read_text())
    if (
        target_payload.get("status") != "PASS"
        or target_payload.get("method") != common.TARGET_METHOD
        or target_payload.get("schema_version") != common.TARGET_SCHEMA_VERSION
        or target_payload.get("design_version") != "causal_branch_pilot_v2"
        or target_payload.get("intervention_outcomes_observed_before_freeze") is not False
        or target_payload.get("primary_estimand")
        != "one_step_local_oracle_minus_no_intervention_v3_baseline"
        or target_payload.get("specificity_estimand")
        != "one_step_local_oracle_minus_magnitude_matched_nonimproving_control"
        or r15.sha256_file(target_payload["source_scenario_file"])
        != target_payload["source_scenario_file_sha256"]
        or r15.sha256_file(target_payload["formal_scenario_file"])
        != target_payload["formal_scenario_file_sha256"]
        or r15.sha256_file(target_payload["smoke_scenario_file"])
        != target_payload["smoke_scenario_file_sha256"]
    ):
        raise RuntimeError("target manifest or frozen scenario files drifted")
    frozen_design_files = {
        "protocol_file": "protocol_file_sha256",
        "target_builder_file": "target_builder_sha256",
        "common_helpers_file": "common_helpers_sha256",
    }
    for file_key, sha_key in frozen_design_files.items():
        if r15.sha256_file(target_payload[file_key]) != target_payload[sha_key]:
            raise RuntimeError(f"frozen target design provenance drifted: {file_key}")

    baseline_a = load_audit(args.baseline_audit_a, "baseline_a", "observe_only", "none")
    baseline_b = load_audit(args.baseline_audit_b, "baseline_b", "observe_only", "none")
    smoke_oracle = load_audit(
        args.smoke_oracle_audit, "intervention_smoke8", "one_shot_oracle", target_sha
    )
    smoke_matched = load_audit(
        args.smoke_matched_audit, "intervention_smoke8", "one_shot_matched", target_sha
    )
    formal_oracle = load_audit(
        args.formal_oracle_audit, "intervention_target", "one_shot_oracle", target_sha
    )
    formal_matched = load_audit(
        args.formal_matched_audit, "intervention_target", "one_shot_matched", target_sha
    )
    audits = [
        baseline_a,
        baseline_b,
        smoke_oracle,
        smoke_matched,
        formal_oracle,
        formal_matched,
    ]
    audit_provenance(audits, target_payload["checkpoint_sha256"])
    if r15.sha256_file(baseline_a[0]) != target_payload["baseline_audit_a_sha256"]:
        raise RuntimeError("baseline A audit changed after target freezing")
    if r15.sha256_file(baseline_b[0]) != target_payload["baseline_audit_b_sha256"]:
        raise RuntimeError("baseline B audit changed after target freezing")
    expected_scenario_shas = {
        "baseline_a": target_payload["source_scenario_file_sha256"],
        "baseline_b": target_payload["source_scenario_file_sha256"],
        "intervention_smoke8": target_payload["smoke_scenario_file_sha256"],
        "intervention_target": target_payload["formal_scenario_file_sha256"],
    }
    for _, audit in audits:
        if audit["scenario_file_sha256"] != expected_scenario_shas[audit["diagnostic_split"]]:
            raise RuntimeError("audit/scenario contract SHA256 drifted")

    base_a_out = outcomes(baseline_a[1])
    base_b_out = outcomes(baseline_b[1])
    if (
        baseline_a[1]["closed_loop_outcome"]["metrics_csv_sha256"]
        != target_payload["baseline_metrics_sha256"]
        or baseline_b[1]["closed_loop_outcome"]["metrics_csv_sha256"]
        != target_payload["baseline_metrics_sha256"]
        or base_a_out != base_b_out
    ):
        raise RuntimeError("frozen no-intervention baseline is not exactly reproducible")
    smoke_oracle_out = outcomes(smoke_oracle[1])
    smoke_matched_out = outcomes(smoke_matched[1])
    formal_oracle_out = outcomes(formal_oracle[1])
    formal_matched_out = outcomes(formal_matched[1])
    smoke_ids = [str(scene) for scene in target_payload["smoke_scene_ids"]]
    reproducibility = {
        "one_shot_oracle": repeatability(
            smoke_oracle_out, formal_oracle_out, smoke_ids
        ),
        "one_shot_matched": repeatability(
            smoke_matched_out, formal_matched_out, smoke_ids
        ),
    }
    all_reproducible = all(row["all_repeatable"] for row in reproducibility.values())

    repeat_oracle = {
        row["scene_id"]: row for row in formal_oracle[1]["target_repeatability"]
    }
    repeat_matched = {
        row["scene_id"]: row for row in formal_matched[1]["target_repeatability"]
    }
    rows = []
    exclusions = []
    for target in target_payload["targets"]:
        scene = str(target["scene_id"])
        required = (
            scene in base_a_out
            and scene in base_b_out
            and scene in formal_oracle_out
            and scene in formal_matched_out
            and scene in repeat_oracle
            and scene in repeat_matched
        )
        if not required:
            exclusions.append({"scene_id": scene, "reason": "missing_outcome_or_target_record"})
            continue
        oracle_eligible = bool(
            repeat_oracle[scene]["current_frozen_oracle_headroom_gt_0p02"]
        )
        matched_eligible = bool(
            repeat_oracle[scene]["current_matched_magnitude_caliper_pass"]
            and repeat_matched[scene]["current_frozen_oracle_headroom_gt_0p02"]
            and repeat_matched[scene]["current_matched_headroom_le_0p005"]
            and repeat_matched[scene]["current_matched_magnitude_caliper_pass"]
        )
        if not (oracle_eligible and matched_eligible):
            exclusions.append({
                "scene_id": scene,
                "reason": "formal_reward_eligibility_failed",
                "oracle_eligible": oracle_eligible,
                "matched_eligible": matched_eligible,
            })
            continue
        baseline_score = (base_a_out[scene]["score"] + base_b_out[scene]["score"]) / 2.0
        oracle_score = formal_oracle_out[scene]["score"]
        matched_score = formal_matched_out[scene]["score"]
        stratum = target["outcome_stratum"]
        baseline_success = bool(base_a_out[scene]["success"])
        oracle_success = bool(formal_oracle_out[scene]["success"])
        matched_success = bool(formal_matched_out[scene]["success"])
        row = {
            "scene_id": scene,
            "outcome_stratum": stratum,
            "origin_log": target["origin_log"],
            "source_kind": target["source_kind"],
            "pairing_method": target["pairing_method"],
            "decision_step": target["decision_step"],
            "baseline_score_mean": baseline_score,
            "oracle_score": oracle_score,
            "matched_score": matched_score,
            "oracle_minus_baseline_score": oracle_score - baseline_score,
            "matched_minus_baseline_score": matched_score - baseline_score,
            "oracle_minus_matched_score": oracle_score - matched_score,
            "frozen_oracle_headroom": repeat_oracle[scene]["current_frozen_oracle_headroom"],
            "matched_headroom": repeat_matched[scene]["current_matched_headroom"],
            "absolute_ade_match_error_m": target["ade_match_error"],
            "relative_ade_match_error": target["relative_ade_match_error"],
            "baseline_success": baseline_success,
            "oracle_success": oracle_success,
            "matched_success": matched_success,
        }
        if stratum == "failed":
            row["baseline_rescue"] = baseline_success
            row["oracle_rescue"] = oracle_success
            row["matched_rescue"] = matched_success
            row["primary_rescue_contrast"] = int(oracle_success) - int(baseline_success)
            row["specificity_rescue_contrast"] = int(oracle_success) - int(matched_success)
        else:
            row["oracle_harm"] = bool(not oracle_success)
            row["matched_harm"] = bool(not matched_success)
            row["harm_benefit"] = int(row["matched_harm"]) - int(row["oracle_harm"])
        rows.append(row)

    by_stratum = {
        name: [row for row in rows if row["outcome_stratum"] == name]
        for name in ("failed", "solved")
    }
    summaries = {
        name: summarize_stratum(
            by_stratum[name], name, args.bootstrap_repetitions, args.bootstrap_seed
        )
        for name in ("failed", "solved")
    }
    coverage = {
        name: {
            "target_count": summaries[name]["count"],
            "origin_log_count": summaries[name]["origin_log_count"],
            "minimum_target_count": 8,
            "minimum_origin_log_count": 6,
            "pass": bool(
                summaries[name]["count"] >= 8
                and summaries[name]["origin_log_count"] >= 6
            ),
        }
        for name in ("failed", "solved")
    }
    coverage_pass = all(row["pass"] for row in coverage.values())
    failed = summaries["failed"]
    primary_rate = failed.get("oracle_minus_baseline_rescue_rate")
    specificity_rate = failed.get("oracle_minus_matched_rescue_rate")
    oracle_rescue_logs = failed.get("oracle_rescue_origin_log_count", 0)
    differential_rescue_logs = failed.get(
        "differential_oracle_rescue_origin_log_count", 0
    )
    primary_lower_95 = failed.get("primary_rescue_contrast_bootstrap", {}).get(
        "lower_95"
    )
    specificity_lower_95 = failed.get(
        "specificity_rescue_contrast_bootstrap", {}
    ).get("lower_95")
    decision_report = decide_causal_route(
        coverage_pass=coverage_pass,
        reproducible=all_reproducible,
        primary_rescue_rate=primary_rate,
        oracle_rescue_origin_logs=oracle_rescue_logs,
        primary_lower_95=primary_lower_95,
        specificity_rescue_rate=specificity_rate,
        differential_rescue_origin_logs=differential_rescue_logs,
        specificity_lower_95=specificity_lower_95,
    )

    return {
        "schema_version": 2,
        "status": "PASS",
        "method": common.GATE_METHOD,
        "decision": decision_report["decision"],
        "scientific_question": "does_correcting_a_reproducible_local_selector_error_causally_improve_closed_loop_outcome_and_outperform_equal_magnitude_nonimproving_perturbation",
        "primary_estimand": "frozen_local_oracle_minus_no_intervention_v3_baseline",
        "specificity_estimand": "frozen_local_oracle_minus_magnitude_matched_nonimproving_control",
        "matched_control_interpretation": "treatment_magnitude_match_not_directional_geometry_match",
        "target_manifest": str(target_path),
        "target_manifest_sha256": target_sha,
        "analyzer_sha256": r15.sha256_file(Path(__file__)),
        "common_helpers_sha256": r15.sha256_file(Path(common.__file__)),
        "audits": {
            f"{row['diagnostic_split']}:{row['oracle_intervention_mode']}": {
                "path": str(path),
                "sha256": r15.sha256_file(path),
            }
            for path, row in audits
        },
        "baseline_no_intervention_exactly_reproducible": True,
        "smoke_formal_outcome_reproducibility": reproducibility,
        "all_smoke_formal_outcomes_reproducible": all_reproducible,
        "formal_eligibility_exclusions": exclusions,
        "coverage": coverage,
        "coverage_pass": coverage_pass,
        "primary_gate": {
            "estimand": "oracle_minus_no_intervention_baseline_rescue_rate",
            "minimum_rescue_rate_delta": 0.15,
            "minimum_oracle_rescue_origin_logs": 4,
            "observed_rescue_rate_delta": primary_rate,
            "observed_oracle_rescue_origin_logs": oracle_rescue_logs,
            "point_pass": decision_report["primary_point_pass"],
            "bootstrap_criterion": "origin_log_cluster_bootstrap_lower95_gt_0",
            "observed_bootstrap_lower_95": primary_lower_95,
            "bootstrap_pass": decision_report["primary_bootstrap_pass"],
        },
        "specificity_gate": {
            "estimand": "oracle_minus_magnitude_matched_control_rescue_rate",
            "minimum_rescue_rate_delta": 0.15,
            "minimum_differential_rescue_origin_logs": 4,
            "observed_rescue_rate_delta": specificity_rate,
            "observed_differential_rescue_origin_logs": differential_rescue_logs,
            "point_pass": decision_report["specificity_point_pass"],
            "bootstrap_criterion": "origin_log_cluster_bootstrap_lower95_gt_0",
            "observed_bootstrap_lower_95": specificity_lower_95,
            "bootstrap_pass": decision_report["specificity_bootstrap_pass"],
        },
        "matching_contract": target_payload["matching_contract"],
        "strata": summaries,
        "formal_rows": rows,
        "decision_contract": {
            "authorize": "both_point_gates_and_both_bootstrap_gates_and_coverage_and_reproducibility",
            "seed1": "both_point_gates_pass_but_at_least_one_bootstrap_gate_is_uncertain",
            "redirect": "oracle_beats_baseline_point_gate_but_not_magnitude_matched_specificity_point_gate",
            "stop": "oracle_does_not_beat_no_intervention_baseline_point_gate",
            "invalid": "smoke_formal_outcomes_are_not_reproducible",
            "insufficient": "fewer_than_8_targets_or_6_origin_logs_in_either_stratum",
        },
        "paper_scope": "diagnostic_only_not_a_deployable_oracle_and_not_a_new_training_result",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--baseline-audit-a", type=Path, required=True)
    parser.add_argument("--baseline-audit-b", type=Path, required=True)
    parser.add_argument("--smoke-oracle-audit", type=Path, required=True)
    parser.add_argument("--smoke-matched-audit", type=Path, required=True)
    parser.add_argument("--formal-oracle-audit", type=Path, required=True)
    parser.add_argument("--formal-matched-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(output, report)
    print(json.dumps({
        "status": "PASS",
        "decision": report["decision"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

