"""Separate engineering validity, information, learning, and novelty decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cfpi_common as common


def pilot_gate(run_root):
    audits = [common.verified_json(run_root / f"cache/{stage}_cache_audit.json", "cache_file")
              for stage in ("pilot64", "repeat8")]
    pilot, repeated = [common.load_pickle(a["cache_file"]) for a in audits]
    rows = common.validate_cache(pilot)
    if (repeated.get("method") != common.CACHE_METHOD or repeated.get("stage") != "repeat8"
            or repeated.get("num_rows") != 8):
        raise RuntimeError("Missing repeat8 surface")
    for key in ("checkpoint_sha256", "selector_state_sha256", "candidate_noise_namespace",
                "target_manifest_sha256", "collection_code_sha", "rollout_implementation_sha256"):
        if pilot[key] != repeated[key]:
            raise RuntimeError(f"Repeat provenance drift: {key}")
    by_id = {r["scene_id"]: r for r in rows}
    _, targets = common.load_target_manifest(pilot["target_manifest"])
    if set(r["scene_id"] for r in repeated["rows"]) != set(targets["repeat_scene_ids"]):
        raise RuntimeError("Repeat subset was not preregistered")
    max_error = 0.0
    for row in repeated["rows"]:
        original = by_id[row["scene_id"]]
        q = common.checked_array(row["branch_returns"], (20,), "repeated return")
        components = common.checked_array(row["causal_outcome_components"], (20, 3), "repeat components")
        original_components = common.checked_array(original["causal_outcome_components"], (20, 3), "components")
        max_error = max(max_error, float(np.max(np.abs(q - original["branch_returns"]))),
                        float(np.max(np.abs(components - original_components))))
        if not np.array_equal(components[:, :2], original_components[:, :2]):
            raise RuntimeError("Repeated discrete outcomes changed")
        if (row["branch_success"] != original["branch_success"]
                or row["branch_first_violation_steps"] != original["branch_first_violation_steps"]):
            raise RuntimeError("Repeated event timing/success changed")
    for row in rows:
        if abs(float(row["branch_returns"][row["policy_index"]]) - row["behavior_score"]) > 1e-3:
            raise RuntimeError("Incumbent branch does not reproduce baseline")
        if (row["branch_success"][row["policy_index"]] != row["behavior_success"]
                or row["branch_first_violation_steps"][row["policy_index"]] != row["first_violation_step"]):
            raise RuntimeError("Incumbent branch event timing/success differs from baseline")
    gaps = [float(np.max(r["branch_returns"]) - r["branch_returns"][r["policy_index"]]) for r in rows]
    count = sum(gap > .02 for gap in gaps)
    valid = max_error <= common.OUTCOME_TOLERANCE
    informative = count >= common.PILOT_MIN_IMPROVABLE
    report = dict(status="PASS" if valid else "FAIL", method="selector_cfpi_pilot_gate_v1",
                  cache_sha256=audits[0]["cache_file_sha256"], repeat_cache_sha256=audits[1]["cache_file_sha256"],
                  repeat_max_absolute_error=max_error, repeat_noise_estimand="fixed_conditions_only",
                  improvable_above_0p02=count, information_gate_pass=informative,
                  training_authorized=valid and informative, expansion_authorized=False,
                  decision=("TRAIN_CV_ONLY" if valid and informative else
                            "STOP_ENGINEERING" if not valid else "STOP_LOW_INFORMATION"),
                  development_consumed=False, test_consumed=False)
    common.atomic_json(run_root / "pilot_gate.json", report)
    return report


def grouped_interval(rows, gains, repetitions=10000):
    logs = sorted({r["origin_log"] for r in rows})
    values = np.array([np.mean([g for r, g in zip(rows, gains) if r["origin_log"] == log]) for log in logs])
    rng = np.random.default_rng(20260905)
    means = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
    return dict(log_weighted_mean=float(values.mean()),
                log_bootstrap_95_percentile=np.quantile(means, [.025, .975]).tolist(),
                num_origin_logs=len(logs))


def summarize_predictions(rows, predictions):
    if set(predictions) != {r["scene_id"] for r in rows}:
        raise RuntimeError("Incomplete out-of-fold predictions")
    gains = []
    for row in rows:
        action = predictions[row["scene_id"]]
        if not isinstance(action, int) or action not in range(20):
            raise RuntimeError("Invalid predicted action")
        gains.append(float(row["branch_returns"][action] - row["branch_returns"][row["policy_index"]]))
    solved = [g for r, g in zip(rows, gains) if r["behavior_success"]]
    failed = [g for r, g in zip(rows, gains) if not r["behavior_success"]]
    folds = [float(np.mean([g for r, g in zip(rows, gains) if r["fold"] == f])) for f in range(4)]
    mean = float(np.mean(gains))
    return dict(mean_gain=mean, solved_mean_gain=float(np.mean(solved)),
                failed_mean_gain=float(np.mean(failed)), fold_mean_gains=folds,
                beneficial_changes=sum(g > .001 for g in gains),
                harmful_changes=sum(g < -.001 for g in gains),
                screening_pass=mean >= .005 and sum(g > 0 for g in folds) >= 3
                and float(np.mean(solved)) >= -.005,
                **grouped_interval(rows, gains))


def cv_report(run_root):
    from selector_cfpi_objectives import METHODS
    from train_selector_cfpi import LEARNING_RATES, SEEDS, CHECKPOINT_STEPS, job_name
    gate = common.verified_json(run_root / "pilot_gate.json")
    if not gate["training_authorized"]:
        stopped = dict(gate, expansion_authorized=False,
                       statistical_role="no_method_training_low_information")
        common.atomic_json(run_root / "pilot_report.json", stopped)
        (run_root / "PILOT_REPORT.md").write_text(
            "# CFPI pilot report\n\n" + gate["decision"] +
            "\n\nNo method training or expansion authorized. This is not a proof of unlearnability.\n")
        return gate
    cache_path = run_root / "cache/pilot64_cache.pkl"
    audit = common.verified_json(run_root / "cache/pilot64_cache_audit.json", "cache_file")
    if (Path(audit["cache_file"]).resolve() != cache_path.resolve()
            or audit["cache_file_sha256"] != gate["cache_sha256"]):
        raise RuntimeError("Report cache differs from authorized pilot")
    rows = common.validate_cache(common.load_pickle(cache_path))
    summaries = []
    for method in METHODS:
        for lr in LEARNING_RATES:
            jobs = {}
            for seed in SEEDS:
                for fold in range(4):
                    name = job_name(method, lr, fold, seed)
                    jobs[(seed, fold)] = common.verified_json(run_root / "cv" / name / "report.json")
                    job = jobs[(seed, fold)]
                    expected = dict(method=method, learning_rate=lr, seed=seed, fold=fold)
                    if any(job.get(k) != v or job["provenance"].get(k) != v
                           for k, v in expected.items()):
                        raise RuntimeError("CV job identity does not match requested configuration")
                    if (job["provenance"]["cache_sha256"] != gate["cache_sha256"]
                            or job["provenance"].get("pilot_gate_sha256")
                            != common.sha256_file(run_root / "pilot_gate.json")):
                        raise RuntimeError("CV result used a different cache/gate")
                    if sorted(c["step"] for c in job["checkpoints"]) != list(CHECKPOINT_STEPS):
                        raise RuntimeError("CV checkpoint schedule is incomplete or duplicated")
                    heldout_ids = {row["scene_id"] for row in rows if row["fold"] == fold}
                    for checkpoint in job["checkpoints"]:
                        if set(checkpoint["predictions"]) != heldout_ids:
                            raise RuntimeError("CV predictions do not match their held-out fold")
                        if common.sha256_file(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
                            raise RuntimeError("CV checkpoint artifact drifted")
            for step in CHECKPOINT_STEPS:
                seed_results = []
                all_predictions = []
                for seed in SEEDS:
                    predictions = {}
                    for fold in range(4):
                        checkpoint = next(c for c in jobs[(seed, fold)]["checkpoints"] if c["step"] == step)
                        if set(predictions) & set(checkpoint["predictions"]):
                            raise RuntimeError("CV fold overlap")
                        predictions.update(checkpoint["predictions"])
                    all_predictions.append(predictions)
                    seed_results.append(summarize_predictions(rows, predictions))
                average_gains = [float(np.mean([
                    row["branch_returns"][p[row["scene_id"]]] - row["branch_returns"][row["policy_index"]]
                    for p in all_predictions])) for row in rows]
                fold_gains = [float(np.mean([g for row, g in zip(rows, average_gains) if row["fold"] == f]))
                              for f in range(4)]
                solved = float(np.mean([g for row, g in zip(rows, average_gains) if row["behavior_success"]]))
                summaries.append(dict(method=method, learning_rate=lr, step=step,
                                      mean_gain=float(np.mean(average_gains)), solved_mean_gain=solved,
                                      fold_mean_gains=fold_gains, seeds=seed_results,
                                      screening_pass=np.mean(average_gains) >= .005
                                      and sum(g > 0 for g in fold_gains) >= 3 and solved >= -.005,
                                      **grouped_interval(rows, average_gains)))
    # Selection is exploratory train-side only; these intervals are not certification.
    ranked = sorted(summaries, key=lambda r: (-r["mean_gain"], -r["solved_mean_gain"],
                                            r["step"], r["learning_rate"], r["method"]))
    passing = [r for r in ranked if r["screening_pass"]]
    selected = passing[0] if passing else ranked[0]
    report = dict(status="PASS", decision="REVIEW_EXPANSION" if passing else "REVIEW_LEARNING_FAILURE",
                  selected_configuration=selected, configurations=ranked,
                  statistical_role="exploratory_train_CV_after_configuration_selection_not_certification",
                  loss_novelty_established=False, continuous_deployment_improvement_established=False,
                  expansion_authorized=False, development_consumed=False, test_consumed=False)
    # Convert NumPy comparison booleans before JSON serialization.
    report = json.loads(json.dumps(report, default=lambda x: x.item()))
    common.atomic_json(run_root / "pilot_report.json", report)
    lines = ["# CFPI pilot report", "", report["decision"], "",
             "Exploratory training-side CV only; no development/test data consumed.",
             "No automatic expansion. Neither loss novelty nor continuous-deployment gain is established.", "",
             "| Method | LR | Steps | Mean gain | Solved gain | Gate |",
             "|---|---:|---:|---:|---:|---|"]
    for row in ranked:
        lines.append(f"| {row['method']} | {row['learning_rate']:g} | {row['step']} | "
                     f"{row['mean_gain']:.6f} | {row['solved_mean_gain']:.6f} | {row['screening_pass']} |")
    (run_root / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pilot_gate", "report"))
    p.add_argument("--run-root", type=Path, required=True)
    args = p.parse_args()
    result = pilot_gate(args.run_root) if args.stage == "pilot_gate" else cv_report(args.run_root)
    print(json.dumps({k: result[k] for k in ("status", "decision")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
