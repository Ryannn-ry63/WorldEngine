"""Paired continuous outcomes, one-shot retention and bounded A->B decisions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import cfpi_common as c
import selector_cfpi_deployment_common as d


def checked_collection(entry):
    collection_path = d.verify(entry)
    collection = d.read(collection_path)
    audit_path = collection_path.parent / "collection_audit.json"
    if not audit_path.exists():
        return None
    audit = c.verified_json(audit_path)
    if audit["collection"] != entry or audit["incomplete_or_dropped_scenes"] != 0:
        raise RuntimeError("Collection audit provenance/coverage mismatch")
    for item in audit["artifacts"]:
        d.verify(item)
    routing = d.verified_read(collection["routing"])
    for model in routing['models'].values():
        d.verify(model)
    metrics = d.metrics_csv(d.verify(audit["merged_metrics"]), {r["scene_id"] for r in routing["routes"]})
    if metrics != audit["metrics"]:
        raise RuntimeError("Audited metric values changed")
    return dict(collection=collection, audit=d.artifact(audit_path), metrics=metrics)


def differences(current, reference, rows):
    ids = [r["scene_id"] for r in rows]
    delta = {key: float(np.mean([current[s][key] - reference[s][key] for s in ids])) for key in d.METRICS}
    return dict(delta=delta, mean={k: float(np.mean([current[s][k] for s in ids])) for k in d.METRICS},
                rescued_failed=sum(reference[s]["success"] == 0 and current[s]["success"] == 1 for s in ids),
                broken_solved=sum(reference[s]["success"] == 1 and current[s]["success"] == 0 for s in ids),
                beneficial_scenes=sum(current[s]["score"] - reference[s]["score"] > .001 for s in ids),
                harmful_scenes=sum(current[s]["score"] - reference[s]["score"] < -.001 for s in ids))


def summarize(method, data, reference, rows):
    per_seed = [dict(seed=s, **differences(data[d.learned_id(method, s)]["metrics"], reference, rows)) for s in d.SEEDS]
    average = {r["scene_id"]: {k: float(np.mean([data[d.learned_id(method, s)]["metrics"][r["scene_id"]][k]
                                                for s in d.SEEDS])) for k in d.METRICS} for r in rows}
    scene_gain = {r["scene_id"]: average[r["scene_id"]]["score"] - reference[r["scene_id"]]["score"] for r in rows}
    strata = {}
    for field in ("scenario_family", "outcome_stratum", "time_stratum"):
        if all(field in r for r in rows):
            strata[field] = {v: differences(average, reference, [r for r in rows if r[field] == v])["delta"]
                            for v in sorted({r[field] for r in rows})}
    return dict(method=method, configuration=d.POLICIES[method], seeds=per_seed,
                **d.advancement([s["delta"] for s in per_seed]), strata=strata,
                paired_log_interval=d.paired_interval(scene_gain, rows)), average


def report(run, phase):
    inventory_path = run / f"{phase}_collections.json"
    if not inventory_path.exists():
        return dict(status="INCOMPLETE", phase=phase, missing=[str(inventory_path)])
    inventory = d.read(inventory_path)
    expected_count = 13 if phase == "phase_a" else 14
    if len(inventory["collections"]) != expected_count:
        raise RuntimeError("Frozen collection inventory has missing controls")
    data, missing = {}, []
    for item in inventory["collections"]:
        checked = checked_collection(item)
        if checked is None:
            missing.append(item["path"])
        else:
            policy = checked["collection"]["policy"]
            if policy in data:
                raise RuntimeError("Duplicate deployment policy")
            data[policy] = checked
    if missing:
        progress = dict(status="INCOMPLETE", phase=phase, complete_collections=len(data), missing=missing,
                        phase_b_authorized=False, expansion_authorized=False)
        c.atomic_json(run / f"{phase}_progress.json", progress)
        return progress
    inputs = d.read(run / "deployment_inputs.json")
    artifacts = inputs["artifacts"]
    if phase == "phase_a":
        rows = d.verified_read(artifacts["targets"])["targets"]
        reference = d.metrics_csv(d.verify(artifacts["baseline_csv"]))
    else:
        rows = d.verified_read(artifacts["natural_membership"])["rows"]
        reference = data["scalar_v3"]["metrics"]
    if len(rows) != (64 if phase == "phase_a" else 128):
        raise RuntimeError("Report scene count differs from the frozen A/B protocol")
    configs, averages = {}, {}
    for method in d.POLICIES:
        configs[method], averages[method] = summarize(method, data, reference, rows)
    comparisons = {}
    for left, right in (("q_grpo_t1", "local_grpo_t1"), ("q_grpo_t5", "q_grpo_t1"), ("q_mse", "q_grpo_t1")):
        comparisons[f"{left}_minus_{right}"] = d.paired_interval(
            {r["scene_id"]: averages[left][r["scene_id"]]["score"] - averages[right][r["scene_id"]]["score"] for r in rows}, rows)
    paired_rows = []
    arms = {}
    if phase == "phase_a":
        for item in inputs["one_shot_arms"]:
            d.verify(item)
            arms[item["candidate_index"]] = d.metrics_csv(d.verify(dict(path=item["metrics_csv"], sha256=item["metrics_csv_sha256"])),
                                                           {r["scene_id"] for r in rows})
    for method in d.POLICIES:
        continuation_deltas = []
        for seed in d.SEEDS:
            policy = d.learned_id(method, seed)
            current = data[policy]["metrics"]
            if phase == "phase_a":
                one_shot = {r["scene_id"]: arms[inputs["oof_predictions"][policy][r["scene_id"]]][r["scene_id"]] for r in rows}
                continuation_deltas.append(dict(seed=seed, **differences(current, one_shot, rows)))
            for row in rows:
                scene = row["scene_id"]
                entry = dict(scene_id=scene, origin_log=row["origin_log"], family=row["scenario_family"],
                             method=method, seed=seed)
                for key in d.METRICS:
                    entry.update({"continuous_" + key: current[scene][key], "scalar_" + key: reference[scene][key]})
                    entry[("one_shot_" if phase == "phase_a" else "gate_") + key] = (
                        one_shot[scene][key] if phase == "phase_a" else data["gate_v3"]["metrics"][scene][key])
                paired_rows.append(entry)
        if phase == "phase_a":
            configs[method]["continuous_minus_one_shot"] = continuation_deltas
        else:
            configs[method]["versus_gate_v3"] = summarize(method, data, data["gate_v3"]["metrics"], rows)[0]
    table = run / f"{phase}_paired_scenes.csv"
    temporary = table.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    temporary.replace(table)
    passing = [m for m in d.POLICIES if configs[m]["pass_screening"]]
    result = dict(status="PASS", method=d.METHOD, phase=phase, configurations=configs, paired_comparisons=comparisons,
                  phase_b_authorized=bool(phase == "phase_a" and passing), passing_configurations=passing,
                  decision=("PROCEED_TO_FIXED_PHASE_B" if passing else "STOP_AND_DIAGNOSE") if phase == "phase_a"
                           else "RESEARCH_REVIEW_REQUIRED_NO_EXPANSION",
                  thresholds=d.ADVANCEMENT, collection_audits={k: v["audit"] for k, v in data.items()},
                  inputs=d.artifact(run / "deployment_inputs.json"), paired_scene_table=d.artifact(table),
                  evaluation_role=("pilot_stratified_oof_continuation_screening" if phase == "phase_a"
                                   else "training_pool_deployment_screening"),
                  independent_development_or_test=False, expansion_authorized=False,
                  passing_configurations_reference="scalar_v3",
                  phase_b_screening_flags_are_descriptive_only=(phase == "phase_b"),
                  temperature_only_causal_claim_authorized=False,
                  limitations=["post-selection exploratory intervals", "one incumbent initialization / fixed noise namespace",
                               "known historical membership is not an exhaustive exposure audit"])
    c.locked_json(run / f"{phase}_report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--phase", choices=("phase_a", "phase_b"), required=True)
    args = p.parse_args()
    result = report(args.run_root.resolve(), args.phase)
    print({k: result[k] for k in ("status", "phase", "decision") if k in result})
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
