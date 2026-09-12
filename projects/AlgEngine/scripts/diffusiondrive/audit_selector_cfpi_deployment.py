"""Audit all frames, routing, prefix, OOF target and actual consumed actions."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import cfpi_common as c
import oracle_r15_common as r15
import selector_cfpi_deployment_common as d
from audit_selector_root_cause_r1_collection import completed_scenes

SHAPES = dict(candidate_features=(20, 256), candidate_trajectories_8=(20, 8, 3),
              route_bev_features=(20, 8, 256), status_tokens=(1, 256), ego_queries=(1, 256),
              agents_queries=(30, 256), reference_logits=(20,), current_logits=(20,))
ACTION_FIELDS = ("waypoints", "velocities", "headings", "angular_velocities")


def load_current_reports(root):
    paths = sorted(p for p in root.rglob("runner_report_*.json")
                   if p.relative_to(root).parts[0] != "attempt_archive")
    outcomes = defaultdict(list)
    if not paths:
        raise RuntimeError("No current-attempt runner reports; archived attempts cannot supply success")
    for path in paths:
        payload = d.read(path)
        if not isinstance(payload, list):
            raise RuntimeError("Invalid current runner report")
        for row in payload:
            outcomes[row["scenario_name"]].append(row.get("succeeded") is True)
    return paths, outcomes


def action_arrays(action):
    values = {key: (None if getattr(action, key, None) is None else
                    np.asarray(getattr(action, key), dtype=np.float64).tolist()) for key in ACTION_FIELDS}
    if values["waypoints"] is None or not np.isfinite(np.asarray(values["waypoints"])).all():
        raise RuntimeError("Simulator did not consume a finite trajectory action")
    return values


def action_error(left, right):
    errors = []
    for key in ACTION_FIELDS:
        a, b = left[key], right[key]
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise RuntimeError("Actual / expected action field mismatch")
        errors.append(r15.max_abs_error(a, b))
    return max(errors, default=float("inf"))


def validate_sidecar(sidecar, collection, routing, code_sha, *, allow_terminal=False):
    audit = sidecar.get("cfpi_deployment", {})
    step = int(sidecar["planner_step"])
    route, active = d.route_for(routing, sidecar["scene_prefix"], step, allow_terminal=allow_terminal)
    scene = route["scene_id"]
    key = route["model_key"] if active else None
    model = routing["models"].get(key)
    expected = dict(schema_version=1, routing_sha256=collection["routing"]["sha256"],
                    terminal_unexecuted=(step == d.TERMINAL_PUBLICATION),
                    scene_id=scene, decision_step=step, fold=route["fold"], active=active, model_key=key,
                    selector_sha256=(model["sha256"] if model else routing["incumbent_selector_sha256"]),
                    score_mode=(model["score_mode"] if model else "residual"),
                    generator_forward_count=1, inference_uses_reward_or_q=False)
    if any(audit.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Deployment frame routing/provenance mismatch: {scene}/{step}")
    contract = sidecar.get("selector_rollout_contract", {})
    if collection.get("research_method") == "selector_rare_retention_v1":
        cohort = collection["cohort"]
        if (contract.get("research_method") != "selector_rare_retention_v1"
                or contract.get("source_data_split") != cohort
                or contract.get("development_consumed") != (cohort == "development")
                or contract.get("test_consumed") != (cohort == "confirmation")
                or contract.get("legacy_exposed_benchmark") is not True
                or contract.get("independent_unseen_test") is not False):
            raise RuntimeError("Rare evaluation exposure/cohort metadata changed")
    if (contract.get("experiment") != d.METHOD or
            contract.get("deployment_routing_sha256") != collection["routing"]["sha256"] or
            contract.get("rollout_implementation_sha256") != code_sha or
            contract.get("diagnostic_split") != collection["collection_id"] or
            contract.get("inference_uses_reward_or_q") is not False or
            sidecar.get("checkpoint_sha256") != c.CHECKPOINT_SHA256 or
            sidecar.get("candidate_noise_namespace") != c.NOISE_NAMESPACE or
            sidecar.get("code_sha") != code_sha):
        raise RuntimeError("Deployment sidecar/full-planner contract changed")
    values = {k: c.checked_array(sidecar[k], shape, k) for k, shape in SHAPES.items()}
    forced = False
    if collection.get('research_method') == 'selector_feedback_repair_v2':
        from selector_feedback_transport import validate_feedback
        forced = validate_feedback(sidecar,collection,route)
    selected = int(sidecar['selected_index']) if forced else int(np.argmax(values["current_logits"]))
    if any(int(value) != selected for value in (sidecar["selected_index"], sidecar["selected_indices"], audit["selected_index"])):
        raise RuntimeError("Published selector index differs from actual score argmax")
    incumbent_logits = c.checked_array(audit["incumbent_logits"], (20,), "incumbent logits")
    if int(audit["incumbent_index"]) != int(incumbent_logits.argmax()):
        raise RuntimeError("Incumbent argmax audit mismatch")
    if not active and (selected != int(audit["incumbent_index"]) or
                       r15.max_abs_error(values["current_logits"], incumbent_logits) != 0):
        raise RuntimeError("New policy changed a prefix action")
    if routing["policy"] == "scalar_v3":
        error = r15.max_abs_error(values["current_logits"], incumbent_logits)
        if (audit.get("same_model_recompute_max_abs") is None or
                error > c.MODEL_RECOMPUTE_TOLERANCE or selected != audit["incumbent_index"]):
            raise RuntimeError("Same-model routing sentinel parity failed")
    target = collection["audit_targets"].get(scene)
    initial = collection.get("initial_contexts", {}).get(scene)
    if initial and step == 4:
        original = c.load_pickle(d.verify(initial))
        error = max(r15.max_abs_error(values[k] if k != "current_logits" else incumbent_logits, original[k]) for k in SHAPES)
        if error > c.ARRAY_TOLERANCE:
            raise RuntimeError(f"Bridge initial context drift: {scene}: {error}")
    prefix_error, target_checked = None, False
    if target and str(step) in target["prefix"]:
        original = c.load_pickle(d.verify(target["prefix"][str(step)]))
        prefix_error = max(r15.max_abs_error(values[k] if k != "current_logits" else incumbent_logits, original[k])
                           for k in SHAPES)
        if prefix_error > c.ARRAY_TOLERANCE:
            raise RuntimeError(f"Frozen prefix/target context drift: {scene}/{step}: {prefix_error}")
        if collection["sentinel"] or step < target["target_decision"]:
            if selected != int(original["selected_index"]) or r15.max_abs_error(
                    sidecar["deployed_trajectory"], original["deployed_trajectory"]) > c.ACTION_TOLERANCE:
                raise RuntimeError("Frozen prefix / sentinel trajectory changed")
    if target and step == target["target_decision"]:
        target_checked = True
        if selected != target["expected_target_action"]:
            raise RuntimeError(f"Target action does not reproduce frozen OOF prediction: {scene}")
    c.checked_array(sidecar["deployed_trajectory"], (40, 3), "published trajectory")
    return dict(scene_id=scene, decision_step=step, selected_index=selected, active=active,
                terminal_unexecuted=(step == d.TERMINAL_PUBLICATION),
                prefix_max_abs=prefix_error, target_checked=target_checked)


def validate_coverage(rows, scenes, completed, reports):
    if set(completed) != set(scenes) or set(reports) != set(scenes) or any(not any(v) for v in reports.values()):
        raise RuntimeError("Completed-scene ledgers / successful runner reports are incomplete")
    found = defaultdict(set)
    for row in rows:
        scene, step = row["scene_id"], row["decision_step"]
        if step in found[scene]:
            raise RuntimeError(f"Duplicate logical frame across workers: {scene}/{step}")
        found[scene].add(step)
    if set(found) != set(scenes) or any(steps != set(c.DECISION_STEPS) for steps in found.values()):
        raise RuntimeError("Deployment requires all 8 decisions for every frozen scene")


def audit_collection(path, *, merge=True):
    path = Path(path).resolve()
    root, collection = path.parent, d.read(path)
    routing = d.verified_read(collection["routing"])
    contract = d.verified_read(collection["run_contract"])
    d.verify(collection["scenario"])
    d.verify(collection["inputs"])
    for model in routing["models"].values():
        d.verify(model)
    scenes = {r["scene_id"] for r in routing["routes"]}
    record_paths = sorted(root.glob("split_*/WE_output/openscene_format/cfpi_deployment_records/*.json"))
    checked, artifacts, workers, plans, consumed_sidecars = [], [], set(), {}, set()
    for record_path in record_paths:
        record = d.read(record_path)
        if record.get("collection_sha256") != c.sha256_file(path):
            raise RuntimeError("Actual action record belongs to a different collection")
        sidecar = c.load_pickle(d.verify(record["sidecar"]))
        consumed_sidecars.add(Path(record["sidecar"]["path"]).resolve())
        row = validate_sidecar(sidecar, collection, routing, contract["code_sha"])
        actual_error = action_error(record["actual_action"], record["expected_action"])
        if not np.isfinite(actual_error) or actual_error > c.ACTION_TOLERANCE:
            raise RuntimeError("Consumed trajectory differs from selected candidate action")
        if (record["state_step"] != row["decision_step"] - 1 or record["scene_id"] != row["scene_id"]
                or record["validation"] != row or not np.isfinite(record["candidate_trajectory_max_abs"])
                or record["candidate_trajectory_max_abs"] > c.ACTION_TOLERANCE):
            raise RuntimeError("Simulator timing / independent online validation mismatch")
        plan = np.load(d.verify(record["plan"]), allow_pickle=False)
        if r15.max_abs_error(plan, sidecar["deployed_trajectory"]) > c.ACTION_TOLERANCE:
            raise RuntimeError("Published plan differs from audited sidecar")
        worker = record_path.relative_to(root).parts[0]
        row.update(worker=worker, actual_action_max_abs=actual_error)
        checked.append(row)
        artifacts.extend((d.artifact(record_path), record["sidecar"], record["plan"]))
        workers.add(worker)
        plans[(sidecar["scene_prefix"], row["decision_step"])] = row["selected_index"]
    report_paths, reports = load_current_reports(root)
    ledger_paths, completed = completed_scenes(root)
    validate_coverage(checked, scenes, completed, reports)
    # The legacy planner emits a ninth, unconsumed publication at decision 12.
    # Preserve the same planner lifecycle, but never count it as an executed action.
    terminal_publications, terminal_scenes = [], set()
    scene_workers = {r["scene_id"]: r["worker"] for r in checked}
    for sidecar_path in sorted(root.glob("split_*/diffusiondrive_candidate_sidecars/*.pkl")):
        if sidecar_path.resolve() in consumed_sidecars:
            continue
        sidecar = c.load_pickle(sidecar_path)
        row = validate_sidecar(sidecar, collection, routing, contract["code_sha"], allow_terminal=True)
        worker = sidecar_path.relative_to(root).parts[0]
        scene = row["scene_id"]
        if (row["decision_step"] != d.TERMINAL_PUBLICATION or scene in terminal_scenes
                or scene_workers.get(scene) != worker):
            raise RuntimeError("Unknown / duplicate publication outside the executed decision horizon")
        plan_path = root / worker / "plan_traj" / f"{sidecar['scene_prefix']}_{d.TERMINAL_PUBLICATION}.npy"
        if r15.max_abs_error(np.load(plan_path, allow_pickle=False), sidecar["deployed_trajectory"]) > c.ACTION_TOLERANCE:
            raise RuntimeError("Terminal sidecar / publication mismatch")
        terminal_scenes.add(scene)
        terminal_publications.append(dict(row, worker=worker))
        plans[(sidecar["scene_prefix"], d.TERMINAL_PUBLICATION)] = row["selected_index"]
        artifacts.extend((d.artifact(sidecar_path), d.artifact(plan_path)))
    if len(workers) != contract["gpu_count"]:
        raise RuntimeError("Unexpected rollout worker coverage")
    seen_plans = {}
    for plan_csv in root.glob("split_*/plan_traj/plan_idx.csv"):
        with plan_csv.open() as stream:
            for item in csv.DictReader(stream):
                key = (item["prefix"], int(item["step"]))
                value = int(item["plan_idx"])
                # Identical retry publications are harmless; conflicting actions are not.
                if key not in plans or value != plans[key] or (key in seen_plans and seen_plans[key] != value):
                    raise RuntimeError("Planner CSV contains an unknown/conflicting decision")
                seen_plans[key] = value
        artifacts.append(d.artifact(plan_csv))
    if seen_plans != plans:
        raise RuntimeError("Missing planner CSV decisions")
    metrics = {}
    for worker in sorted(workers):
        mode = collection.get('react_type','R')
        if mode not in ('NR','R'):
            raise RuntimeError('Unknown metric mode')
        metric_path = root / worker / f"WE_output/openscene_format/all_scenes_pdm_averages_{mode}.csv"
        subset = d.metrics_csv(metric_path)
        if metrics.keys() & subset.keys():
            raise RuntimeError("Duplicate metric scene across workers")
        metrics.update(subset)
        artifacts.append(d.artifact(metric_path))
    if set(metrics) != scenes:
        raise RuntimeError("Final metric scene coverage differs from actions")
    if collection["sentinel"]:
        inputs = d.verified_read(collection["inputs"])
        reference = d.metrics_csv(d.verify(inputs["artifacts"]["baseline_csv"]))
        if any(abs(metrics[s][k] - reference[s][k]) > c.OUTCOME_TOLERANCE for s in scenes for k in d.METRICS):
            raise RuntimeError("Scalar routing sentinel changed closed-loop outcomes")
    artifacts.extend(d.artifact(p) for p in report_paths + ledger_paths)
    output = dict(status="PASS", method=d.METHOD, collection=d.artifact(path), records=len(checked),
                  scenes=len(scenes), workers=sorted(workers), artifacts=artifacts,
                  metrics=metrics, decisions=checked, incomplete_or_dropped_scenes=0,
                  terminal_unexecuted_publications=terminal_publications,
                  maximum_action_error=max(r["actual_action_max_abs"] for r in checked))
    if merge:
        destination = root / "continuous_metrics.csv"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["token", *d.METRICS])
            writer.writeheader()
            writer.writerows(dict(token=s, **metrics[s]) for s in sorted(metrics))
        temporary.replace(destination)
        output["merged_metrics"] = d.artifact(destination)
        c.locked_json(root / "collection_audit.json", output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    args = p.parse_args()
    result = audit_collection(args.collection)
    print(f"PASS: {result['scenes']} scenes / {result['records']} decisions / actual actions audited")


if __name__ == "__main__":
    main()
