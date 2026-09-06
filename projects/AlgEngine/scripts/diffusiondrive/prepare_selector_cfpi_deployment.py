"""Freeze inputs/routing before outcomes; materialize only the approved A/B scenes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cfpi_common as c
import selector_cfpi_deployment_common as d
from prepare_selector_cfpi import select_source_joint, historical_exposure


def fixed_models(targets, cv, cache_sha):
    bank, predictions, reports = {}, {}, {}
    for method, config in d.POLICIES.items():
        for seed in d.SEEDS:
            policy = d.learned_id(method, seed)
            predictions[policy] = {}
            for fold in range(4):
                key = f"{policy}_fold{fold}"
                path = cv / "cv" / f"{method}_lr{config['lr']:g}_fold{fold}_seed{seed}" / "report.json"
                report = d.read(path)
                expected = {r["scene_id"] for r in targets if r["fold"] == fold}
                provenance = dict(cache_sha256=cache_sha, method=method, learning_rate=config["lr"],
                                  fold=fold, seed=seed, step=config["steps"],
                                  incumbent_sha256=c.CHECKPOINT_SHA256)
                if (report.get("status") != "PASS" or report.get("train_scene_count") != 48
                        or report.get("heldout_scene_count") != 16 or len(expected) != 16
                        or report.get("score_mode") != config["score_mode"]
                        or report["incumbent_recompute_parity"]["argmax_mismatch_count"] != 0
                        or any(report["provenance"].get(k) != v for k, v in provenance.items()
                               if k not in {"step", "incumbent_sha256"})):
                    raise RuntimeError(f"Invalid frozen CV report: {path}")
                checkpoints = [r for r in report["checkpoints"] if r["step"] == config["steps"]]
                if len(checkpoints) != 1 or set(checkpoints[0]["predictions"]) != expected:
                    raise RuntimeError("CV checkpoint / held-out coverage drifted")
                checkpoint = checkpoints[0]
                item = dict(path=checkpoint["checkpoint"], sha256=checkpoint["checkpoint_sha256"])
                d.verify(item)
                if any(type(v) is not int or v not in range(20) for v in checkpoint["predictions"].values()):
                    raise RuntimeError("Invalid OOF action")
                bank[key] = dict(item, kind="cfpi", score_mode=config["score_mode"], provenance=provenance)
                predictions[policy].update(checkpoint["predictions"])
                reports[key] = d.artifact(path)
    return bank, predictions, reports


def natural_membership(source_audit, targets, baseline_csv):
    # Only the ID column is consumed. No success/event/return-dependent sampling.
    with Path(baseline_csv).open() as stream:
        scene_ids = [row["token"] for row in csv.DictReader(stream) if row["token"] != "overall_average"]
    if len(scene_ids) != 1024 or len(set(scene_ids)) != 1024:
        raise RuntimeError("Source ID coverage changed")
    excluded = {r["origin_log"] for r in targets}
    history = source_audit["historical_v3_membership_audit"]
    for split in ("development", "certification"):
        item = history["splits"][split]
        path = d.verify(dict(path=item["source"], sha256=item["sha256"]))
        import json
        excluded.update(json.loads(line)["log_name"] for line in path.read_text().splitlines() if line.strip())
    candidates, seen = {}, set()
    for entry in source_audit["source_inventories"]:
        index = d.verified_read(dict(path=entry["index"], sha256=entry["index_sha256"]))
        source_ids = {scene for shard in index["shards"] for scene in shard["scenario_ids"]}
        members = set(scene_ids) & source_ids
        if seen & members:
            raise RuntimeError("Ambiguous family membership")
        seen |= members
        candidates[entry["family"]] = [s for s in members if c.scene_origin_log(s) not in excluded]
    if seen != set(scene_ids):
        raise RuntimeError("Source IDs missing from upstream family indexes")
    selected, allocation = select_source_joint(candidates, per_family=64, maximum_per_log=1)
    rows = [dict(scene_id=scene, origin_token=c.scene_token(scene), origin_log=c.scene_origin_log(scene),
                 scenario_family=family, fold=None) for family in c.ALLOWED_FAMILIES for scene in selected[family]]
    if len(rows) != 128 or len({r["origin_log"] for r in rows}) != 128:
        raise RuntimeError("Natural128 must use exactly one scene per log")
    return dict(status="PASS", rows=rows, allocation=allocation, excluded_origin_logs=sorted(excluded),
                selection="cfpi-source-v1 deterministic joint ID-hash allocation; 64/family; 1/log",
                label_based_sampling=False, evaluation_role="training_pool_deployment_screening",
                historical_membership=historical_exposure({r["origin_log"] for r in rows}, Path(history["source"])),
                independent_development_or_test=False)


def freeze(args):
    pilot, cv, run = args.pilot_run.resolve(), args.cv_run.resolve(), args.run_root.resolve()
    inputs_path = run / "deployment_inputs.json"
    if inputs_path.exists():
        inputs = d.read(inputs_path)
        if inputs["pilot_run"] != str(pilot) or inputs["cv_run"] != str(cv):
            raise RuntimeError("Deployment input run changed")
        for item in [*inputs["artifacts"].values(), *inputs["cv_reports"].values(), *inputs["models"].values()]:
            d.verify(item)
        return inputs
    targets_path = pilot / "targets/target_manifest.json"
    targets = d.read(targets_path)
    cache_audit_path = pilot / "cache/pilot64_cache_audit.json"
    cache_audit = c.verified_json(cache_audit_path, "cache_file")
    gate = c.verified_json(pilot / "pilot_gate.json")
    if not gate.get("training_authorized") or gate["cache_sha256"] != cache_audit["cache_file_sha256"]:
        raise RuntimeError("Original pilot does not authorize these training data")
    for old in (pilot, cv):
        ledger = d.read(old / "decision_ledger.json")
        if ledger.get("active") or ledger.get("development_consumed") or ledger.get("test_consumed"):
            raise RuntimeError("Source run still active or outside train-only contract")
    if d.read(cv / "pilot_report.json").get("status") != "PASS":
        raise RuntimeError("CV report is incomplete")
    rows = targets["targets"]
    if targets.get("status") != "PASS" or len(rows) != 64 or len({r["scene_id"] for r in rows}) != 64:
        raise RuntimeError("Invalid pilot target membership")
    for row in rows:
        if targets["fold_by_origin_log"][row["origin_log"]] != row["fold"]:
            raise RuntimeError("Pilot log/fold routing mismatch")
    source = d.verified_read(dict(path=targets["source_audit"], sha256=targets["source_audit_sha256"]))
    baseline_path = pilot / "collections/baseline_train_a/collection_audit.json"
    baseline = c.verified_json(baseline_path)
    outcome = baseline["closed_loop_outcome"]
    d.verify(dict(path=outcome["metrics_csv"], sha256=outcome["metrics_csv_sha256"]))
    bank, predictions, reports = fixed_models(rows, cv, gate["cache_sha256"])
    legacy = {}
    for name, path, expected in [("scalar_v3", args.checkpoint_manifest, c.CHECKPOINT_SHA256),
                                 ("gate_v3", args.gate_manifest, d.GATE_CHECKPOINT_SHA)]:
        manifest = c.verified_json(path)
        if manifest["checkpoint_sha256"] != expected:
            raise RuntimeError("Wrong preregistered baseline")
        state = dict(path=manifest["scene_selector_state"], sha256=manifest["scene_selector_state_sha256"])
        d.verify(state)
        if name == "gate_v3" and state["sha256"] != d.GATE_SELECTOR_SHA:
            raise RuntimeError("Wrong preregistered gate selector")
        legacy[name] = dict(state, kind="legacy", score_mode="residual", provenance={})
    arms = []
    for item in cache_audit["arm_collection_audits"]:
        d.verify(item)
        d.verify(dict(path=item["metrics_csv"], sha256=item["metrics_csv_sha256"]))
        arms.append(item)
    if sorted(a["candidate_index"] for a in arms) != list(range(20)):
        raise RuntimeError("One-shot reference requires all 20 audited branches")
    membership = natural_membership(source, rows, outcome["metrics_csv"])
    c.locked_json(run / "natural128_membership.json", membership)
    files = dict(targets=targets_path, cache_audit=cache_audit_path, cache=cache_audit["cache_file"],
                 pilot_gate=pilot / "pilot_gate.json", source_audit=targets["source_audit"],
                 cv_report=cv / "pilot_report.json", scalar_manifest=args.checkpoint_manifest,
                 gate_manifest=args.gate_manifest, baseline_audit=baseline_path,
                 baseline_csv=outcome["metrics_csv"], natural_membership=run / "natural128_membership.json")
    inputs = dict(status="PASS", method=d.METHOD, pilot_run=str(pilot), cv_run=str(cv),
                  artifacts={k: d.artifact(v) for k, v in files.items()}, cv_reports=reports,
                  models=dict(bank, **legacy), oof_predictions=predictions, one_shot_arms=arms,
                  policies=d.POLICIES, seeds=list(d.SEEDS), advancement=d.ADVANCEMENT,
                  phase_limits=d.PHASE_LIMITS, expansion_authorized=False,
                  development_consumed=False, test_consumed=False)
    c.locked_json(inputs_path, inputs)
    return inputs


def save_subset(destination, source, scenes):
    """Only deserialize locally collected scenario files after manifest hash validation."""
    audit_path = destination.with_suffix(".json")
    expected_ids = sorted(scenes)
    if audit_path.exists():
        existing = d.read(audit_path)
        if existing["scene_ids"] != expected_ids or existing["source"] != source:
            raise RuntimeError("Scenario resume membership drift")
        d.verify(existing["scenario"])
        return existing["scenario"]
    payload = c.load_pickle(d.verify(source))
    subset = {s: payload[s] for s in expected_ids}
    if any(v["id"] != s or int(v["log_length"]) < 20 for s, v in subset.items()):
        raise RuntimeError("Scene materialization identity/length drift")
    c.atomic_pickle(destination, subset)
    item = d.artifact(destination)
    c.locked_json(audit_path, dict(status="PASS", scene_ids=expected_ids, source=source, scenario=item))
    return item


def collection_manifest(run, inputs, phase, policy, rows, scenario, models, predictions=None, sentinel=False):
    collection = f"{phase}_{policy}"
    folder = run / "collections" / collection
    routes, audits = [], {}
    for row in rows:
        fold = row["fold"]
        key = (f"{policy}_fold{fold}" if phase == "phase_a" and not sentinel else policy)
        start = row["decision_step"] if phase == "phase_a" and not sentinel else 4
        routes.append(dict(scene_id=row["scene_id"], origin_token=row["origin_token"], fold=fold,
                           start_decision=start, model_key=key))
        if phase == "phase_a":
            prefix = dict(row["prefix_records"])
            if sentinel:
                first = Path(prefix["4"]["path"])
                prefix = {str(step): d.artifact(first.with_name(f"{row['origin_token']}_{step}_cfpicausal.pkl"))
                          for step in c.DECISION_STEPS}
            for item in prefix.values():
                d.verify(item)
            audits[row["scene_id"]] = dict(prefix=prefix, target_decision=row["decision_step"],
                expected_target_action=(row["policy_index"] if sentinel else predictions[row["scene_id"]]),
                baseline_score=row["baseline_score"])
    routing = dict(status="PASS", method=d.METHOD, phase=phase, policy=("scalar_v3" if sentinel else policy),
                   terminal_publication_decision=d.TERMINAL_PUBLICATION,
                   models={k: models[k] for k in sorted({r["model_key"] for r in routes})}, routes=routes,
                   incumbent_checkpoint_sha256=c.CHECKPOINT_SHA256,
                   incumbent_selector_sha256=inputs["models"]["scalar_v3"]["sha256"],
                   inference_uses_reward_or_q=False)
    c.locked_json(folder / "routing.json", routing)
    contract = dict(status="PASS", method=d.METHOD, collection_id=collection, phase=phase, policy=policy,
                    sentinel=sentinel, routing=d.artifact(folder / "routing.json"), scenario=scenario,
                    inputs=d.artifact(run / "deployment_inputs.json"),
                    run_contract=d.artifact(run / "run_contract.json"), audit_targets=audits,
                    checkpoint_sha256=c.CHECKPOINT_SHA256, noise_namespace=c.NOISE_NAMESPACE,
                    expected_decisions=list(c.DECISION_STEPS))
    c.locked_json(folder / "deployment_collection.json", contract)
    return d.artifact(folder / "deployment_collection.json")


def phase_a(run, inputs):
    targets = d.verified_read(inputs["artifacts"]["targets"])
    rows = targets["targets"]
    scenario = dict(path=targets["pilot64_scenario_file"], sha256=targets["pilot64_scenario_file_sha256"])
    d.verify(scenario)
    sentinel_rows = [r for r in rows if r["scene_id"] in targets["repeat_scene_ids"]]
    if len(sentinel_rows) != 8:
        raise RuntimeError("Sentinel membership must be the frozen repeat8")
    subset = save_subset(run / "scenarios/sentinel8.pkl", scenario, [r["scene_id"] for r in sentinel_rows])
    collections = [collection_manifest(run, inputs, "phase_a", "scalar_v3", sentinel_rows, subset,
                                       inputs["models"], sentinel=True)]
    for method in d.POLICIES:
        for seed in d.SEEDS:
            policy = d.learned_id(method, seed)
            collections.append(collection_manifest(run, inputs, "phase_a", policy, rows, scenario,
                                                    inputs["models"], inputs["oof_predictions"][policy]))
    c.locked_json(run / "phase_a_collections.json", dict(status="PASS", collections=collections))


def phase_b(run, inputs):
    gate = d.read(run / "phase_a_report.json")
    if gate.get("status") != "PASS" or gate.get("phase_b_authorized") is not True:
        raise RuntimeError("Phase A has not authorized phase B")
    membership = d.verified_read(inputs["artifacts"]["natural_membership"])
    rows = membership["rows"]
    source = d.verified_read(inputs["artifacts"]["source_audit"])
    scenario = save_subset(run / "scenarios/natural128.pkl",
                           dict(path=source["train_scenario_file"], sha256=source["train_scenario_file_sha256"]),
                           [r["scene_id"] for r in rows])
    models, collections = dict(inputs["models"]), []
    for policy in ("scalar_v3", "gate_v3"):
        collections.append(collection_manifest(run, inputs, "phase_b", policy, rows, scenario, models))
    for method, config in d.POLICIES.items():
        for seed in d.SEEDS:
            policy = d.learned_id(method, seed)
            report = c.verified_json(run / "refit" / policy / "report.json")
            if report["method"] != method or report["seed"] != seed or report["train_scene_count"] != 64:
                raise RuntimeError("Refit report identity drift")
            models[policy] = dict(report["selector"], kind="cfpi", score_mode=config["score_mode"],
                                  provenance=report["provenance"])
            collections.append(collection_manifest(run, inputs, "phase_b", policy, rows, scenario, models))
    c.locked_json(run / "phase_b_collections.json", dict(status="PASS", collections=collections,
                  phase_a_authorization=d.artifact(run / "phase_a_report.json")))


def verify_baselines(run, inputs):
    import torch
    from materialize_grpo_selector_v3 import state_dict, PREFIX
    manifests = [d.verified_read(inputs["artifacts"][k]) for k in ("scalar_manifest", "gate_manifest")]
    states = []
    for manifest in manifests:
        path = d.verify(dict(path=manifest["checkpoint"], sha256=manifest["checkpoint_sha256"]))
        state = state_dict(torch.load(path, map_location="cpu"))
        selector = torch.load(manifest["scene_selector_state"], map_location="cpu")["scene_selector_state"]
        keys = {k for k in state if k.startswith(PREFIX)}
        if keys != {PREFIX + k for k in selector} or any(not torch.equal(state[PREFIX + k], v) for k, v in selector.items()):
            raise RuntimeError("Full baseline / selector bank tensor mismatch")
        states.append({k: v for k, v in state.items() if not k.startswith(PREFIX)})
    if states[0].keys() != states[1].keys() or any(not torch.equal(states[0][k], v) for k, v in states[1].items()):
        raise RuntimeError("Gate baseline differs outside selector; cannot use shared frozen forward")
    c.locked_json(run / "baseline_tensor_audit.json", dict(status="PASS", unchanged_nonselector_tensors=len(states[0]),
                  shared_forward_checkpoint_sha256=c.CHECKPOINT_SHA256, gate_checkpoint_sha256=d.GATE_CHECKPOINT_SHA,
                  inputs=d.artifact(run / "deployment_inputs.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "phase_a", "phase_b", "verify_baselines"))
    for key in ("run-root", "pilot-run", "cv-run", "checkpoint-manifest", "gate-manifest"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    inputs = freeze(args)
    if args.stage != "freeze":
        globals()[args.stage](args.run_root.resolve(), inputs)
    print(f"PASS: deployment preparation {args.stage}")


if __name__ == "__main__":
    main()
