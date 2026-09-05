#!/usr/bin/env python3
"""Freeze reward/Q-blind CFPI causal targets from repeatable V3 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import cfpi_common as common


def load_outcomes(path: Path) -> dict[str, dict]:
    rows = {}
    with path.expanduser().resolve().open(newline="") as stream:
        for row in csv.DictReader(stream):
            scene = str(row["token"])
            if scene == "overall_average":
                continue
            if scene in rows:
                raise RuntimeError(f"duplicate outcome row: {scene}")
            collision = float(row["no_at_fault_collisions"])
            drivable = float(row["drivable_area_compliance"])
            raw_violation = str(row.get("first_violation_step", "") or "").strip()
            rows[scene] = {
                "score": float(row["score"]),
                "success": bool(collision >= 1.0 and drivable >= 1.0),
                "no_at_fault_collisions": collision,
                "drivable_area_compliance": drivable,
                "ego_progress": float(row["ego_progress"]),
                "first_violation_step": (
                    None if not raw_violation else int(float(raw_violation))
                ),
            }
            if not all(np.isfinite(rows[scene][key]) for key in (
                    "score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress")):
                raise RuntimeError(f"Nonfinite closed-loop outcome: {scene}")
            if not 0 <= rows[scene]["score"] <= 1:
                raise RuntimeError(f"Out-of-range closed-loop return: {scene}")
    if not rows:
        raise RuntimeError(f"empty outcome file: {path}")
    return rows


def load_baseline(path: Path, collection_id: str, split: str) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != common.COLLECTION_AUDIT_METHOD
        or payload.get("layout") != "merged"
        or payload.get("collection_id") != collection_id
        or payload.get("source_split") != split
        or payload.get("intervention_mode") != "observe_only"
        or int(payload.get("intervention_count", -1)) != 0
        or not payload.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"invalid CFPI baseline audit: {path}")
    return path, payload


def load_records(audit: dict) -> dict[str, dict[int, tuple[Path, dict]]]:
    root = Path(audit["rollout_root"])
    record_root = root / "WE_output/openscene_format/diffusiondrive_cfpi_causal_records"
    result = defaultdict(dict)
    for path in sorted(record_root.glob("*_cfpicausal.pkl")):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        scene = str(row["rollout_scene_id"])
        step = int(row["decision_step"])
        if step in result[scene]:
            raise RuntimeError(f"duplicate baseline frame: {(scene, step)}")
        result[scene][step] = (path, row)
    if not result:
        raise RuntimeError(f"no CFPI baseline records under {record_root}")
    return result


def outcomes_agree(left: dict, right: dict) -> bool:
    discrete = (
        left["success"] == right["success"]
        and left["first_violation_step"] == right["first_violation_step"]
        and left["no_at_fault_collisions"] == right["no_at_fault_collisions"]
        and left["drivable_area_compliance"] == right["drivable_area_compliance"]
    )
    continuous = max(
        abs(float(left[key]) - float(right[key]))
        for key in ("score", "ego_progress")
    )
    return bool(discrete and continuous <= common.OUTCOME_TOLERANCE)


def validate_context_pair(row_a: dict, row_b: dict) -> tuple[np.ndarray, np.ndarray, int]:
    for key, shape in {
        "candidate_features": (20, 256), "route_bev_features": (20, 8, 256),
        "status_tokens": (1, 256), "ego_queries": (1, 256), "agents_queries": (30, 256),
        "reference_logits": (20,),
    }.items():
        left = common.checked_array(row_a[key], shape, key)
        right = common.checked_array(row_b[key], shape, key)
        if r15.max_abs_error(left, right) > common.ARRAY_TOLERANCE:
            raise RuntimeError(f"A/B selector-visible context drift: {key}")
    candidates_a = common.checked_array(
        row_a["candidate_trajectories_8"], (20, 8, 3), "baseline A candidates"
    )
    candidates_b = common.checked_array(
        row_b["candidate_trajectories_8"], (20, 8, 3), "baseline B candidates"
    )
    logits_a = common.checked_array(
        row_a["current_logits"], (20,), "baseline A logits"
    )
    logits_b = common.checked_array(
        row_b["current_logits"], (20,), "baseline B logits"
    )
    errors = {
        "candidate_trajectories": r15.max_abs_error(candidates_a, candidates_b),
        "current_logits": r15.max_abs_error(logits_a, logits_b),
    }
    if any(value > common.ARRAY_TOLERANCE for value in errors.values()):
        raise RuntimeError(f"baseline candidate/logit rerun drift: {errors}")
    policy_a = int(row_a["policy_selected_index"])
    policy_b = int(row_b["policy_selected_index"])
    if (
        policy_a != policy_b
        or policy_a != int(np.argmax(logits_a))
        or policy_b != int(np.argmax(logits_b))
    ):
        raise RuntimeError("baseline policy action rerun drift")
    return candidates_a, logits_a, policy_a


def make_target(
    row: dict,
    frames_a: dict[str, dict[int, tuple[Path, dict]]],
    frames_b: dict[str, dict[int, tuple[Path, dict]]],
) -> dict:
    scene_id = row["scene_id"]
    step = int(row["decision_step"])
    path_a, record_a = frames_a[scene_id][step]
    path_b, record_b = frames_b[scene_id][step]
    candidates, logits, policy = validate_context_pair(record_a, record_b)
    rewards_a = common.checked_array(
        record_a["candidate_rewards"], (20,), "baseline A local rewards"
    )
    rewards_b = common.checked_array(
        record_b["candidate_rewards"], (20,), "baseline B local rewards"
    )
    reward_error = r15.max_abs_error(rewards_a, rewards_b)
    # Local reward is not a target-eligibility signal.  The pairwise-progress
    # diagnostic is known to have small rerun drift even when candidates,
    # logits, deployed actions and outcomes are stable.  Freeze A for the
    # matched local-label controls and report (but never select on) A/B drift.
    outcome = row["outcome_a"]
    return {
        "scene_id": scene_id,
        "split": row["split"],
        "scenario_family": row["scenario_family"],
        "origin_log": row["origin_log"],
        "origin_token": row["origin_token"],
        "outcome_stratum": row["outcome_stratum"],
        "decision_step": step,
        "state_step": int(record_a["state_step"]),
        "policy_index": policy,
        "candidate_trajectories_8": candidates.astype(np.float32).tolist(),
        "current_logits": logits.astype(np.float32).tolist(),
        "candidate_rewards": rewards_a.astype(np.float32).tolist(),
        "candidate_reward_components": common.checked_array(
            record_a["candidate_reward_components"], (20, 6),
            "baseline A reward components",
        ).astype(np.float32).tolist(),
        "local_reward_rerun_max_abs_error": reward_error,
        "local_reward_rerun_stable_at_array_tolerance": bool(
            reward_error <= common.ARRAY_TOLERANCE
        ),
        "baseline_score": float(outcome["score"]),
        "baseline_success": bool(outcome["success"]),
        "baseline_ego_progress": float(outcome["ego_progress"]),
        "first_violation_step": outcome["first_violation_step"],
        "source_record_a": str(path_a.resolve()),
        "source_record_a_sha256": common.sha256_file(path_a),
        "source_record_b": str(path_b.resolve()),
        "source_record_b_sha256": common.sha256_file(path_b),
        "prefix_records": {
            str(s): {"path": str(frames_a[scene_id][s][0].resolve()),
                     "sha256": common.sha256_file(frames_a[scene_id][s][0])}
            for s in common.DECISION_STEPS if s <= step
        },
    }


def select_source_joint(candidates, per_family=512, maximum_per_log=4):
    """Complete hash-greedy quotas by deterministic residual-network rerouting.

    A family-first greedy failure is not a proof of joint infeasibility. Seed
    the family -> origin-log capacity network with that greedy allocation,
    then augment it. Only source IDs are consumed: no outcomes or Q values.
    """
    families = common.ALLOWED_FAMILIES
    if set(candidates) != set(families) or per_family <= 0 or maximum_per_log <= 0:
        raise ValueError("Invalid joint source contract")
    ranked, available, greedy, allocated = {}, {}, {}, {}
    seen, used = set(), Counter()
    for family in families:
        values = list(candidates[family])
        if len(values) != len(set(values)) or seen & set(values):
            raise RuntimeError("Duplicate or overlapping source scene")
        seen.update(values)
        ranked[family] = sorted(values, key=lambda scene: (
            common.stable_digest("cfpi-source-v1", family, scene), scene))
        available[family] = Counter(common.scene_origin_log(s) for s in ranked[family])
        greedy[family], allocated[family] = [], Counter()
        for scene in ranked[family]:
            log = common.scene_origin_log(scene)
            if used[log] < maximum_per_log:
                greedy[family].append(scene)
                used[log] += 1
                allocated[family][log] += 1
            if len(greedy[family]) == per_family:
                break

    source, sink = ("source",), ("sink",)
    graph = defaultdict(dict)

    def edge(left, right, capacity, initial):
        if not 0 <= initial <= capacity:
            raise RuntimeError("Invalid initial source allocation")
        graph[left][right] = capacity - initial
        graph[right][left] = initial

    logs = sorted({log for family in families for log in available[family]},
                  key=lambda log: (common.stable_digest("cfpi-source-capacity-v1", log), log))
    for family in families:
        node = ("family", family)
        edge(source, node, per_family, len(greedy[family]))
        # Counter insertion order is the first scene's frozen hash rank.
        for log, count in available[family].items():
            edge(node, ("log", log), min(maximum_per_log, count), allocated[family][log])
    for log in logs:
        edge(("log", log), sink, maximum_per_log, used[log])
    total = sum(map(len, greedy.values()))
    augmentations = 0
    while total < per_family * len(families):
        parents, queue = {source: None}, deque([source])
        while queue and sink not in parents:
            left = queue.popleft()
            for right, capacity in graph[left].items():
                if capacity > 0 and right not in parents:
                    parents[right] = left
                    queue.append(right)
        if sink not in parents:
            raise RuntimeError(f"Joint clean source infeasible: maximum {total} < "
                               f"{per_family * len(families)} with per-log cap {maximum_per_log}; "
                               "do not relax quotas or exclusions")
        amount, right = per_family * len(families), sink
        while parents[right] is not None:
            left = parents[right]
            amount = min(amount, graph[left][right])
            right = left
        right = sink
        while parents[right] is not None:
            left = parents[right]
            graph[left][right] -= amount
            graph[right][left] += amount
            right = left
        total += amount
        augmentations += 1

    selected, used = {}, Counter()
    for family in families:
        remaining = {log: graph[("log", log)][("family", family)] for log in available[family]}
        selected[family] = []
        for scene in ranked[family]:
            log = common.scene_origin_log(scene)
            if remaining[log] > 0:
                remaining[log] -= 1
                selected[family].append(scene)
                used[log] += 1
        if len(selected[family]) != per_family or any(remaining.values()):
            raise RuntimeError("Joint source materialization disagrees with allocation")
    if max(used.values()) > maximum_per_log:
        raise RuntimeError("Joint source log cap violated")
    details = dict(method="hash_greedy_residual_capacity_v1", per_family=per_family,
                   maximum_per_log=maximum_per_log, augmentations=augmentations,
                   selection_uses_outcomes_or_q=False,
                   eligible_counts={f: len(ranked[f]) for f in families},
                   independent_capacities={f: sum(min(maximum_per_log, n)
                                                 for n in available[f].values()) for f in families},
                   joint_capacity_upper_bound=sum(min(maximum_per_log, sum(available[f][log]
                                                       for f in families)) for log in logs),
                   legacy_greedy_counts={f: len(greedy[f]) for f in families},
                   selected_counts={f: len(selected[f]) for f in families},
                   removed_from_greedy={f: len(set(greedy[f]) - set(selected[f])) for f in families},
                   actual_maximum_per_log=max(used.values()), selected_origin_logs=len(used))
    return selected, details


def source_pool(args):
    """Read only train shard membership; exclude previous diagnostic origin logs."""
    from prepare_selector_v4_causal_source import validate_split_contract, validate_scene_id
    output = args.run_root / "source"
    audit_path = output / "source_audit.json"
    if audit_path.exists():
        _, audit = common.load_source_audit(audit_path)
        if common.sha256_file(args.exclusions) != audit["exclusions_sha256"]:
            raise RuntimeError("Exclusion inventory drifted")
        for inventory in audit["source_inventories"]:
            if common.sha256_file(inventory["index"]) != inventory["index_sha256"]:
                raise RuntimeError("Upstream train index drifted")
            for path, expected in inventory["source_shards"].items():
                if common.sha256_file(path) != expected:
                    raise RuntimeError("Upstream train shard drifted")
        return audit
    _, token_splits, log_sets = validate_split_contract(args.source_root)
    exclusion = json.loads(args.exclusions.read_text())
    excluded = set(exclusion["excluded_origin_logs"])
    if len(excluded) < 15:
        raise RuntimeError("Missing prior diagnostic log exclusions")
    pool, inventories, candidates_by_family = {}, [], {}
    for family in common.ALLOWED_FAMILIES:
        index_path = args.source_root / "p2_shards_v1" / family / "index.json"
        index = json.loads(index_path.read_text())
        if (index["scenario_family"] != family or index["scenario_variant"] != "original"
                or index["revision"] != "diffusiondrive_grpo_v4_p2_shards_v1"):
            raise RuntimeError("Unexpected upstream shard contract")
        candidates, shard_hashes = {}, {}
        for shard in index["shards"]:
            shard_path = Path(shard["path"])
            content = common.load_pickle(shard_path)
            if set(content) != set(shard["scenario_ids"]):
                raise RuntimeError("Upstream shard membership mismatch")
            shard_hashes[str(shard_path)] = common.sha256_file(shard_path)
            for scene in shard["scenario_ids"]:
                validate_scene_id(scene, "train", token_splits, log_sets)
                if scene in candidates:
                    raise RuntimeError("Duplicate upstream scene")
                if content[scene].get("id") != scene:
                    raise RuntimeError("Upstream scenario identity mismatch")
                if (common.scene_origin_log(scene) not in excluded
                        and int(content[scene].get("log_length", -1)) >= 20):
                    candidates[scene] = shard_path
            del content
        candidates_by_family[family] = candidates
        inventories.append({"family": family, "index": str(index_path),
                            "index_sha256": common.sha256_file(index_path),
                            "source_shards": shard_hashes})
        print(json.dumps(dict(stage="source_inventory", family=family,
                              eligible_count=len(candidates))), flush=True)
    selection, selection_audit = select_source_joint(
        candidates_by_family, common.TRAIN_POOL_PER_FAMILY, common.MAXIMUM_POOL_SCENES_PER_LOG)
    print(json.dumps(dict(stage="source_allocation", **selection_audit)), flush=True)
    for family in common.ALLOWED_FAMILIES:
        candidates, selected = candidates_by_family[family], selection[family]
        grouped = defaultdict(list)
        for scene in selected:
            grouped[candidates[scene]].append(scene)
        materialized = {}
        for path, scenes in grouped.items():
            content = common.load_pickle(path)
            for scene in scenes:
                materialized[scene] = common.annotate_scenario(content[scene], family, "train")
            del content
        pool.update({scene: materialized[scene] for scene in selected})
    scenario_path = output / "train_pool_1024.pkl"
    common.atomic_pickle(scenario_path, pool)
    logs = {common.scene_origin_log(s) for s in pool}
    history = historical_exposure(
        logs, args.source_root.parent / "diffusiondrive_selector_rare_original_v1/tuning_split/split_audit.json")
    audit = dict(schema_version=common.SCHEMA_VERSION, design_version=common.DESIGN_VERSION,
                 status="PASS", method=common.SOURCE_METHOD, train_scenario_count=len(pool),
                 train_scenario_file=str(scenario_path.resolve()),
                 train_scenario_file_sha256=common.sha256_file(scenario_path),
                 train_origin_log_count=len(logs), excluded_origin_log_overlap=len(logs & excluded),
                 origin_log_overlap=0, source_inventories=inventories, source_selection=selection_audit,
                 exclusions=str(args.exclusions.resolve()),
                 exclusions_sha256=common.sha256_file(args.exclusions),
                 historical_exposure_status=exclusion["historical_exposure_status"],
                 historical_v3_membership_audit=history,
                 validation_collected=False, development_consumed=False, test_consumed=False)
    common.locked_json(audit_path, audit)
    return audit


def historical_exposure(logs, split_audit):
    """Membership-only audit, never reads historical evaluation outcomes."""
    payload = common.verified_json(split_audit)
    result = {"source": str(split_audit.resolve()), "sha256": common.sha256_file(split_audit),
              "coverage": "known V3 rare-tuning splits; not an exhaustive pretraining/history audit",
              "fully_unseen_claim_authorized": False, "splits": {}}
    for split in ("train", "development", "certification"):
        row = payload["splits"][split]
        path = Path(row["pairs"])
        if common.sha256_file(path) != row["pairs_sha256"]:
            raise RuntimeError("Historical membership inventory drifted")
        with path.open() as stream:
            known_logs = {json.loads(line)["log_name"] for line in stream if line.strip()}
        result["splits"][split] = dict(source=str(path), sha256=common.sha256_file(path),
                                      source_log_count=len(known_logs),
                                      overlapping_origin_logs=sorted(logs & known_logs),
                                      overlapping_log_count=len(logs & known_logs))
    return result


def select_stratified(rows, per_cell=8):
    """Hash-stable 2 family x 2 outcome x 2 timing cells, globally log-capped."""
    selected, used, counts = [], set(), Counter()
    for family in common.ALLOWED_FAMILIES:
        for outcome in ("failed", "solved"):
            matching = [r for r in rows if r["scenario_family"] == family
                        and r["outcome_stratum"] == outcome]
            failed_near = [r for r in selected if r["scenario_family"] == family
                           and r["outcome_stratum"] == "failed" and r["time_stratum"] == "near"]
            for timing in ("near", "ordinary"):
                chosen = []
                for row in sorted(matching, key=lambda x: common.stable_digest(
                        "cfpi-target-v1", family, outcome, timing, x["scene_id"])):
                    scene, log = row["scene_id"], row["origin_log"]
                    if scene in used or counts[log] >= 2:
                        continue
                    if timing == "near" and outcome == "solved":
                        step = sorted(r["decision_step"] for r in failed_near)[len(chosen)]
                        choices = [step] if step in row["all_decisions"] else []
                    else:
                        choices = row["near_decisions"] if timing == "near" else row["all_decisions"]
                    if not choices:
                        continue
                    offset = int(common.stable_digest("cfpi-step-v1", timing, scene)[:16], 16)
                    item = dict(row, time_stratum=timing, decision_step=int(choices[offset % len(choices)]))
                    chosen.append(item)
                    used.add(scene)
                    counts[log] += 1
                    if len(chosen) == per_cell:
                        break
                if len(chosen) != per_cell:
                    raise RuntimeError(f"Insufficient target cell {family}/{outcome}/{timing}: "
                                       f"{len(chosen)} != {per_cell}; do not alter quotas")
                selected += chosen
    return selected


def assign_folds(targets):
    """Deterministic grouped balancing without branch labels."""
    logs = sorted({r["origin_log"] for r in targets},
                  key=lambda s: common.stable_digest("cfpi-folds-v1", s))
    counts, cell_counts, mapping = [0] * 4, [Counter() for _ in range(4)], {}
    for log in logs:
        group = [r for r in targets if r["origin_log"] == log]
        cells = [(r["scenario_family"], r["outcome_stratum"], r["time_stratum"]) for r in group]
        fold = min(range(4), key=lambda f: (counts[f], sum(cell_counts[f][c] for c in cells), f))
        mapping[log] = fold
        counts[fold] += len(group)
        cell_counts[fold].update(cells)
    if min(counts) == 0:
        raise RuntimeError("Insufficient logs for four-fold CV")
    for target in targets:
        target["fold"] = mapping[target["origin_log"]]
    return mapping


def freeze_targets(args):
    audit = source_pool(args)
    destination = args.run_root / "targets"
    target_path = destination / "target_manifest.json"
    if target_path.exists():
        _, existing = common.load_target_manifest(target_path)
        if common.sha256_file(args.protocol) != existing["protocol_file_sha256"]:
            raise RuntimeError("Target protocol changed")
        return target_path
    source = common.load_pickle(audit["train_scenario_file"])
    audits = []
    for label in ("a", "b"):
        path = args.run_root / "collections" / f"baseline_train_{label}" / "collection_audit.json"
        _, collected = load_baseline(path, f"baseline_train_{label}", "train")
        outcome = collected["closed_loop_outcome"]
        if common.sha256_file(outcome["metrics_csv"]) != outcome["metrics_csv_sha256"]:
            raise RuntimeError("Baseline metrics changed")
        if collected["scenario_file_sha256"] != audit["train_scenario_file_sha256"]:
            raise RuntimeError("Baseline used a different source")
        audits.append(collected)
    for key in ("checkpoint_sha256", "selector_state_sha256", "candidate_noise_namespace",
                "rollout_implementation_sha256", "code_sha"):
        if audits[0][key] != audits[1][key]:
            raise RuntimeError(f"A/B provenance differs: {key}")
    outcomes = [load_outcomes(Path(a["closed_loop_outcome"]["metrics_csv"])) for a in audits]
    frames = [load_records(a) for a in audits]
    if any(set(x) != set(source) for x in outcomes + frames):
        raise RuntimeError("A/B scenario coverage differs")
    eligible, reasons = [], Counter()
    for scene, scenario in sorted(source.items()):
        a, b = outcomes[0][scene], outcomes[1][scene]
        if not outcomes_agree(a, b):
            raise RuntimeError(f"A/B closed-loop outcome drift: {scene}")
        if any(set(f[scene]) != set(common.DECISION_STEPS) for f in frames):
            raise RuntimeError(f"Missing baseline decisions: {scene}")
        for step in common.DECISION_STEPS:
            validate_context_pair(frames[0][scene][step][1], frames[1][scene][step][1])
        decisions = [s for s in common.DECISION_STEPS if a["success"]
                     or (a["first_violation_step"] is not None and s < a["first_violation_step"])]
        if not decisions:
            reasons["no_previolation_decision"] += 1
            continue
        metadata = common.source_metadata(scenario)
        eligible.append(dict(metadata, scene_id=scene,
                             outcome_stratum="solved" if a["success"] else "failed",
                             all_decisions=decisions, near_decisions=decisions[-3:],
                             outcome_a=a, outcome_b=b))
    chosen = select_stratified(eligible)
    targets = [dict(make_target(row, frames[0], frames[1]),
                    time_stratum=row["time_stratum"], stage="pilot64") for row in chosen]
    fold_map = assign_folds(targets)
    repeats = []
    for family in common.ALLOWED_FAMILIES:
        for outcome in ("failed", "solved"):
            for timing in ("near", "ordinary"):
                group = [r for r in targets if (r["scenario_family"], r["outcome_stratum"], r["time_stratum"])
                         == (family, outcome, timing)]
                repeats.append(min(group, key=lambda r: common.stable_digest("cfpi-repeat-v1", r["scene_id"]))["scene_id"])
    scenario_path = destination / "pilot64_scenarios.pkl"
    common.atomic_pickle(scenario_path, {r["scene_id"]: source[r["scene_id"]] for r in targets})
    payload = dict(schema_version=common.SCHEMA_VERSION, design_version=common.DESIGN_VERSION,
                   status="PASS", method=common.TARGET_METHOD,
                   intervention_outcomes_observed_before_freeze=False, target_count=64,
                   targets=targets, repeat_scene_ids=repeats, fold_by_origin_log=fold_map,
                   protocol_file=str(args.protocol.resolve()),
                   protocol_file_sha256=common.sha256_file(args.protocol),
                   source_audit=str((args.run_root / "source/source_audit.json").resolve()),
                   source_audit_sha256=common.sha256_file(args.run_root / "source/source_audit.json"),
                   pilot64_scenario_file=str(scenario_path.resolve()),
                   pilot64_scenario_file_sha256=common.sha256_file(scenario_path),
                   selector_state_sha256=audits[0]["selector_state_sha256"],
                   baseline_rollout_implementation_sha256=audits[0]["rollout_implementation_sha256"],
                   excluded_eligibility_reasons=dict(reasons),
                   development_consumed=False, test_consumed=False)
    common.locked_json(target_path, payload)
    common.load_target_manifest(target_path)
    return target_path


def treatments(args):
    target_path = freeze_targets(args)
    _, target = common.load_target_manifest(target_path)
    output = args.run_root / "manifests"
    source = common.load_pickle(target["pilot64_scenario_file"])
    repeat_path = output / "repeat8_scenarios.pkl"
    common.atomic_pickle(repeat_path, {s: source[s] for s in target["repeat_scene_ids"]})
    collections = {}
    for collection_id, index, stage in [("sentinel_policy", None, "sentinel")] + [
            (f"{stage}_arm_{i:02d}", i, stage) for stage in ("pilot64", "repeat8") for i in range(20)]:
        subset = target["targets"] if stage == "pilot64" else [
            r for r in target["targets"] if r["scene_id"] in target["repeat_scene_ids"]]
        scenario = Path(target["pilot64_scenario_file"]) if stage == "pilot64" else repeat_path
        payload = dict(schema_version=common.SCHEMA_VERSION, design_version=common.DESIGN_VERSION,
                       status="PASS", method=common.TREATMENT_METHOD, collection_id=collection_id,
                       stage=stage, source_split="train",
                       treatment_kind="policy_sentinel" if index is None else "fixed_candidate",
                       treatment_index=index, future_outcome_used_to_choose_treatment=False,
                       target_manifest=str(target_path.resolve()), target_manifest_sha256=common.sha256_file(target_path),
                       protocol_file=str(args.protocol.resolve()), protocol_file_sha256=common.sha256_file(args.protocol),
                       scenario_file=str(scenario.resolve()), scenario_file_sha256=common.sha256_file(scenario),
                       target_count=len(subset), targets=[dict(scene_id=r["scene_id"], decision_step=r["decision_step"],
                       treatment_index=r["policy_index"] if index is None else index) for r in subset])
        path = output / (collection_id + ".json")
        common.locked_json(path, payload)
        common.load_treatment(path)
        collections[collection_id] = dict(path=str(path.resolve()), sha256=common.sha256_file(path),
                                          stage=stage, source_split="train", treatment_index=index)
    master = dict(schema_version=common.SCHEMA_VERSION, design_version=common.DESIGN_VERSION,
                  status="PASS", method=common.MASTER_METHOD,
                  target_manifest=str(target_path.resolve()), target_manifest_sha256=common.sha256_file(target_path),
                  collections=collections, method_training_authorized=False,
                  expansion_authorized=False, development_consumed=False, test_consumed=False)
    common.locked_json(output / "master_manifest.json", master)
    return master


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "freeze"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    result = source_pool(args) if args.stage == "source" else treatments(args)
    print(json.dumps({"status": "PASS", "stage": args.stage, "run_root": str(args.run_root)}))


if __name__ == "__main__":
    main()
