#!/usr/bin/env python3
"""Freeze reward/Q-blind V4 causal targets from repeatable V3 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import oracle_r15_common as r15
import v4_causal_cache_common as common


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
        raise RuntimeError(f"invalid V4 baseline audit: {path}")
    return path, payload


def load_records(audit: dict) -> dict[str, dict[int, tuple[Path, dict]]]:
    root = Path(audit["rollout_root"])
    record_root = root / "WE_output/openscene_format/diffusiondrive_v4_causal_records"
    result = defaultdict(dict)
    for path in sorted(record_root.glob("*_v4causal.pkl")):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        scene = str(row["rollout_scene_id"])
        step = int(row["decision_step"])
        if step in result[scene]:
            raise RuntimeError(f"duplicate baseline frame: {(scene, step)}")
        result[scene][step] = (path, row)
    if not result:
        raise RuntimeError(f"no V4 baseline records under {record_root}")
    return result


def outcomes_agree(left: dict, right: dict) -> bool:
    discrete = (
        left["success"] == right["success"]
        and left["first_violation_step"] == right["first_violation_step"]
    )
    continuous = max(
        abs(float(left[key]) - float(right[key]))
        for key in ("score", "ego_progress")
    )
    return bool(discrete and continuous <= common.OUTCOME_TOLERANCE)


def validate_context_pair(row_a: dict, row_b: dict) -> tuple[np.ndarray, np.ndarray, int]:
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


def eligible_scene(
    scene_id: str,
    scenario: dict,
    frames_a: dict[int, tuple[Path, dict]],
    frames_b: dict[int, tuple[Path, dict]],
    outcome_a: dict,
    outcome_b: dict,
):
    if not outcomes_agree(outcome_a, outcome_b):
        return None, "unstable_outcome"
    if (
        set(frames_a) != set(common.DECISION_STEPS)
        or set(frames_b) != set(common.DECISION_STEPS)
    ):
        return None, "incomplete_frames"
    for step in common.DECISION_STEPS:
        try:
            validate_context_pair(frames_a[step][1], frames_b[step][1])
        except RuntimeError:
            return None, "unstable_candidate_context"
    metadata = common.source_metadata(scenario)
    if metadata["origin_log"] != common.scene_origin_log(scene_id):
        return None, "source_metadata_drift"
    if outcome_a["success"]:
        decisions = list(common.DECISION_STEPS)
        stratum = "solved"
    else:
        boundary = outcome_a["first_violation_step"]
        if boundary is None:
            return None, "failed_without_violation_boundary"
        decisions = [step for step in common.DECISION_STEPS if step < boundary]
        if not decisions:
            return None, "failed_without_previolation_decision"
        decisions = decisions[-3:]
        stratum = "failed"
    return {
        "scene_id": scene_id,
        "scenario_family": metadata["scenario_family"],
        "split": metadata["split"],
        "origin_log": metadata["origin_log"],
        "origin_token": metadata["origin_token"],
        "outcome_stratum": stratum,
        "eligible_decisions": decisions,
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
    }, "eligible"


def select_balanced(
    rows: list[dict],
    count: int,
    per_log: Counter,
    salt: str,
) -> list[dict]:
    if count < 0:
        raise ValueError("target count must be non-negative")
    by_family = {
        family: sorted(
            [row for row in rows if row["scenario_family"] == family],
            key=lambda row: (
                common.stable_digest(salt, family, row["scene_id"]),
                row["scene_id"],
            ),
        )
        for family in common.ALLOWED_FAMILIES
    }
    quotas = {
        common.ALLOWED_FAMILIES[0]: (count + 1) // 2,
        common.ALLOWED_FAMILIES[1]: count // 2,
    }
    selected = []
    selected_ids = set()

    def take(pool, maximum):
        taken = 0
        for row in pool:
            if taken >= maximum:
                break
            if row["scene_id"] in selected_ids:
                continue
            if per_log[row["origin_log"]] >= common.MAXIMUM_TARGETS_PER_LOG:
                continue
            selected.append(row)
            selected_ids.add(row["scene_id"])
            per_log[row["origin_log"]] += 1
            taken += 1
        return taken

    for family in common.ALLOWED_FAMILIES:
        take(by_family[family], quotas[family])
    if len(selected) < count:
        remainder = sorted(
            [
                row for row in rows
                if row["scene_id"] not in selected_ids
            ],
            key=lambda row: (
                common.stable_digest(salt, "deficit", row["scene_id"]),
                row["scene_id"],
            ),
        )
        take(remainder, count - len(selected))
    if len(selected) != count:
        raise RuntimeError(
            f"insufficient balanced/log-capped targets: {len(selected)} != {count}"
        )
    return selected


def choose_failed_step(row: dict, salt: str) -> int:
    choices = row["eligible_decisions"]
    offset = int(common.stable_digest(salt, row["scene_id"])[:16], 16)
    return int(choices[offset % len(choices)])


def solved_step_allocation(failed: list[dict], count: int) -> list[int]:
    histogram = Counter(int(row["decision_step"]) for row in failed)
    if not histogram:
        raise RuntimeError("cannot match solved decisions without failed targets")
    raw = {
        step: count * histogram[step] / len(failed)
        for step in sorted(histogram)
    }
    allocation = {step: int(np.floor(value)) for step, value in raw.items()}
    remainder = count - sum(allocation.values())
    order = sorted(
        raw,
        key=lambda step: (-(raw[step] - allocation[step]), step),
    )
    for step in order[:remainder]:
        allocation[step] += 1
    return [
        step
        for step in sorted(allocation)
        for _ in range(allocation[step])
    ]


def assign_solved_steps(rows: list[dict], failed: list[dict], salt: str) -> None:
    steps = solved_step_allocation(failed, len(rows))
    ranked = sorted(
        rows,
        key=lambda row: (
            common.stable_digest(salt, row["scene_id"]),
            row["scene_id"],
        ),
    )
    step_order = sorted(
        steps,
        key=lambda step: common.stable_digest(salt, "step", step),
    )
    for row, step in zip(ranked, step_order):
        if step not in row["eligible_decisions"]:
            raise RuntimeError("solved target lacks matched decision step")
        row["decision_step"] = int(step)


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
    }


def pilot_prefix(train_targets: list[dict]) -> list[dict]:
    failed = [row for row in train_targets if row["outcome_stratum"] == "failed"]
    solved = [row for row in train_targets if row["outcome_stratum"] == "solved"]
    failed_count = min(32, len(failed))
    solved_count = common.PILOT_TARGETS - failed_count
    if solved_count > len(solved):
        solved_count = len(solved)
        failed_count = common.PILOT_TARGETS - solved_count
    counts = Counter()
    selected = select_balanced(
        failed, failed_count, counts, f"{common.TARGET_SELECTION_SALT}:pilot:failed"
    )
    selected += select_balanced(
        solved, solved_count, counts, f"{common.TARGET_SELECTION_SALT}:pilot:solved"
    )
    if len(selected) != common.PILOT_TARGETS:
        raise RuntimeError("failed to construct immutable pilot64 prefix")
    return selected


def select_split(
    split: str,
    target_count: int,
    failure_cap: int,
    failure_minimum: int,
    source: dict,
    audit_a: dict,
    audit_b: dict,
) -> tuple[list[dict], dict]:
    outcomes_a = load_outcomes(Path(audit_a["closed_loop_outcome"]["metrics_csv"]))
    outcomes_b = load_outcomes(Path(audit_b["closed_loop_outcome"]["metrics_csv"]))
    if set(outcomes_a) != set(source) or set(outcomes_b) != set(source):
        raise RuntimeError(f"{split} baseline outcome/source coverage drifted")
    frames_a = load_records(audit_a)
    frames_b = load_records(audit_b)
    if set(frames_a) != set(source) or set(frames_b) != set(source):
        raise RuntimeError(f"{split} baseline record/source coverage drifted")

    reasons = Counter()
    eligible = {"failed": [], "solved": []}
    for scene_id in sorted(source):
        row, reason = eligible_scene(
            scene_id,
            source[scene_id],
            frames_a[scene_id],
            frames_b[scene_id],
            outcomes_a[scene_id],
            outcomes_b[scene_id],
        )
        reasons[reason] += 1
        if row is not None:
            eligible[row["outcome_stratum"]].append(row)
    failure_count = min(failure_cap, len(eligible["failed"]))
    if failure_count < failure_minimum:
        raise RuntimeError(
            f"{split} failed-target coverage {failure_count} < {failure_minimum}"
        )
    solved_count = target_count - failure_count
    per_log = Counter()
    failed = select_balanced(
        eligible["failed"], failure_count, per_log,
        f"{common.TARGET_SELECTION_SALT}:{split}:failed",
    )
    for row in failed:
        row["decision_step"] = choose_failed_step(
            row, f"{common.TARGET_SELECTION_SALT}:{split}:failed-step"
        )
    solved = select_balanced(
        eligible["solved"], solved_count, per_log,
        f"{common.TARGET_SELECTION_SALT}:{split}:solved",
    )
    assign_solved_steps(
        solved, failed, f"{common.TARGET_SELECTION_SALT}:{split}:solved-step"
    )
    targets = [
        make_target(row, frames_a, frames_b)
        for row in failed + solved
    ]
    reward_errors = np.asarray([
        float(row["local_reward_rerun_max_abs_error"]) for row in targets
    ], dtype=np.float64)
    return targets, {
        "eligibility_reasons": dict(sorted(reasons.items())),
        "eligible_counts": {
            key: len(value) for key, value in eligible.items()
        },
        "selected_counts": dict(sorted(Counter(
            row["outcome_stratum"] for row in targets
        ).items())),
        "selected_family_counts": dict(sorted(Counter(
            row["scenario_family"] for row in targets
        ).items())),
        "selected_decision_histogram": {
            str(key): value for key, value in sorted(Counter(
                row["decision_step"] for row in targets
            ).items())
        },
        "selected_origin_logs": len({row["origin_log"] for row in targets}),
        "local_reward_rerun_diagnostic_only": {
            "stable_at_array_tolerance": int(np.sum(
                reward_errors <= common.ARRAY_TOLERANCE
            )),
            "drifted_at_array_tolerance": int(np.sum(
                reward_errors > common.ARRAY_TOLERANCE
            )),
            "median_max_abs_error": float(np.median(reward_errors)),
            "maximum_max_abs_error": float(np.max(reward_errors)),
            "used_for_target_selection": False,
            "local_control_label_source": "baseline_a",
        },
    }


def write_scenarios(path: Path, source: dict, targets: list[dict]) -> None:
    common.atomic_pickle(
        path, {row["scene_id"]: source[row["scene_id"]] for row in targets}
    )


def build(args) -> Path:
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "target_manifest.json"
    if manifest_path.exists():
        _, payload = common.load_target_manifest(manifest_path)
        for key, sha_key in (
            ("pilot64_scenario_file", "pilot64_scenario_file_sha256"),
            ("expand192_scenario_file", "expand192_scenario_file_sha256"),
            ("dev64_scenario_file", "dev64_scenario_file_sha256"),
        ):
            if common.sha256_file(payload[key]) != payload[sha_key]:
                raise RuntimeError(f"existing V4 target artifact drifted: {key}")
        return manifest_path
    # Without a final manifest, incomplete generated files are regenerated
    # atomically below using the same frozen baseline inputs.

    source_audit_path, source_audit = common.load_source_audit(args.source_audit)
    train_source = common.load_pickle(source_audit["train_scenario_file"])
    validation_source = common.load_pickle(
        source_audit["validation_scenario_file"]
    )
    train_a_path, train_a = load_baseline(
        args.train_baseline_a, "baseline_train_a", "train"
    )
    train_b_path, train_b = load_baseline(
        args.train_baseline_b, "baseline_train_b", "train"
    )
    val_a_path, val_a = load_baseline(
        args.validation_baseline_a, "baseline_validation_a", "validation"
    )
    val_b_path, val_b = load_baseline(
        args.validation_baseline_b, "baseline_validation_b", "validation"
    )
    expected_source_sha = {
        "train": source_audit["train_scenario_file_sha256"],
        "validation": source_audit["validation_scenario_file_sha256"],
    }
    for split, audits in (
        ("train", (train_a, train_b)),
        ("validation", (val_a, val_b)),
    ):
        if any(
            row["scenario_file_sha256"] != expected_source_sha[split]
            for row in audits
        ):
            raise RuntimeError(f"{split} baseline/source provenance drifted")
        keys = (
            "checkpoint_sha256", "selector_state_sha256",
            "candidate_noise_namespace", "rollout_implementation_sha256",
        )
        if any(audits[0][key] != audits[1][key] for key in keys):
            raise RuntimeError(f"{split} baseline A/B provenance drifted")

    train_targets, train_summary = select_split(
        "train", common.TRAIN_TARGETS, 128, 64,
        train_source, train_a, train_b,
    )
    development_targets, development_summary = select_split(
        "validation", common.DEV_TARGETS, 32, 8,
        validation_source, val_a, val_b,
    )
    pilot = pilot_prefix(train_targets)
    pilot_ids = {row["scene_id"] for row in pilot}
    expand = [row for row in train_targets if row["scene_id"] not in pilot_ids]
    if len(expand) != common.TRAIN_TARGETS - common.PILOT_TARGETS:
        raise RuntimeError("V4 expand192 complement drifted")
    for row in pilot:
        row["stage"] = "pilot64"
    for row in expand:
        row["stage"] = "expand192"
    for row in development_targets:
        row["stage"] = "dev64"
    ordered_targets = pilot + expand + development_targets

    output_root.mkdir(parents=True, exist_ok=True)
    pilot_path = output_root / "pilot64_scenarios.pkl"
    expand_path = output_root / "expand192_scenarios.pkl"
    dev_path = output_root / "sealed_dev64_scenarios.pkl"
    write_scenarios(pilot_path, train_source, pilot)
    write_scenarios(expand_path, train_source, expand)
    write_scenarios(dev_path, validation_source, development_targets)

    protocol_path = args.protocol.expanduser().resolve()
    baseline_paths = {
        "train_a": train_a_path,
        "train_b": train_b_path,
        "validation_a": val_a_path,
        "validation_b": val_b_path,
    }
    payload = {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS",
        "method": common.TARGET_METHOD,
        "scientific_role": "reward_and_causal_value_blind_target_freeze",
        "selection_uses_local_candidate_reward": False,
        "selection_uses_causal_value": False,
        "intervention_outcomes_observed_before_freeze": False,
        "source_audit": str(source_audit_path),
        "source_audit_sha256": common.sha256_file(source_audit_path),
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": common.sha256_file(protocol_path),
        "checkpoint_sha256": train_a["checkpoint_sha256"],
        "selector_state_sha256": train_a["selector_state_sha256"],
        "candidate_noise_namespace": train_a["candidate_noise_namespace"],
        "rollout_implementation_sha256": train_a[
            "rollout_implementation_sha256"
        ],
        "maximum_targets_per_origin_log": common.MAXIMUM_TARGETS_PER_LOG,
        "target_selection_salt": common.TARGET_SELECTION_SALT,
        "train_target_count": len(pilot) + len(expand),
        "pilot64_target_count": len(pilot),
        "expand192_target_count": len(expand),
        "development_target_count": len(development_targets),
        "development_consumed": False,
        "test_consumed": False,
        "pilot64_scenario_file": str(pilot_path.resolve()),
        "pilot64_scenario_file_sha256": common.sha256_file(pilot_path),
        "expand192_scenario_file": str(expand_path.resolve()),
        "expand192_scenario_file_sha256": common.sha256_file(expand_path),
        "dev64_scenario_file": str(dev_path.resolve()),
        "dev64_scenario_file_sha256": common.sha256_file(dev_path),
        "baseline_audits": {
            key: {"path": str(path), "sha256": common.sha256_file(path)}
            for key, path in baseline_paths.items()
        },
        "selection_summary": {
            "train": train_summary,
            "validation": development_summary,
        },
        "target_count": len(ordered_targets),
        "targets": ordered_targets,
    }
    common.atomic_json(manifest_path, payload)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--train-baseline-a", type=Path, required=True)
    parser.add_argument("--train-baseline-b", type=Path, required=True)
    parser.add_argument("--validation-baseline-a", type=Path, required=True)
    parser.add_argument("--validation-baseline-b", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = build(args)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
