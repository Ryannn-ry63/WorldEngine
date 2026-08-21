#!/usr/bin/env python3
"""Fail-closed audit for one resume-safe rare-rollout collection lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASELINE_SHA256 = "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
AUDIT_METHOD = "diffusiondrive_v3_rare_rollout_collection_audit_v2"
REQUIRED_MINIMUM_LOG_LENGTH = 20
COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenario_contract(
    path: Path, maximum_scenarios: int | None = None
) -> tuple[set[str], set[str], dict[str, int]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"invalid or empty scenario shard: {path}")
    values = list(payload.items())
    if maximum_scenarios is not None:
        values = values[:maximum_scenarios]
    input_scenes = {str(scene_id) for scene_id, _ in values}
    excluded_short: dict[str, int] = {}
    for scene_id, scene in values:
        if not isinstance(scene, dict) or "log_length" not in scene:
            raise RuntimeError(f"scenario has no auditable log_length: {scene_id}")
        try:
            log_length = int(scene["log_length"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"scenario has invalid log_length: {scene_id}"
            ) from error
        if log_length < 0:
            raise RuntimeError(f"scenario has negative log_length: {scene_id}")
        if log_length < REQUIRED_MINIMUM_LOG_LENGTH:
            excluded_short[str(scene_id)] = log_length
    collectable_scenes = input_scenes - set(excluded_short)
    if not collectable_scenes:
        raise RuntimeError("scenario shard has no collectable scenarios")
    return input_scenes, collectable_scenes, excluded_short


def short_exclusion_digest(excluded_short: dict[str, int]) -> str:
    payload = "".join(
        f"{scene_id}\t{excluded_short[scene_id]}\n"
        for scene_id in sorted(excluded_short)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_reports(root: Path) -> tuple[list[Path], dict[str, list[bool]]]:
    paths = sorted(root.rglob("runner_report_*.json"))
    if not paths:
        raise RuntimeError(f"no WorldEngine runner reports under {root}")
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise RuntimeError(f"invalid runner report: {path}")
        for row in payload:
            outcomes[str(row["scenario_name"])].append(row.get("succeeded") is True)
    return paths, outcomes


def load_completed_scenarios(root: Path) -> tuple[list[Path], set[str]]:
    paths = sorted(root.glob("split_*/completed_scenarios/completed_*.txt"))
    if not paths:
        raise RuntimeError(f"no persistent completed-scenario ledgers under {root}")
    completed = set()
    for path in paths:
        for line in path.read_text().splitlines():
            scenario = line.strip()
            if scenario:
                completed.add(scenario)
    return paths, completed


def record_paths(root: Path, layout: str) -> list[Path]:
    if layout == "merged":
        candidate = root / "WE_output/openscene_format/diffusiondrive_rollout_records"
        paths = sorted(candidate.glob("*_reward.pkl")) if candidate.is_dir() else []
    else:
        paths = sorted(
            root.glob(
                "split_*/WE_output/openscene_format/"
                "diffusiondrive_rollout_records/*_reward.pkl"
            )
        )
    if not paths:
        raise RuntimeError(f"no {layout} rollout records under {root}")
    return paths


def senior_v1_filter(rewards: np.ndarray, components: np.ndarray, selected: int):
    oracle_feasible = float(rewards.max()) >= 0.9
    deployed_failure = float(rewards[selected]) <= (2.0 / 3.0)
    low_ep = (
        float(components[selected, 0]) == 1.0
        and float(components[selected, 1]) == 1.0
        and float(components[selected, 2]) < 0.2
    )
    return oracle_feasible and (deployed_failure or low_ep), deployed_failure, low_ep


def audit_lane(
    root: Path,
    scenario_file: Path,
    layout: str,
    expected_namespace: str,
    expected_code_sha: str | None,
    expected_records_per_scene: int,
    expected_workers: int,
    maximum_scenarios: int | None,
) -> dict:
    input_scenes, expected_scenes, excluded_short = load_scenario_contract(
        scenario_file, maximum_scenarios
    )
    report_paths, outcomes = load_reports(root)
    completed_paths, completed_scenes = load_completed_scenarios(root)
    unknown_reports = set(outcomes) - expected_scenes
    missing_completed = expected_scenes - completed_scenes
    unknown_completed = completed_scenes - expected_scenes
    if unknown_reports or missing_completed or unknown_completed:
        raise RuntimeError(
            "persistent runner coverage failed: "
            f"unknown_reports={len(unknown_reports)} "
            f"missing_completed={len(missing_completed)} "
            f"unknown_completed={len(unknown_completed)}"
        )

    records = record_paths(root, layout)
    seen: set[tuple[str, int]] = set()
    steps: dict[str, list[int]] = defaultdict(list)
    workers: set[str] = set()
    config_shas: set[str] = set()
    code_shas: set[str] = set()
    resolved_by_worker: dict[str, set[str]] = defaultdict(set)
    selected_rewards = []
    oracle_rewards = []
    eligible = 0
    eligible_plan_failure = 0
    eligible_low_ep = 0
    maximum_parity = 0.0
    for path in records:
        with path.open("rb") as stream:
            row = pickle.load(stream)
        if row.get("schema_version") != 2 or row.get("record_type") != (
            "diffusiondrive_closed_loop_candidate_reward"
        ):
            raise RuntimeError(f"record schema drifted: {path}")
        if row.get("checkpoint_sha256") != BASELINE_SHA256:
            raise RuntimeError(f"behavior checkpoint drifted: {path}")
        if row.get("candidate_noise_namespace") != expected_namespace:
            raise RuntimeError(f"candidate-noise namespace drifted: {path}")
        scene = str(row["rollout_scene_id"])
        step = int(row["worldengine_step"])
        if scene not in expected_scenes:
            raise RuntimeError(f"record belongs to another lane: {path}")
        origin = scene.rsplit("-", 1)[-1]
        if str(row.get("rollout_origin_token")) != origin:
            raise RuntimeError(f"rollout origin token drifted: {path}")
        identity = (scene, step)
        if identity in seen:
            raise RuntimeError(f"duplicate logical rollout frame: {identity}")
        seen.add(identity)
        steps[scene].append(step)

        rewards = np.asarray(row["candidate_rewards"], dtype=np.float64)
        components = np.asarray(row["candidate_reward_components"], dtype=np.float64)
        valid = np.asarray(row["candidate_reward_valid_mask"], dtype=np.bool_)
        reference = np.asarray(row["reference_logits"], dtype=np.float64)
        current = np.asarray(row["current_logits"], dtype=np.float64)
        if rewards.shape != (20,) or components.shape != (20, 6):
            raise RuntimeError(f"candidate reward shape drifted: {path}")
        if valid.shape != (20,) or not valid.all():
            raise RuntimeError(f"candidate validity drifted: {path}")
        if reference.shape != (20,) or current.shape != (20,):
            raise RuntimeError(f"selector logit shape drifted: {path}")
        if not all(
            np.isfinite(value).all()
            for value in (rewards, components, reference, current)
        ):
            raise RuntimeError(f"non-finite rollout record: {path}")
        if tuple(row.get("reward_component_names", ())) != COMPONENT_NAMES:
            raise RuntimeError(f"reward component order drifted: {path}")
        if float(np.max(np.abs(current - reference))) > 1e-5:
            raise RuntimeError(f"rollout did not deploy the epoch-100 policy: {path}")
        selected = int(row["selected_index"])
        if selected != int(reference.argmax()):
            raise RuntimeError(f"deployed action/reference selector drifted: {path}")
        parity = float(row["deployed_candidate_parity_max_abs_error"])
        if parity > 1e-4:
            raise RuntimeError(f"deployed trajectory parity failed: {path}")
        maximum_parity = max(maximum_parity, parity)
        raw_path = row.get("raw_observation_path")
        if not raw_path or not Path(raw_path).is_file():
            raise RuntimeError(f"raw observation provenance is missing: {path}")
        sidecar = Path(str(row.get("sidecar_path", "")))
        if not sidecar.is_file():
            raise RuntimeError(f"candidate sidecar provenance is missing: {path}")

        worker_source = (
            path
            if layout == "split"
            else Path(str(row.get("sidecar_path", "")))
        )
        parts = [part for part in worker_source.parts if part.startswith("split_")]
        if len(parts) != 1:
            raise RuntimeError(f"ambiguous worker provenance: {path}")
        worker = parts[0]
        workers.add(worker)
        config_shas.add(str(row.get("config_sha256")))
        code_shas.add(str(row.get("code_sha")))
        resolved_by_worker[worker].add(str(row.get("resolved_config_sha256")))
        selected_rewards.append(float(rewards[selected]))
        oracle_rewards.append(float(rewards.max()))
        keep, plan_failure, low_ep = senior_v1_filter(rewards, components, selected)
        eligible += int(keep)
        eligible_plan_failure += int(keep and plan_failure)
        eligible_low_ep += int(keep and low_ep)

    expected_steps = list(range(4, 4 + expected_records_per_scene))
    bad_steps = {
        scene: sorted(values)
        for scene, values in steps.items()
        if sorted(values) != expected_steps
    }
    missing_record_scenes = expected_scenes - set(steps)
    if bad_steps or missing_record_scenes:
        raise RuntimeError(
            "rollout record coverage failed: "
            f"bad_steps={len(bad_steps)} missing_scenes={len(missing_record_scenes)}"
        )
    if len(config_shas) != 1 or None in config_shas or "None" in config_shas:
        raise RuntimeError("source config provenance drifted")
    if len(code_shas) != 1 or "None" in code_shas:
        raise RuntimeError("code provenance drifted")
    actual_code_sha = next(iter(code_shas))
    if expected_code_sha is not None and actual_code_sha != expected_code_sha:
        raise RuntimeError("collection code SHA does not match the submitted revision")
    unstable = {key: value for key, value in resolved_by_worker.items() if len(value) != 1}
    if unstable:
        raise RuntimeError(f"resolved config drifted within workers: {unstable}")
    if len(workers) != expected_workers:
        raise RuntimeError(
            f"expected records from {expected_workers} workers, found {sorted(workers)}"
        )

    return {
        "schema_version": 2,
        "status": "PASS",
        "method": AUDIT_METHOD,
        "layout": layout,
        "source_policy": "immutable_epoch100_diffusiondrive",
        "checkpoint_sha256": BASELINE_SHA256,
        "candidate_noise_namespace": expected_namespace,
        "code_sha": actual_code_sha,
        "config_sha256": next(iter(config_shas)),
        "scenario_file": str(scenario_file),
        "scenario_file_sha256": sha256_file(scenario_file),
        "num_input_scenarios": len(input_scenes),
        "num_scenarios": len(expected_scenes),
        "num_records": len(seen),
        "records_per_scene": expected_records_per_scene,
        "num_workers": len(workers),
        "workers": sorted(workers),
        "runner_report_files": [str(path) for path in report_paths],
        "completed_scenario_ledgers": [str(path) for path in completed_paths],
        "completed_scenarios": len(completed_scenes),
        "runner_attempts": sum(len(value) for value in outcomes.values()),
        "scenarios_with_prior_failed_attempts": sum(
            1 for value in outcomes.values() if any(not status for status in value)
        ),
        "senior_v1_filter": {
            "oracle_score_minimum": 0.9,
            "deployed_score_maximum": 2.0 / 3.0,
            "low_ep_maximum_exclusive": 0.2,
            "eligible_records": eligible,
            "eligible_deployed_failure_records": eligible_plan_failure,
            "eligible_low_ep_records": eligible_low_ep,
        },
        "short_scenario_exclusion": {
            "policy": "exclude_only_when_log_length_is_below_required_minimum_v1",
            "required_minimum_log_length": REQUIRED_MINIMUM_LOG_LENGTH,
            "count": len(excluded_short),
            "scenes": [
                {"scene_id": scene_id, "log_length": excluded_short[scene_id]}
                for scene_id in sorted(excluded_short)
            ],
            "sha256": short_exclusion_digest(excluded_short),
        },
        "mean_deployed_reward": float(np.mean(selected_rewards)),
        "mean_oracle_reward": float(np.mean(oracle_rewards)),
        "mean_oracle_headroom": float(
            np.mean(np.asarray(oracle_rewards) - np.asarray(selected_rewards))
        ),
        "maximum_deployed_candidate_parity_error": maximum_parity,
        "resolved_config_sha256_by_worker": {
            key: next(iter(value)) for key, value in sorted(resolved_by_worker.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--layout", choices=("split", "merged"), required=True)
    parser.add_argument("--expected-noise-namespace", required=True)
    parser.add_argument("--expected-code-sha")
    parser.add_argument("--expected-records-per-scene", type=int, default=8)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--maximum-scenarios", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.rollout_root.expanduser().resolve()
    scenario_file = args.scenario_file.expanduser().resolve()
    report = audit_lane(
        root,
        scenario_file,
        args.layout,
        args.expected_noise_namespace,
        args.expected_code_sha,
        args.expected_records_per_scene,
        args.expected_workers,
        args.maximum_scenarios,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
