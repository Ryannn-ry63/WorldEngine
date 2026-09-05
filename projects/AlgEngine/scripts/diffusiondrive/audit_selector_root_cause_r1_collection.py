#!/usr/bin/env python3
"""Fail-closed audit for trained-selector R1 Reactive rollout collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import selector_root_cause_metrics as metrics_common


REQUIRED_MINIMUM_LOG_LENGTH = 20
AUDIT_METHOD = "diffusiondrive_selector_root_cause_r1_collection_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, family: str, seed: int) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "PASS" or manifest.get("schema_version") != 3:
        raise RuntimeError(f"invalid checkpoint manifest: {path}")
    if int(manifest.get("selector_payload", {}).get("train_seed", -1)) != seed:
        raise RuntimeError("behavior checkpoint train seed drifted")
    method = str(manifest.get("method", ""))
    expected_fragment = {
        "scalar_v3": "rare_tuned",
        "gate_conditioned": "gate_conditioned",
    }[family]
    if expected_fragment not in method:
        raise RuntimeError(f"checkpoint method does not match {family}: {method}")
    checkpoint = Path(manifest["checkpoint"]).expanduser().resolve()
    state = Path(manifest["scene_selector_state"]).expanduser().resolve()
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("materialized behavior checkpoint SHA256 drifted")
    if sha256_file(state) != manifest["scene_selector_state_sha256"]:
        raise RuntimeError("behavior selector-state SHA256 drifted")
    return path, manifest


def load_scenario_contract(
    path: Path, maximum_scenarios: int | None
) -> tuple[set[str], dict[str, int]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"invalid scenario file: {path}")
    items = list(payload.items())
    if maximum_scenarios is not None:
        items = items[:maximum_scenarios]
    excluded = {}
    selected = set()
    for scene_id, scene in items:
        if not isinstance(scene, dict) or "log_length" not in scene:
            raise RuntimeError(f"scenario has no log_length: {scene_id}")
        length = int(scene["log_length"])
        if length < REQUIRED_MINIMUM_LOG_LENGTH:
            excluded[str(scene_id)] = length
        else:
            selected.add(str(scene_id))
    if not selected:
        raise RuntimeError("scenario contract contains no collectable scenes")
    return selected, excluded


def record_paths(root: Path, layout: str) -> list[Path]:
    if layout == "merged":
        paths = sorted(
            (root / "WE_output/openscene_format/diffusiondrive_rollout_records").glob(
                "*_reward.pkl"
            )
        )
    else:
        paths = sorted(
            root.glob(
                "split_*/WE_output/openscene_format/"
                "diffusiondrive_rollout_records/*_reward.pkl"
            )
        )
    if not paths:
        raise RuntimeError(f"no {layout} candidate reward records under {root}")
    return paths


def load_reports(root: Path) -> tuple[list[Path], dict[str, list[bool]]]:
    paths = sorted(root.rglob("runner_report_*.json"))
    if not paths:
        raise RuntimeError(f"no runner reports under {root}")
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise RuntimeError(f"invalid runner report: {path}")
        for row in payload:
            outcomes[str(row["scenario_name"])].append(row.get("succeeded") is True)
    return paths, outcomes


def completed_scenes(root: Path) -> tuple[list[Path], set[str]]:
    paths = sorted(root.glob("split_*/completed_scenarios/completed_*.txt"))
    if not paths:
        raise RuntimeError(f"no completed-scenario ledgers under {root}")
    scenes = set()
    for path in paths:
        scenes.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return paths, scenes


def expected_contract(
    *,
    family: str,
    seed: int,
    checkpoint_sha: str,
    manifest_sha: str,
    split: str,
    namespace: str,
    implementation_sha: str,
) -> dict:
    return {
        "schema_version": 3,
        "experiment": "diffusiondrive_selector_root_cause_r1",
        "source_policy": "trained_v3_selector",
        "behavior_policy_family": family,
        "behavior_policy_train_seed": seed,
        "expected_checkpoint_sha256": checkpoint_sha,
        "behavior_checkpoint_manifest_sha256": manifest_sha,
        "rollout_implementation_sha256": implementation_sha,
        "trained_v3_checkpoint_loaded": True,
        "deployed_action_parity_required": False,
        "num_dynamic_candidates": 20,
        "candidate_context_export": True,
        "reward_owner": "simengine_dynamic_candidate_reward",
        "reward_scalar": "official_pairwise_pdm",
        "reward_components_role": "diagnostics_only",
        "generator_frozen": True,
        "perception_frozen": True,
        "base_selector_frozen": True,
        "diagnostic_split": split,
        "react_type": "R",
        "training_data_consumed": False,
        "development_only": True,
    }


def validate_record(
    path: Path,
    row: dict,
    contract: dict,
    namespace: str,
    checkpoint_sha: str,
) -> tuple[str, int, float, float, float]:
    if row.get("schema_version") != 2 or row.get("record_type") != (
        "diffusiondrive_closed_loop_candidate_reward"
    ):
        raise RuntimeError(f"record schema drifted: {path}")
    if row.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"behavior checkpoint drifted: {path}")
    if row.get("candidate_noise_namespace") != namespace:
        raise RuntimeError(f"candidate namespace drifted: {path}")
    if row.get("selector_rollout_contract") != contract:
        raise RuntimeError(f"selector rollout contract drifted: {path}")
    flat = {
        "behavior_policy_family": contract["behavior_policy_family"],
        "behavior_policy_train_seed": contract["behavior_policy_train_seed"],
        "behavior_checkpoint_manifest_sha256": contract[
            "behavior_checkpoint_manifest_sha256"
        ],
        "diagnostic_split": contract["diagnostic_split"],
        "react_type": "R",
    }
    for key, expected in flat.items():
        if row.get(key) != expected:
            raise RuntimeError(f"flat provenance {key} drifted: {path}")
    shapes = {
        "candidate_features": (20, 256),
        "candidate_trajectories_8": (20, 8, 3),
        "route_bev_features": (20, 8, 256),
        "status_tokens": (1, 256),
        "ego_queries": (1, 256),
        "agents_queries": (30, 256),
        "reference_logits": (20,),
        "current_logits": (20,),
        "candidate_rewards": (20,),
        "candidate_reward_components": (20, 6),
        "candidate_reward_valid_mask": (20,),
    }
    arrays = {}
    for key, shape in shapes.items():
        value = np.asarray(row.get(key))
        if value.shape != shape:
            raise RuntimeError(f"{path}: {key} shape {value.shape} != {shape}")
        if key != "candidate_reward_valid_mask" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite {key}: {path}")
        arrays[key] = value
    if not np.asarray(arrays["candidate_reward_valid_mask"], dtype=np.bool_).all():
        raise RuntimeError(f"candidate validity drifted: {path}")
    if tuple(row.get("reward_component_names", ())) != metrics_common.COMPONENT_NAMES:
        raise RuntimeError(f"component order drifted: {path}")
    selected = int(row["selected_index"])
    if int(np.asarray(row["selected_indices"]).item()) != selected:
        raise RuntimeError(f"selected-index export drifted: {path}")
    current = np.asarray(arrays["current_logits"], dtype=np.float64)
    reference = np.asarray(arrays["reference_logits"], dtype=np.float64)
    if selected != int(current.argmax()):
        raise RuntimeError(f"deployed action/current selector drifted: {path}")
    parity = float(row["deployed_candidate_parity_max_abs_error"])
    if parity > 1e-4:
        raise RuntimeError(f"deployed trajectory parity failed: {path}")
    raw_path = Path(str(row.get("raw_observation_path", "")))
    sidecar_path = Path(str(row.get("sidecar_path", "")))
    if not raw_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"raw observation or sidecar provenance missing: {path}")
    scene = str(row["rollout_scene_id"])
    origin = scene.rsplit("-", 1)[-1]
    if row.get("rollout_origin_token") != origin:
        raise RuntimeError(f"origin-token provenance drifted: {path}")
    rewards = np.asarray(arrays["candidate_rewards"], dtype=np.float64)
    return (
        scene,
        int(row["worldengine_step"]),
        float(np.max(np.abs(current - reference))),
        float(rewards[selected]),
        float(rewards.max()),
    )


def audit_collection(
    *,
    root: Path,
    scenario_file: Path,
    manifest_path: Path,
    family: str,
    seed: int,
    split: str,
    layout: str,
    namespace: str,
    implementation_sha: str,
    expected_code_sha: str,
    expected_workers: int,
    expected_records_per_scene: int,
    maximum_scenarios: int | None,
    metrics_csv: Path | None,
) -> dict:
    root = root.expanduser().resolve()
    scenario_file = scenario_file.expanduser().resolve()
    manifest_path, manifest = load_manifest(manifest_path, family, seed)
    manifest_sha = sha256_file(manifest_path)
    checkpoint_sha = str(manifest["checkpoint_sha256"])
    contract = expected_contract(
        family=family,
        seed=seed,
        checkpoint_sha=checkpoint_sha,
        manifest_sha=manifest_sha,
        split=split,
        namespace=namespace,
        implementation_sha=implementation_sha,
    )
    expected_scenes, excluded_short = load_scenario_contract(
        scenario_file, maximum_scenarios
    )
    report_paths, reports = load_reports(root)
    ledger_paths, ledger_scenes = completed_scenes(root)
    unknown_reports = set(reports) - expected_scenes
    missing_reports = expected_scenes - set(reports)
    unsuccessful_reports = {
        scene for scene in expected_scenes if scene in reports and not any(reports[scene])
    }
    missing_ledgers = expected_scenes - ledger_scenes
    if unknown_reports or missing_reports or unsuccessful_reports or missing_ledgers:
        raise RuntimeError(
            "runner/ledger scenario coverage drifted: "
            f"unknown_reports={len(unknown_reports)} missing_reports={len(missing_reports)} "
            f"unsuccessful_reports={len(unsuccessful_reports)} missing_ledgers={len(missing_ledgers)}"
        )
    if ledger_scenes - expected_scenes:
        raise RuntimeError("completed ledger contains out-of-contract scenarios")

    seen = set()
    steps: dict[str, list[int]] = defaultdict(list)
    workers = set()
    config_shas = set()
    resolved_by_worker: dict[str, set[str]] = defaultdict(set)
    code_shas = set()
    residuals = []
    selected_rewards = []
    oracle_rewards = []
    maximum_parity = 0.0
    for path in record_paths(root, layout):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        scene, step, residual, selected_reward, oracle_reward = validate_record(
            path, row, contract, namespace, checkpoint_sha
        )
        if scene not in expected_scenes:
            raise RuntimeError(f"record outside scenario contract: {path}")
        identity = (scene, step)
        if identity in seen:
            raise RuntimeError(f"duplicate logical rollout frame: {identity}")
        seen.add(identity)
        steps[scene].append(step)
        residuals.append(residual)
        selected_rewards.append(selected_reward)
        oracle_rewards.append(oracle_reward)
        maximum_parity = max(
            maximum_parity,
            float(row["deployed_candidate_parity_max_abs_error"]),
        )
        worker_source = path if layout == "split" else Path(row["sidecar_path"])
        parts = [part for part in worker_source.parts if part.startswith("split_")]
        if len(parts) != 1:
            raise RuntimeError(f"ambiguous worker provenance: {path}")
        worker = parts[0]
        workers.add(worker)
        config_shas.add(str(row.get("config_sha256")))
        code_shas.add(str(row.get("code_sha")))
        resolved_by_worker[worker].add(str(row.get("resolved_config_sha256")))

    expected_steps = list(range(4, 4 + expected_records_per_scene))
    bad_steps = {
        scene: sorted(values)
        for scene, values in steps.items()
        if sorted(values) != expected_steps
    }
    if bad_steps or expected_scenes - set(steps):
        raise RuntimeError(
            f"record coverage failed: bad_steps={len(bad_steps)} "
            f"missing={len(expected_scenes - set(steps))}"
        )
    if len(workers) != expected_workers:
        raise RuntimeError(
            f"expected {expected_workers} contributing workers, got {sorted(workers)}"
        )
    if len(config_shas) != 1 or "None" in config_shas:
        raise RuntimeError("source config provenance drifted")
    if code_shas != {expected_code_sha}:
        raise RuntimeError(f"code provenance drifted: {code_shas}")
    unstable = {key: value for key, value in resolved_by_worker.items() if len(value) != 1}
    if unstable:
        raise RuntimeError(f"resolved config drifted within worker: {unstable}")
    if max(residuals) <= 1e-5:
        raise RuntimeError("trained selector produced no detectable residual logits")

    outcome_summary = None
    if metrics_csv is not None:
        outcomes = metrics_common.load_outcomes(metrics_csv)
        if set(outcomes) != expected_scenes:
            raise RuntimeError(
                "merged metric scenario coverage drifted: "
                f"missing={len(expected_scenes - set(outcomes))} "
                f"extra={len(set(outcomes) - expected_scenes)}"
            )
        outcome_summary = {
            "metrics_csv": str(metrics_csv.expanduser().resolve()),
            "metrics_csv_sha256": sha256_file(metrics_csv.expanduser().resolve()),
            "mean_score": float(np.mean([row["score"] for row in outcomes.values()])),
            "success_rate": float(
                np.mean([row["success"] for row in outcomes.values()])
            ),
            "success_definition": "NC==1_and_DAC==1",
            "failed_scenarios": sum(not row["success"] for row in outcomes.values()),
        }

    return {
        "schema_version": 1,
        "status": "PASS",
        "method": AUDIT_METHOD,
        "layout": layout,
        "rollout_root": str(root),
        "diagnostic_split": split,
        "react_type": "R",
        "behavior_policy_family": family,
        "behavior_policy_train_seed": seed,
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "selector_state_sha256": manifest["scene_selector_state_sha256"],
        "candidate_noise_namespace": namespace,
        "rollout_implementation_sha256": implementation_sha,
        "code_sha": expected_code_sha,
        "selector_rollout_contract": contract,
        "scenario_file": str(scenario_file),
        "scenario_file_sha256": sha256_file(scenario_file),
        "num_scenarios": len(expected_scenes),
        "num_records": len(seen),
        "records_per_scene": expected_records_per_scene,
        "num_workers": len(workers),
        "workers": sorted(workers),
        "runner_report_files": [str(path) for path in report_paths],
        "completed_scenario_ledgers": [str(path) for path in ledger_paths],
        "excluded_short_scenarios": excluded_short,
        "maximum_residual_logit_max_abs": max(residuals),
        "mean_residual_logit_max_abs": float(np.mean(residuals)),
        "maximum_deployed_candidate_parity_error": maximum_parity,
        "mean_selected_candidate_reward": float(np.mean(selected_rewards)),
        "mean_oracle_candidate_reward": float(np.mean(oracle_rewards)),
        "mean_candidate_headroom": float(
            np.mean(np.asarray(oracle_rewards) - np.asarray(selected_rewards))
        ),
        "resolved_config_sha256_by_worker": {
            key: next(iter(value)) for key, value in sorted(resolved_by_worker.items())
        },
        "closed_loop_outcome": outcome_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument(
        "--policy-family", choices=("scalar_v3", "gate_conditioned"), required=True
    )
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--diagnostic-split", choices=("smoke8", "cl_dev58"), required=True)
    parser.add_argument("--layout", choices=("split", "merged"), required=True)
    parser.add_argument("--expected-noise-namespace", required=True)
    parser.add_argument("--expected-implementation-sha256", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--expected-records-per-scene", type=int, default=8)
    parser.add_argument("--maximum-scenarios", type=int)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_collection(
        root=args.rollout_root,
        scenario_file=args.scenario_file,
        manifest_path=args.checkpoint_manifest,
        family=args.policy_family,
        seed=args.train_seed,
        split=args.diagnostic_split,
        layout=args.layout,
        namespace=args.expected_noise_namespace,
        implementation_sha=args.expected_implementation_sha256,
        expected_code_sha=args.expected_code_sha,
        expected_workers=args.expected_workers,
        expected_records_per_scene=args.expected_records_per_scene,
        maximum_scenarios=args.maximum_scenarios,
        metrics_csv=args.metrics_csv,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

