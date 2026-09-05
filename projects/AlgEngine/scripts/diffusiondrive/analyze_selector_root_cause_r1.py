#!/usr/bin/env python3
"""Analyze on-policy R1 candidate headroom and issue the next-stage decision."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import selector_root_cause_metrics as metrics_common


AUDIT_METHOD = "diffusiondrive_selector_root_cause_r1_collection_audit_v1"
HEADROOM_GATE = 0.01
RECOVERABLE_GATE = 0.25


def load_audit(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != AUDIT_METHOD
        or payload.get("layout") != "merged"
        or payload.get("diagnostic_split") != "cl_dev58"
        or payload.get("react_type") != "R"
        or not payload.get("closed_loop_outcome")
    ):
        raise RuntimeError(f"not a completed R1 development collection: {path}")
    return path, payload


def record_paths(root: Path) -> list[Path]:
    paths = sorted(
        (root / "WE_output/openscene_format/diffusiondrive_rollout_records").glob(
            "*_reward.pkl"
        )
    )
    if not paths:
        raise RuntimeError(f"no merged reward records under {root}")
    return paths


def append(target: dict[str, list[np.ndarray]], values: dict[str, np.ndarray]):
    for key, value in values.items():
        target[key].append(np.asarray(value))


def analyze_collection(
    audit_path: Path,
    audit: dict,
    *,
    temperature: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict:
    root = Path(audit["rollout_root"]).expanduser().resolve()
    metric_csv = Path(audit["closed_loop_outcome"]["metrics_csv"])
    outcomes = metrics_common.load_outcomes(metric_csv)
    scenes = []
    current_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    reference_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    seen = set()
    for path in record_paths(root):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        if row.get("selector_rollout_contract") != audit["selector_rollout_contract"]:
            raise RuntimeError(f"record/audit selector contract drifted: {path}")
        scene = str(row["rollout_scene_id"])
        identity = (scene, int(row["worldengine_step"]))
        if identity in seen:
            raise RuntimeError(f"duplicate logical R1 frame: {identity}")
        seen.add(identity)
        scenes.append(scene)
        current = np.asarray(row["current_logits"], dtype=np.float64)[None, :]
        reference = np.asarray(row["reference_logits"], dtype=np.float64)[None, :]
        rewards = np.asarray(row["candidate_rewards"], dtype=np.float64)[None, :]
        components = np.asarray(
            row["candidate_reward_components"], dtype=np.float64
        )[None, :, :]
        deployed = np.asarray([int(row["selected_index"])], dtype=np.int64)
        if int(current.argmax(axis=1)[0]) != int(deployed[0]):
            raise RuntimeError(f"R1 deployed action drifted: {path}")
        append(
            current_chunks,
            metrics_common.frame_metric_arrays(
                current,
                rewards,
                components,
                reference_logits=reference,
                behavior_indices=deployed,
                temperature=temperature,
            ),
        )
        append(
            reference_chunks,
            metrics_common.frame_metric_arrays(
                reference,
                rewards,
                components,
                reference_logits=reference,
                behavior_indices=deployed,
                temperature=temperature,
            ),
        )
    if len(seen) != int(audit["num_records"]):
        raise RuntimeError("analysis/audit record count drifted")
    current_values = metrics_common.concatenate_metric_chunks(current_chunks)
    reference_values = metrics_common.concatenate_metric_chunks(reference_chunks)
    return {
        "audit": str(audit_path),
        "audit_checkpoint_sha256": audit["checkpoint_sha256"],
        "policy_family": audit["behavior_policy_family"],
        "train_seed": audit["behavior_policy_train_seed"],
        "closed_loop_outcome": audit["closed_loop_outcome"],
        "on_policy_selector": metrics_common.summarize_policy(
            current_values,
            scenes,
            outcomes,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        ),
        "frozen_reference_replay_on_same_visited_states": (
            metrics_common.summarize_policy(
                reference_values,
                scenes,
                outcomes,
                bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_seed=bootstrap_seed + 100,
            )
        ),
        "interpretation": {
            "on_policy_candidate_headroom": True,
            "reference_replay_is_counterfactual": True,
            "reference_replay_is_not_a_closed_loop_baseline": True,
            "reward_components_are_descriptive_only": True,
        },
    }


def decide_headroom(failure_stratum: dict) -> dict:
    if int(failure_stratum.get("scenario_count", 0)) == 0:
        return {
            "decision": "STOP_R1_NO_FAILED_REACTIVE_SCENARIOS",
            "reason": "headroom_on_failures_is_not_identifiable",
        }
    intervals = failure_stratum["scenario_bootstrap_95"]
    headroom = intervals["headroom"]
    recoverable = intervals["recoverable_0p005"]
    passes = (
        headroom["lower95"] > HEADROOM_GATE
        and recoverable["lower95"] > RECOVERABLE_GATE
    )
    clear_failure = (
        headroom["upper95"] <= HEADROOM_GATE
        and recoverable["upper95"] <= RECOVERABLE_GATE
    )
    if passes:
        decision = "AUTHORIZE_R1_SCALAR_SEED1_SEED2"
        reason = "both_scalar_seed0_lower_bounds_pass"
    elif clear_failure:
        decision = "STOP_SELECTOR_ONLY_NO_ON_POLICY_HEADROOM"
        reason = "both_scalar_seed0_upper_bounds_fail"
    else:
        decision = "AUTHORIZE_R1_SCALAR_SEED1_BORDERLINE"
        reason = "scalar_seed0_confidence_intervals_overlap_a_gate"
    return {
        "decision": decision,
        "reason": reason,
        "thresholds": {
            "failed_scenario_mean_oracle20_minus_selected_lower95_gt": HEADROOM_GATE,
            "failed_scenario_recoverable_0p005_fraction_lower95_gt": RECOVERABLE_GATE,
            "sensitivity_recoverable_headroom": 0.02,
        },
        "observed": {
            "headroom": headroom,
            "recoverable_0p005": recoverable,
            "recoverable_0p02": intervals["recoverable_0p02"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    loaded = [load_audit(path) for path in args.audit]
    keys = [
        (row["behavior_policy_family"], int(row["behavior_policy_train_seed"]))
        for _, row in loaded
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate policy-family/train-seed collection")
    results = [
        analyze_collection(
            path,
            audit,
            temperature=args.temperature,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed + index * 1000,
        )
        for index, (path, audit) in enumerate(loaded)
    ]
    scalar_seed0 = [
        row
        for row in results
        if row["policy_family"] == "scalar_v3" and row["train_seed"] == 0
    ]
    if len(scalar_seed0) != 1:
        raise RuntimeError("R1 decision requires exactly one scalar_v3 seed0 collection")
    primary_gate = decide_headroom(
        scalar_seed0[0]["on_policy_selector"]["strata"]["closed_loop_failure"]
    )
    gate_secondary = [
        row
        for row in results
        if row["policy_family"] == "gate_conditioned" and row["train_seed"] == 0
    ]
    decision = primary_gate["decision"]
    followups = {
        "AUTHORIZE_R1_SCALAR_SEED1_SEED2": (
            "collect scalar_v3 seeds1/2, then diagnose probability support before R2"
        ),
        "AUTHORIZE_R1_SCALAR_SEED1_BORDERLINE": (
            "collect scalar_v3 seed1 only and rerun this gate"
        ),
        "STOP_SELECTOR_ONLY_NO_ON_POLICY_HEADROOM": (
            "stop selector-only optimization and revisit candidate generation"
        ),
        "STOP_R1_NO_FAILED_REACTIVE_SCENARIOS": (
            "report the closed-loop ceiling; do not infer a selector mechanism"
        ),
    }
    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_selector_root_cause_r1_gate_v1",
        "decision": decision,
        "primary_gate": primary_gate,
        "policy_results": results,
        "scientific_contract": {
            "primary_attribution_policy": "scalar_v3_seed0",
            "gate_conditioned_role": "performance_baseline_only",
            "gate_conditioned_can_authorize_followup": False,
            "fixed_dynamic_candidate_count": 20,
            "generator_perception_and_base_selector_frozen": True,
            "reward_scalar": "official_pairwise_pdm",
            "reward_components": "diagnostics_only",
            "development_consumed": True,
            "certification_consumed": False,
            "new_training_performed": False,
            "new_loss_implemented": False,
            "proposal_history_scaffold_run": False,
            "nr_r_occupancy_shift_claim_allowed": False,
            "reason_nr_r_shift_is_deferred": (
                "R0_is_epoch100_NR_while_R1_is_trained_selector_R"
            ),
        },
        "secondary_gate_conditioned_seed0_present": len(gate_secondary) == 1,
        "authorized_followup": followups[decision],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable R1 result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "decision": decision, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

