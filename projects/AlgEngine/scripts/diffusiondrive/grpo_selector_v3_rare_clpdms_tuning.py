#!/usr/bin/env python3
"""Audited closed-loop split, selection, and reporting for rare V3 CL-PDMS tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path


PDM_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
)
AVERAGE_TOKENS = {"average", "overall_average"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_values(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.sha256(payload).hexdigest()
        if sha256_file(path) != digest:
            raise RuntimeError(f"immutable output drifted: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(
        Path(path),
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def load_pass_json(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"report did not pass: {path}")
    return payload


def deterministic_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}|{namespace}|{value}".encode("utf-8")
    ).hexdigest()


def scene_log_name(scene_id: str, scene: dict) -> str:
    prefix, separator, suffix = str(scene_id).rpartition("-")
    if not separator or len(suffix) != 16:
        raise RuntimeError(f"unexpected scenario id: {scene_id}")
    metadata = scene.get("metadata", {})
    infos = metadata.get("openscene_data_infos_dict", {})
    recorded = {
        str(info["log_name"])
        for info in infos.values()
        if isinstance(info, dict) and info.get("log_name")
    }
    if recorded and recorded != {prefix}:
        raise RuntimeError(
            f"scenario/log metadata mismatch: {scene_id} -> {sorted(recorded)}"
        )
    for field in ("id", "name", "token"):
        if str(scene.get(field)) != str(scene_id):
            raise RuntimeError(f"scenario {field} mismatch: {scene_id}")
    return prefix


def choose_development_logs(
    by_log: dict[str, list[str]], split_seed: int, development_fraction: float
) -> tuple[list[str], int]:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development fraction must be between zero and one")
    if len(by_log) < 2:
        raise RuntimeError("log-disjoint split requires at least two logs")
    ordered = sorted(
        by_log,
        key=lambda name: deterministic_rank(split_seed, "cl_log_split", name),
    )
    target_scenes = round(
        sum(len(tokens) for tokens in by_log.values()) * development_fraction
    )
    target_logs = round(len(ordered) * development_fraction)
    cumulative = 0
    choices = []
    for index, log_name in enumerate(ordered[:-1], 1):
        cumulative += len(by_log[log_name])
        choices.append(
            (
                abs(cumulative - target_scenes),
                abs(index - target_logs),
                index,
                cumulative,
            )
        )
    _, _, prefix_count, scene_count = min(choices)
    return ordered[:prefix_count], scene_count


def validate_reusable_split(
    audit_path: Path,
    source_sha256: str,
    expected_counts: dict[str, int],
    expected_code_commit: str,
) -> dict | None:
    if not audit_path.is_file():
        return None
    audit = load_pass_json(audit_path)
    if audit.get("source_sha256") != source_sha256:
        raise RuntimeError("closed-loop source scenario SHA256 drifted")
    if audit.get("code_commit") != expected_code_commit:
        raise RuntimeError("closed-loop split code commit drifted")
    for split_name, expected in expected_counts.items():
        row = audit["splits"][split_name]
        if int(row["num_scenarios"]) != int(expected):
            raise RuntimeError(f"{split_name} scenario count drifted")
        path = Path(row["scenario_file"])
        if not path.is_file() or sha256_file(path) != row["scenario_file_sha256"]:
            raise RuntimeError(f"{split_name} scenario file drifted")
    return audit


def build_split(args) -> dict:
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_root = args.output_root.expanduser().resolve()
    audit_path = output_root / "split_audit.json"
    source_sha256 = sha256_file(source)
    reusable = validate_reusable_split(
        audit_path,
        source_sha256,
        {
            "development": args.expected_development_scenes,
            "confirmation": args.expected_confirmation_scenes,
            "smoke": 1,
        },
        str(args.code_commit),
    )
    if reusable is not None:
        return reusable

    with source.open("rb") as stream:
        scenes = pickle.load(stream)
    if not isinstance(scenes, dict) or not scenes:
        raise RuntimeError("source scenario pickle must be a non-empty dict")
    if len(scenes) != args.expected_source_scenes:
        raise RuntimeError(
            f"source scenario count {len(scenes)} != {args.expected_source_scenes}"
        )

    by_log = defaultdict(list)
    for scene_id, scene in scenes.items():
        if not isinstance(scene, dict):
            raise RuntimeError(f"scenario payload is not a dict: {scene_id}")
        by_log[scene_log_name(str(scene_id), scene)].append(str(scene_id))
    if len(by_log) != args.expected_source_logs:
        raise RuntimeError(f"source log count {len(by_log)} != {args.expected_source_logs}")

    development_logs, development_count = choose_development_logs(
        by_log, args.split_seed, args.development_fraction
    )
    development_log_set = set(development_logs)
    development_tokens = {
        token
        for log_name in development_logs
        for token in by_log[log_name]
    }
    confirmation_tokens = set(scenes) - development_tokens
    confirmation_logs = set(by_log) - development_log_set
    if development_count != len(development_tokens):
        raise RuntimeError("development count bookkeeping drifted")
    if development_tokens.intersection(confirmation_tokens):
        raise RuntimeError("closed-loop token leakage")
    if development_log_set.intersection(confirmation_logs):
        raise RuntimeError("closed-loop log leakage")
    if development_tokens | confirmation_tokens != set(scenes):
        raise RuntimeError("closed-loop split does not cover the source")

    if len(development_tokens) != args.expected_development_scenes:
        raise RuntimeError("unexpected development scene count")
    if len(confirmation_tokens) != args.expected_confirmation_scenes:
        raise RuntimeError("unexpected confirmation scene count")
    if len(development_log_set) != args.expected_development_logs:
        raise RuntimeError("unexpected development log count")
    if len(confirmation_logs) != args.expected_confirmation_logs:
        raise RuntimeError("unexpected confirmation log count")

    split_tokens = {
        "development": development_tokens,
        "confirmation": confirmation_tokens,
    }
    smoke_token = sorted(
        development_tokens,
        key=lambda token: deterministic_rank(args.split_seed, "cl_smoke", token),
    )[0]
    split_tokens["smoke"] = {smoke_token}
    split_reports = {}
    for split_name, tokens in split_tokens.items():
        payload = {
            scene_id: scenes[scene_id]
            for scene_id in scenes
            if scene_id in tokens
        }
        scenario_file = output_root / split_name / "all_scenarios.pkl"
        atomic_write_bytes(
            scenario_file, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
        logs = {
            scene_log_name(scene_id, scenes[scene_id]) for scene_id in tokens
        }
        split_reports[split_name] = {
            "scenario_file": str(scenario_file),
            "scenario_file_sha256": sha256_file(scenario_file),
            "num_scenarios": len(tokens),
            "num_logs": len(logs),
            "tokens": sorted(tokens),
            "tokens_sha256": digest_values(tokens),
            "log_names": sorted(logs),
            "log_names_sha256": digest_values(logs),
        }

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_log_disjoint_split_v1",
        "source": str(source),
        "source_sha256": source_sha256,
        "source_scenarios": len(scenes),
        "source_logs": len(by_log),
        "split_seed": int(args.split_seed),
        "development_fraction": float(args.development_fraction),
        "selection_rule": (
            "SHA256-ordered log prefix minimizing scene-count error, then "
            "log-count error"
        ),
        "code_commit": str(args.code_commit),
        "log_disjoint": True,
        "token_disjoint": True,
        "all_source_scenarios_accounted": True,
        "splits": split_reports,
    }
    write_json(audit_path, audit)
    return audit


def read_metric_csv(path: Path) -> dict[str, dict[str, float]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = str(row.get("token", ""))
            if token in AVERAGE_TOKENS:
                continue
            if not token:
                raise RuntimeError(f"metric row without token: {path}")
            if token in rows:
                raise RuntimeError(f"duplicate metric token {token}: {path}")
            values = {key: float(row[key]) for key in PDM_KEYS}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric row {token}: {path}")
            rows[token] = values
    if not rows:
        raise RuntimeError(f"empty metric CSV: {path}")
    return rows


def summarize_rows(rows: dict[str, dict[str, float]]) -> dict:
    metrics = {
        key: statistics.fmean(row[key] for row in rows.values())
        for key in PDM_KEYS
    }
    successes = sum(
        row["no_at_fault_collisions"] == 1.0
        and row["drivable_area_compliance"] == 1.0
        for row in rows.values()
    )
    return {
        **metrics,
        "success_rate": successes / len(rows),
        "successful_scenarios": successes,
        "num_scenarios": len(rows),
    }


def build_metrics(args) -> dict:
    audit = load_pass_json(args.split_audit)
    split_row = audit["splits"][args.split]
    expected = set(split_row["tokens"])
    rows = read_metric_csv(args.csv)
    actual = set(rows)
    if args.allow_superset:
        missing = expected - actual
        if missing:
            raise RuntimeError(f"metric CSV is missing {len(missing)} split tokens")
    elif actual != expected:
        raise RuntimeError(
            f"metric token set mismatch: actual={len(actual)} expected={len(expected)}"
        )
    selected = {token: rows[token] for token in sorted(expected)}
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_split_metrics_v1",
        "model": args.model,
        "seed": int(args.seed),
        "split": args.split,
        "metrics": summarize_rows(selected),
        "metric_csv": str(args.csv.expanduser().resolve()),
        "metric_csv_sha256": sha256_file(args.csv),
        "split_audit": str(args.split_audit.expanduser().resolve()),
        "split_audit_sha256": sha256_file(args.split_audit),
        "tokens_sha256": digest_values(selected),
        "source_csv_rows": len(rows),
        "allow_superset": bool(args.allow_superset),
    }
    if args.checkpoint_manifest:
        manifest = load_pass_json(args.checkpoint_manifest)
        checkpoint = Path(manifest["checkpoint"]).expanduser().resolve()
        if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("checkpoint manifest SHA256 drifted")
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = manifest["checkpoint_sha256"]
        report["checkpoint_manifest"] = str(args.checkpoint_manifest.resolve())
        report["checkpoint_manifest_sha256"] = sha256_file(args.checkpoint_manifest)
    write_json(args.output, report)
    return report


def load_recipe_config(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported CL-PDMS recipe schema")
    challenger_names = [row["name"] for row in payload["challengers"]]
    if len(challenger_names) != len(set(challenger_names)):
        raise RuntimeError("duplicate challenger names")
    if payload["incumbent"] not in payload["controls"]:
        raise RuntimeError("incumbent is not a control")
    return payload


def recipe_map(config: dict) -> dict[str, dict]:
    output = {
        name: {"name": name, "kind": "control", **row}
        for name, row in config["controls"].items()
    }
    for row in config["challengers"]:
        output[row["name"]] = {"kind": "challenger", **row}
    return output


def load_metric_report(metrics_root: Path, name: str, seed: int) -> dict:
    report = load_pass_json(metrics_root / name / f"seed{seed}.json")
    if report["model"] != name or int(report["seed"]) != seed:
        raise RuntimeError(f"metric provenance mismatch for {name} seed {seed}")
    if int(report["metrics"]["num_scenarios"]) != 58:
        raise RuntimeError(f"incomplete CL-dev metrics for {name} seed {seed}")
    return report


def screen_candidates(args) -> dict:
    config = load_recipe_config(args.recipe_config)
    recipes = recipe_map(config)
    challenger_names = [row["name"] for row in config["challengers"]]
    rows = []
    for name in [*config["controls"], *challenger_names]:
        metrics = load_metric_report(args.metrics_root, name, 0)
        rows.append(
            {
                "name": name,
                "kind": recipes[name]["kind"],
                "score": metrics["metrics"]["score"],
                "success_rate": metrics["metrics"]["success_rate"],
                "metrics_report": str(
                    (args.metrics_root / name / "seed0.json").resolve()
                ),
                "metrics_report_sha256": sha256_file(
                    args.metrics_root / name / "seed0.json"
                ),
            }
        )
    challengers = [row for row in rows if row["kind"] == "challenger"]
    challengers.sort(
        key=lambda row: (row["score"], row["success_rate"], row["name"]),
        reverse=True,
    )
    shortlist = [row["name"] for row in challengers[: args.shortlist_size]]
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_seed0_screen_v1",
        "ranking_metric": "closedloop_reactive.score",
        "seed": 0,
        "num_candidates": len(rows),
        "shortlist_size": int(args.shortlist_size),
        "shortlist": shortlist,
        "ranked": sorted(
            rows,
            key=lambda row: (row["score"], row["success_rate"], row["name"]),
            reverse=True,
        ),
        "recipe_config_sha256": sha256_file(args.recipe_config),
    }
    write_json(args.output, report)
    return report


def aggregate_dev(metrics_root: Path, name: str) -> dict:
    reports = [load_metric_report(metrics_root, name, seed) for seed in range(3)]
    keys = (*PDM_KEYS, "success_rate")
    return {
        "name": name,
        "by_seed": {
            str(seed): reports[seed]["metrics"] for seed in range(3)
        },
        "mean": {
            key: statistics.fmean(
                report["metrics"][key] for report in reports
            )
            for key in keys
        },
        "population_std": {
            key: statistics.pstdev(
                report["metrics"][key] for report in reports
            )
            for key in keys
        },
        "metric_reports": [
            {
                "path": str((metrics_root / name / f"seed{seed}.json").resolve()),
                "sha256": sha256_file(metrics_root / name / f"seed{seed}.json"),
            }
            for seed in range(3)
        ],
    }


def choose_final(
    aggregates: dict[str, dict],
    challenger_names: list[str],
    incumbent_name: str,
    minimum_improvement: float,
    minimum_nonnegative_seeds: int,
) -> tuple[str, list[dict]]:
    incumbent = aggregates[incumbent_name]
    eligible = []
    for name in challenger_names:
        row = aggregates[name]
        deltas = [
            row["by_seed"][str(seed)]["score"]
            - incumbent["by_seed"][str(seed)]["score"]
            for seed in range(3)
        ]
        record = {
            "name": name,
            "mean_score": row["mean"]["score"],
            "mean_score_delta_vs_incumbent": (
                row["mean"]["score"] - incumbent["mean"]["score"]
            ),
            "score_deltas_by_seed": deltas,
            "nonnegative_seed_count": sum(value >= 0.0 for value in deltas),
            "mean_success_rate": row["mean"]["success_rate"],
        }
        record["promotion_gate_passed"] = (
            record["mean_score_delta_vs_incumbent"] >= minimum_improvement
            and record["nonnegative_seed_count"] >= minimum_nonnegative_seeds
        )
        eligible.append(record)
    passed = [row for row in eligible if row["promotion_gate_passed"]]
    if not passed:
        return incumbent_name, eligible
    passed.sort(
        key=lambda row: (
            row["mean_score"],
            row["mean_success_rate"],
            row["name"],
        ),
        reverse=True,
    )
    return passed[0]["name"], eligible


def resolve_checkpoint_record(
    repo_root: Path,
    model_root: Path,
    recipe: dict,
    name: str,
    seed: int,
) -> dict:
    if recipe["kind"] == "control":
        manifest_path = repo_root / recipe["model_root"] / f"seed{seed}" / "checkpoint_manifest.json"
        report_path = repo_root / recipe["model_root"] / f"seed{seed}" / "train" / "report.json"
        audit_path = repo_root / recipe["model_root"] / f"seed{seed}" / "checkpoint_audit.json"
    else:
        manifest_path = model_root / name / f"seed{seed}" / "checkpoint_manifest.json"
        audit_path = model_root / name / f"seed{seed}" / "checkpoint_audit.json"
        report_path = model_root / name / f"seed{seed}" / "train" / "report.json"
    manifest = load_pass_json(manifest_path)
    audit = load_pass_json(audit_path)
    report = load_pass_json(report_path)
    checkpoint = Path(manifest["checkpoint"]).expanduser().resolve()
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint SHA256 drifted: {checkpoint}")
    if int(audit["changed_baseline_tensor_count"]) != 0:
        raise RuntimeError(f"baseline tensors changed: {checkpoint}")
    if audit["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint audit SHA256 drifted: {checkpoint}")
    for key in ("temperature", "learning_rate", "kl_weight", "epochs"):
        expected_key = "epoch" if key == "epochs" else key
        expected = recipe[expected_key]
        actual = report[key]
        if float(actual) != float(expected):
            raise RuntimeError(f"{name} {key} drifted: {actual} != {expected}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_manifest": str(manifest_path.resolve()),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "training_report": str(report_path.resolve()),
        "training_report_sha256": sha256_file(report_path),
        "checkpoint_audit": str(audit_path.resolve()),
        "checkpoint_audit_sha256": sha256_file(audit_path),
    }

def audit_lane(args) -> dict:
    config = load_recipe_config(args.recipe_config)
    recipes = recipe_map(config)
    split = load_pass_json(args.split_audit)
    if split["code_commit"] != args.code_commit:
        raise RuntimeError("lane split/code commit drifted")
    split_sha256 = sha256_file(args.split_audit)
    seed = int(args.seed)
    names = [
        *[row["name"] for row in config["challengers"]],
        *config["controls"],
    ]
    records = {}
    for name in names:
        metric_path = args.metrics_root / name / f"seed{seed}.json"
        metric = load_metric_report(args.metrics_root, name, seed)
        if metric["split_audit_sha256"] != split_sha256:
            raise RuntimeError(f"{name} seed {seed} split audit drifted")
        checkpoint = resolve_checkpoint_record(
            args.repo_root,
            args.model_root,
            recipes[name],
            name,
            seed,
        )
        if metric.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            raise RuntimeError(f"{name} seed {seed} metric checkpoint drifted")
        records[name] = {
            "kind": recipes[name]["kind"],
            "checkpoint": checkpoint,
            "metrics_report": str(metric_path.resolve()),
            "metrics_report_sha256": sha256_file(metric_path),
            "score": metric["metrics"]["score"],
            "success_rate": metric["metrics"]["success_rate"],
        }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_parallel_lane_v1",
        "seed": seed,
        "code_commit": str(args.code_commit),
        "recipe_config_sha256": sha256_file(args.recipe_config),
        "split_audit": str(args.split_audit.resolve()),
        "split_audit_sha256": split_sha256,
        "num_challengers": len(config["challengers"]),
        "num_controls": len(config["controls"]),
        "records": records,
    }
    write_json(args.output, report)
    return report


def load_lane_reports(
    args, recipe_sha256: str, split_audit_sha256: str
) -> list[dict]:
    lanes = {}
    for path in args.lane_report:
        report = load_pass_json(path)
        seed = int(report["seed"])
        if seed in lanes:
            raise RuntimeError(f"duplicate lane seed {seed}")
        if report["code_commit"] != args.code_commit:
            raise RuntimeError(f"lane seed {seed} code commit drifted")
        if report["recipe_config_sha256"] != recipe_sha256:
            raise RuntimeError(f"lane seed {seed} recipe config drifted")
        if report["split_audit_sha256"] != split_audit_sha256:
            raise RuntimeError(f"lane seed {seed} split audit drifted")
        lanes[seed] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    if set(lanes) != {0, 1, 2}:
        raise RuntimeError("selection requires passing lanes 0, 1, and 2")
    return [lanes[seed] for seed in range(3)]


def select_candidate(args) -> dict:
    config = load_recipe_config(args.recipe_config)
    recipes = recipe_map(config)
    recipe_sha256 = sha256_file(args.recipe_config)
    split = load_pass_json(args.split_audit)
    if split["code_commit"] != args.code_commit:
        raise RuntimeError("selection split/code commit drifted")
    split_sha256 = sha256_file(args.split_audit)
    lane_reports = load_lane_reports(args, recipe_sha256, split_sha256)
    screen = load_pass_json(args.screen)
    if screen["recipe_config_sha256"] != recipe_sha256:
        raise RuntimeError("screen recipe config drifted")
    shortlist = list(screen["shortlist"])
    compared = [config["incumbent"], *config["controls"]]
    names = list(dict.fromkeys([*compared, *shortlist]))
    aggregates = {
        name: aggregate_dev(args.metrics_root, name) for name in names
    }
    selected_name, challenger_diagnostics = choose_final(
        aggregates,
        shortlist,
        config["incumbent"],
        args.minimum_improvement,
        args.minimum_nonnegative_seeds,
    )
    selected_recipe = recipes[selected_name]
    checkpoints = {
        str(seed): resolve_checkpoint_record(
            args.repo_root,
            args.model_root,
            selected_recipe,
            selected_name,
            seed,
        )
        for seed in range(3)
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_selection_v1",
        "ranking_metric": "three_seed_mean_closedloop_reactive.score",
        "selection_data": "58_scenes_22_logs_log_disjoint_cl_dev",
        "formal_data_consumed": False,
        "minimum_improvement": float(args.minimum_improvement),
        "minimum_nonnegative_seeds": int(args.minimum_nonnegative_seeds),
        "incumbent": config["incumbent"],
        "shortlist": shortlist,
        "selected": {
            "name": selected_name,
            "kind": selected_recipe["kind"],
            "temperature": selected_recipe["temperature"],
            "learning_rate": selected_recipe["learning_rate"],
            "kl_weight": selected_recipe["kl_weight"],
            "epoch": selected_recipe["epoch"],
            "promotion_from_incumbent": selected_name != config["incumbent"],
        },
        "checkpoints": checkpoints,
        "aggregates": aggregates,
        "challenger_diagnostics": challenger_diagnostics,
        "lane_reports": lane_reports,
        "recipe_config_sha256": recipe_sha256,
        "split_audit": str(args.split_audit.resolve()),
        "split_audit_sha256": split_sha256,
        "screen_sha256": sha256_file(args.screen),
        "code_commit": str(args.code_commit),
    }
    write_json(args.output, report)
    return report


def parse_seed_paths(values, label: str) -> dict[int, Path]:
    output = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"{label}: expected SEED=PATH")
        seed = int(seed_text)
        path = Path(path_text).expanduser().resolve()
        if seed in output:
            raise ValueError(f"{label}: duplicate seed {seed}")
        output[seed] = path
    if set(output) != {0, 1, 2}:
        raise ValueError(f"{label}: expected seeds 0,1,2")
    return output


def load_formal_summary(path: Path, seed: int) -> dict:
    payload = load_pass_json(path)
    if int(payload["eval_seed"]) != seed:
        raise RuntimeError(f"formal summary seed mismatch: {path}")
    checkpoint = Path(payload["checkpoint"])
    if sha256_file(checkpoint) != payload["checkpoint_sha256"]:
        raise RuntimeError(f"formal checkpoint SHA256 drifted: {path}")
    return payload


def confirmation_metrics(summary: dict, confirmation_tokens: set[str]) -> dict:
    row = summary["input_files"]["closedloop_r"]
    path = Path(row["path"]).expanduser().resolve()
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"formal reactive CSV drifted: {path}")
    rows = read_metric_csv(path)
    missing = confirmation_tokens - set(rows)
    if missing:
        raise RuntimeError(f"formal reactive CSV missing {len(missing)} confirmation tokens")
    selected = {token: rows[token] for token in sorted(confirmation_tokens)}
    return summarize_rows(selected)


def finalize(args) -> dict:
    selection = load_pass_json(args.selection)
    split = load_pass_json(args.split_audit)
    confirmation_tokens = set(split["splits"]["confirmation"]["tokens"])
    selected_paths = parse_seed_paths(args.selected_summary, "selected-summary")
    incumbent_paths = parse_seed_paths(args.incumbent_summary, "incumbent-summary")
    selected_rows = {}
    incumbent_rows = {}
    for seed in range(3):
        selected = load_formal_summary(selected_paths[seed], seed)
        incumbent = load_formal_summary(incumbent_paths[seed], seed)
        expected_sha = selection["checkpoints"][str(seed)]["checkpoint_sha256"]
        if selected["checkpoint_sha256"] != expected_sha:
            raise RuntimeError("selected formal summary checkpoint drifted")
        selected_rows[str(seed)] = {
            "summary": str(selected_paths[seed]),
            "summary_sha256": sha256_file(selected_paths[seed]),
            "full_score": selected["metrics"]["closedloop_reactive"]["score"],
            "full_success_rate": selected["metrics"]["success_rate"],
            "confirmation": confirmation_metrics(selected, confirmation_tokens),
        }
        incumbent_rows[str(seed)] = {
            "summary": str(incumbent_paths[seed]),
            "summary_sha256": sha256_file(incumbent_paths[seed]),
            "full_score": incumbent["metrics"]["closedloop_reactive"]["score"],
            "full_success_rate": incumbent["metrics"]["success_rate"],
            "confirmation": confirmation_metrics(incumbent, confirmation_tokens),
        }

    def aggregate(rows):
        return {
            "full_score_mean": statistics.fmean(
                row["full_score"] for row in rows.values()
            ),
            "full_score_population_std": statistics.pstdev(
                row["full_score"] for row in rows.values()
            ),
            "full_success_rate_mean": statistics.fmean(
                row["full_success_rate"] for row in rows.values()
            ),
            "confirmation_score_mean": statistics.fmean(
                row["confirmation"]["score"] for row in rows.values()
            ),
            "confirmation_success_rate_mean": statistics.fmean(
                row["confirmation"]["success_rate"] for row in rows.values()
            ),
        }

    selected_aggregate = aggregate(selected_rows)
    incumbent_aggregate = aggregate(incumbent_rows)
    confirmation_seed_deltas = [
        selected_rows[str(seed)]["confirmation"]["score"]
        - incumbent_rows[str(seed)]["confirmation"]["score"]
        for seed in range(3)
    ]
    full_seed_deltas = [
        selected_rows[str(seed)]["full_score"]
        - incumbent_rows[str(seed)]["full_score"]
        for seed in range(3)
    ]
    best_seed = max(
        range(3), key=lambda seed: selected_rows[str(seed)]["full_score"]
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_clpdms_formal_summary_v1",
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "selected": selection["selected"],
        "selected_by_seed": selected_rows,
        "incumbent_by_seed": incumbent_rows,
        "selected_aggregate": selected_aggregate,
        "incumbent_aggregate": incumbent_aggregate,
        "confirmation_score_delta_mean": statistics.fmean(
            confirmation_seed_deltas
        ),
        "confirmation_score_deltas_by_seed": confirmation_seed_deltas,
        "confirmation_nonnegative_seed_count": sum(
            value >= 0.0 for value in confirmation_seed_deltas
        ),
        "full_score_delta_mean": statistics.fmean(full_seed_deltas),
        "full_score_deltas_by_seed": full_seed_deltas,
        "best_seed": best_seed,
        "best_seed_full_score": selected_rows[str(best_seed)]["full_score"],
        "confirmation_generalized": (
            statistics.fmean(confirmation_seed_deltas) >= 0.0
            and sum(value >= 0.0 for value in confirmation_seed_deltas) >= 2
        ),
    }
    write_json(args.output, report)
    markdown = args.output.with_suffix(".md")
    lines = [
        "# DiffusionDrive V3 rare CL-PDMS tuning",
        "",
        "| Model | CL-confirm PDMS | Full CL-PDMS | CL Valid Rate |",
        "|---|---:|---:|---:|",
        (
            f"| incumbent rare_tuned | "
            f"{100 * incumbent_aggregate['confirmation_score_mean']:.2f} | "
            f"{100 * incumbent_aggregate['full_score_mean']:.2f} | "
            f"{100 * incumbent_aggregate['full_success_rate_mean']:.2f} |"
        ),
        (
            f"| selected {selection['selected']['name']} | "
            f"{100 * selected_aggregate['confirmation_score_mean']:.2f} | "
            f"{100 * selected_aggregate['full_score_mean']:.2f} | "
            f"{100 * selected_aggregate['full_success_rate_mean']:.2f} |"
        ),
        "",
        (
            f"Best seed: {best_seed}; full CL-PDMS="
            f"{100 * report['best_seed_full_score']:.2f}."
        ),
        (
            "Confirmation generalized: "
            f"{str(report['confirmation_generalized']).lower()}."
        ),
    ]
    atomic_write_bytes(markdown, ("\n".join(lines) + "\n").encode("utf-8"))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--output-root", type=Path, required=True)
    split.add_argument("--split-seed", type=int, default=20260820)
    split.add_argument("--development-fraction", type=float, default=0.2)
    split.add_argument("--code-commit", required=True)
    split.add_argument("--expected-source-scenes", type=int, default=288)
    split.add_argument("--expected-source-logs", type=int, default=89)
    split.add_argument("--expected-development-scenes", type=int, default=58)
    split.add_argument("--expected-development-logs", type=int, default=22)
    split.add_argument("--expected-confirmation-scenes", type=int, default=230)
    split.add_argument("--expected-confirmation-logs", type=int, default=67)
    split.set_defaults(func=build_split)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--split-audit", type=Path, required=True)
    metrics.add_argument("--split", choices=("development", "confirmation", "smoke"), required=True)
    metrics.add_argument("--csv", type=Path, required=True)
    metrics.add_argument("--model", required=True)
    metrics.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    metrics.add_argument("--checkpoint-manifest", type=Path)
    metrics.add_argument("--allow-superset", action="store_true")
    metrics.add_argument("--output", type=Path, required=True)
    metrics.set_defaults(func=build_metrics)

    screen = subparsers.add_parser("screen")
    screen.add_argument("--recipe-config", type=Path, required=True)
    screen.add_argument("--metrics-root", type=Path, required=True)
    screen.add_argument("--shortlist-size", type=int, default=2)
    screen.add_argument("--output", type=Path, required=True)
    screen.set_defaults(func=screen_candidates)

    lane = subparsers.add_parser("lane")
    lane.add_argument("--recipe-config", type=Path, required=True)
    lane.add_argument("--metrics-root", type=Path, required=True)
    lane.add_argument("--repo-root", type=Path, required=True)
    lane.add_argument("--model-root", type=Path, required=True)
    lane.add_argument("--split-audit", type=Path, required=True)
    lane.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    lane.add_argument("--code-commit", required=True)
    lane.add_argument("--output", type=Path, required=True)
    lane.set_defaults(func=audit_lane)

    select = subparsers.add_parser("select")
    select.add_argument("--recipe-config", type=Path, required=True)
    select.add_argument("--screen", type=Path, required=True)
    select.add_argument("--metrics-root", type=Path, required=True)
    select.add_argument("--repo-root", type=Path, required=True)
    select.add_argument("--model-root", type=Path, required=True)
    select.add_argument("--split-audit", type=Path, required=True)
    select.add_argument("--lane-report", type=Path, action="append", required=True)
    select.add_argument("--minimum-improvement", type=float, default=0.005)
    select.add_argument("--minimum-nonnegative-seeds", type=int, default=2)
    select.add_argument("--code-commit", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(func=select_candidate)

    final = subparsers.add_parser("finalize")
    final.add_argument("--selection", type=Path, required=True)
    final.add_argument("--split-audit", type=Path, required=True)
    final.add_argument("--selected-summary", action="append", required=True)
    final.add_argument("--incumbent-summary", action="append", required=True)
    final.add_argument("--output", type=Path, required=True)
    final.set_defaults(func=finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = args.func(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
