#!/usr/bin/env python3
"""Prepare the full-navtrain DiffusionDrive V3 rare-log data contract.

The commands in this module deliberately keep the large NAVSIM metric-cache
payloads immutable.  They build an index-only view, export deployed
DiffusionDrive trajectories, audit official PDM scores, and construct the
all-rare plus same-log-common training manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PDM_COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_values(values):
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_rank(seed, *values):
    text = ":".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_immutable_text(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite non-identical file: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path


def write_immutable_bytes(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite non-identical file: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def load_yaml(path):
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return payload


def load_filter_tokens(path):
    payload = load_yaml(path)
    tokens = payload.get("tokens") or payload.get("scenario_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"filter contains no tokens: {path}")
    values = [str(token) for token in tokens]
    if len(values) != len(set(values)):
        raise ValueError(f"filter contains duplicate tokens: {path}")
    return values


def load_annotation(path):
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise TypeError(f"unsupported annotation payload: {path}")
    token_to_log = {}
    token_to_scene = {}
    for info in infos:
        token = str(info["token"])
        log_name = str(info["log_name"])
        scene = str(info.get("scene_token") or info.get("log_token") or log_name)
        previous = token_to_log.get(token)
        if previous is not None and previous != log_name:
            raise RuntimeError(f"annotation token {token} belongs to multiple logs")
        token_to_log[token] = log_name
        token_to_scene[token] = scene
    return token_to_log, token_to_scene


def metadata_csv(index_root):
    metadata_dir = Path(index_root) / "metadata"
    files = sorted(metadata_dir.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(
            f"expected exactly one metadata CSV under {metadata_dir}, found {len(files)}"
        )
    return files[0]


def load_cache_index(index_root):
    index_root = Path(index_root)
    result = {}
    with metadata_csv(index_root).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file_name"]:
            raise ValueError(f"unexpected metric-cache metadata columns: {reader.fieldnames}")
        for row in reader:
            cache_path = Path(row["file_name"])
            if not cache_path.is_absolute():
                cache_path = index_root / cache_path
            token = cache_path.parent.name
            if token in result:
                raise ValueError(f"duplicate cache token in metadata: {token}")
            result[token] = cache_path
    if not result:
        raise ValueError(f"empty metric-cache index: {index_root}")
    return result


def build_full_index(
    physical_cache_root,
    navtrain_filter,
    navtest_filter,
    annotation_file,
    output_dir,
    expected_tokens,
    expected_logs,
    expected_physical_tokens=None,
    expected_navtest_tokens=None,
    expected_navtest_logs=None,
):
    physical_cache_root = Path(physical_cache_root).expanduser().resolve()
    navtrain_filter = Path(navtrain_filter).expanduser().resolve()
    navtest_filter = Path(navtest_filter).expanduser().resolve()
    annotation_file = Path(annotation_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (
        physical_cache_root,
        navtrain_filter,
        navtest_filter,
        annotation_file,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_payload = load_yaml(navtrain_filter)
    log_names = source_payload.get("log_names")
    if not isinstance(log_names, list) or not log_names:
        raise ValueError("full navtrain filter must contain log_names")
    log_names = {str(log_name) for log_name in log_names}
    if len(log_names) != expected_logs:
        raise RuntimeError(
            f"navtrain log coverage {len(log_names)} != expected {expected_logs}"
        )

    physical = {}
    physical_logs = {}
    selected = {}
    selected_logs = set()
    for cache_path in physical_cache_root.rglob("metric_cache.pkl"):
        relative = cache_path.relative_to(physical_cache_root)
        if len(relative.parts) < 4:
            raise RuntimeError(f"unexpected metric-cache path: {cache_path}")
        log_name = relative.parts[-4]
        token = cache_path.parent.name
        if token in physical:
            raise RuntimeError(f"duplicate physical metric-cache token: {token}")
        absolute = Path(str(cache_path.absolute()))
        physical[token] = absolute
        physical_logs[token] = log_name
        if log_name in log_names:
            selected[token] = absolute
            selected_logs.add(log_name)

    if expected_physical_tokens is not None and len(physical) != expected_physical_tokens:
        raise RuntimeError(
            f"physical cache coverage {len(physical)} != {expected_physical_tokens}"
        )
    if len(selected) != expected_tokens:
        raise RuntimeError(
            f"full navtrain cache coverage {len(selected)} != expected {expected_tokens}"
        )
    if selected_logs != log_names:
        missing = sorted(log_names - selected_logs)
        raise RuntimeError(
            f"physical cache omitted {len(missing)} navtrain logs: {missing[:5]}"
        )

    token_to_log, _ = load_annotation(annotation_file)
    annotation_tokens = set(token_to_log)
    selected_tokens = set(selected)
    missing_annotation = selected_tokens - annotation_tokens
    if missing_annotation:
        raise RuntimeError(
            f"annotation omitted {len(missing_annotation)} navtrain cache tokens"
        )
    wrong_logs = [
        token
        for token in selected_tokens
        if token_to_log[token] not in log_names
    ]
    if wrong_logs:
        raise RuntimeError(f"{len(wrong_logs)} selected tokens map to non-navtrain logs")

    navtest_payload = load_yaml(navtest_filter)
    filter_tokens = navtest_payload.get("tokens") or navtest_payload.get(
        "scenario_tokens"
    )
    filter_logs = navtest_payload.get("log_names")
    if isinstance(filter_tokens, list) and filter_tokens:
        navtest_filter_kind = "tokens"
        navtest_tokens = {str(token) for token in filter_tokens}
    elif isinstance(filter_logs, list) and filter_logs:
        navtest_filter_kind = "log_names"
        filter_log_names = {str(log_name) for log_name in filter_logs}
        navtest_tokens = {
            token
            for token, log_name in physical_logs.items()
            if log_name in filter_log_names
        }
    else:
        raise ValueError("navtest filter must contain tokens or log_names")
    missing_navtest_cache = navtest_tokens - set(physical)
    if missing_navtest_cache:
        raise RuntimeError(
            f"physical cache omitted {len(missing_navtest_cache)} navtest tokens"
        )
    navtest_logs = {physical_logs[token] for token in navtest_tokens}
    if expected_navtest_tokens is not None and len(navtest_tokens) != expected_navtest_tokens:
        raise RuntimeError(
            f"navtest cache coverage {len(navtest_tokens)} != {expected_navtest_tokens}"
        )
    if expected_navtest_logs is not None and len(navtest_logs) != expected_navtest_logs:
        raise RuntimeError(
            f"navtest log coverage {len(navtest_logs)} != {expected_navtest_logs}"
        )
    log_overlap = log_names & navtest_logs
    if log_overlap:
        raise RuntimeError(f"navtrain/navtest log overlap: {len(log_overlap)}")
    overlap = selected_tokens & navtest_tokens
    if overlap:
        raise RuntimeError(f"full navtrain cache overlaps navtest by {len(overlap)} tokens")

    index_path = output_dir / "metadata" / "metric_cache.csv"
    rows = ["file_name", *(str(selected[token]) for token in sorted(selected))]
    write_immutable_text(index_path, "\n".join(rows) + "\n")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "full_navtrain_metric_cache_index",
        "physical_cache_root": str(physical_cache_root),
        "physical_cache_tokens": len(physical),
        "navtrain_filter": str(navtrain_filter),
        "navtrain_filter_sha256": sha256_file(navtrain_filter),
        "navtrain_logs": len(log_names),
        "navtrain_tokens": len(selected),
        "navtrain_tokens_sha256": digest_values(selected),
        "annotation_file": str(annotation_file),
        "annotation_sha256": sha256_file(annotation_file),
        "annotation_tokens": len(annotation_tokens),
        "navtest_filter": str(navtest_filter),
        "navtest_filter_sha256": sha256_file(navtest_filter),
        "navtest_filter_kind": navtest_filter_kind,
        "navtest_logs": len(navtest_logs),
        "navtest_tokens": len(navtest_tokens),
        "navtest_overlap": 0,
        "metadata_csv": str(index_path),
        "metadata_sha256": sha256_file(index_path),
    }
    audit_path = output_dir / "index_audit.json"
    write_immutable_text(
        audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def flatten_results(payload):
    if not isinstance(payload, list):
        raise TypeError("inference result must be a list")
    rows = []
    for item in payload:
        if isinstance(item, list):
            rows.extend(item)
        else:
            rows.append(item)
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("inference result rows must be dictionaries")
    return rows


def build_submission(
    results_path,
    cache_index,
    checkpoint,
    config,
    provenance_files,
    expected_checkpoint_sha256,
    noise_seed,
    noise_namespace,
    output,
    manifest_path,
):
    results_path = Path(results_path).expanduser().resolve()
    cache_index = Path(cache_index).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    config = Path(config).expanduser().resolve()
    provenance_files = [
        Path(path).expanduser().resolve() for path in provenance_files
    ]
    output = Path(output).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    for path in (results_path, cache_index, checkpoint, config, *provenance_files):
        if not path.exists():
            raise FileNotFoundError(path)
    if not provenance_files:
        raise ValueError("at least one implementation provenance file is required")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError("rare-mining checkpoint SHA256 mismatch")

    with results_path.open("rb") as stream:
        rows = flatten_results(pickle.load(stream))
    expected_tokens = set(load_cache_index(cache_index))
    by_token = {}
    for row in rows:
        token = str(row.get("token"))
        if token in by_token:
            raise RuntimeError(f"duplicate inference token: {token}")
        trajectory = np.asarray(row.get("trajectory"), dtype=np.float32)
        if trajectory.shape != (40, 3):
            raise RuntimeError(f"{token}: trajectory shape {trajectory.shape} != (40, 3)")
        if not np.isfinite(trajectory).all():
            raise RuntimeError(f"{token}: non-finite trajectory")
        chosen = int(row.get("chosen_ind", -1))
        if not 0 <= chosen < 20:
            raise RuntimeError(f"{token}: invalid chosen candidate {chosen}")
        by_token[token] = trajectory
    if set(by_token) != expected_tokens:
        missing = expected_tokens - set(by_token)
        extra = set(by_token) - expected_tokens
        raise RuntimeError(
            f"inference coverage drifted: missing={len(missing)} extra={len(extra)}"
        )

    from navsim.common.dataclasses import Trajectory

    predictions = {
        token: Trajectory(by_token[token][4::5].astype(np.float32))
        for token in sorted(by_token)
    }
    submission = {
        "team_name": "DiffusionDrive-V3-rare-log-mining",
        "authors": ["WorldEngine"],
        "email": "placeholder@example.com",
        "institution": "WorldEngine",
        "country / region": "N/A",
        "predictions": [predictions],
    }
    write_immutable_bytes(
        output, pickle.dumps(submission, protocol=pickle.HIGHEST_PROTOCOL)
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_epoch100_full_navtrain_deployed_submission",
        "noise_seed": int(noise_seed),
        "noise_namespace": str(noise_namespace),
        "num_tokens": len(predictions),
        "tokens_sha256": digest_values(predictions),
        "results": str(results_path),
        "results_sha256": sha256_file(results_path),
        "cache_index": str(cache_index),
        "cache_index_sha256": sha256_file(metadata_csv(cache_index)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "implementation_files": {
            str(path): sha256_file(path) for path in provenance_files
        },
        "submission": str(output),
        "submission_sha256": sha256_file(output),
    }
    write_immutable_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_score_rows(path):
    path = Path(path)
    rows = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = set(("token", "valid", *PDM_COMPONENTS)) - set(
            reader.fieldnames or []
        )
        if missing_columns:
            raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
        for row in reader:
            token = str(row["token"])
            if token == "average":
                continue
            if token in rows:
                raise RuntimeError(f"duplicate score token in {path}: {token}")
            if row["valid"].strip().lower() != "true":
                raise RuntimeError(f"invalid official PDM score for {token}")
            values = {name: float(row[name]) for name in PDM_COMPONENTS}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite official PDM score for {token}")
            rows[token] = values
    if not rows:
        raise RuntimeError(f"empty score CSV: {path}")
    return rows


def audit_score(score_csv, submission_path, cache_index, output, expected_tokens):
    score_csv = Path(score_csv).expanduser().resolve()
    submission_path = Path(submission_path).expanduser().resolve()
    cache_index = Path(cache_index).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    for path in (score_csv, submission_path, cache_index):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = set(load_cache_index(cache_index))
    if len(expected) != expected_tokens:
        raise RuntimeError(f"cache index contains {len(expected)} != {expected_tokens}")
    with submission_path.open("rb") as stream:
        submission = pickle.load(stream)
    predictions = submission.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise RuntimeError("submission must contain exactly one prediction mapping")
    submitted = set(str(token) for token in predictions[0])
    rows = load_score_rows(score_csv)
    if submitted != expected or set(rows) != expected:
        raise RuntimeError(
            "submission, score CSV and full-navtrain cache token sets disagree"
        )
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "num_tokens": len(expected),
        "tokens_sha256": digest_values(expected),
        "cache_index": str(cache_index),
        "cache_index_sha256": sha256_file(metadata_csv(cache_index)),
        "submission": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "score_csv": str(score_csv),
        "score_csv_sha256": sha256_file(score_csv),
        "all_rows_valid_and_finite": True,
    }
    write_immutable_text(output, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def parse_seed_paths(values):
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"expected SEED=PATH, got {value}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate seed {seed}")
        result[seed] = Path(path_text).expanduser().resolve()
    if set(result) != {0, 1, 2}:
        raise ValueError(f"rare mining requires score seeds 0,1,2; got {sorted(result)}")
    return result


def classify_rare(rows, percentile):
    collisions = {
        token
        for token, values in rows.items()
        if values["no_at_fault_collisions"] == 0.0
    }
    offroad = {
        token
        for token, values in rows.items()
        if values["drivable_area_compliance"] == 0.0
    }
    safe_progress = [
        values["ego_progress"]
        for values in rows.values()
        if values["no_at_fault_collisions"] == 1.0
        and values["drivable_area_compliance"] == 1.0
        and values["ego_progress"] > 0.0
    ]
    if not safe_progress:
        raise RuntimeError("no safe positive-progress samples for rare mining")
    threshold = float(np.percentile(np.asarray(safe_progress), percentile))
    low_progress = {
        token
        for token, values in rows.items()
        if values["no_at_fault_collisions"] == 1.0
        and values["drivable_area_compliance"] == 1.0
        and values["ego_progress"] > 0.0
        and values["ego_progress"] <= threshold
    }
    modes = {
        "collision": collisions,
        "offroad": offroad,
        "low_ego_progress": low_progress,
    }
    return modes, threshold


def render_filter(base_payload, tokens):
    payload = dict(base_payload)
    payload["tokens"] = sorted(tokens)
    return yaml.safe_dump(payload, sort_keys=False)


def build_rare_dataset(
    score_paths,
    cache_index,
    annotation_file,
    base_filter,
    output_dir,
    expected_tokens,
    ep_percentile,
    pair_seed,
):
    score_paths = parse_seed_paths(score_paths)
    cache_index = Path(cache_index).expanduser().resolve()
    annotation_file = Path(annotation_file).expanduser().resolve()
    base_filter = Path(base_filter).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (*score_paths.values(), cache_index, annotation_file, base_filter):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = set(load_cache_index(cache_index))
    if len(expected) != expected_tokens:
        raise RuntimeError(f"full cache index contains {len(expected)} != {expected_tokens}")

    by_seed = {}
    modes_by_seed = {}
    thresholds = {}
    for seed, path in sorted(score_paths.items()):
        rows = load_score_rows(path)
        if set(rows) != expected:
            raise RuntimeError(f"score seed {seed} token coverage drifted")
        by_seed[seed] = rows
        modes_by_seed[seed], thresholds[seed] = classify_rare(rows, ep_percentile)

    rare_by_seed = {
        seed: set().union(*modes.values())
        for seed, modes in modes_by_seed.items()
    }
    votes = {
        token: sum(token in rare_by_seed[seed] for seed in sorted(rare_by_seed))
        for token in expected
    }
    rare_tokens = {token for token, count in votes.items() if count >= 1}
    strict_common = expected - rare_tokens
    if not rare_tokens:
        raise RuntimeError("full-navtrain mining produced no rare tokens")
    if not strict_common:
        raise RuntimeError("full-navtrain mining produced no strict-common tokens")

    token_to_log, token_to_scene = load_annotation(annotation_file)
    missing_annotation = expected - set(token_to_log)
    if missing_annotation:
        raise RuntimeError(
            f"annotation omitted {len(missing_annotation)} full-navtrain tokens"
        )
    common_by_log = defaultdict(list)
    rare_by_log = defaultdict(list)
    for token in strict_common:
        common_by_log[token_to_log[token]].append(token)
    for token in rare_tokens:
        rare_by_log[token_to_log[token]].append(token)

    pairs = []
    for log_name in sorted(rare_by_log):
        rare_pool = sorted(
            rare_by_log[log_name],
            key=lambda token: deterministic_rank(pair_seed, "rare", log_name, token),
        )
        common_pool = sorted(
            common_by_log.get(log_name, []),
            key=lambda token: deterministic_rank(pair_seed, "common", log_name, token),
        )
        if not common_pool:
            raise RuntimeError(
                f"rare log {log_name} has no strict-common token across three seeds"
            )
        for index, rare_token in enumerate(rare_pool):
            common_token = common_pool[index % len(common_pool)]
            failure_modes = {
                str(seed): sorted(
                    name
                    for name, tokens in modes_by_seed[seed].items()
                    if rare_token in tokens
                )
                for seed in sorted(modes_by_seed)
            }
            pairs.append(
                {
                    "rare_token": rare_token,
                    "common_token": common_token,
                    "log_name": log_name,
                    "scene": token_to_scene[rare_token],
                    "rare_votes": votes[rare_token],
                    "failure_modes_by_seed": failure_modes,
                }
            )
    pairs.sort(key=lambda row: row["rare_token"])
    selected_common = [row["common_token"] for row in pairs]
    unique_common = set(selected_common)
    union_tokens = rare_tokens | unique_common
    if rare_tokens & unique_common:
        raise RuntimeError("rare/common pairing overlap")
    if len(pairs) != len(rare_tokens):
        raise RuntimeError("one-to-one rare/common pair accounting drifted")

    base_payload = load_yaml(base_filter)
    rare_filter = output_dir / "rare_tokens.yaml"
    common_filter = output_dir / "common_tokens.yaml"
    union_filter = output_dir / "rare_common_union.yaml"
    pairs_path = output_dir / "pairs.jsonl"
    write_immutable_text(rare_filter, render_filter(base_payload, rare_tokens))
    write_immutable_text(common_filter, render_filter(base_payload, unique_common))
    write_immutable_text(union_filter, render_filter(base_payload, union_tokens))
    pair_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in pairs
    )
    write_immutable_text(pairs_path, pair_text)

    vote_histogram = Counter(votes.values())
    mode_counts = {
        str(seed): {
            name: len(tokens) for name, tokens in sorted(modes.items())
        }
        for seed, modes in sorted(modes_by_seed.items())
    }
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_full_navtrain_rare_log_v1",
        "rare_definition": {
            "collision": "no_at_fault_collisions == 0",
            "offroad": "drivable_area_compliance == 0",
            "low_ego_progress": (
                "safe positive ego_progress at or below the per-seed "
                f"{float(ep_percentile):g}th percentile"
            ),
            "seed_union": "rare in at least one of fixed noise seeds 0,1,2",
            "common": "rare vote count == 0 across seeds 0,1,2",
            "pairing": "one deterministic same-log strict-common row per rare row",
        },
        "expected_tokens": int(expected_tokens),
        "score_seeds": [0, 1, 2],
        "score_csvs": {
            str(seed): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for seed, path in sorted(score_paths.items())
        },
        "ego_progress_percentile": float(ep_percentile),
        "ego_progress_threshold_by_seed": {
            str(seed): threshold for seed, threshold in sorted(thresholds.items())
        },
        "failure_mode_counts_by_seed": mode_counts,
        "rare_count_by_seed": {
            str(seed): len(tokens) for seed, tokens in sorted(rare_by_seed.items())
        },
        "rare_vote_histogram": {
            str(vote): count for vote, count in sorted(vote_histogram.items())
        },
        "rare_count": len(rare_tokens),
        "strict_common_population": len(strict_common),
        "paired_common_rows": len(selected_common),
        "paired_common_unique": len(unique_common),
        "paired_common_reuse_count": len(selected_common) - len(unique_common),
        "union_count": len(union_tokens),
        "rare_tokens_sha256": digest_values(rare_tokens),
        "paired_common_tokens_sha256": digest_values(unique_common),
        "union_tokens_sha256": digest_values(union_tokens),
        "pair_seed": int(pair_seed),
        "cache_index": str(cache_index),
        "cache_index_sha256": sha256_file(metadata_csv(cache_index)),
        "annotation_file": str(annotation_file),
        "annotation_sha256": sha256_file(annotation_file),
        "base_filter": str(base_filter),
        "base_filter_sha256": sha256_file(base_filter),
        "outputs": {
            "rare_filter": str(rare_filter),
            "rare_filter_sha256": sha256_file(rare_filter),
            "common_filter": str(common_filter),
            "common_filter_sha256": sha256_file(common_filter),
            "union_filter": str(union_filter),
            "union_filter_sha256": sha256_file(union_filter),
            "pairs": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
        },
    }
    audit_path = output_dir / "rare_data_audit.json"
    write_immutable_text(
        audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser(
        "index", help="build an immutable full-navtrain metric-cache index view"
    )
    index.add_argument("--physical-cache-root", type=Path, required=True)
    index.add_argument("--navtrain-filter", type=Path, required=True)
    index.add_argument("--navtest-filter", type=Path, required=True)
    index.add_argument("--annotation-file", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.add_argument("--expected-tokens", type=int, default=103288)
    index.add_argument("--expected-logs", type=int, default=1192)
    index.add_argument("--expected-physical-tokens", type=int, default=115434)
    index.add_argument("--expected-navtest-tokens", type=int, default=12146)
    index.add_argument("--expected-navtest-logs", type=int, default=136)

    submission = subparsers.add_parser(
        "submission", help="convert deployed test.py results to a NAVSIM submission"
    )
    submission.add_argument("--results", type=Path, required=True)
    submission.add_argument("--cache-index", type=Path, required=True)
    submission.add_argument("--checkpoint", type=Path, required=True)
    submission.add_argument("--config", type=Path, required=True)
    submission.add_argument(
        "--provenance-file",
        type=Path,
        action="append",
        required=True,
    )
    submission.add_argument("--expected-checkpoint-sha256", required=True)
    submission.add_argument("--noise-seed", type=int, required=True)
    submission.add_argument("--noise-namespace", required=True)
    submission.add_argument("--output", type=Path, required=True)
    submission.add_argument("--manifest", type=Path, required=True)

    score = subparsers.add_parser(
        "audit-score", help="audit one complete official full-navtrain score"
    )
    score.add_argument("--score-csv", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)
    score.add_argument("--cache-index", type=Path, required=True)
    score.add_argument("--expected-tokens", type=int, default=103288)
    score.add_argument("--output", type=Path, required=True)

    mine = subparsers.add_parser(
        "mine", help="mine the all-rare union and deterministic same-log common pairs"
    )
    mine.add_argument(
        "--score", action="append", required=True, metavar="SEED=CSV"
    )
    mine.add_argument("--cache-index", type=Path, required=True)
    mine.add_argument("--annotation-file", type=Path, required=True)
    mine.add_argument("--base-filter", type=Path, required=True)
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.add_argument("--expected-tokens", type=int, default=103288)
    mine.add_argument("--ego-progress-percentile", type=float, default=1.0)
    mine.add_argument("--pair-seed", type=int, default=20260818)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "index":
        report = build_full_index(
            args.physical_cache_root,
            args.navtrain_filter,
            args.navtest_filter,
            args.annotation_file,
            args.output_dir,
            args.expected_tokens,
            args.expected_logs,
            args.expected_physical_tokens,
            args.expected_navtest_tokens,
            args.expected_navtest_logs,
        )
    elif args.command == "submission":
        report = build_submission(
            args.results,
            args.cache_index,
            args.checkpoint,
            args.config,
            args.provenance_file,
            args.expected_checkpoint_sha256,
            args.noise_seed,
            args.noise_namespace,
            args.output,
            args.manifest,
        )
    elif args.command == "audit-score":
        report = audit_score(
            args.score_csv,
            args.submission,
            args.cache_index,
            args.output,
            args.expected_tokens,
        )
    elif args.command == "mine":
        if not 0.0 < args.ego_progress_percentile < 100.0:
            raise ValueError("--ego-progress-percentile must be in (0, 100)")
        report = build_rare_dataset(
            args.score,
            args.cache_index,
            args.annotation_file,
            args.base_filter,
            args.output_dir,
            args.expected_tokens,
            args.ego_progress_percentile,
            args.pair_seed,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
