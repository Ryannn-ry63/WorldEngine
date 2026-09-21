#!/usr/bin/env python3
"""Fail-closed audits for rollout-v1 development through formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHOD = "scene_conditioned_exact_group_grpo_rollout_v1"
RELEASE = "e2e_diffusiondrive_grpo_selector_rollout_v1"
EXPECTED_CACHE_LAYOUT = {
    "train": ((0, 1, 2), 348, 2784),
    "development": ((3, 4, 5), 43, 344),
    "certification": ((6, 7, 8), 21, 168),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        fail(f"invalid JSON file {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def require_pass(row: dict, path: Path, schema_version: int | None = None) -> None:
    require_equal(row.get("status"), "PASS", f"{path} status")
    if schema_version is not None:
        require_equal(
            row.get("schema_version"), schema_version, f"{path} schema_version"
        )


def require_file_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    require_equal(actual, expected, f"{label} SHA256")


def resolve_recorded_path(value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} path is missing")
    return Path(value).expanduser().resolve()


def validate_inputs(root: Path, worldengine_root: Path) -> None:
    split_path = root / "splits/navtrain_50pct_collision_r1full.json"
    split = read_json(split_path)
    require_equal(split.get("rollout_seeds"), list(range(9)), "split rollout seeds")
    require_equal(split.get("num_scenes"), 412, "split scene count")
    require_equal(split.get("num_records_per_seed"), 3296, "split record count")
    split_sha = sha256_file(split_path)

    identities = set()
    baselines = set()
    for split_name, (seeds, expected_scenes, expected_tokens) in EXPECTED_CACHE_LAYOUT.items():
        for seed in seeds:
            cache_dir = root / f"cache/{split_name}_seed{seed}"
            cache_path = cache_dir / "cache.pt"
            manifest_path = cache_dir / "manifest.json"
            manifest = read_json(manifest_path)
            require_pass(manifest, manifest_path, 3)
            require_equal(manifest.get("source_kind"), "base_policy_rollout", f"seed{seed} source")
            require_equal(manifest.get("split"), split_name, f"seed{seed} split")
            require_equal(manifest.get("noise_seed"), seed, f"seed{seed} noise seed")
            require_equal(manifest.get("num_scenes"), expected_scenes, f"seed{seed} scenes")
            require_equal(manifest.get("num_tokens"), expected_tokens, f"seed{seed} tokens")
            require_equal(manifest.get("num_candidates"), 20, f"seed{seed} candidates")
            require_equal(
                manifest.get("split_manifest_sha256"), split_sha, f"seed{seed} split manifest"
            )
            require_file_sha(cache_path, manifest.get("cache_sha256"), f"seed{seed} cache")
            identities.add(
                (
                    manifest.get("checkpoint_sha256"),
                    manifest.get("config_sha256"),
                    manifest.get("annotation_sha256"),
                    manifest.get("code_sha"),
                )
            )
            baselines.add(
                (
                    resolve_recorded_path(manifest.get("checkpoint"), f"seed{seed} baseline"),
                    manifest.get("checkpoint_sha256"),
                )
            )
    require_equal(len(identities), 1, "cache identity count")
    require_equal(len(baselines), 1, "cache baseline count")
    baseline, baseline_sha = next(iter(baselines))
    require_file_sha(baseline, baseline_sha, "baseline checkpoint")

    for seed in (0, 1, 2):
        path = (
            worldengine_root
            / f"experiments/diffusiondrive/grpo_selector_formal/formal_eval/"
            f"e2e_diffusiondrive_reference_paired_s{seed}/summary.json"
        )
        row = read_json(path)
        require_pass(row, path, 1)
        require_equal(row.get("eval_seed"), seed, f"reference seed{seed} eval seed")
        require_equal(row.get("counts", {}).get("openloop_navtest"), 12147, f"reference seed{seed} navtest count")
        require_equal(row.get("counts", {}).get("openloop_failures"), 289, f"reference seed{seed} failure count")


def checkpoint_for_epoch(report: dict, epoch: int, label: str) -> dict:
    matches = [row for row in report.get("checkpoints", []) if row.get("epoch") == epoch]
    require_equal(len(matches), 1, f"{label} epoch {epoch} checkpoint count")
    return matches[0]


def validate_development(root: Path) -> dict:
    status_path = root / "development/status.txt"
    if not status_path.is_file() or status_path.read_text().strip() != "PASS":
        fail(f"development status is not PASS: {status_path}")
    selection_path = root / "development/selection.json"
    selection = read_json(selection_path)
    require_pass(selection, selection_path, 3)
    require_equal(selection.get("method"), METHOD, "development method")
    require_equal(selection.get("certification_consumed"), False, "development certification flag")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        fail("development selection has no selected row")
    epoch = selected.get("epoch")
    if epoch not in (1, 2, 4, 8, 16):
        fail(f"unsupported selected epoch: {epoch!r}")

    trial_rows = {}
    for seed in (0, 1, 2):
        report_path = root / f"development/trials/optimizer_seed{seed}/report.json"
        report = read_json(report_path)
        require_pass(report, report_path, 3)
        require_equal(report.get("method"), METHOD, f"optimizer seed{seed} method")
        require_equal(report.get("ablation"), "full", f"optimizer seed{seed} ablation")
        require_equal(report.get("train_seed"), seed, f"optimizer seed{seed} train seed")
        require_equal(report.get("temperature"), selected.get("temperature"), f"optimizer seed{seed} temperature")
        require_equal(report.get("learning_rate"), selected.get("learning_rate"), f"optimizer seed{seed} learning rate")
        require_equal(report.get("kl_weight"), selected.get("kl_weight"), f"optimizer seed{seed} KL weight")
        checkpoint = checkpoint_for_epoch(report, epoch, f"optimizer seed{seed}")
        state_path = resolve_recorded_path(
            checkpoint.get("scene_selector_state"), f"optimizer seed{seed} selector"
        )
        require_file_sha(
            state_path,
            checkpoint.get("scene_selector_state_sha256"),
            f"optimizer seed{seed} selector",
        )
        trial_rows[seed] = {
            "report_path": report_path.resolve(),
            "state_path": state_path,
            "state_sha256": checkpoint.get("scene_selector_state_sha256"),
        }

    selected_seed = selected.get("train_seed")
    if selected_seed not in trial_rows:
        fail(f"invalid selected train seed: {selected_seed!r}")
    selected_trial = trial_rows[selected_seed]
    require_equal(
        resolve_recorded_path(selected.get("report"), "selected report"),
        selected_trial["report_path"],
        "selected report",
    )
    require_equal(
        resolve_recorded_path(selected.get("scene_selector_state"), "selected state"),
        selected_trial["state_path"],
        "selected state",
    )
    require_equal(
        selected.get("scene_selector_state_sha256"),
        selected_trial["state_sha256"],
        "selected state SHA256",
    )
    return {"selection": selection, "selection_path": selection_path, "trials": trial_rows}


def validate_certification(root: Path) -> dict:
    development = validate_development(root)
    cert_root = root / "certification"
    status_path = cert_root / "status.txt"
    if not status_path.is_file() or status_path.read_text().strip() != "PASS":
        fail(f"certification status is not PASS: {status_path}")
    report_path = cert_root / "report.json"
    report = read_json(report_path)
    require_pass(report, report_path, 3)
    require_equal(report.get("method"), METHOD, "certification method")
    require_equal(report.get("certification_consumed"), True, "certification consumption")
    require_equal(report.get("certification_seeds"), [6, 7, 8], "certification seeds")
    selected = development["selection"]["selected"]
    require_equal(
        report.get("selection_sha256"),
        sha256_file(development["selection_path"]),
        "certification selection SHA256",
    )
    require_equal(
        resolve_recorded_path(report.get("scene_selector_state"), "certification selector"),
        resolve_recorded_path(selected.get("scene_selector_state"), "selected selector"),
        "certification selected state",
    )
    require_equal(
        report.get("scene_selector_state_sha256"),
        selected.get("scene_selector_state_sha256"),
        "certification selected state SHA256",
    )
    gate = report.get("scientific_gate")
    if not isinstance(gate, dict) or not all(
        isinstance(gate.get(key), bool)
        for key in ("mean_top1_gain_positive", "lower_95_top1_gain_positive")
    ):
        fail("certification scientific_gate is missing or malformed")

    manifest_path = cert_root / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    require_pass(manifest, manifest_path, 3)
    require_equal(manifest.get("method"), RELEASE, "certification release name")
    require_equal(
        manifest.get("scene_selector_state_sha256"),
        selected.get("scene_selector_state_sha256"),
        "certification manifest selector SHA256",
    )
    checkpoint = resolve_recorded_path(manifest.get("checkpoint"), "certification checkpoint")
    require_file_sha(checkpoint, manifest.get("checkpoint_sha256"), "certification checkpoint")
    return {"development": development, "report": report, "manifest": manifest}


def validate_formal(root: Path, seed: int) -> None:
    certification = validate_certification(root)
    selected = certification["development"]["selection"]["selected"]
    trial = certification["development"]["trials"][seed]
    replica_root = root / f"formal/replicas/seed{seed}"
    manifest_path = replica_root / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    require_pass(manifest, manifest_path, 3)
    require_equal(manifest.get("method"), RELEASE, f"formal seed{seed} release name")
    require_equal(
        resolve_recorded_path(manifest.get("scene_selector_state"), f"formal seed{seed} selector"),
        trial["state_path"],
        f"formal seed{seed} development state",
    )
    require_equal(
        manifest.get("scene_selector_state_sha256"),
        trial["state_sha256"],
        f"formal seed{seed} selector SHA256",
    )
    payload = manifest.get("selector_payload", {})
    require_equal(payload.get("train_seed"), seed, f"formal seed{seed} train seed")
    require_equal(payload.get("epoch"), selected.get("epoch"), f"formal seed{seed} epoch")
    checkpoint = resolve_recorded_path(manifest.get("checkpoint"), f"formal seed{seed} checkpoint")
    require_file_sha(checkpoint, manifest.get("checkpoint_sha256"), f"formal seed{seed} checkpoint")

    audit_path = replica_root / "checkpoint_audit.json"
    audit = read_json(audit_path)
    require_pass(audit, audit_path, 3)
    require_equal(audit.get("checkpoint_sha256"), manifest.get("checkpoint_sha256"), f"formal seed{seed} audit SHA256")

    model_name = f"e2e_diffusiondrive_grpo_selector_rollout_v1_s{seed}"
    summary_path = root / f"formal/formal_eval/{model_name}/summary.json"
    summary = read_json(summary_path)
    require_pass(summary, summary_path, 1)
    require_equal(summary.get("model_name"), model_name, f"formal seed{seed} model name")
    require_equal(summary.get("eval_seed"), seed, f"formal seed{seed} eval seed")
    require_equal(summary.get("checkpoint_sha256"), manifest.get("checkpoint_sha256"), f"formal seed{seed} summary SHA256")
    counts = summary.get("counts", {})
    require_equal(counts.get("openloop_navtest"), 12147, f"formal seed{seed} navtest count")
    require_equal(counts.get("openloop_failures"), 289, f"formal seed{seed} failures count")
    require_equal(counts.get("closedloop_nr"), 289, f"formal seed{seed} NR count")
    require_equal(counts.get("closedloop_r"), 289, f"formal seed{seed} R count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worldengine-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("inputs", "development", "certification", "formal", "complete"),
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    worldengine_root = args.worldengine_root.expanduser().resolve()
    if args.stage == "inputs":
        validate_inputs(root, worldengine_root)
    elif args.stage == "development":
        validate_development(root)
    elif args.stage == "certification":
        validate_certification(root)
    elif args.stage == "formal":
        if args.seed is None:
            parser.error("--stage formal requires --seed")
        validate_formal(root, args.seed)
    else:
        validate_inputs(root, worldengine_root)
        for seed in (0, 1, 2):
            validate_formal(root, seed)
    suffix = f" seed={args.seed}" if args.seed is not None else ""
    print(f"PASS rollout-v1 finish audit stage={args.stage}{suffix}")


if __name__ == "__main__":
    main()
