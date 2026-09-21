#!/usr/bin/env python3
"""Shared fail-closed provenance checks for rollout-v1 records."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


WORKER_PATTERN = re.compile(r"split_([0-9]+)")


def worker_id_from_sidecar(row):
    sidecar_path = row.get("sidecar_path")
    if not sidecar_path:
        raise RuntimeError("rollout record has no sidecar_path provenance")
    matches = [
        part
        for part in Path(str(sidecar_path)).parts
        if WORKER_PATTERN.fullmatch(part)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"rollout sidecar path has ambiguous worker provenance: {sidecar_path}"
        )
    return matches[0]


def validate_rollout_provenance(rows, expected_workers=None):
    """Require one source config/code SHA and one stable resolved SHA per worker."""
    source_config_shas = set()
    code_shas = set()
    resolved_by_worker = defaultdict(set)
    for row in rows:
        source_config_sha = row.get("config_sha256")
        resolved_config_sha = row.get("resolved_config_sha256")
        code_sha = row.get("code_sha")
        if not source_config_sha or not resolved_config_sha or not code_sha:
            raise RuntimeError("rollout record has incomplete config/code provenance")
        worker_id = worker_id_from_sidecar(row)
        source_config_shas.add(str(source_config_sha))
        code_shas.add(str(code_sha))
        resolved_by_worker[worker_id].add(str(resolved_config_sha))

    if len(source_config_shas) != 1:
        raise RuntimeError("source config provenance drifted within rollout")
    if len(code_shas) != 1:
        raise RuntimeError("code provenance drifted within rollout")
    unstable_workers = {
        worker_id: sorted(shas)
        for worker_id, shas in resolved_by_worker.items()
        if len(shas) != 1
    }
    if unstable_workers:
        raise RuntimeError(
            f"resolved config provenance drifted within workers: {unstable_workers}"
        )
    if expected_workers is not None:
        expected = {f"split_{index}" for index in range(expected_workers)}
        actual = set(resolved_by_worker)
        if actual != expected:
            raise RuntimeError(
                "rollout worker provenance drifted: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )

    return {
        "config_sha256": next(iter(source_config_shas)),
        "code_sha": next(iter(code_shas)),
        "resolved_config_sha256_by_worker": {
            worker_id: next(iter(shas))
            for worker_id, shas in sorted(resolved_by_worker.items())
        },
    }
