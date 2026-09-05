#!/usr/bin/env python3
"""Apply the frozen method-independent information gate to the V4 pilot64."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import v4_causal_cache_common as common


def verify(args) -> dict:
    cache_path = args.cache.expanduser().resolve()
    audit_path = args.cache_audit.expanduser().resolve()
    payload = common.load_pickle(cache_path)
    audit = json.loads(audit_path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != common.CACHE_METHOD
        or payload.get("stage") != "pilot64"
        or int(payload.get("num_rows", -1)) != common.PILOT_TARGETS
        or audit.get("status") != "PASS"
        or audit.get("stage") != "pilot64"
        or audit.get("cache_file_sha256") != common.sha256_file(cache_path)
    ):
        raise RuntimeError("invalid V4 pilot64 causal cache")
    rows = payload["rows"]
    if len({row["scene_id"] for row in rows}) != common.PILOT_TARGETS:
        raise RuntimeError("V4 pilot64 scene identities are not unique")

    spans = []
    gaps = []
    for row in rows:
        q = common.checked_array(row["causal_q_v3"], (20,), "pilot causal Q")
        policy = int(row["policy_index"])
        if policy not in range(common.NUM_CANDIDATES):
            raise RuntimeError("invalid V4 pilot policy index")
        spans.append(float(np.max(q) - np.min(q)))
        gaps.append(float(np.max(q) - q[policy]))
    span_count = int(np.sum(np.asarray(spans) > common.Q_SPAN_THRESHOLD))
    gap_count = int(np.sum(np.asarray(gaps) > common.Q_POLICY_GAP_THRESHOLD))
    complete = len(rows) == common.PILOT_TARGETS
    informative = bool(complete and span_count >= 32 and gap_count >= 16)
    decision = (
        "AUTHORIZE_V4_EXPAND192_AND_SEALED_DEV64"
        if informative
        else "STOP_V4_CAUSAL_CACHE"
    )
    return {
        "schema_version": common.SCHEMA_VERSION,
        "design_version": common.DESIGN_VERSION,
        "status": "PASS" if informative else "FAIL",
        "method": common.PILOT_GATE_METHOD,
        "decision": decision,
        "complete_target_surfaces": len(rows),
        "required_complete_target_surfaces": common.PILOT_TARGETS,
        "q_span_strictly_above_0p02": span_count,
        "required_q_span_count": 32,
        "q_oracle_minus_policy_strictly_above_0p02": gap_count,
        "required_q_oracle_minus_policy_count": 16,
        "q_span_median": float(np.median(spans)),
        "q_oracle_minus_policy_median": float(np.median(gaps)),
        "outcome_strata": dict(sorted(Counter(
            row["behavior_outcome_stratum"] for row in rows
        ).items())),
        "source_families": dict(sorted(Counter(
            row["scenario_family"] for row in rows
        ).items())),
        "cache_file": str(cache_path),
        "cache_file_sha256": common.sha256_file(cache_path),
        "cache_audit": str(audit_path),
        "cache_audit_sha256": common.sha256_file(audit_path),
        "method_training_performed": False,
        "development_consumed": False,
        "test_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args)
    common.atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(report["decision"])


if __name__ == "__main__":
    main()
