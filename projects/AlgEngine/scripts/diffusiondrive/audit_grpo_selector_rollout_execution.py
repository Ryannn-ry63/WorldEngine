#!/usr/bin/env python3
"""Fail before merge when WorldEngine reports a failed rollout scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reports(root):
    paths = sorted(root.rglob("runner_report_*.json"))
    if not paths:
        raise RuntimeError(f"no WorldEngine runner report found under {root}")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise RuntimeError(f"invalid WorldEngine runner report: {path}")
        rows.extend(payload)
    return paths, rows


def validate_reports(rows, minimum_scenarios=1, maximum_scenarios=None):
    if not rows:
        raise RuntimeError("WorldEngine runner reports contain no scenarios")
    if minimum_scenarios < 1:
        raise ValueError("minimum scenarios must be positive")
    if maximum_scenarios is not None and maximum_scenarios < minimum_scenarios:
        raise ValueError("maximum scenarios must be >= minimum scenarios")
    if len(rows) < minimum_scenarios:
        raise RuntimeError(
            f"WorldEngine reported {len(rows)} scenarios; "
            f"minimum is {minimum_scenarios}"
        )
    if maximum_scenarios is not None and len(rows) > maximum_scenarios:
        raise RuntimeError(
            f"WorldEngine reported {len(rows)} scenarios; "
            f"maximum is {maximum_scenarios}"
        )
    failures = [row for row in rows if row.get("succeeded") is not True]
    if failures:
        first = failures[0]
        raise RuntimeError(
            "WorldEngine rollout failed before merge: "
            f"scenario={first.get('scenario_name')} "
            f"worker={first.get('log_name')}\n"
            f"{first.get('error_message')}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--minimum-scenarios", type=int, default=1)
    parser.add_argument("--maximum-scenarios", type=int)
    args = parser.parse_args()

    root = args.rollout_root.expanduser().resolve()
    paths, rows = load_reports(root)
    validate_reports(rows, args.minimum_scenarios, args.maximum_scenarios)
    print(
        "PASS WorldEngine rollout execution: "
        f"scenarios={len(rows)} reports={len(paths)}"
    )


if __name__ == "__main__":
    main()
