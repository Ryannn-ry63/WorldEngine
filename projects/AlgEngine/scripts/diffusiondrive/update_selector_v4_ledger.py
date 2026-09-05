#!/usr/bin/env python3
"""Update a run-local copy of the frozen V4 decision ledger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import v4_causal_cache_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--next-stage")
    args = parser.parse_args()

    ledger_path = args.ledger.expanduser().resolve()
    source = ledger_path if ledger_path.exists() else args.template.expanduser().resolve()
    payload = json.loads(source.read_text())
    matching = [row for row in payload["stages"] if row["name"] == args.stage]
    if len(matching) != 1:
        raise RuntimeError(f"unknown or duplicate V4 ledger stage: {args.stage}")
    matching[0]["status"] = args.status
    matching[0]["decision"] = args.decision
    payload["run_id"] = args.run_id
    payload["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.next_stage:
        payload["next_authorized_stage"] = args.next_stage
    if args.artifact is not None:
        artifact = args.artifact.expanduser().resolve()
        payload.setdefault("artifacts", {})[args.stage] = {
            "path": str(artifact),
            "sha256": common.sha256_file(artifact),
        }
    common.atomic_json(ledger_path, payload)
    print(json.dumps({"status": "PASS", "output": str(ledger_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
