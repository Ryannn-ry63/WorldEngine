#!/usr/bin/env bash
set -eo pipefail
SNAPSHOT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOT_ENV="$DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE"
if [ -z "$SNAPSHOT_ENV" ]; then
    SNAPSHOT_ENV=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine
fi
set -u
exec "$SNAPSHOT_ENV/bin/python" "$SNAPSHOT_ROOT/projects/AlgEngine/scripts/diffusiondrive/run_selector_snapshot.py" "$@"
