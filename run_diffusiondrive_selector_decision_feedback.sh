#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
DECISION_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECISION_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${DECISION_PYTHON}" "${DECISION_WORKTREE}/projects/AlgEngine/scripts/diffusiondrive/run_selector_decision.py" "$@"
