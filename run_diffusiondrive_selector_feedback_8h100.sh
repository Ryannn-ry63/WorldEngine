#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
FEEDBACK_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEEDBACK_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${FEEDBACK_PYTHON}" "${FEEDBACK_WORKTREE}/projects/AlgEngine/scripts/diffusiondrive/run_selector_feedback.py" "$@"
