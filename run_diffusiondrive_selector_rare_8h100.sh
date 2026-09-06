#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
RARE_RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RARE_RUN_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${RARE_RUN_PYTHON}" "${RARE_RUN_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_selector_rare.py" "$@"
