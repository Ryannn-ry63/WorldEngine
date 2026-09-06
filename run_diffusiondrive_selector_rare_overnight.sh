#!/usr/bin/env bash
# External composition only: preserve the already-frozen scientific implementation.
set -Eeuo pipefail
ulimit -c 0
RARE_AUTO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RARE_AUTO_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${RARE_AUTO_PYTHON}" "${RARE_AUTO_ROOT}/selector_rare_overnight.py" "$@"
