#!/usr/bin/env bash
# Default first phase ends at pilot report, never expands or consumes dev/test.
set -Eeuo pipefail
ulimit -c 0
CFPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFPI_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${CFPI_PYTHON}" "${CFPI_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_selector_cfpi.py" "$@"
