#!/usr/bin/env bash
# Independent continuous-deployment A/B endpoint. Old pilot runs remain immutable.
set -Eeuo pipefail
ulimit -c 0
CFPI_DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFPI_DEPLOY_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
exec "${CFPI_DEPLOY_PYTHON}" "${CFPI_DEPLOY_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_selector_cfpi_deployment.py" "$@"
