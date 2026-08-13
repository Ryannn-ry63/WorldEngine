#!/usr/bin/env bash
# Run rollout-v1 with one visible H100 and one worker.

set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE=/root/miniconda3/envs/algengine
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export ALGENGINE_TORCHRUN="${ALGENGINE_ENV}/bin/torchrun"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=1

exec "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_h100.sh" "$@"
