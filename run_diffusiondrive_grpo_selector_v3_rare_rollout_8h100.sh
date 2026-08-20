#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:?usage: $0 collect-lane LANE | finish}"
WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"

case "${MODE}" in
    collect-lane)
        LANE="${2:?usage: $0 collect-lane LANE}"
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_collect_h100.sh" \
            "${LANE}" -1 formal
        ;;
    finish)
        if [[ "${#}" -ne 1 ]]; then
            echo "finish takes no additional arguments" >&2
            exit 2
        fi
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh"
        ;;
    *)
        echo "usage: $0 collect-lane LANE | finish" >&2
        exit 2
        ;;
esac
