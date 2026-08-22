#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:?usage: $0 collect-lane LANE | collect-all | finish-seed SEED | finish-all}"
WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT
export DIFFUSIONDRIVE_WORLDENGINE_ROOT_OVERRIDE="${WORLDENGINE_ROOT}"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"

case "${MODE}" in
    collect-lane)
        LANE="${2:?usage: $0 collect-lane LANE}"
        [[ "${#}" -eq 2 && "${LANE}" =~ ^[0-2]$ ]] || {
            echo "lane must be one of 0, 1, 2" >&2
            exit 2
        }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_bwm_collect_h100.sh" \
            "${LANE}" -1 formal
        ;;
    collect-all)
        [[ "${#}" -eq 1 ]] || { echo "collect-all takes no arguments" >&2; exit 2; }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=8
        for lane in 0 1 2; do
            "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_bwm_collect_h100.sh" \
                "${lane}" -1 formal
        done
        ;;
    finish-seed)
        SEED="${2:?usage: $0 finish-seed SEED}"
        [[ "${#}" -eq 2 && "${SEED}" =~ ^[0-2]$ ]] || {
            echo "seed must be one of 0, 1, 2" >&2
            exit 2
        }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_bwm_finish_h100.sh" \
            seed "${SEED}"
        ;;
    finish-all)
        [[ "${#}" -eq 1 ]] || { echo "finish-all takes no arguments" >&2; exit 2; }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_bwm_finish_h100.sh"
        ;;
    *)
        echo "usage: $0 collect-lane LANE | collect-all | finish-seed SEED | finish-all" >&2
        exit 2
        ;;
esac
