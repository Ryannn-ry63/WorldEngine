#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:?usage: $0 collect-lane LANE | collect-wait LANE | recover-collection | finish | finish-seed SEED | summarize | finalize}"
WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"

wait_for_prepare_smoke() {
    local status_file="${DIFFUSIONDRIVE_RARE_ROLLOUT_PREPARE_SMOKE_STATUS:-${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/prepare_smoke_status.txt}"
    local code_sha timeout_seconds poll_seconds started_epoch now_epoch elapsed status_line
    code_sha="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    timeout_seconds="${DIFFUSIONDRIVE_RARE_ROLLOUT_WAIT_TIMEOUT_SECONDS:-43200}"
    poll_seconds="${DIFFUSIONDRIVE_RARE_ROLLOUT_WAIT_POLL_SECONDS:-60}"
    if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
        echo "DIFFUSIONDRIVE_RARE_ROLLOUT_WAIT_TIMEOUT_SECONDS must be positive" >&2
        return 2
    fi
    if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || ((poll_seconds > 60)); then
        echo "DIFFUSIONDRIVE_RARE_ROLLOUT_WAIT_POLL_SECONDS must be in [1, 60]" >&2
        return 2
    fi
    started_epoch="$(date +%s)"
    echo "WAIT rare-rollout prepare+smoke code_sha=${code_sha} timeout=${timeout_seconds}s poll=${poll_seconds}s"
    while true; do
        if [[ -f "${status_file}" ]]; then
            IFS= read -r status_line < "${status_file}" || true
            if [[ "${status_line}" == PASS\ * && "${status_line}" == *"code_sha=${code_sha}"* ]]; then
                echo "PASS upstream prepare+smoke gate: ${status_line}"
                return 0
            fi
            if [[ "${status_line}" == FAIL\ * && "${status_line}" == *"code_sha=${code_sha}"* ]]; then
                echo "Upstream prepare+smoke failed: ${status_line}" >&2
                return 1
            fi
        fi
        now_epoch="$(date +%s)"
        elapsed=$((now_epoch - started_epoch))
        if ((elapsed >= timeout_seconds)); then
            echo "Timed out after ${elapsed}s waiting for ${status_file}" >&2
            return 1
        fi
        echo "WAIT prepare+smoke not ready elapsed=${elapsed}s"
        sleep "${poll_seconds}"
    done
}

case "${MODE}" in
    collect-lane)
        LANE="${2:?usage: $0 collect-lane LANE}"
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_collect_h100.sh" \
            "${LANE}" -1 formal
        ;;
    collect-wait)
        LANE="${2:?usage: $0 collect-wait LANE}"
        if [[ ! "${LANE}" =~ ^[0-2]$ ]]; then
            echo "LANE must be 0, 1, or 2" >&2
            exit 2
        fi
        wait_for_prepare_smoke
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_collect_h100.sh" \
            "${LANE}" -1 formal
        ;;
    recover-collection)
        if [[ "${#}" -ne 1 ]]; then
            echo "recover-collection takes no additional arguments" >&2
            exit 2
        fi
        exec "${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_recover_local.sh"
        ;;
    finish)
        if [[ "${#}" -ne 1 ]]; then
            echo "finish takes no additional arguments" >&2
            exit 2
        fi
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh"
        ;;
    finish-seed)
        SEED="${2:?usage: $0 finish-seed SEED}"
        if [[ "${#}" -ne 2 || ! "${SEED}" =~ ^[0-2]$ ]]; then
            echo "finish-seed requires exactly one seed in {0,1,2}" >&2
            exit 2
        fi
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh" seed "${SEED}"
        ;;
    summarize)
        if [[ "${#}" -ne 1 ]]; then
            echo "summarize takes no additional arguments" >&2
            exit 2
        fi
        exec "${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_summarize_local.sh"
        ;;
    finalize)
        if [[ "${#}" -ne 1 ]]; then
            echo "finalize takes no additional arguments" >&2
            exit 2
        fi
        # Fail closed: training/evaluation starts only after all three formal
        # collection audits pass and the filtered 50/50 mixture is materialized.
        "${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_prepare_local.sh"
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh"
        ;;
    *)
        echo "usage: $0 collect-lane LANE | collect-wait LANE | recover-collection | finish | finish-seed SEED | summarize | finalize" >&2
        exit 2
        ;;
esac
