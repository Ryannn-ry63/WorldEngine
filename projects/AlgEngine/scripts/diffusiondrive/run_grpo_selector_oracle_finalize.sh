#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
ALGENGINE_PYTHON="${ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
FORMAL_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
ORACLE_ROOT="${FORMAL_ROOT}/oracle_audit"
FAILURES_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml"
OUTPUT="${ORACLE_ROOT}/final/oracle_audit_three_seed.json"
WAIT_MODE=0
case "${1:-}" in
    --wait)
        WAIT_MODE=1
        shift
        ;;
    "")
        ;;
    *)
        echo "usage: run_grpo_selector_oracle_finalize.sh [--wait]" >&2
        exit 2
        ;;
esac
if [[ "$#" -ne 0 ]]; then
    echo "usage: run_grpo_selector_oracle_finalize.sh [--wait]" >&2
    exit 2
fi

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
if ! [[ "${WAIT_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WAIT_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${WAIT_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WAIT_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi

seed_ready() {
    local seed="$1"
    local seed_dir="${ORACLE_ROOT}/seed${seed}"
    local status_file="${seed_dir}/status.txt"
    local status_line selection result_csv row_count
    if [[ ! -f "${status_file}" ]]; then
        return 1
    fi
    status_line="$(head -n 1 "${status_file}")"
    case "${status_line}" in
        FAIL*)
            echo "Seed ${seed} reported failure: ${status_line}" >&2
            return 2
            ;;
        PASS*)
            ;;
        *)
            return 1
            ;;
    esac
    for path in "${seed_dir}/audit.json" "${seed_dir}/records.jsonl.gz"; do
        if [[ ! -f "${path}" ]]; then
            return 1
        fi
    done
    for selection in reference current oracle; do
        result_csv="${seed_dir}/${selection}_official_pdms/pdm_scores_merged.csv"
        if [[ ! -f "${result_csv}" ]]; then
            return 1
        fi
        # pdm_scores_merged.csv ends with an aggregate "average" row.
        row_count="$(awk -F, 'NR > 1 && $1 != "average" { count += 1 } END { print count + 0 }' "${result_csv}")"
        if [[ "${row_count}" -ne 12146 ]]; then
            return 1
        fi
    done
    return 0
}

if [[ "${WAIT_MODE}" -eq 1 ]]; then
    wait_started="$(date +%s)"
    while true; do
        ready=0
        for seed in 0 1 2; do
            if seed_ready "${seed}"; then
                ((ready += 1))
            else
                rc=$?
                if [[ "${rc}" -eq 2 ]]; then
                    exit 1
                fi
            fi
        done
        if [[ "${ready}" -eq 3 ]]; then
            echo "All three Oracle audit seeds are complete; finalizing."
            break
        fi
        now="$(date +%s)"
        elapsed="$((now - wait_started))"
        if [[ "${elapsed}" -ge "${WAIT_TIMEOUT_SECONDS}" ]]; then
            echo "Timed out after ${elapsed}s waiting for Oracle audit seeds" >&2
            exit 124
        fi
        echo "WAIT Oracle audit seeds: ready=${ready}/3 elapsed=${elapsed}s"
        sleep "${WAIT_POLL_SECONDS}"
    done
fi

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/finalize_grpo_selector_oracle.py" \
    --seed-dir "${ORACLE_ROOT}/seed0" \
    --seed-dir "${ORACLE_ROOT}/seed1" \
    --seed-dir "${ORACLE_ROOT}/seed2" \
    --failures-filter "${FAILURES_FILTER}" \
    --formal-reference-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s0/summary.json" \
    --formal-reference-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s1/summary.json" \
    --formal-reference-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s2/summary.json" \
    --formal-current-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_s0/summary.json" \
    --formal-current-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_s1/summary.json" \
    --formal-current-summary "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_s2/summary.json" \
    --output "${OUTPUT}"

echo "PASS finalized DiffusionDrive three-seed oracle audit"
echo "json: ${OUTPUT}"
echo "markdown: ${OUTPUT%.json}.md"
