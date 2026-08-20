#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:-preflight}"
FORMAL_SEED="${2:-}"
case "${MODE}" in
    preflight|prepare-seed0|formal|formal-wait|overnight|summarize) ;;
    *)
        echo "Usage: $0 [preflight|prepare-seed0|formal SEED|formal-wait SEED|overnight|summarize]" >&2
        exit 2
        ;;
esac
if [[ "${MODE}" =~ ^formal(-wait)?$ && ! "${FORMAL_SEED}" =~ ^[012]$ ]]; then
    echo "${MODE} mode requires seed 0, 1, or 2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
V2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_grpo_selector_v2_progress_fix.py"

if [[ "${MODE}" == summarize ]]; then
    ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}/bin/python"
    [[ -x "${ALGENGINE_PYTHON}" ]] || { echo "Missing AlgEngine Python: ${ALGENGINE_PYTHON}" >&2; exit 1; }
    "${ALGENGINE_PYTHON}" "${SUMMARY_SCRIPT}" \
        --fixed-root "${V2_ROOT}" \
        --old-v2-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2" \
        --reference-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal" \
        --output-json "${V2_ROOT}/formal/progress_fix_comparison.json" \
        --output-md "${V2_ROOT}/formal/progress_fix_comparison.md"
    echo "PASS corrected V2 formal summary: ${V2_ROOT}/formal/progress_fix_comparison.json"
    exit 0
fi

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
export DIFFUSIONDRIVE_GRPO_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

TRACKED_INPUTS=(
    projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py
    projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py
    projects/AlgEngine/scripts/diffusiondrive
    projects/AlgEngine/scripts/tests/test_navsim_online_pdm_sampling.py
    projects/AlgEngine/tests/test_grpo_selector_v2_progress_fix.py
    projects/SimEngine/scripts/merge_simulation_results.py
    run_diffusiondrive_grpo_selector_v2_progress_fix_1h100_smoke.sh
    run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh
)
if ! git -C "${WORLDENGINE_ROOT}" diff --quiet HEAD -- "${TRACKED_INPUTS[@]}"; then
    echo "Tracked corrected-V2 inputs differ from HEAD; commit them before a formal run" >&2
    exit 1
fi

STATUS_SUFFIX="${MODE}${FORMAL_SEED:+_s${FORMAL_SEED}}"
STATUS_FILE="${V2_ROOT}/status_${STATUS_SUFFIX}.txt"
LOG_DIR="${V2_ROOT}/integrated_logs"
RECEIPT_DIR="${V2_ROOT}/receipts"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_DIR}" "${RECEIPT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}_${MODE}${FORMAL_SEED:+_s${FORMAL_SEED}}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
printf 'RUNNING mode=%s seed=%s run_id=%s code_sha=%s started_utc=%s\n' \
    "${MODE}" "${FORMAL_SEED:-none}" "${RUN_ID}" "${CODE_SHA}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap 'rc=$?; printf "FAIL mode=%s seed=%s code_sha=%s stage=%s exit=%s\n" "${MODE}" "${FORMAL_SEED:-none}" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL corrected V2 stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

write_receipt() {
    local stage="$1"
    local path="${RECEIPT_DIR}/${stage}.json"
    "${ALGENGINE_PYTHON}" -c '
import datetime, json, pathlib, sys
p=pathlib.Path(sys.argv[1]); payload={"schema_version":1,"status":"PASS","stage":sys.argv[2],"code_sha":sys.argv[3],"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
' "${path}" "${stage}" "${CODE_SHA}"
}

run_preflight() {
    CURRENT_STAGE=unit_regression
    "${ALGENGINE_PYTHON}" -m pytest -q \
        "${ALGENGINE_ROOT}/scripts/tests/test_navsim_online_pdm_sampling.py" \
        "${ALGENGINE_ROOT}/tests/test_grpo_selector_v2_progress_fix.py" \
        "${ALGENGINE_ROOT}/tests/test_diffusion_grpo_selector_v2_sweep.py" \
        "${ALGENGINE_ROOT}/tests/test_diffusion_grpo_selector_v2_formal.py"

    CURRENT_STAGE=mmcv_cuda_preflight
    "${ALGENGINE_PYTHON}" \
        "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
        --extension "${WORLDENGINE_MMCV_EXTENSION}" \
        --expected-capability sm_90 --all-visible

    local parity_root="${V2_ROOT}/preflight"
    local parity_report="${parity_root}/parity_report.json"
    local validator="${SCRIPT_DIR}/validate_grpo_selector_pdm_progress_fix.py"
    local reward_file="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    local nav_filter="${DIFFUSIONDRIVE_GRPO_NAV_FILTER_PATH}"
    local parity_tokens="${V2_PROGRESS_FIX_PARITY_TOKENS:-16}"
    local reward_sha validator_sha config_sha nav_filter_sha
    reward_sha="$(sha256sum "${reward_file}" | awk '{print $1}')"
    validator_sha="$(sha256sum "${validator}" | awk '{print $1}')"
    config_sha="$(sha256sum "${DIFFUSIONDRIVE_GRPO_CONFIG}" | awk '{print $1}')"
    nav_filter_sha="$(sha256sum "${nav_filter}" | awk '{print $1}')"
    mkdir -p "${parity_root}"
    if [[ -f "${parity_report}" ]] && "${ALGENGINE_PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1])); ok=(p.get("status")=="PASS" and p.get("reward_implementation_sha256")==sys.argv[2] and p.get("validator_sha256")==sys.argv[3] and p.get("config_sha256")==sys.argv[4] and p.get("nav_filter_sha256")==sys.argv[5] and p.get("num_tokens")==int(sys.argv[6]) and p.get("atol")==1e-5)
raise SystemExit(0 if ok else 1)
' "${parity_report}" "${reward_sha}" "${validator_sha}" "${config_sha}" "${nav_filter_sha}" "${parity_tokens}"; then
        echo "REUSE PASS CPU/CUDA/official parity: ${parity_report}"
    else
        CURRENT_STAGE=real_candidate_cpu_cuda_parity
        cd "${ALGENGINE_ROOT}"
        CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${validator}" \
            "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
            --nav-filter "${nav_filter}" --num-tokens "${parity_tokens}" \
            --noise-seed 20260820 --atol 1e-5 --output "${parity_report}"
    fi
    write_receipt preflight
}

run_prepare_seed0() {
    run_preflight
    for cache_name in train_seed0 calibration_seed0 calibration_seed1 calibration_seed2; do
        CURRENT_STAGE="rescore_${cache_name}"
        "${SCRIPT_DIR}/run_grpo_selector_v2_progress_fix_rescore_cache_h100.sh" "${cache_name}"
    done
    write_receipt corrected_cache

    CURRENT_STAGE=corrected_27_grid_sweep
    "${SCRIPT_DIR}/run_grpo_selector_v2_progress_fix_sweep_h100.sh"
    write_receipt corrected_sweep

    CURRENT_STAGE=formal_seed0
    "${SCRIPT_DIR}/run_grpo_selector_v2_progress_fix_formal_seed_h100.sh" 0
    write_receipt formal_seed0
}

run_formal_seed() {
    local seed="$1"
    CURRENT_STAGE="formal_seed${seed}"
    "${SCRIPT_DIR}/run_grpo_selector_v2_progress_fix_formal_seed_h100.sh" "${seed}"
    write_receipt "formal_seed${seed}"
}

wait_for_corrected_selection() {
    local timeout_seconds="${V2_PROGRESS_FIX_WAIT_TIMEOUT_SECONDS:-43200}"
    local poll_seconds="${V2_PROGRESS_FIX_WAIT_POLL_SECONDS:-60}"
    local started_epoch now_epoch elapsed rc upstream_mtime
    local selection="${V2_ROOT}/selection/selection.json"
    local receipt="${RECEIPT_DIR}/corrected_sweep.json"
    local upstream_status="${V2_ROOT}/status_prepare-seed0.txt"
    local reward_file="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    local reward_sha
    if ! [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
        echo "V2_PROGRESS_FIX_WAIT_TIMEOUT_SECONDS must be a positive integer" >&2
        return 2
    fi
    if ! [[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || ((poll_seconds > 60)); then
        echo "V2_PROGRESS_FIX_WAIT_POLL_SECONDS must be in [1, 60]" >&2
        return 2
    fi
    reward_sha="$(sha256sum "${reward_file}" | awk '{print $1}')"
    started_epoch="$(date +%s)"
    CURRENT_STAGE=wait_for_corrected_selection
    echo "WAIT corrected V2 selection timeout=${timeout_seconds}s poll=${poll_seconds}s"
    while true; do
        if "${ALGENGINE_PYTHON}" -c '
import json, pathlib, sys
selection_path, receipt_path = map(pathlib.Path, sys.argv[1:3])
if not selection_path.is_file() or not receipt_path.is_file():
    raise SystemExit(1)
selection = json.loads(selection_path.read_text())
receipt = json.loads(receipt_path.read_text())
# A receipt from an older commit is stale, not a fatal error: a newly queued
# prepare job will replace it when its sweep completes.
if receipt.get("code_sha") != sys.argv[3]:
    raise SystemExit(1)
if receipt.get("status") != "PASS" or receipt.get("stage") != "corrected_sweep":
    raise SystemExit(2)
if selection.get("status") != "PASS":
    raise SystemExit(2)
if selection.get("reward_contract") != sys.argv[4]:
    raise SystemExit(2)
report_path = pathlib.Path(selection.get("selected", {}).get("report", ""))
if not report_path.is_file():
    raise SystemExit(2)
report = json.loads(report_path.read_text())
if report.get("reward_implementation_sha256") != sys.argv[5]:
    raise SystemExit(2)
' "${selection}" "${receipt}" "${CODE_SHA}" \
                navsim_pairwise_raw_progress_then_candidate_gate_v1 "${reward_sha}"; then
            echo "READY corrected V2 selection: ${selection}"
            return 0
        else
            rc=$?
        fi
        if [[ "${rc}" -eq 2 ]]; then
            echo "Selection/receipt exists but failed provenance validation" >&2
            return 1
        fi
        if [[ -f "${upstream_status}" ]]; then
            upstream_mtime="$(stat -c %Y "${upstream_status}")"
            if ((upstream_mtime >= started_epoch)) \
                && grep -q "^FAIL .*code_sha=${CODE_SHA} " "${upstream_status}"; then
                echo "Upstream prepare-seed0 failed: $(<"${upstream_status}")" >&2
                return 1
            fi
        fi
        now_epoch="$(date +%s)"
        elapsed=$((now_epoch - started_epoch))
        if ((elapsed >= timeout_seconds)); then
            echo "Timed out waiting ${elapsed}s for corrected V2 selection" >&2
            return 1
        fi
        echo "WAIT selection not ready elapsed=${elapsed}s"
        sleep "${poll_seconds}"
    done
}

run_summary() {
    CURRENT_STAGE=formal_summary
    "${ALGENGINE_PYTHON}" "${SUMMARY_SCRIPT}" \
        --fixed-root "${V2_ROOT}" \
        --old-v2-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2" \
        --reference-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal" \
        --output-json "${V2_ROOT}/formal/progress_fix_comparison.json" \
        --output-md "${V2_ROOT}/formal/progress_fix_comparison.md"
}

case "${MODE}" in
    preflight) run_preflight ;;
    prepare-seed0) run_prepare_seed0 ;;
    formal) run_formal_seed "${FORMAL_SEED}" ;;
    formal-wait)
        wait_for_corrected_selection
        run_formal_seed "${FORMAL_SEED}"
        ;;
    overnight)
        run_prepare_seed0
        run_formal_seed 1
        run_formal_seed 2
        run_summary
        ;;
esac

CURRENT_STAGE=complete
printf 'PASS mode=%s seed=%s run_id=%s code_sha=%s completed_utc=%s\n' \
    "${MODE}" "${FORMAL_SEED:-none}" "${RUN_ID}" "${CODE_SHA}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS corrected V2 pipeline mode=${MODE} seed=${FORMAL_SEED:-none}"
echo "output: ${V2_ROOT}"
echo "persistent_log: ${LOG_FILE}"
