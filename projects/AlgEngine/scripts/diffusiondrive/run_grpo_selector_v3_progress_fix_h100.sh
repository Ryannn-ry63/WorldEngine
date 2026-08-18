#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:-all}"
case "${MODE}" in
    preflight|cache|tune|eval|all) ;;
    *)
        echo "Usage: $0 [preflight|cache|tune|eval|all]" >&2
        exit 2
        ;;
esac

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
export DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME=grpo_selector_v3_progress_fix_v1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"

if ! git -C "${WORLDENGINE_ROOT}" diff --quiet HEAD -- \
    projects/AlgEngine projects/SimEngine \
    run_diffusiondrive_grpo_selector_v3_progress_fix_1h100_smoke.sh \
    run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh; then
    echo "Tracked progress-fix inputs differ from HEAD; commit them before a formal run" >&2
    exit 1
fi

INTEGRATED_ROOT="${V3_ROOT}/integrated"
STATUS_FILE="${INTEGRATED_ROOT}/status.txt"
LOG_DIR="${INTEGRATED_ROOT}/logs"
RECEIPT_DIR="${INTEGRATED_ROOT}/receipts"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_DIR}" "${RECEIPT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}_${MODE}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
printf 'RUNNING mode=%s run_id=%s code_sha=%s started_utc=%s\n' \
    "${MODE}" "${RUN_ID}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUS_FILE}"
trap 'rc=$?; printf "FAIL mode=%s run_id=%s code_sha=%s stage=%s exit=%s\n" "${MODE}" "${RUN_ID}" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL V3 progress-fix stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

json_pass() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("status") == "PASS" else 1)
' "${path}"
}

receipt_pass() {
    local path="$1"
    local stage="$2"
    [[ -f "${path}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import json, sys
payload = json.load(open(sys.argv[1]))
ok = (
    payload.get("status") == "PASS"
    and payload.get("stage") == sys.argv[2]
    and payload.get("code_sha") == sys.argv[3]
)
raise SystemExit(0 if ok else 1)
' "${path}" "${stage}" "${CODE_SHA}"
}

write_receipt() {
    local path="$1"
    local stage="$2"
    "${ALGENGINE_PYTHON}" -c '
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "PASS",
    "stage": sys.argv[2],
    "code_sha": sys.argv[3],
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
' "${path}" "${stage}" "${CODE_SHA}"
}

run_preflight() {
    CURRENT_STAGE=unit_regression
    "${ALGENGINE_PYTHON}" -m pytest -q \
        "${ALGENGINE_ROOT}/scripts/tests/test_navsim_online_pdm_sampling.py"

    CURRENT_STAGE=mmcv_cuda_preflight
    "${ALGENGINE_PYTHON}" \
        "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
        --extension "${WORLDENGINE_MMCV_EXTENSION}" \
        --expected-capability sm_90 --all-visible

    local parity_root="${V3_ROOT}/preflight"
    local parity_report="${parity_root}/parity_report.json"
    local validator="${SCRIPT_DIR}/validate_grpo_selector_pdm_progress_fix.py"
    local reward_file="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    local nav_filter="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v2/navtrain_grpo_train.yaml"
    local parity_tokens="${V3_PROGRESS_FIX_SMOKE_TOKENS:-16}"
    local reward_sha validator_sha config_sha nav_filter_sha
    if ! [[ "${parity_tokens}" =~ ^[1-9][0-9]*$ ]]; then
        echo "V3_PROGRESS_FIX_SMOKE_TOKENS must be a positive integer" >&2
        return 2
    fi
    reward_sha="$(sha256sum "${reward_file}" | awk '{print $1}')"
    validator_sha="$(sha256sum "${validator}" | awk '{print $1}')"
    config_sha="$(sha256sum "${DIFFUSIONDRIVE_GRPO_CONFIG}" | awk '{print $1}')"
    nav_filter_sha="$(sha256sum "${nav_filter}" | awk '{print $1}')"
    mkdir -p "${parity_root}"
    if [[ -f "${parity_report}" ]] && "${ALGENGINE_PYTHON}" -c '
import json, sys
payload = json.load(open(sys.argv[1]))
ok = (
    payload.get("status") == "PASS"
    and payload.get("reward_implementation_sha256") == sys.argv[2]
    and payload.get("validator_sha256") == sys.argv[3]
    and payload.get("config_sha256") == sys.argv[4]
    and payload.get("nav_filter_sha256") == sys.argv[5]
    and payload.get("num_tokens") == int(sys.argv[6])
    and payload.get("noise_seed") == 20260818
    and payload.get("atol") == 1e-5
)
raise SystemExit(0 if ok else 1)
' "${parity_report}" "${reward_sha}" "${validator_sha}" "${config_sha}" \
            "${nav_filter_sha}" "${parity_tokens}"; then
        echo "REUSE verified CPU/CUDA/official parity report: ${parity_report}"
    else
        CURRENT_STAGE=real_candidate_cpu_cuda_parity
        cd "${ALGENGINE_ROOT}"
        CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
            "${validator}" "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
            --nav-filter "${nav_filter}" \
            --num-tokens "${parity_tokens}" \
            --noise-seed 20260818 --atol 1e-5 \
            --output "${parity_report}"
    fi
}

run_cache() {
    local bundle
    for bundle in 0 1 2; do
        CURRENT_STAGE="cache_bundle${bundle}"
        "${SCRIPT_DIR}/run_grpo_selector_v3_cache_bundle_h100.sh" "${bundle}"
    done
}

run_tune() {
    local receipt

    CURRENT_STAGE=h100_optimizer_preflight
    "${SCRIPT_DIR}/run_grpo_selector_v3_optimizer_preflight_h100.sh"

    receipt="${RECEIPT_DIR}/representation_ablation.json"
    if receipt_pass "${receipt}" representation_ablation \
        && [[ -f "${V3_ROOT}/ablation/status.txt" ]] \
        && grep -q '^PASS ' "${V3_ROOT}/ablation/status.txt" \
        && json_pass "${V3_ROOT}/ablation/report.json"; then
        echo "REUSE verified V3 progress-fix ablation"
    else
        CURRENT_STAGE=representation_ablation
        "${SCRIPT_DIR}/run_grpo_selector_v3_ablation_h100.sh"
        write_receipt "${receipt}" representation_ablation
    fi

    receipt="${RECEIPT_DIR}/development_sweep.json"
    if receipt_pass "${receipt}" development_sweep \
        && [[ -f "${V3_ROOT}/sweep/status.txt" ]] \
        && grep -q '^PASS ' "${V3_ROOT}/sweep/status.txt" \
        && json_pass "${V3_ROOT}/sweep/selection.json"; then
        echo "REUSE verified V3 progress-fix sweep"
    else
        CURRENT_STAGE=development_sweep
        "${SCRIPT_DIR}/run_grpo_selector_v3_sweep_h100.sh"
        write_receipt "${receipt}" development_sweep
    fi

    receipt="${RECEIPT_DIR}/certification.json"
    if receipt_pass "${receipt}" certification \
        && [[ -f "${V3_ROOT}/certification/status.txt" ]] \
        && grep -q '^PASS ' "${V3_ROOT}/certification/status.txt" \
        && json_pass "${V3_ROOT}/certification/report.json" \
        && [[ -f "${V3_ROOT}/certification/selected_checkpoint.pth" ]] \
        && [[ -f "${V3_ROOT}/certification/checkpoint_manifest.json" ]]; then
        echo "REUSE verified V3 progress-fix certification"
    else
        CURRENT_STAGE=certification
        "${SCRIPT_DIR}/run_grpo_selector_v3_certify_h100.sh"
        write_receipt "${receipt}" certification
    fi
}

run_eval() {
    local seed model summary receipt
    for seed in 0 1 2; do
        model="e2e_diffusiondrive_${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME}_s${seed}"
        summary="${V3_ROOT}/formal/formal_eval/${model}/summary.json"
        receipt="${RECEIPT_DIR}/formal_seed${seed}.json"
        if receipt_pass "${receipt}" "formal_seed${seed}" \
            && json_pass "${summary}" \
            && "${ALGENGINE_PYTHON}" -c '
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("eval_seed") == int(sys.argv[2]) else 1)
' "${summary}" "${seed}"; then
            echo "REUSE verified formal seed ${seed}: ${summary}"
        else
            CURRENT_STAGE="formal_seed${seed}"
            "${SCRIPT_DIR}/run_grpo_selector_v3_formal_seed_h100.sh" "${seed}"
            write_receipt "${receipt}" "formal_seed${seed}"
        fi
    done

    CURRENT_STAGE=formal_comparison
    "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/summarize_grpo_selector_v3_progress_fix.py" \
        --fixed-root "${V3_ROOT}/formal" \
        --old-v3-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3/formal" \
        --reference-root "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal" \
        --fixed-prefix "e2e_diffusiondrive_${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME}" \
        --output-json "${V3_ROOT}/formal/progress_fix_comparison.json" \
        --output-md "${V3_ROOT}/formal/progress_fix_comparison.md"
}

run_preflight
case "${MODE}" in
    preflight) ;;
    cache) run_cache ;;
    tune) run_tune ;;
    eval) run_eval ;;
    all)
        run_cache
        run_tune
        run_eval
        ;;
esac

CURRENT_STAGE=complete
printf 'PASS mode=%s run_id=%s code_sha=%s completed_utc=%s\n' \
    "${MODE}" "${RUN_ID}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUS_FILE}"
trap - ERR
echo "PASS V3 progress normalization pipeline mode=${MODE}"
echo "output: ${V3_ROOT}"
echo "persistent_log: ${LOG_FILE}"
