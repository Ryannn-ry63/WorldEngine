#!/usr/bin/env bash
set -Eeo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <bundle: 0|1|2>" >&2
    exit 2
fi
BUNDLE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
TRAIN_FILTER="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v2/navtrain_grpo_train.yaml"
V3_SPLIT_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v3"
OUTPUT_ROOT="${V3_ROOT}/cache"
STATUS_FILE="${OUTPUT_ROOT}/bundle${BUNDLE}.status"
LOG_DIR="${OUTPUT_ROOT}/logs"
IMPLEMENTATION_FILES=(
    "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
    "${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py"
    "${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    "${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
    "${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    "${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
)
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/bundle${BUNDLE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

for file in "${CONFIG}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" "${TRAIN_FILTER}" \
    "${V3_SPLIT_ROOT}/navtrain_grpo_development.yaml" \
    "${V3_SPLIT_ROOT}/navtrain_grpo_certification.yaml" \
    "${IMPLEMENTATION_FILES[@]}"; do
    [[ -f "${file}" ]] || { echo "Missing required V3 file: ${file}" >&2; exit 1; }
done
[[ "$(sha256sum "${DIFFUSIONDRIVE_GRPO_BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2; exit 1;
}
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/preflight_grpo_online_selector.py" --config "${CONFIG}"
"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible

run_extract() {
    local split="$1"
    local seed="$2"
    local count="$3"
    local filter="$4"
    local output="${OUTPUT_ROOT}/${split}_seed${seed}"
    CURRENT_STAGE="cache_${split}_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        local expected_sha actual_sha config_sha filter_sha
        expected_sha="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_sha256"])' "${output}/manifest.json")"
        actual_sha="$(sha256sum "${output}/cache.pt" | awk '{print $1}')"
        config_sha="$(sha256sum "${CONFIG}" | awk '{print $1}')"
        filter_sha="$(sha256sum "${filter}" | awk '{print $1}')"
        if [[ "${expected_sha}" == "${actual_sha}" ]] && "${ALGENGINE_PYTHON}" -c '
import hashlib, json, sys
digest = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
m = json.load(open(sys.argv[1]))
expected = dict(schema_version=2, status="PASS", split=sys.argv[2],
                noise_seed=int(sys.argv[3]), num_tokens=int(sys.argv[4]),
                checkpoint_sha256=sys.argv[5], config_sha256=sys.argv[6],
                nav_filter_sha256=sys.argv[7])
bad = {key: (m.get(key), value) for key, value in expected.items()
       if m.get(key) != value}
recorded = m.get("implementation_files", {})
bad_impl = {
    path: (recorded.get(path), digest(path))
    for path in sys.argv[8:]
    if recorded.get(path) != digest(path)
}
if bad or bad_impl:
    raise SystemExit(
        "V3 cache manifest identity mismatch: "
        + repr({"metadata": bad, "implementation_files": bad_impl})
    )
' "${output}/manifest.json" "${split}" "${seed}" "${count}" \
                "${BASELINE_SHA256}" "${config_sha}" "${filter_sha}" \
                "${IMPLEMENTATION_FILES[@]}"; then
            echo "REUSE verified V3 cache identity: ${output}/cache.pt"
            return
        fi
        echo "REBUILD V3 cache because content or provenance drifted: ${output}"
    fi
    mkdir -p "${output}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}" --nproc_per_node=8 --master_port="$((29600 + BUNDLE * 10 + seed))" \
        "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
        "${CONFIG}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
        --expected-checkpoint-sha256 "${BASELINE_SHA256}" \
        --nav-filter "${filter}" --split "${split}" --noise-seed "${seed}" \
        --expected-num-tokens "${count}" --output-dir "${output}" --launcher pytorch
}

run_extract train "${BUNDLE}" 6339 "${TRAIN_FILTER}"
run_extract development "$((BUNDLE + 3))" 671 "${V3_SPLIT_ROOT}/navtrain_grpo_development.yaml"
run_extract certification "$((BUNDLE + 6))" 447 "${V3_SPLIT_ROOT}/navtrain_grpo_certification.yaml"

CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 context cache bundle ${BUNDLE}"
echo "persistent_log: ${LOG_FILE}"
