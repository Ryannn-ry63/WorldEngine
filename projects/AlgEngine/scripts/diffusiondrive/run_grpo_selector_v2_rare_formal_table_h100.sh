#!/usr/bin/env bash
set -Eeo pipefail

CHECKPOINT="${1:?usage: $0 CHECKPOINT EXPECTED_SHA MODEL_NAME EVAL_SEED NOTE}"
EXPECTED_SHA256="${2:?usage: $0 CHECKPOINT EXPECTED_SHA MODEL_NAME EVAL_SEED NOTE}"
MODEL_NAME="${3:?usage: $0 CHECKPOINT EXPECTED_SHA MODEL_NAME EVAL_SEED NOTE}"
EVAL_SEED="${4:?usage: $0 CHECKPOINT EXPECTED_SHA MODEL_NAME EVAL_SEED NOTE}"
NOTE="${5:?usage: $0 CHECKPOINT EXPECTED_SHA MODEL_NAME EVAL_SEED NOTE}"
[[ "${MODEL_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_v2_rare_h100_env.sh"
export ALGENGINE_PYTHON=/root/miniconda3/envs/algengine/bin/python
export SIMENGINE_PYTHON=/root/miniconda3/envs/simengine/bin/python
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_GSPLAT_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"
export NAVSIM_OFFICIAL_RESCORE=auto
export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="formal_navtest_seed${EVAL_SEED}"

CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
SCENARIO="${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl"
ASSETS="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets"
RAY_RUNNER="${ALGENGINE_ROOT}/scripts/diffusiondrive/run_ray_distributed_testing_diffusiondrive_h100.sh"
FORMAL_ROOT="${DIFFUSIONDRIVE_GRPO_FORMAL_ROOT:-${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal}"
RESULT_ROOT="${FORMAL_ROOT}/formal_eval/${MODEL_NAME}"
LOG_DIR="${RESULT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL V2 formal stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${CHECKPOINT}" "${CONFIG}" "${SCENARIO}" \
    "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ASSETS}" ]] || { echo "Missing assets: ${ASSETS}" >&2; exit 1; }
ACTUAL_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256}" ]] || {
    echo "Checkpoint SHA256 mismatch" >&2
    exit 1
}

CURRENT_STAGE=cuda_preflight
"${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
"${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py" \
    --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90 --ray-workers 8

cd "${ALGENGINE_ROOT}"
TEST_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/test"
CURRENT_STAGE=openloop_navtest
NAV_MARKER="$(mktemp)"
MASTER_PORT="${MASTER_PORT_NAVTEST:-28620}" \
    ./scripts/e2e_dist_eval.sh "${CONFIG}" "${CHECKPOINT}" 8
mapfile -t NAV_DIRS < <(find "${TEST_DIR}" -mindepth 1 -maxdepth 1 \
    -type d -name '*_official_pdms' -newer "${NAV_MARKER}")
[[ "${#NAV_DIRS[@]}" -eq 1 ]] || {
    echo "Expected one new navtest result, found ${#NAV_DIRS[@]}" >&2
    exit 1
}
NAV_PDM="${NAV_DIRS[0]}/pdm_scores_merged.csv"
NAV_PREFIX="${NAV_DIRS[0]%_official_pdms}"
NAV_ADE="${NAV_PREFIX}.csv"

CURRENT_STAGE=openloop_failures
FAIL_MARKER="$(mktemp)"
MASTER_PORT="${MASTER_PORT_FAILURES:-28621}" \
    ./scripts/e2e_dist_eval_navtest_failures.sh "${CONFIG}" "${CHECKPOINT}" 8
mapfile -t FAIL_DIRS < <(find "${TEST_DIR}" -mindepth 1 -maxdepth 1 \
    -type d -name '*_official_pdms' -newer "${FAIL_MARKER}")
[[ "${#FAIL_DIRS[@]}" -eq 1 ]] || {
    echo "Expected one new failures result, found ${#FAIL_DIRS[@]}" >&2
    exit 1
}
FAIL_PDM="${FAIL_DIRS[0]}/pdm_scores_merged.csv"

CURRENT_STAGE=closed_loop_nonreactive
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures NR
CURRENT_STAGE=closed_loop_reactive
sleep 10
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures R

NR_CSV="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_NR/WE_output/openscene_format/all_scenes_pdm_averages_NR.csv"
R_CSV="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_R/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
CURRENT_STAGE=summarize
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/summarize_grpo_selector_table.py \
    --model-name "${MODEL_NAME}" --note "${NOTE}" \
    --checkpoint "${CHECKPOINT}" --eval-seed "${EVAL_SEED}" \
    --openloop-navtest-ade "${NAV_ADE}" --openloop-navtest-pdm "${NAV_PDM}" \
    --openloop-failures-pdm "${FAIL_PDM}" --closedloop-nr "${NR_CSV}" \
    --closedloop-r "${R_CSV}" --output "${RESULT_ROOT}/summary.json"

rm -f "${NAV_MARKER}" "${FAIL_MARKER}"
CURRENT_STAGE=complete
trap - ERR
echo "PASS V2 formal four-block evaluation"
echo "summary: ${RESULT_ROOT}/summary.json"
echo "persistent_log: ${LOG_FILE}"
