#!/usr/bin/env bash
# One independent RAPG checkpoint: navtest, rare, CL-NR and CL-R.
set -Eeo pipefail

CHECKPOINT="${1:?usage: run_rapg_formal_eval_8hopper.sh CHECKPOINT SHA MODEL SEED NOTE ARCHITECTURE}"
EXPECTED_SHA256="${2:?missing expected checkpoint SHA256}"
MODEL_NAME="${3:?missing model name}"
EVAL_SEED="${4:?missing eval seed}"
NOTE="${5:?missing note}"
ARCHITECTURE="${6:?missing selector architecture}"
if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "MODEL_NAME contains invalid characters" >&2
    exit 2
fi
case "${ARCHITECTURE}" in
    reference_anchored_preference_graph|trajectory_set_reasoner|capacity_matched_unary) ;;
    *) echo "unsupported selector architecture: ${ARCHITECTURE}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8
export DIFFUSIONDRIVE_RAPG_ARCHITECTURE="${ARCHITECTURE}"
export DIFFUSIONDRIVE_RAPG_ABLATION=relational_only
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_GSPLAT_EXTENSION="${RAPG_SOURCE_WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"
export NAVSIM_OFFICIAL_RESCORE=auto
export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="rapg_formal_navtest_seed${EVAL_SEED}"

CONFIG="${RAPG_CONFIG}"
SCENARIO="${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl"
ASSETS="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets"
RAY_RUNNER="${ALGENGINE_ROOT}/scripts/diffusiondrive/run_ray_distributed_testing_diffusiondrive_h100.sh"
FORMAL_ROOT="${RAPG_FORMAL_ROOT:-${RAPG_EXPERIMENT_ROOT}/formal}"
RESULT_ROOT="${FORMAL_ROOT}/formal_eval/${MODEL_NAME}"
[[ ! -e "${RESULT_ROOT}/summary.json" ]] || {
    echo "immutable formal result already exists: ${RESULT_ROOT}/summary.json" >&2
    exit 1
}
mkdir -p "${RESULT_ROOT}/logs"
LOG_FILE="${RESULT_ROOT}/logs/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; echo "FAIL RAPG formal eval stage=${CURRENT_STAGE} exit=${rc}" >&2' ERR

CURRENT_STAGE=static_preflight
for path in "${CHECKPOINT}" "${CONFIG}" "${SCENARIO}" "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ASSETS}" ]] || { echo "Missing navtest_failures assets" >&2; exit 1; }
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]] || {
    echo "Checkpoint SHA256 mismatch" >&2
    exit 1
}
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
mapfile -t NAV_DIRS < <(find "${TEST_DIR}" -mindepth 1 -maxdepth 1 -type d \
    -name '*_official_pdms' -newer "${NAV_MARKER}")
[[ "${#NAV_DIRS[@]}" -eq 1 ]] || { echo "expected one navtest result" >&2; exit 1; }
NAV_PDM="${NAV_DIRS[0]}/pdm_scores_merged.csv"
NAV_ADE="${NAV_DIRS[0]%_official_pdms}.csv"

CURRENT_STAGE=openloop_rare
RARE_MARKER="$(mktemp)"
MASTER_PORT="${MASTER_PORT_FAILURES:-28621}" \
    ./scripts/e2e_dist_eval_navtest_failures.sh "${CONFIG}" "${CHECKPOINT}" 8
mapfile -t RARE_DIRS < <(find "${TEST_DIR}" -mindepth 1 -maxdepth 1 -type d \
    -name '*_official_pdms' -newer "${RARE_MARKER}")
[[ "${#RARE_DIRS[@]}" -eq 1 ]] || { echo "expected one rare result" >&2; exit 1; }
RARE_PDM="${RARE_DIRS[0]}/pdm_scores_merged.csv"

CURRENT_STAGE=closed_loop_nonreactive
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures NR
CURRENT_STAGE=closed_loop_reactive
sleep 10
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures R
NR_CSV="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_NR/WE_output/openscene_format/all_scenes_pdm_averages_NR.csv"
R_CSV="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_R/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"

CURRENT_STAGE=summarize
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/summarize_grpo_selector_table.py \
    --model-name "${MODEL_NAME}" --note "${NOTE}" --checkpoint "${CHECKPOINT}" \
    --eval-seed "${EVAL_SEED}" --openloop-navtest-ade "${NAV_ADE}" \
    --openloop-navtest-pdm "${NAV_PDM}" --openloop-failures-pdm "${RARE_PDM}" \
    --closedloop-nr "${NR_CSV}" --closedloop-r "${R_CSV}" \
    --output "${RESULT_ROOT}/summary.json"
rm -f "${NAV_MARKER}" "${RARE_MARKER}"
trap - ERR
echo "PASS RAPG formal four-block evaluation: ${RESULT_ROOT}/summary.json"
