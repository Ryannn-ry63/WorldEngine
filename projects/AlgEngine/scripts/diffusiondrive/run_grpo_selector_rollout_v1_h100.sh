#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:?usage: run_grpo_selector_rollout_v1_h100.sh MODE SEED DATA_TYPE [ASSET_NAME] [RUN_ID]}"
ROLLOUT_SEED="${2:?usage: run_grpo_selector_rollout_v1_h100.sh MODE SEED DATA_TYPE [ASSET_NAME] [RUN_ID]}"
DATA_TYPE="${3:?usage: run_grpo_selector_rollout_v1_h100.sh MODE SEED DATA_TYPE [ASSET_NAME] [RUN_ID]}"
ASSET_NAME="${4:-navtrain}"
RUN_ID="${5:-${MODE}}"
case "${MODE}" in
    preflight) MAX_SCENARIOS=0 ;;
    smoke) MAX_SCENARIOS=1 ;;
    pilot) MAX_SCENARIOS=8 ;;
    collect) MAX_SCENARIOS=-1 ;;
    *) echo "MODE must be preflight, smoke, pilot, or collect" >&2; exit 2 ;;
esac
if [[ ! "${ROLLOUT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi
for value in "${DATA_TYPE}" "${ASSET_NAME}" "${RUN_ID}"; do
    if [[ ! "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "unsafe identifier: ${value}" >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
export ALGENGINE_PYTHON=/root/miniconda3/envs/algengine/bin/python
export SIMENGINE_PYTHON=/root/miniconda3/envs/simengine/bin/python
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_GSPLAT_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"
export DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE="diffusiondrive_rollout_v1_seed${ROLLOUT_SEED}"
export DIFFUSIONDRIVE_ROLLOUT_CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_rollout_base.py"
CHECKPOINT="${DIFFUSIONDRIVE_GRPO_BASELINE}"
EXPECTED_SHA256=1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514
SCENARIO_FILE="${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/${DATA_TYPE}/all_scenarios.pkl"
ASSET_ROOT="${WORLDENGINE_ROOT}/data/sim_engine/assets/${ASSET_NAME}/assets"
ROLLOUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1/rollouts/${DATA_TYPE}/seed${ROLLOUT_SEED}/${RUN_ID}"
LOG_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1/logs"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/$(date -u +%Y%m%dT%H%M%SZ)_${DATA_TYPE}_s${ROLLOUT_SEED}_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL rollout-v1 stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${CONFIG}" "${CHECKPOINT}" "${SCENARIO_FILE}" "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ASSET_ROOT}" ]] || { echo "Missing asset root: ${ASSET_ROOT}" >&2; exit 1; }
ACTUAL_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch: ${ACTUAL_SHA256}" >&2
    exit 1
}

CURRENT_STAGE=cuda_preflight
"${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
"${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py" \
    --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90 --ray-workers 8
"${ALGENGINE_PYTHON}" -c "from mmcv import Config; c=Config.fromfile('${CONFIG}'); assert c.model.planning_head.export_rollout_context; assert c.model.planning_head.online_reward is None; assert c.selector_rollout_contract.deployed_action_parity_required"
"${SIMENGINE_PYTHON}" -c "from worldengine.manager.diffusiondrive_dynamic_reward_manager import expand_candidates_to_40; import numpy as np; assert expand_candidates_to_40(np.zeros((20,8,3))).shape == (20,40,3)"
if [[ "${MODE}" == preflight ]]; then
    CURRENT_STAGE=complete
    trap - ERR
    echo "PASS DiffusionDrive rollout-v1 H100 preflight; no rollout started"
    exit 0
fi
if [[ -e "${ROLLOUT_ROOT}" ]]; then
    echo "Immutable rollout target already exists: ${ROLLOUT_ROOT}" >&2
    exit 1
fi
mkdir -p "${ROLLOUT_ROOT}"

we_pid=""
planner_pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${planner_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${we_pid}" ]]; then
        kill "${we_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

CURRENT_STAGE=worldengine_rollout
MAX_ARGUMENT=()
if [[ "${MAX_SCENARIOS}" -gt 0 ]]; then
    MAX_ARGUMENT=("max_successful_scenarios=${MAX_SCENARIOS}")
fi
cd "${SIMENGINE_ROOT}"
"${SIMENGINE_PYTHON}" worldengine/runner/run_simulation.py \
    debug_mode=True \
    debug_scene_name=null \
    data_file_path="${SCENARIO_FILE}" \
    asset_folder_path="${ASSET_ROOT}" \
    output_dir="${ROLLOUT_ROOT}/__WORKER_ID__/WE_output" \
    job_name="dd_rollout_v1_${DATA_TYPE}_s${ROLLOUT_SEED}_${RUN_ID}" \
    use_planner_actions=true \
    ego_policy=env_input_policy \
    ego_client=navformer_client \
    ego_controller=log_play_controller \
    ego_navigation=trajectory_navigation \
    planner_data_path="${ROLLOUT_ROOT}/__WORKER_ID__/plan_traj" \
    planner_client_folder="${ROLLOUT_ROOT}/__WORKER_ID__/frames" \
    with_metric_manager=true \
    with_dense_reward_manager=true \
    diffusiondrive_dynamic_candidate_reward=true \
    diffusiondrive_candidate_sidecar_path="${ROLLOUT_ROOT}/__WORKER_ID__/diffusiondrive_candidate_sidecars" \
    distributed_mode=SCENARIO_BASED \
    worker=ray_distributed \
    worker_id_prefix=split_ \
    enable_resume=false \
    "${MAX_ARGUMENT[@]}" &
we_pid=$!
sleep 30

run_planner() {
    local split_id="$1"
    local worker_root="${ROLLOUT_ROOT}/split_${split_id}"
    mkdir -p "${worker_root}/plan_traj" "${worker_root}/frames" \
        "${worker_root}/merged_ann_files" "${worker_root}/diffusiondrive_candidate_sidecars"
    cd "${ALGENGINE_ROOT}"
    CUDA_VISIBLE_DEVICES="${split_id}" "${ALGENGINE_PYTHON}" closed_loop/sim_test.py \
        "${CONFIG}" "${CHECKPOINT}" --seed "${ROLLOUT_SEED}" --log-dir "${worker_root}" \
        --cfg-options sim.monitored_folder="${worker_root}/frames" \
        sim.plan_save_path="${worker_root}/plan_traj" \
        sim.merged_ann_save_dir="${worker_root}/merged_ann_files" \
        sim.diffusiondrive_rollout_sidecar_path="${worker_root}/diffusiondrive_candidate_sidecars" \
        sim.clean_temp_files=False sim.clean_record_data=False \
        data_root="${worker_root}/WE_output/openscene_format/"
}

CURRENT_STAGE=algengine_clients
for split_id in {0..7}; do
    run_planner "${split_id}" &
    planner_pids+=("$!")
done
for pid in "${planner_pids[@]}"; do
    wait "${pid}"
done
wait "${we_pid}"
we_pid=""
planner_pids=()
trap - EXIT INT TERM

CURRENT_STAGE=merge
cd "${SIMENGINE_ROOT}"
"${SIMENGINE_PYTHON}" scripts/merge_simulation_results.py \
    --test_path "${ROLLOUT_ROOT}" --react_type NR
CURRENT_STAGE=audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_rollout_v1.py" \
    --rollout-root "${ROLLOUT_ROOT}" \
    --expected-checkpoint-sha256 "${EXPECTED_SHA256}" \
    --expected-noise-namespace "${DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE}" \
    --minimum-records 8 \
    --output "${ROLLOUT_ROOT}/rollout_audit.json"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive selector rollout-v1 mode=${MODE} seed=${ROLLOUT_SEED}"
echo "rollout_root: ${ROLLOUT_ROOT}"
echo "persistent_log: ${LOG_FILE}"
