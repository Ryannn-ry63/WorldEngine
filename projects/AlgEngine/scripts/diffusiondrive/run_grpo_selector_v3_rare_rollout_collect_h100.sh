#!/usr/bin/env bash
# Resume-safe online collection for one disjoint rare-scene lane.

set -Eeo pipefail

LANE="${1:?usage: run_grpo_selector_v3_rare_rollout_collect_h100.sh LANE [MAX_SCENARIOS] [RUN_KIND]}"
MAX_SCENARIOS="${2:--1}"
RUN_KIND="${3:-formal}"
if [[ ! "${LANE}" =~ ^[0-2]$ ]]; then
    echo "LANE must be 0, 1, or 2" >&2
    exit 2
fi
if [[ ! "${MAX_SCENARIOS}" =~ ^-?[0-9]+$ ]] || [[ "${MAX_SCENARIOS}" -eq 0 ]]; then
    echo "MAX_SCENARIOS must be -1 or a positive integer" >&2
    exit 2
fi
if [[ ! "${RUN_KIND}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsafe RUN_KIND: ${RUN_KIND}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_base.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

ROLLOUT_GPU_COUNT="${DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT:-8}"
if [[ "${ROLLOUT_GPU_COUNT}" -ne "${GPU_COUNT}" ]]; then
    echo "rollout worker count ${ROLLOUT_GPU_COUNT} != visible GPU count ${GPU_COUNT}" >&2
    exit 1
fi
export SIMENGINE_PYTHON="${DIFFUSIONDRIVE_SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_GSPLAT_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"
export DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE="diffusiondrive_v3_rare_rollout_v1"

BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_v1"
SCENARIO_ROOT="${SOURCE_ROOT}/scenarios"
SCENARIO_AUDIT="${SCENARIO_ROOT}/rare_rollout_scenario_audit.json"
SCENARIO_FILE="${SCENARIO_ROOT}/scenario_shard_0${LANE}_of_03.pkl"
ASSET_ROOT="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtrain/assets"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
ROLLOUT_ROOT="${ROOT}/collection/${RUN_KIND}/lane${LANE}"
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_v3_rare_rollout_collection.py"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/collect_${RUN_KIND}_lane${LANE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL rare-rollout collection lane=${LANE} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${CONFIG}" "${BASELINE}" "${SCENARIO_FILE}" "${SCENARIO_AUDIT}" \
    "${AUDITOR}" "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ASSET_ROOT}" ]] || { echo "Missing asset root: ${ASSET_ROOT}" >&2; exit 1; }
[[ -x "${SIMENGINE_PYTHON}" ]] || { echo "Missing SimEngine Python: ${SIMENGINE_PYTHON}" >&2; exit 1; }
ACTUAL_BASELINE_SHA="$(sha256sum "${BASELINE}" | awk '{print $1}')"
[[ "${ACTUAL_BASELINE_SHA}" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch: ${ACTUAL_BASELINE_SHA}" >&2
    exit 1
}
SCENARIO_SHA="$(sha256sum "${SCENARIO_FILE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" -c '
import json,sys
audit=json.load(open(sys.argv[1]))
assert audit["status"]=="PASS"
lane=audit["lanes"][int(sys.argv[2])]
assert lane["scenario_file_sha256"]==sys.argv[3]
' "${SCENARIO_AUDIT}" "${LANE}" "${SCENARIO_SHA}"

if ! git -C "${WORLDENGINE_ROOT}" diff --quiet -- \
    projects/AlgEngine projects/SimEngine; then
    echo "Tracked AlgEngine/SimEngine files are dirty; refusing unaudited collection" >&2
    exit 1
fi
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
export DIFFUSIONDRIVE_ROLLOUT_CODE_SHA="${CODE_SHA}"

FINAL_AUDIT="${ROLLOUT_ROOT}/collection_audit.json"
if [[ "${MAX_SCENARIOS}" -lt 0 && -f "${FINAL_AUDIT}" ]]; then
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["layout"]=="merged"
assert row["code_sha"]==sys.argv[2]
assert row["scenario_file_sha256"]==sys.argv[3]
' "${FINAL_AUDIT}" "${CODE_SHA}" "${SCENARIO_SHA}"
    trap - ERR
    echo "SKIP verified completed rare-rollout lane ${LANE}"
    echo "audit: ${FINAL_AUDIT}"
    exit 0
fi

CURRENT_STAGE=cuda_preflight
"${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
"${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py" \
    --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90 \
    --ray-workers "${ROLLOUT_GPU_COUNT}"
"${ALGENGINE_PYTHON}" -c "from mmcv import Config; c=Config.fromfile('${CONFIG}'); assert c.selector_rollout_contract.source_policy=='immutable_epoch100_diffusiondrive'; assert not c.selector_rollout_contract.trained_v3_checkpoint_loaded; assert c.model.planning_head.export_rollout_context"

mkdir -p "${ROLLOUT_ROOT}"
# A completion flag is a process-lifetime signal, whereas completed_scenarios
# is the persistent resume ledger.  Remove only stale signals before restart.
"${ALGENGINE_PYTHON}" -c '
import pathlib,sys
root=pathlib.Path(sys.argv[1])
for path in root.rglob("simulation_completed.flag"):
    path.unlink()
' "${ROLLOUT_ROOT}"

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
    # This is an opt-in smoke/resume limiter, not a field in the base Hydra
    # schema.  The leading '+' is therefore required by Hydra struct mode.
    MAX_ARGUMENT=("+max_successful_scenarios=${MAX_SCENARIOS}")
fi
cd "${SIMENGINE_ROOT}"
"${SIMENGINE_PYTHON}" worldengine/runner/run_simulation.py \
    debug_mode=True \
    debug_scene_name=null \
    data_file_path="${SCENARIO_FILE}" \
    asset_folder_path="${ASSET_ROOT}" \
    output_dir="${ROLLOUT_ROOT}/__WORKER_ID__/WE_output" \
    job_name="dd_v3_rare_rollout_${RUN_KIND}_lane${LANE}" \
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
    diffusiondrive_sidecar_timeout_s=300 \
    distributed_mode=SCENARIO_BASED \
    worker=ray_distributed \
    worker_id_prefix=split_ \
    completed_scenarios_dir="${ROLLOUT_ROOT}/__WORKER_ID__/completed_scenarios" \
    enable_resume=true \
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
        "${CONFIG}" "${BASELINE}" --seed 0 --log-dir "${worker_root}" \
        --cfg-options sim.monitored_folder="${worker_root}/frames" \
        sim.plan_save_path="${worker_root}/plan_traj" \
        sim.merged_ann_save_dir="${worker_root}/merged_ann_files" \
        sim.diffusiondrive_rollout_sidecar_path="${worker_root}/diffusiondrive_candidate_sidecars" \
        sim.clean_temp_files=False sim.clean_record_data=False \
        data_root="${worker_root}/WE_output/openscene_format/"
}

CURRENT_STAGE=algengine_clients
for (( split_id=0; split_id<ROLLOUT_GPU_COUNT; split_id++ )); do
    run_planner "${split_id}" &
    planner_pids+=("$!")
done
client_failure=0
for pid in "${planner_pids[@]}"; do
    if ! wait "${pid}"; then
        client_failure=1
    fi
done
if ! wait "${we_pid}"; then
    client_failure=1
fi
we_pid=""
planner_pids=()
trap - EXIT INT TERM
[[ "${client_failure}" -eq 0 ]] || {
    echo "WorldEngine or an AlgEngine client failed; rerun the same lane to resume" >&2
    exit 1
}

CURRENT_STAGE=premerge_audit
AUDIT_ARGS=(
    --rollout-root "${ROLLOUT_ROOT}"
    --scenario-file "${SCENARIO_FILE}"
    --layout split
    --expected-noise-namespace "${DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE}"
    --expected-code-sha "${CODE_SHA}"
    --expected-workers "${ROLLOUT_GPU_COUNT}"
    --output "${ROLLOUT_ROOT}/premerge_collection_audit.json"
)
if [[ "${MAX_SCENARIOS}" -gt 0 ]]; then
    AUDIT_ARGS+=(--maximum-scenarios "${MAX_SCENARIOS}")
fi
"${ALGENGINE_PYTHON}" "${AUDITOR}" "${AUDIT_ARGS[@]}"

CURRENT_STAGE=merge
cd "${SIMENGINE_ROOT}"
"${SIMENGINE_PYTHON}" scripts/merge_simulation_results.py \
    --test_path "${ROLLOUT_ROOT}" --react_type NR --num-splits "${ROLLOUT_GPU_COUNT}"

CURRENT_STAGE=final_audit
FINAL_AUDIT_ARGS=()
if [[ "${MAX_SCENARIOS}" -gt 0 ]]; then
    FINAL_AUDIT_ARGS=(--maximum-scenarios "${MAX_SCENARIOS}")
fi
"${ALGENGINE_PYTHON}" "${AUDITOR}" \
    --rollout-root "${ROLLOUT_ROOT}" \
    --scenario-file "${SCENARIO_FILE}" \
    --layout merged \
    --expected-noise-namespace "${DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE}" \
    --expected-code-sha "${CODE_SHA}" \
    --expected-workers "${ROLLOUT_GPU_COUNT}" \
    "${FINAL_AUDIT_ARGS[@]}" \
    --output "${FINAL_AUDIT}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive V3 rare rollout lane=${LANE} kind=${RUN_KIND}"
echo "rollout_root: ${ROLLOUT_ROOT}"
echo "audit: ${FINAL_AUDIT}"
echo "persistent_log: ${LOG_FILE}"
