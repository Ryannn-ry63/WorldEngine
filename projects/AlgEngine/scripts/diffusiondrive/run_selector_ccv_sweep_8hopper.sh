#!/usr/bin/env bash
# Resume-safe 33x20 candidate causal-value sweep for frozen scalar V3.
set -Eeuo pipefail

set +u
PS1="${PS1:-selector-ccv-sweep}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u

MODE="${1:-}"
RUN_ID="${2:-ccv_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
ARG3="${3:-}"
ARG4="${4:-}"
case "${MODE}" in
    prepare|preflight|sentinel|arm|sweep|analyze|all) ;;
    *)
        echo "Usage: $0 {prepare|preflight|sentinel|arm|sweep|analyze|all} [RUN_ID] [INDEX|START] [STRIDE]" >&2
        exit 2
        ;;
esac
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run id: ${RUN_ID}" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export SOURCE_WORLDENGINE_ROOT="${SELECTOR_CCV_SOURCE_WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export SIMENGINE_PYTHON="${DIFFUSIONDRIVE_SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"
export PATH="${ALGENGINE_ENV}/bin:${PATH}"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export MMCV_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/mmcv
export NUPLAN_DEVKIT_ROOT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit
export SENIOR_NAVSIM_PARENT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export DIFFUSIONDRIVE_BOOTSTRAP="${H100_SUPPORT_DIR}/algengine_worker_bootstrap"
export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${SOURCE_WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export WORLDENGINE_GSPLAT_EXTENSION="${SOURCE_WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

SOURCE_RUN="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_causal_branch_pilot_v1/runs/causal_prefixfix_20260903"
SOURCE_TARGET="${SOURCE_RUN}/targets/target_manifest.json"
SOURCE_GATE="${SOURCE_RUN}/causal_gate.json"
PROTOCOL="${WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_PROTOCOL.md"
SENTINEL_AMENDMENT="${WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_SENTINEL_AMENDMENT.md"
ANALYSIS_NOTE="${WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_CCV_SWEEP_V1_ANALYSIS_NOTE.md"
EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ccv_sweep_v1"
RUN_ROOT="${EXPERIMENT_ROOT}/runs/${RUN_ID}"
MANIFEST_ROOT="${RUN_ROOT}/manifests"
MASTER_MANIFEST="${MANIFEST_ROOT}/master_manifest.json"
SENTINEL_GATE="${RUN_ROOT}/sentinel_gate.json"
MATRIX_CSV="${RUN_ROOT}/candidate_causal_value_matrix.csv"
FINAL_GATE="${RUN_ROOT}/ccv_gate.json"

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_ccv_sweep.py"
BUILDER="${SCRIPT_DIR}/build_selector_ccv_sweep.py"
AUDITOR="${SCRIPT_DIR}/audit_selector_ccv_sweep_collection.py"
SENTINEL_VERIFIER="${SCRIPT_DIR}/verify_selector_ccv_sentinel.py"
ANALYZER="${SCRIPT_DIR}/analyze_selector_ccv_sweep.py"
CCV_COMMON="${SCRIPT_DIR}/ccv_sweep_common.py"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_selector_ccv_sweep_8hopper.sh"
SIM_TEST="${ALGENGINE_ROOT}/closed_loop/sim_test.py"
PLANNING_HEAD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
SCENE_SELECTOR="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
NAVFORMER="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
NAVFORMER_CLIENT="${SIMENGINE_ROOT}/worldengine/components/agents/client/navformer_client.py"
CCV_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_candidate_sweep_manager.py"
PREACTION_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_preaction_oracle_manager.py"
DYNAMIC_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_dynamic_reward_manager.py"
SIDECAR_CONTRACT="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_sidecar_contract.py"
BASE_ENV="${SIMENGINE_ROOT}/worldengine/envs/base_env.py"
DEFAULT_RUNNER="${SIMENGINE_ROOT}/worldengine/configs/default_runner.yaml"
MERGE_RESULTS="${SIMENGINE_ROOT}/scripts/merge_simulation_results.py"
ASSET_ROOT="${SOURCE_WORLDENGINE_ROOT}/data/sim_engine/assets/navtrain/assets"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }; }
json_value() {
    "${ALGENGINE_PYTHON}" -c 'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): value=value[key]
print(value)' "$1" "$2"
}
manifest_path() {
    echo "${SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json"
}

static_preflight() {
    local required=(
        "${ALGENGINE_PYTHON}" "${SIMENGINE_PYTHON}" "${CONFIG}" "${BUILDER}"
        "${AUDITOR}" "${SENTINEL_VERIFIER}" "${ANALYZER}" "${CCV_COMMON}"
        "${RUNNER_SCRIPT}" "${PROTOCOL}" "${SENTINEL_AMENDMENT}" "${ANALYSIS_NOTE}" "${SOURCE_TARGET}" "${SOURCE_GATE}"
        "${SIM_TEST}" "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${NAVFORMER}"
        "${NAVFORMER_CLIENT}" "${CCV_MANAGER}" "${PREACTION_MANAGER}"
        "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}" "${BASE_ENV}"
        "${DEFAULT_RUNNER}" "${MERGE_RESULTS}" "${WORLDENGINE_MMCV_EXTENSION}"
        "${WORLDENGINE_GSPLAT_EXTENSION}" "$(manifest_path)"
    )
    local path
    for path in "${required[@]}"; do require_file "${path}"; done
    [[ -d "${ASSET_ROOT}" ]] || { echo "Missing asset root: ${ASSET_ROOT}" >&2; exit 1; }
    mkdir -p "${RUN_ROOT}"
}

hopper_preflight() {
    "${ALGENGINE_PYTHON}" - <<'PY'
import json,torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names)!=8: raise RuntimeError(f"expected 8 visible GPUs, got {len(names)}")
if any(not any(x in name.upper() for x in ("H100","H200")) for name in names): raise RuntimeError(f"requires Hopper, got {names}")
if any(torch.cuda.get_device_capability(i)!=(9,0) for i in range(8)): raise RuntimeError("requires sm_90")
if str(torch.__version__)!="2.0.1+cu118" or torch.version.cuda!="11.8": raise RuntimeError(f"environment drift: {torch.__version__}/{torch.version.cuda}")
print(json.dumps({"status":"PASS","hardware_contract":"hopper8","devices":names,"torch":torch.__version__},sort_keys=True))
PY
    "${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
    "${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py" --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90 --ray-workers 8
}

verify_master() {
    require_file "${MASTER_MANIFEST}"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" -c 'import sys,ccv_sweep_common as c; p,d=c.load_master(sys.argv[1]); assert d["failed_target_count"]==9 and d["solved_target_count"]==24 and not d["causal_values_authorized_for_training"]' "${MASTER_MANIFEST}"
}

prepare_manifests() {
    if [[ ! -f "${MASTER_MANIFEST}" ]]; then
        mkdir -p "${MANIFEST_ROOT}"
        PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" "${BUILDER}" \
            --source-target-manifest "${SOURCE_TARGET}" \
            --source-causal-gate "${SOURCE_GATE}" \
            --protocol "${PROTOCOL}" \
            --output-root "${MANIFEST_ROOT}"
    fi
    verify_master
    echo "PASS frozen CCV manifests: ${MASTER_MANIFEST}"
}

prepare_environment() {
    local treatment_id="$1" treatment_manifest="$2" treatment_sha="$3"
    local manifest checkpoint
    manifest="$(manifest_path)"
    checkpoint="$(json_value "${manifest}" checkpoint)"
    require_file "${checkpoint}"
    export DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED=0
    export DIFFUSIONDRIVE_CCV_TREATMENT_ID="${treatment_id}"
    export DIFFUSIONDRIVE_CCV_TREATMENT_MANIFEST_SHA256="${treatment_sha}"
    export DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256="$(json_value "${manifest}" checkpoint_sha256)"
    export DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256="$(sha256sum "${manifest}" | awk '{print $1}')"
    [[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256}" ]] || { echo "Checkpoint SHA256 drifted" >&2; exit 1; }
    export DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE=selector_causal_branch_pilot_evalnoise0
    export DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256="$({ sha256sum "${SIM_TEST}" "${NAVFORMER_CLIENT}" "${CCV_MANAGER}" "${PREACTION_MANAGER}" "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}" "${BASE_ENV}" "${DEFAULT_RUNNER}" "${CCV_COMMON}"; } | sha256sum | awk '{print $1}')"
    export DIFFUSIONDRIVE_ROLLOUT_CODE_SHA="$({ sha256sum "${RUNNER_SCRIPT}" "${CONFIG}" "${BUILDER}" "${AUDITOR}" "${SENTINEL_VERIFIER}" "${ANALYZER}" "${CCV_COMMON}" "${SIM_TEST}" "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${NAVFORMER}" "${NAVFORMER_CLIENT}" "${CCV_MANAGER}" "${PREACTION_MANAGER}" "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}" "${BASE_ENV}" "${DEFAULT_RUNNER}" "${MERGE_RESULTS}"; } | sha256sum | awk '{print $1}')"
    POLICY_MANIFEST="${manifest}"
    POLICY_CHECKPOINT="${checkpoint}"
    TREATMENT_MANIFEST="${treatment_manifest}"
}

verify_audit() {
    local audit="$1" treatment_id="$2" treatment_sha="$3"
    "${ALGENGINE_PYTHON}" -c 'import json,os,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS" and r["method"]=="diffusiondrive_selector_ccv_collection_audit_v1" and r["layout"]=="merged"
assert r["treatment_id"]==sys.argv[2] and r["treatment_manifest_sha256"]==sys.argv[3]
assert r["behavior_policy_train_seed"]==0 and r["code_sha"]==os.environ["DIFFUSIONDRIVE_ROLLOUT_CODE_SHA"]
assert r["rollout_implementation_sha256"]==os.environ["DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"]
assert r["intervention_count"]==r["num_scenarios"] and r["closed_loop_outcome"] is not None' "${audit}" "${treatment_id}" "${treatment_sha}"
}

run_collection() {
    local treatment_id="$1" treatment_manifest="$2"
    require_file "${treatment_manifest}"
    local treatment_sha scenario root final_audit partial_audit
    treatment_sha="$(sha256sum "${treatment_manifest}" | awk '{print $1}')"
    scenario="$(json_value "${treatment_manifest}" scenario_file)"
    require_file "${scenario}"
    prepare_environment "${treatment_id}" "${treatment_manifest}" "${treatment_sha}"
    root="${RUN_ROOT}/collections/${treatment_id}"
    final_audit="${root}/collection_audit.json"
    partial_audit="${root}/premerge_collection_audit.json"
    if [[ -f "${final_audit}" ]]; then
        verify_audit "${final_audit}" "${treatment_id}" "${treatment_sha}"
        echo "SKIP verified CCV ${treatment_id}: ${final_audit}"
        return 0
    fi
    if [[ -f "${partial_audit}" ]] && ! "${ALGENGINE_PYTHON}" -c 'import json,os,sys
r=json.load(open(sys.argv[1]))
valid=(r.get("status")=="PASS" and r.get("layout")=="split"
       and r.get("treatment_id")==sys.argv[2]
       and r.get("treatment_manifest_sha256")==sys.argv[3]
       and r.get("code_sha")==os.environ["DIFFUSIONDRIVE_ROLLOUT_CODE_SHA"]
       and r.get("rollout_implementation_sha256")==os.environ["DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"])
raise SystemExit(0 if valid else 1)' "${partial_audit}" "${treatment_id}" "${treatment_sha}"; then
        echo "STALE_CCV_PARTIAL_PROVENANCE: ${partial_audit}" >&2
        echo "The existing split records were produced by another code hash." >&2
        echo "Keep this run as evidence and restart sentinel with a new RUN_ID." >&2
        exit 1
    fi
    mkdir -p "${root}/logs"
    find "${root}" -name simulation_completed.flag -type f -delete

    "${ALGENGINE_PYTHON}" -c 'from mmcv import Config; import sys
c=Config.fromfile(sys.argv[1]); x=c.selector_rollout_contract
assert c.model.planning_head.export_rollout_context and c.model.planning_head.online_reward is None
assert x.schema_version==8 and x.train_only and x.diagnostic_only and not x.training_data_consumed
assert x.action_score_timing=="pre_action" and x.intervention_mode=="one_shot_manifest"
assert x.treatment_id==sys.argv[2] and x.treatment_manifest_sha256==sys.argv[3]
assert not x.causal_value_authorized_for_training' "${CONFIG}" "${treatment_id}" "${treatment_sha}"

    local we_pid="" failure=0 planner_pids=()
    cleanup_collection() {
        trap - EXIT INT TERM
        local pid
        for pid in "${planner_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
        [[ -z "${we_pid}" ]] || kill "${we_pid}" 2>/dev/null || true
    }
    trap cleanup_collection EXIT INT TERM
    cd "${SIMENGINE_ROOT}"
    "${SIMENGINE_PYTHON}" worldengine/runner/run_simulation.py \
        debug_mode=True debug_scene_name=null data_file_path="${scenario}" asset_folder_path="${ASSET_ROOT}" \
        output_dir="${root}/__WORKER_ID__/WE_output" job_name="selector_ccv_${treatment_id}" \
        use_planner_actions=true ego_policy=env_input_policy ego_client=navformer_client ego_controller=log_play_controller ego_navigation=trajectory_navigation \
        agent_policy=idm_policy agent_navigation=idm_navigation planner_data_path="${root}/__WORKER_ID__/plan_traj" planner_client_folder="${root}/__WORKER_ID__/frames" \
        with_metric_manager=true with_dense_reward_manager=true diffusiondrive_candidate_sweep=true diffusiondrive_preaction_oracle=false diffusiondrive_dynamic_candidate_reward=false \
        diffusiondrive_behavior_train_seed=0 diffusiondrive_ccv_treatment_manifest="${treatment_manifest}" diffusiondrive_ccv_treatment_manifest_sha256="${treatment_sha}" \
        diffusiondrive_candidate_sidecar_path="${root}/__WORKER_ID__/diffusiondrive_candidate_sidecars" diffusiondrive_sidecar_timeout_s=300 \
        distributed_mode=SCENARIO_BASED worker=ray_distributed worker_id_prefix=split_ enable_resume=true completed_scenarios_dir="${root}/__WORKER_ID__/completed_scenarios" \
        >"${root}/logs/worldengine.log" 2>&1 &
    we_pid=$!
    sleep 30

    run_planner() {
        local split_id="$1" worker_root="${root}/split_$1"
        mkdir -p "${worker_root}/plan_traj" "${worker_root}/frames" "${worker_root}/merged_ann_files" "${worker_root}/diffusiondrive_candidate_sidecars"
        cd "${ALGENGINE_ROOT}"
        CUDA_VISIBLE_DEVICES="${split_id}" "${ALGENGINE_PYTHON}" closed_loop/sim_test.py "${CONFIG}" "${POLICY_CHECKPOINT}" --seed 0 --log-dir "${worker_root}" \
            --cfg-options sim.monitored_folder="${worker_root}/frames" sim.plan_save_path="${worker_root}/plan_traj" sim.merged_ann_save_dir="${worker_root}/merged_ann_files" \
            sim.diffusiondrive_rollout_sidecar_path="${worker_root}/diffusiondrive_candidate_sidecars" sim.clean_temp_files=False sim.clean_record_data=False data_root="${worker_root}/WE_output/openscene_format/" \
            >"${root}/logs/planner_split${split_id}.log" 2>&1
    }
    local split_id
    for split_id in 0 1 2 3 4 5 6 7; do run_planner "${split_id}" & planner_pids+=("$!"); done
    local pid
    for pid in "${planner_pids[@]}"; do wait "${pid}" || failure=1; done
    wait "${we_pid}" || failure=1
    we_pid=""; planner_pids=(); trap - EXIT INT TERM
    if [[ "${failure}" -ne 0 ]]; then
        echo "FAIL CCV collection ${treatment_id}; rerun the same treatment and RUN_ID to resume" >&2
        tail -80 "${root}/logs/worldengine.log" >&2 || true
        exit 1
    fi

    local audit_common=(
        --rollout-root "${root}" --scenario-file "${scenario}"
        --checkpoint-manifest "${POLICY_MANIFEST}" --train-seed 0
        --treatment-id "${treatment_id}" --treatment-manifest "${treatment_manifest}"
        --expected-noise-namespace "${DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE}"
        --expected-implementation-sha256 "${DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256}"
        --expected-code-sha "${DIFFUSIONDRIVE_ROLLOUT_CODE_SHA}" --expected-workers 8
    )
    "${ALGENGINE_PYTHON}" "${AUDITOR}" "${audit_common[@]}" --layout split --output "${root}/premerge_collection_audit.json"
    # Split records are authoritative. Remove only merged hard links so a
    # resumed collection cannot retain records from an earlier code hash.
    local merged_ccv_records="${root}/WE_output/openscene_format/diffusiondrive_ccv_records"
    if [[ -d "${merged_ccv_records}" ]]; then
        find "${merged_ccv_records}" -maxdepth 1 -type f -name '*_ccv.pkl' -delete
    fi
    cd "${SIMENGINE_ROOT}"
    "${SIMENGINE_PYTHON}" scripts/merge_simulation_results.py --test_path "${root}" --react_type R --num-splits 8
    local metrics="${root}/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
    require_file "${metrics}"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" "${audit_common[@]}" --layout merged --metrics-csv "${metrics}" --output "${final_audit}"
    verify_audit "${final_audit}" "${treatment_id}" "${treatment_sha}"
    echo "PASS CCV ${treatment_id}: ${final_audit}"
}

run_sentinel() {
    prepare_manifests
    local label manifest
    for label in sentinel_policy sentinel_oracle sentinel_matched; do
        manifest="$(json_value "${MASTER_MANIFEST}" "treatment_manifests.sentinels.${label}.path")"
        run_collection "${label}" "${manifest}"
    done
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" "${SENTINEL_VERIFIER}" \
        --master-manifest "${MASTER_MANIFEST}" --run-root "${RUN_ROOT}" \
        --amendment "${SENTINEL_AMENDMENT}" --output "${SENTINEL_GATE}"
}

verify_sentinel() {
    require_file "${SENTINEL_GATE}"
    "${ALGENGINE_PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASS" and r["decision"]=="AUTHORIZE_CCV_20_ARM_SWEEP" and r["all_sentinels_reproducible"] and r["sentinel_amendment_sha256"]==sys.argv[2]' "${SENTINEL_GATE}" "$(sha256sum "${SENTINEL_AMENDMENT}" | awk '{print $1}')"
}

run_arm() {
    local index="$1"
    [[ "${index}" =~ ^[0-9]+$ ]] && (( index >= 0 && index < 20 )) || { echo "CCV arm index must be 0..19" >&2; exit 2; }
    verify_master
    verify_sentinel
    local manifest
    manifest="$(json_value "${MASTER_MANIFEST}" "treatment_manifests.arms.${index}.path")"
    run_collection "$(printf 'arm_%02d' "${index}")" "${manifest}"
}

run_sweep() {
    local start="${1:-0}" stride="${2:-1}" index
    [[ "${start}" =~ ^[0-9]+$ && "${stride}" =~ ^[1-9][0-9]*$ ]] || { echo "CCV START must be >=0 and STRIDE >=1" >&2; exit 2; }
    (( start < 20 )) || { echo "CCV START must be below 20" >&2; exit 2; }
    for ((index=start; index<20; index+=stride)); do run_arm "${index}"; done
}

run_analysis() {
    verify_master
    verify_sentinel
    local index
    for index in $(seq 0 19); do
        require_file "${RUN_ROOT}/collections/$(printf 'arm_%02d' "${index}")/collection_audit.json"
    done
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" "${ANALYZER}" \
        --master-manifest "${MASTER_MANIFEST}" --run-root "${RUN_ROOT}" \
        --sentinel-gate "${SENTINEL_GATE}" --analysis-note "${ANALYSIS_NOTE}" \
        --matrix-csv "${MATRIX_CSV}" --output "${FINAL_GATE}"
}

static_preflight
case "${MODE}" in
    prepare) prepare_manifests ;;
    preflight) hopper_preflight; prepare_manifests; echo "PASS CCV preflight" ;;
    sentinel) hopper_preflight; run_sentinel ;;
    arm) hopper_preflight; prepare_manifests; run_arm "${ARG3}" ;;
    sweep) hopper_preflight; prepare_manifests; run_sweep "${ARG3:-0}" "${ARG4:-1}" ;;
    analyze) run_analysis ;;
    all) hopper_preflight; prepare_manifests; run_sentinel; run_sweep 0 1; run_analysis ;;
esac

