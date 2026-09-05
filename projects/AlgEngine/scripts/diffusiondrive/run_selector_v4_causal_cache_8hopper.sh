#!/usr/bin/env bash
# Resume-safe V4 method-neutral causal-cache collection on one 8-Hopper instance.
set -Eeuo pipefail

set +u
PS1="${PS1:-selector-v4-causal-cache}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u

MODE="${1:-}"
RUN_ID="${2:-v4_causal_cache_$(date -u +%Y%m%dT%H%M%SZ)}"
ARG3="${3:-}"
ARG4="${4:-}"
case "${MODE}" in
    preflight|source|baseline|freeze|sentinel|pilot64|pilot_audit|expand192|dev64|arm|all) ;;
    *)
        echo "Usage: $0 {preflight|source|baseline|freeze|sentinel|pilot64|pilot_audit|expand192|dev64|arm|all} [RUN_ID] [STAGE] [INDEX]" >&2
        exit 2
        ;;
esac
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Invalid run id: ${RUN_ID}" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export SOURCE_WORLDENGINE_ROOT="${SELECTOR_V4_SOURCE_WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
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

SOURCE_ROOT="${SOURCE_WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_v4_split0"
PROTOCOL="${WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_V4_CAUSAL_CACHE_PROTOCOL_20260904.md"
LEDGER_TEMPLATE="${WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_V4_DECISION_LEDGER.json"
EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_v4_causal_cache_v1"
RUN_ROOT="${EXPERIMENT_ROOT}/runs/${RUN_ID}"
SOURCE_OUT="${RUN_ROOT}/source"
SOURCE_AUDIT="${SOURCE_OUT}/source_audit.json"
TARGET_ROOT="${RUN_ROOT}/targets"
TARGET_MANIFEST="${TARGET_ROOT}/target_manifest.json"
MANIFEST_ROOT="${RUN_ROOT}/manifests"
MASTER_MANIFEST="${MANIFEST_ROOT}/master_manifest.json"
CACHE_ROOT="${RUN_ROOT}/cache"
SENTINEL_GATE="${RUN_ROOT}/sentinel_gate.json"
PILOT_GATE="${RUN_ROOT}/pilot_information_gate.json"
RUN_LEDGER="${RUN_ROOT}/decision_ledger.json"

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_v4_causal_cache.py"
SOURCE_BUILDER="${SCRIPT_DIR}/prepare_selector_v4_causal_source.py"
TARGET_BUILDER="${SCRIPT_DIR}/build_selector_v4_causal_targets.py"
TREATMENT_BUILDER="${SCRIPT_DIR}/build_selector_v4_causal_treatments.py"
AUDITOR="${SCRIPT_DIR}/audit_selector_v4_causal_collection.py"
SENTINEL_VERIFIER="${SCRIPT_DIR}/verify_selector_v4_causal_sentinel.py"
ASSEMBLER="${SCRIPT_DIR}/assemble_selector_v4_causal_cache.py"
PILOT_VERIFIER="${SCRIPT_DIR}/verify_selector_v4_causal_pilot.py"
COMBINER="${SCRIPT_DIR}/combine_selector_v4_train_cache.py"
LEDGER_UPDATER="${SCRIPT_DIR}/update_selector_v4_ledger.py"
V4_COMMON="${SCRIPT_DIR}/v4_causal_cache_common.py"
V4_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_v4_causal_cache_manager.py"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_selector_v4_causal_cache_8hopper.sh"
SIM_TEST="${ALGENGINE_ROOT}/closed_loop/sim_test.py"
PLANNING_HEAD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
SCENE_SELECTOR="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
NAVFORMER="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
NAVFORMER_CLIENT="${SIMENGINE_ROOT}/worldengine/components/agents/client/navformer_client.py"
PREACTION_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_preaction_oracle_manager.py"
DYNAMIC_MANAGER="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_dynamic_reward_manager.py"
SIDECAR_CONTRACT="${SIMENGINE_ROOT}/worldengine/manager/diffusiondrive_sidecar_contract.py"
BASE_ENV="${SIMENGINE_ROOT}/worldengine/envs/base_env.py"
DEFAULT_RUNNER="${SIMENGINE_ROOT}/worldengine/configs/default_runner.yaml"
MERGE_RESULTS="${SIMENGINE_ROOT}/scripts/merge_simulation_results.py"
ASSET_ROOT="${SOURCE_WORLDENGINE_ROOT}/data/sim_engine/assets/navtrain/assets"

require_file() {
    [[ -f "$1" ]] || {
        echo "Missing required file: $1" >&2
        exit 1
    }
}

json_value() {
    "${ALGENGINE_PYTHON}" -c 'import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value=value[key]
print(value)' "$1" "$2"
}

checkpoint_manifest() {
    echo "${SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json"
}

ledger_update() {
    local stage="$1" status="$2" decision="$3" artifact="$4" next="$5"
    local args=(
        --template "${LEDGER_TEMPLATE}" --ledger "${RUN_LEDGER}"
        --run-id "${RUN_ID}" --stage "${stage}" --status "${status}"
        --decision "${decision}" --next-stage "${next}"
    )
    [[ -z "${artifact}" ]] || args+=(--artifact "${artifact}")
    "${ALGENGINE_PYTHON}" "${LEDGER_UPDATER}" "${args[@]}" >/dev/null
}

static_preflight() {
    local required=(
        "${ALGENGINE_PYTHON}" "${SIMENGINE_PYTHON}" "${CONFIG}"
        "${SOURCE_BUILDER}" "${TARGET_BUILDER}" "${TREATMENT_BUILDER}"
        "${AUDITOR}" "${SENTINEL_VERIFIER}" "${ASSEMBLER}"
        "${PILOT_VERIFIER}" "${COMBINER}" "${LEDGER_UPDATER}"
        "${V4_COMMON}" "${V4_MANAGER}" "${RUNNER_SCRIPT}" "${PROTOCOL}"
        "${LEDGER_TEMPLATE}" "${SIM_TEST}" "${PLANNING_HEAD}"
        "${SCENE_SELECTOR}" "${NAVFORMER}" "${NAVFORMER_CLIENT}"
        "${PREACTION_MANAGER}" "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}"
        "${BASE_ENV}" "${DEFAULT_RUNNER}" "${MERGE_RESULTS}"
        "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}"
        "$(checkpoint_manifest)" "${SOURCE_ROOT}/split_audit.json"
        "${SOURCE_ROOT}/token_split_map.json"
    )
    local path
    for path in "${required[@]}"; do
        require_file "${path}"
    done
    [[ -d "${ASSET_ROOT}" ]] || {
        echo "Missing asset root: ${ASSET_ROOT}" >&2
        exit 1
    }
    mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}"
}

hopper_preflight() {
    "${ALGENGINE_PYTHON}" - <<'PY'
import json
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8:
    raise RuntimeError(f"expected 8 visible GPUs, got {len(names)}")
if any(not any(x in name.upper() for x in ("H100", "H200")) for name in names):
    raise RuntimeError(f"requires Hopper, got {names}")
if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(8)):
    raise RuntimeError("requires sm_90")
if str(torch.__version__) != "2.0.1+cu118" or torch.version.cuda != "11.8":
    raise RuntimeError(f"environment drift: {torch.__version__}/{torch.version.cuda}")
print(json.dumps({
    "status": "PASS", "hardware_contract": "hopper8",
    "devices": names, "torch": torch.__version__,
}, sort_keys=True))
PY
    "${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py"         --extension "${WORLDENGINE_MMCV_EXTENSION}"         --expected-capability sm_90 --all-visible
    "${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py"         --extension "${WORLDENGINE_GSPLAT_EXTENSION}"         --expected-capability sm_90 --ray-workers 8
}

prepare_environment() {
    local collection_id="$1" source_split="$2" mode="$3" treatment_manifest="$4"
    local manifest checkpoint treatment_sha
    manifest="$(checkpoint_manifest)"
    checkpoint="$(json_value "${manifest}" checkpoint)"
    require_file "${checkpoint}"
    treatment_sha=none
    if [[ -n "${treatment_manifest}" ]]; then
        require_file "${treatment_manifest}"
        treatment_sha="$(sha256sum "${treatment_manifest}" | awk '{print $1}')"
    fi
    export DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED=0
    export DIFFUSIONDRIVE_V4_COLLECTION_ID="${collection_id}"
    export DIFFUSIONDRIVE_V4_SOURCE_SPLIT="${source_split}"
    export DIFFUSIONDRIVE_V4_INTERVENTION_MODE="${mode}"
    export DIFFUSIONDRIVE_V4_TREATMENT_MANIFEST_SHA256="${treatment_sha}"
    export DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256="$(json_value "${manifest}" checkpoint_sha256)"
    export DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256="$(sha256sum "${manifest}" | awk '{print $1}')"
    [[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256}" ]] || {
        echo "Checkpoint SHA256 drifted" >&2
        exit 1
    }
    export DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE=selector_v4_causal_cache_noise0
    export DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256="$({
        sha256sum "${SIM_TEST}" "${NAVFORMER_CLIENT}" "${V4_MANAGER}"             "${PREACTION_MANAGER}" "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}"             "${BASE_ENV}" "${DEFAULT_RUNNER}" "${V4_COMMON}"
    } | sha256sum | awk '{print $1}')"
    export DIFFUSIONDRIVE_ROLLOUT_CODE_SHA="$({
        sha256sum "${RUNNER_SCRIPT}" "${CONFIG}" "${SOURCE_BUILDER}"             "${TARGET_BUILDER}" "${TREATMENT_BUILDER}" "${AUDITOR}"             "${SENTINEL_VERIFIER}" "${ASSEMBLER}" "${PILOT_VERIFIER}"             "${COMBINER}" "${LEDGER_UPDATER}" "${V4_COMMON}" "${SIM_TEST}"             "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${NAVFORMER}"             "${NAVFORMER_CLIENT}" "${V4_MANAGER}" "${PREACTION_MANAGER}"             "${DYNAMIC_MANAGER}" "${SIDECAR_CONTRACT}" "${BASE_ENV}"             "${DEFAULT_RUNNER}" "${MERGE_RESULTS}"
    } | sha256sum | awk '{print $1}')"
    POLICY_MANIFEST="${manifest}"
    POLICY_CHECKPOINT="${checkpoint}"
    TREATMENT_SHA="${treatment_sha}"
}

verify_collection_audit() {
    local audit="$1" collection_id="$2" source_split="$3" mode="$4"
    "${ALGENGINE_PYTHON}" -c 'import json,os,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS"
assert r["method"]=="diffusiondrive_selector_v4_collection_audit_v1"
assert r["layout"]=="merged" and r["collection_id"]==sys.argv[2]
assert r["source_split"]==sys.argv[3] and r["intervention_mode"]==sys.argv[4]
assert r["behavior_policy_train_seed"]==0
assert r["code_sha"]==os.environ["DIFFUSIONDRIVE_ROLLOUT_CODE_SHA"]
assert r["rollout_implementation_sha256"]==os.environ["DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"]
assert r["closed_loop_outcome"] is not None'         "${audit}" "${collection_id}" "${source_split}" "${mode}"
}

run_collection() {
    local collection_id="$1" source_split="$2" mode="$3" scenario="$4"
    local treatment_manifest="${5:-}"
    require_file "${scenario}"
    prepare_environment "${collection_id}" "${source_split}" "${mode}" "${treatment_manifest}"
    local root="${RUN_ROOT}/collections/${collection_id}"
    local final_audit="${root}/collection_audit.json"
    local partial_audit="${root}/premerge_collection_audit.json"
    if [[ -f "${final_audit}" ]]; then
        verify_collection_audit "${final_audit}" "${collection_id}" "${source_split}" "${mode}"
        echo "SKIP verified V4 ${collection_id}: ${final_audit}"
        return 0
    fi
    if [[ -f "${partial_audit}" ]] && ! "${ALGENGINE_PYTHON}" -c 'import json,os,sys
r=json.load(open(sys.argv[1]))
ok=(r.get("status")=="PASS" and r.get("layout")=="split"
    and r.get("collection_id")==sys.argv[2]
    and r.get("code_sha")==os.environ["DIFFUSIONDRIVE_ROLLOUT_CODE_SHA"]
    and r.get("rollout_implementation_sha256")==os.environ["DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256"])
raise SystemExit(0 if ok else 1)' "${partial_audit}" "${collection_id}"; then
        echo "STALE_V4_PARTIAL_PROVENANCE: ${partial_audit}" >&2
        echo "Keep this run as evidence and restart with a new RUN_ID." >&2
        exit 1
    fi

    mkdir -p "${root}/logs"
    find "${root}" -name simulation_completed.flag -type f -delete
    "${ALGENGINE_PYTHON}" -c 'from mmcv import Config; import os,sys
c=Config.fromfile(sys.argv[1]); x=c.selector_rollout_contract
assert x.schema_version==9 and x.collection_id==sys.argv[2]
assert x.source_data_split==sys.argv[3] and x.intervention_mode==sys.argv[4]
assert not x.training_data_consumed and not x.development_consumed
assert not x.method_training_performed and not x.inference_uses_reward_or_q
assert c.model.planning_head.export_rollout_context
assert c.model.planning_head.online_reward is None
assert x.treatment_manifest_sha256==os.environ["DIFFUSIONDRIVE_V4_TREATMENT_MANIFEST_SHA256"]
_ = c.pretty_text'         "${CONFIG}" "${collection_id}" "${source_split}" "${mode}"

    local world_args=(
        debug_mode=True debug_scene_name=null
        data_file_path="${scenario}" asset_folder_path="${ASSET_ROOT}"
        output_dir="${root}/__WORKER_ID__/WE_output"
        job_name="selector_v4_${collection_id}"
        use_planner_actions=true ego_policy=env_input_policy
        ego_client=navformer_client ego_controller=log_play_controller
        ego_navigation=trajectory_navigation agent_policy=idm_policy
        agent_navigation=idm_navigation
        planner_data_path="${root}/__WORKER_ID__/plan_traj"
        planner_client_folder="${root}/__WORKER_ID__/frames"
        with_metric_manager=true with_dense_reward_manager=true
        diffusiondrive_v4_causal_cache=true
        diffusiondrive_candidate_sweep=false
        diffusiondrive_preaction_oracle=false
        diffusiondrive_dynamic_candidate_reward=false
        diffusiondrive_behavior_train_seed=0
        diffusiondrive_v4_intervention_mode="${mode}"
        diffusiondrive_v4_collection_id="${collection_id}"
        diffusiondrive_v4_treatment_manifest_sha256="${TREATMENT_SHA}"
        diffusiondrive_candidate_sidecar_path="${root}/__WORKER_ID__/diffusiondrive_candidate_sidecars"
        diffusiondrive_sidecar_timeout_s=300
        distributed_mode=SCENARIO_BASED worker=ray_distributed
        worker_id_prefix=split_ enable_resume=true
        completed_scenarios_dir="${root}/__WORKER_ID__/completed_scenarios"
    )
    if [[ -n "${treatment_manifest}" ]]; then
        world_args+=(diffusiondrive_v4_treatment_manifest="${treatment_manifest}")
    fi

    local we_pid="" failure=0 planner_pids=()
    cleanup_collection() {
        trap - EXIT INT TERM
        local pid
        for pid in "${planner_pids[@]:-}"; do
            kill "${pid}" 2>/dev/null || true
        done
        [[ -z "${we_pid}" ]] || kill "${we_pid}" 2>/dev/null || true
    }
    trap cleanup_collection EXIT INT TERM
    cd "${SIMENGINE_ROOT}"
    "${SIMENGINE_PYTHON}" worldengine/runner/run_simulation.py         "${world_args[@]}" >"${root}/logs/worldengine.log" 2>&1 &
    we_pid=$!
    sleep 30

    run_planner() {
        local split_id="$1" worker_root="${root}/split_$1"
        mkdir -p "${worker_root}/plan_traj" "${worker_root}/frames"             "${worker_root}/merged_ann_files"             "${worker_root}/diffusiondrive_candidate_sidecars"
        cd "${ALGENGINE_ROOT}"
        CUDA_VISIBLE_DEVICES="${split_id}" "${ALGENGINE_PYTHON}"             closed_loop/sim_test.py "${CONFIG}" "${POLICY_CHECKPOINT}"             --seed 0 --log-dir "${worker_root}"             --cfg-options             sim.monitored_folder="${worker_root}/frames"             sim.plan_save_path="${worker_root}/plan_traj"             sim.merged_ann_save_dir="${worker_root}/merged_ann_files"             sim.diffusiondrive_rollout_sidecar_path="${worker_root}/diffusiondrive_candidate_sidecars"             sim.clean_temp_files=False sim.clean_record_data=False             data_root="${worker_root}/WE_output/openscene_format/"             >"${root}/logs/planner_split${split_id}.log" 2>&1
    }

    local split_id
    for split_id in 0 1 2 3 4 5 6 7; do
        run_planner "${split_id}" &
        planner_pids+=("$!")
    done
    local pid
    for pid in "${planner_pids[@]}"; do
        wait "${pid}" || failure=1
    done
    wait "${we_pid}" || failure=1
    we_pid=""
    planner_pids=()
    trap - EXIT INT TERM
    if [[ "${failure}" -ne 0 ]]; then
        echo "FAIL V4 collection ${collection_id}; rerun same mode/RUN_ID to resume" >&2
        tail -80 "${root}/logs/worldengine.log" >&2 || true
        exit 1
    fi

    local audit_args=(
        --rollout-root "${root}" --scenario-file "${scenario}"
        --checkpoint-manifest "${POLICY_MANIFEST}"
        --collection-id "${collection_id}" --source-split "${source_split}"
        --mode "${mode}" --treatment-sha256 "${TREATMENT_SHA}"
        --expected-noise-namespace "${DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE}"
        --expected-implementation-sha256 "${DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256}"
        --expected-code-sha "${DIFFUSIONDRIVE_ROLLOUT_CODE_SHA}"
        --expected-workers 8
    )
    if [[ -n "${treatment_manifest}" ]]; then
        audit_args+=(--treatment-manifest "${treatment_manifest}")
    fi
    "${ALGENGINE_PYTHON}" "${AUDITOR}" "${audit_args[@]}"         --layout split --output "${partial_audit}"

    local merged_records="${root}/WE_output/openscene_format/diffusiondrive_v4_causal_records"
    if [[ -d "${merged_records}" ]]; then
        find "${merged_records}" -maxdepth 1 -type f -name '*_v4causal.pkl' -delete
    fi
    cd "${SIMENGINE_ROOT}"
    "${SIMENGINE_PYTHON}" scripts/merge_simulation_results.py         --test_path "${root}" --react_type R --num-splits 8
    local metrics="${root}/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
    require_file "${metrics}"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" "${audit_args[@]}"         --layout merged --metrics-csv "${metrics}" --output "${final_audit}"
    verify_collection_audit "${final_audit}" "${collection_id}" "${source_split}" "${mode}"
    echo "PASS V4 ${collection_id}: ${final_audit}"
}

prepare_source() {
    if [[ ! -f "${SOURCE_AUDIT}" ]]; then
        "${ALGENGINE_PYTHON}" "${SOURCE_BUILDER}"             --source-root "${SOURCE_ROOT}" --output-root "${SOURCE_OUT}"
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" -c         'import sys,v4_causal_cache_common as c
_,r=c.load_source_audit(sys.argv[1])
assert r["train_scenario_count"]==1024
assert r["validation_scenario_count"]==256
assert r["origin_log_overlap"]==0 and not r["test_consumed"]' "${SOURCE_AUDIT}"
    ledger_update source_preflight COMPLETE PROCEED_TO_BASELINE_A_B         "${SOURCE_AUDIT}" baseline_a_b
    echo "PASS V4 source: ${SOURCE_AUDIT}"
}

run_baselines() {
    prepare_source
    local train_scenarios validation_scenarios
    train_scenarios="$(json_value "${SOURCE_AUDIT}" train_scenario_file)"
    validation_scenarios="$(json_value "${SOURCE_AUDIT}" validation_scenario_file)"
    run_collection baseline_train_a train observe_only "${train_scenarios}"
    run_collection baseline_train_b train observe_only "${train_scenarios}"
    run_collection baseline_validation_a validation observe_only "${validation_scenarios}"
    run_collection baseline_validation_b validation observe_only "${validation_scenarios}"
    ledger_update baseline_a_b COMPLETE PROCEED_TO_TARGET_FREEZE         "${RUN_ROOT}/collections/baseline_train_a/collection_audit.json" target_freeze
}

freeze_targets() {
    prepare_source
    local audits=(
        "${RUN_ROOT}/collections/baseline_train_a/collection_audit.json"
        "${RUN_ROOT}/collections/baseline_train_b/collection_audit.json"
        "${RUN_ROOT}/collections/baseline_validation_a/collection_audit.json"
        "${RUN_ROOT}/collections/baseline_validation_b/collection_audit.json"
    )
    local path
    for path in "${audits[@]}"; do
        require_file "${path}"
    done
    if [[ ! -f "${TARGET_MANIFEST}" ]]; then
        "${ALGENGINE_PYTHON}" "${TARGET_BUILDER}"             --source-audit "${SOURCE_AUDIT}"             --train-baseline-a "${audits[0]}"             --train-baseline-b "${audits[1]}"             --validation-baseline-a "${audits[2]}"             --validation-baseline-b "${audits[3]}"             --protocol "${PROTOCOL}" --output-root "${TARGET_ROOT}"
    fi
    if [[ ! -f "${MASTER_MANIFEST}" ]]; then
        "${ALGENGINE_PYTHON}" "${TREATMENT_BUILDER}"             --target-manifest "${TARGET_MANIFEST}"             --protocol "${PROTOCOL}" --output-root "${MANIFEST_ROOT}"
    fi
    require_file "${MASTER_MANIFEST}"
    ledger_update target_freeze COMPLETE PROCEED_TO_SENTINEL         "${TARGET_MANIFEST}" sentinel
    echo "PASS frozen V4 targets/treatments: ${MASTER_MANIFEST}"
}

treatment_path() {
    json_value "${MASTER_MANIFEST}" "collections.$1.path"
}

run_sentinel() {
    freeze_targets
    local treatment scenario
    treatment="$(treatment_path sentinel_policy)"
    scenario="$(json_value "${treatment}" scenario_file)"
    run_collection sentinel_policy train one_shot_manifest         "${scenario}" "${treatment}"
    "${ALGENGINE_PYTHON}" "${SENTINEL_VERIFIER}"         --baseline-audit "${RUN_ROOT}/collections/baseline_train_a/collection_audit.json"         --sentinel-audit "${RUN_ROOT}/collections/sentinel_policy/collection_audit.json"         --treatment-manifest "${treatment}" --output "${SENTINEL_GATE}"
    ledger_update sentinel COMPLETE AUTHORIZE_V4_PILOT64_CAUSAL_COLLECTION         "${SENTINEL_GATE}" pilot64_causal_cache
}

verify_sentinel() {
    require_file "${SENTINEL_GATE}"
    "${ALGENGINE_PYTHON}" -c 'import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS"
assert r["decision"]=="AUTHORIZE_V4_PILOT64_CAUSAL_COLLECTION"' "${SENTINEL_GATE}"
}

run_arm() {
    local stage="$1" index="$2"
    [[ "${stage}" =~ ^(pilot64|expand192|dev64)$ ]] || {
        echo "V4 arm stage must be pilot64, expand192 or dev64" >&2
        exit 2
    }
    [[ "${index}" =~ ^[0-9]+$ ]] && (( index >= 0 && index < 20 )) || {
        echo "V4 arm index must be 0..19" >&2
        exit 2
    }
    freeze_targets
    verify_sentinel
    if [[ "${stage}" != pilot64 ]]; then
        verify_pilot_gate
    fi
    local collection_id treatment scenario source_split
    collection_id="$(printf '%s_arm_%02d' "${stage}" "${index}")"
    treatment="$(treatment_path "${collection_id}")"
    scenario="$(json_value "${treatment}" scenario_file)"
    source_split=train
    [[ "${stage}" == dev64 ]] && source_split=validation
    run_collection "${collection_id}" "${source_split}"         one_shot_manifest "${scenario}" "${treatment}"
}

assemble_stage() {
    local stage="$1"
    mkdir -p "${CACHE_ROOT}"
    "${ALGENGINE_PYTHON}" "${ASSEMBLER}"         --run-root "${RUN_ROOT}" --master-manifest "${MASTER_MANIFEST}"         --stage "${stage}" --output "${CACHE_ROOT}/${stage}_cache.pkl"         --audit-output "${CACHE_ROOT}/${stage}_cache_audit.json"
}

run_stage_arms() {
    local stage="$1" index
    for index in $(seq 0 19); do
        run_arm "${stage}" "${index}"
    done
}

run_pilot_audit() {
    verify_sentinel
    assemble_stage pilot64
    "${ALGENGINE_PYTHON}" "${PILOT_VERIFIER}"         --cache "${CACHE_ROOT}/pilot64_cache.pkl"         --cache-audit "${CACHE_ROOT}/pilot64_cache_audit.json"         --output "${PILOT_GATE}"
    ledger_update pilot64_causal_cache COMPLETE PROCEED_TO_INFORMATION_GATE         "${CACHE_ROOT}/pilot64_cache_audit.json" pilot_information_gate
    ledger_update pilot_information_gate COMPLETE AUTHORIZE_EXPANSION         "${PILOT_GATE}" expand192_causal_cache
}

verify_pilot_gate() {
    require_file "${PILOT_GATE}"
    "${ALGENGINE_PYTHON}" -c 'import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS"
assert r["decision"]=="AUTHORIZE_V4_EXPAND192_AND_SEALED_DEV64"' "${PILOT_GATE}"
}

run_pilot() {
    run_sentinel
    run_stage_arms pilot64
    run_pilot_audit
}

run_expand() {
    freeze_targets
    verify_sentinel
    verify_pilot_gate
    run_stage_arms expand192
    assemble_stage expand192
    "${ALGENGINE_PYTHON}" "${COMBINER}"         --pilot-cache "${CACHE_ROOT}/pilot64_cache.pkl"         --expand-cache "${CACHE_ROOT}/expand192_cache.pkl"         --output "${CACHE_ROOT}/train256_cache.pkl"         --audit-output "${CACHE_ROOT}/train256_cache_audit.json"
    ledger_update expand192_causal_cache COMPLETE TRAIN256_CAUSAL_CACHE_READY         "${CACHE_ROOT}/train256_cache_audit.json" sealed_dev64_causal_cache
}

run_dev() {
    freeze_targets
    verify_sentinel
    verify_pilot_gate
    run_stage_arms dev64
    assemble_stage dev64
    ledger_update sealed_dev64_causal_cache COMPLETE SEALED_WITHOUT_MODEL_EVALUATION         "${CACHE_ROOT}/dev64_cache_audit.json" astra_review
}

static_preflight
case "${MODE}" in
    preflight)
        hopper_preflight
        "${ALGENGINE_PYTHON}" -m py_compile             "${SOURCE_BUILDER}" "${TARGET_BUILDER}" "${TREATMENT_BUILDER}"             "${AUDITOR}" "${ASSEMBLER}" "${V4_MANAGER}"
        echo "PASS V4 preflight"
        ;;
    source)
        prepare_source
        ;;
    baseline)
        hopper_preflight
        run_baselines
        ;;
    freeze)
        freeze_targets
        ;;
    sentinel)
        hopper_preflight
        run_sentinel
        ;;
    pilot64)
        hopper_preflight
        run_pilot
        ;;
    pilot_audit)
        run_pilot_audit
        ;;
    expand192)
        hopper_preflight
        run_expand
        ;;
    dev64)
        hopper_preflight
        run_dev
        ;;
    arm)
        hopper_preflight
        run_arm "${ARG3}" "${ARG4}"
        ;;
    all)
        hopper_preflight
        prepare_source
        run_baselines
        freeze_targets
        run_sentinel
        run_stage_arms pilot64
        run_pilot_audit
        run_expand
        run_dev
        echo "PASS complete V4 causal cache: ${RUN_ROOT}"
        ;;
esac
