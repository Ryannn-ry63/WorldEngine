#!/usr/bin/env bash
# One-H100 BWM collection -> mixture -> train -> materialize smoke.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ALGENGINE_PYTHON="/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python"
BASELINE="/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1"
BASE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
REAL_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
SCENARIO_AUDIT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_bwm_v1/scenarios/bwm_scenario_audit.json"
SMOKE_ROOT="${ROOT}/local_smoke/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
COLLECTION_ROOT="${SMOKE_ROOT}/collection/smoke/lane0"
METHOD="scene_conditioned_exact_group_grpo_v3_rare_rollout_bwm_v1"
mkdir -p "${SMOKE_ROOT}"
LOG_FILE="${SMOKE_ROOT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export PYTHONPATH="${WORLDENGINE_ROOT}/projects/AlgEngine:${WORLDENGINE_ROOT}/projects/SimEngine:${PYTHONPATH:-}"

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL one-H100 BWM smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
"${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_prepare_local.sh"

CURRENT_STAGE=online_collection
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
export DIFFUSIONDRIVE_ROLLOUT_GPU_COUNT=1
export DIFFUSIONDRIVE_RARE_ROLLOUT_COLLECTION_ROOT="${SMOKE_ROOT}"
"${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_bwm_collect_h100.sh" 0 8 smoke

CURRENT_STAGE=build_mixture
"${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/build_grpo_selector_v3_rare_rollout_bwm_data.py" \
    --base-manifest "${BASE_ROOT}/data/manifest.json" \
    --scenario-audit "${SCENARIO_AUDIT}" \
    --lane-root "${COLLECTION_ROOT}" \
    --lane-audit "${COLLECTION_ROOT}/collection_audit.json" \
    --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt" \
    --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt" \
    --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt" \
    --output-dir "${SMOKE_ROOT}/data" --minimum-new-synthetic 1 --smoke

CURRENT_STAGE=optimizer_preflight
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py" --ablation full

CURRENT_STAGE=train
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_rollout.py" \
    --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt" \
    --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt" \
    --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt" \
    --synthetic-cache "${SMOKE_ROOT}/data/synthetic_cache.pt" \
    --data-manifest "${SMOKE_ROOT}/data/manifest.json" \
    --hard-pool "${SMOKE_ROOT}/data/hard_pool.jsonl" \
    --output-dir "${SMOKE_ROOT}/train" \
    --temperature 1.0 --learning-rate 1e-4 --kl-weight 1e-3 \
    --seed 0 --epochs 1 --examples-per-cache-epoch 4 \
    --checkpoint-epochs 1 --batch-size 4 --device cuda \
    --method-name "${METHOD}" \
    --expected-data-method diffusiondrive_v3_rare_rollout_bwm_mixture_v1 \
    --smoke-limit-hard-pool 2 --smoke-require-bwm

CURRENT_STAGE=materialize
STATE="${SMOKE_ROOT}/train/epoch_1_scene_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" --expected-method "${METHOD}" \
    --release-name e2e_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_v1_smoke \
    --output "${SMOKE_ROOT}/checkpoint.pth" \
    --manifest "${SMOKE_ROOT}/checkpoint_manifest.json"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${BASELINE}" --checkpoint "${SMOKE_ROOT}/checkpoint.pth" \
    --output "${SMOKE_ROOT}/checkpoint_audit.json"

CURRENT_STAGE=final_audit
"${ALGENGINE_PYTHON}" -c '
import json,sys
data=json.load(open(sys.argv[1])); train=json.load(open(sys.argv[2])); audit=json.load(open(sys.argv[3]))
assert data["status"]==train["status"]==audit["status"]=="PASS"
assert data["new_bwm_filtered_synthetic_records"]>0
assert train["source_kind"]=="diffusiondrive_v3_rare_rollout_bwm_mixture_v1"
assert train["fresh_selector_initialization"]=="exact_zero"
assert train["sampling"]["class_examples"]=={"common":6,"hard":6}
assert audit["changed_baseline_tensor_count"]==0
assert audit["scene_selector_tensor_count"]==54
' "${SMOKE_ROOT}/data/manifest.json" "${SMOKE_ROOT}/train/report.json" \
    "${SMOKE_ROOT}/checkpoint_audit.json"

CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 DiffusionDrive V3 rare-rollout BWM smoke"
echo "output: ${SMOKE_ROOT}"
echo "persistent_log: ${LOG_FILE}"
