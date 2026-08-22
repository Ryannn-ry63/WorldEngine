#!/usr/bin/env bash
# One-H100 gate-conditioned GRPO smoke using frozen rare-rollout-v1 data.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ALGENGINE_PYTHON="/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python"
BASELINE="/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
DATA_ROOT="${SOURCE_ROOT}/data"
REAL_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_gate_conditioned_v1"
SMOKE_ROOT="${ROOT}/local_smoke/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
METHOD="scene_conditioned_gate_conditioned_group_grpo_v3_rare_rollout_v1"
mkdir -p "${SMOKE_ROOT}"
LOG_FILE="${SMOKE_ROOT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export PYTHONPATH="${WORLDENGINE_ROOT}/projects/AlgEngine:${WORLDENGINE_ROOT}/projects/SimEngine:${PYTHONPATH:-}"

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL one-H100 gate-conditioned smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${BASELINE}" "${DATA_ROOT}/manifest.json" \
    "${DATA_ROOT}/hard_pool.jsonl" "${DATA_ROOT}/synthetic_cache.pt" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_rollout.py"; do
    [[ -f "${path}" ]] || { echo "Missing gate-conditioned smoke input: ${path}" >&2; exit 1; }
done
"${ALGENGINE_PYTHON}" -c '
import hashlib,json,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert sha(pathlib.Path(row["hard_pool"]))==row["hard_pool_sha256"]
assert sha(pathlib.Path(row["synthetic_cache"]))==row["synthetic_cache_sha256"]
' "${DATA_ROOT}/manifest.json"

CURRENT_STAGE=optimizer_preflight
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py" --ablation full

CURRENT_STAGE=train
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_rollout.py" \
    --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt" \
    --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt" \
    --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt" \
    --synthetic-cache "${DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${DATA_ROOT}/manifest.json" \
    --hard-pool "${DATA_ROOT}/hard_pool.jsonl" \
    --output-dir "${SMOKE_ROOT}/train" \
    --temperature 1.0 --learning-rate 1e-4 --kl-weight 1e-3 \
    --seed 0 --epochs 1 --examples-per-cache-epoch 4 \
    --checkpoint-epochs 1 --batch-size 4 --device cuda \
    --objective gate_conditioned_pdm --method-name "${METHOD}" \
    --smoke-limit-hard-pool 2

CURRENT_STAGE=materialize
STATE="${SMOKE_ROOT}/train/epoch_1_scene_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" --expected-method "${METHOD}" \
    --release-name e2e_diffusiondrive_grpo_selector_v3_gate_conditioned_v1_smoke \
    --output "${SMOKE_ROOT}/checkpoint.pth" \
    --manifest "${SMOKE_ROOT}/checkpoint_manifest.json"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${BASELINE}" --checkpoint "${SMOKE_ROOT}/checkpoint.pth" \
    --output "${SMOKE_ROOT}/checkpoint_audit.json"

CURRENT_STAGE=final_audit
"${ALGENGINE_PYTHON}" -c '
import json,sys
train=json.load(open(sys.argv[1])); audit=json.load(open(sys.argv[2]))
assert train["status"]==audit["status"]=="PASS"
assert train["objective"]=="gate_conditioned_pdm"
assert train["fresh_selector_initialization"]=="exact_zero"
assert train["sampling"]["class_examples"]=={"common":6,"hard":6}
assert train["gate_conditioned_diagnostics"]["quality_active_groups"]>0
assert audit["changed_baseline_tensor_count"]==0
assert audit["scene_selector_tensor_count"]==54
' "${SMOKE_ROOT}/train/report.json" "${SMOKE_ROOT}/checkpoint_audit.json"

CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 DiffusionDrive V3 gate-conditioned GRPO smoke"
echo "output: ${SMOKE_ROOT}"
echo "persistent_log: ${LOG_FILE}"
