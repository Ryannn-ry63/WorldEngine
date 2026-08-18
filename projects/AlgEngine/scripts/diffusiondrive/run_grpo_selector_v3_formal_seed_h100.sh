#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: run_grpo_selector_v3_formal_seed_h100.sh SEED}"
if [[ ! "${SEED}" =~ ^[012]$ ]]; then
    echo "SEED must be 0, 1, or 2" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"

CACHE_ROOT="${V3_ROOT}/cache"
SELECTION="${V3_ROOT}/sweep/selection.json"
CERT_ROOT="${V3_ROOT}/certification"
REFERENCE_SUMMARY="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${SEED}/summary.json"
FORMAL_ROOT="${V3_ROOT}/formal"
FORMAL_MODEL="e2e_diffusiondrive_${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME}_s${SEED}"
LOG_DIR="${FORMAL_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL V3 formal seed=${SEED} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for file in "${SELECTION}" "${REFERENCE_SUMMARY}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
    "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh"; do
    [[ -f "${file}" ]] || { echo "Missing V3 formal input: ${file}" >&2; exit 1; }
done

readarray -t SELECTED_VALUES < <("${ALGENGINE_PYTHON}" -c '
import json, sys
row = json.load(open(sys.argv[1]))
if row.get("status") != "PASS" or row.get("certification_consumed"):
    raise SystemExit("invalid V3 selection")
s = row["selected"]
for key in ("temperature", "learning_rate", "kl_weight", "epoch"):
    print(s[key])
' "${SELECTION}")
TEMPERATURE="${SELECTED_VALUES[0]}"
LEARNING_RATE="${SELECTED_VALUES[1]}"
KL_WEIGHT="${SELECTED_VALUES[2]}"
EPOCH="${SELECTED_VALUES[3]}"

if [[ "${SEED}" == "0" ]]; then
    CHECKPOINT="${CERT_ROOT}/selected_checkpoint.pth"
    MANIFEST="${CERT_ROOT}/checkpoint_manifest.json"
    [[ -f "${CERT_ROOT}/report.json" ]] || { echo "V3 certification has not run" >&2; exit 1; }
else
    CURRENT_STAGE=replica_train
    REPLICA_ROOT="${FORMAL_ROOT}/replicas/seed${SEED}"
    TRAIN_ROOT="${REPLICA_ROOT}/train"
    CHECKPOINT="${REPLICA_ROOT}/checkpoint.pth"
    MANIFEST="${REPLICA_ROOT}/checkpoint_manifest.json"
    mkdir -p "${TRAIN_ROOT}"
    CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
        --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed3/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed4/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed5/cache.pt" \
        --output-dir "${TRAIN_ROOT}" --temperature "${TEMPERATURE}" \
        --learning-rate "${LEARNING_RATE}" --kl-weight "${KL_WEIGHT}" \
        --seed "${SEED}" --epochs "${EPOCH}" --checkpoint-epochs "${EPOCH}" \
        --batch-size 64 --device cuda --ablation full
    SELECTOR_STATE="${TRAIN_ROOT}/epoch_${EPOCH}_scene_selector.pt"
    SELECTOR_SHA="$(sha256sum "${SELECTOR_STATE}" | awk '{print $1}')"
    CURRENT_STAGE=replica_materialize
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
        --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
        --scene-selector-state "${SELECTOR_STATE}" \
        --expected-selector-sha256 "${SELECTOR_SHA}" \
        --output "${CHECKPOINT}" --manifest "${MANIFEST}"
fi

EXPECTED_SHA="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "${MANIFEST}")"
CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" --checkpoint "${CHECKPOINT}" \
    --output "$(dirname "${CHECKPOINT}")/checkpoint_audit.json"

CURRENT_STAGE=four_block_evaluation
export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="${TEMPERATURE}"
export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="${KL_WEIGHT}"
"${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
    "${CHECKPOINT}" "${EXPECTED_SHA}" "${FORMAL_MODEL}" "${SEED}" \
    "scene-conditioned selector-only exact-group GRPO V3; train seed=${SEED}; eval seed=${SEED}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive selector GRPO V3 formal seed ${SEED}"
echo "summary: ${FORMAL_ROOT}/formal_eval/${FORMAL_MODEL}/summary.json"
echo "persistent_log: ${LOG_FILE}"
