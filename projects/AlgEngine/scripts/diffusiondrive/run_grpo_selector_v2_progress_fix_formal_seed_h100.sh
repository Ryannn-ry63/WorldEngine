#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: $0 SEED}"
if [[ ! "${SEED}" =~ ^[012]$ ]]; then
    echo "SEED must be 0, 1, or 2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT="${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT:-8}"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
REWARD_CONTRACT="navsim_pairwise_raw_progress_then_candidate_gate_v1"
V2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1"
SELECTION="${V2_ROOT}/selection/selection.json"
REFERENCE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
REFERENCE_SUMMARY="${REFERENCE_ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s${SEED}/summary.json"
FORMAL_ROOT="${V2_ROOT}/formal"
FORMAL_MODEL="e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_s${SEED}"
CURRENT_SUMMARY="${FORMAL_ROOT}/formal_eval/${FORMAL_MODEL}/summary.json"
LOG_ROOT="${FORMAL_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL corrected V2 formal seed=${SEED} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for required in "${SELECTION}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    "${DIFFUSIONDRIVE_GRPO_CONFIG}" "${REFERENCE_SUMMARY}" \
    "${SCRIPT_DIR}/materialize_grpo_selector_v2_replica.py" \
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${DIFFUSIONDRIVE_GRPO_BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2
    exit 1
}

mapfile -t SELECTED_VALUES < <("${ALGENGINE_PYTHON}" -c '
import json, sys
selection=json.load(open(sys.argv[1])); reference=json.load(open(sys.argv[2])); seed=int(sys.argv[3])
if selection.get("status") != "PASS": raise SystemExit("selection pipeline did not pass")
if selection.get("reward_contract") != sys.argv[5]: raise SystemExit("reward contract drifted")
if selection.get("baseline_sha256") != sys.argv[4]: raise SystemExit("baseline drifted")
if reference.get("status") != "PASS" or int(reference.get("eval_seed", -1)) != seed: raise SystemExit("paired reference drifted")
if reference.get("checkpoint_sha256") != sys.argv[4]: raise SystemExit("reference checkpoint drifted")
s=selection["selected"]
for key in ("temperature", "learning_rate", "kl_weight", "epoch"):
    print(s[key])
print(selection["selection_mode"])
' "${SELECTION}" "${REFERENCE_SUMMARY}" "${SEED}" "${BASELINE_SHA256}" "${REWARD_CONTRACT}")
TEMPERATURE="${SELECTED_VALUES[0]}"
LEARNING_RATE="${SELECTED_VALUES[1]}"
KL_WEIGHT="${SELECTED_VALUES[2]}"
EPOCH="${SELECTED_VALUES[3]}"
SELECTION_MODE="${SELECTED_VALUES[4]}"
REWARD_FILE="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
REWARD_SHA="$(sha256sum "${REWARD_FILE}" | awk '{print $1}')"

if [[ "${SEED}" == "0" ]]; then
    CHECKPOINT="${V2_ROOT}/selection/selected_checkpoint.pth"
    EXPECTED_SHA256="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint_sha256"])' "${SELECTION}")"
    AUDIT="${V2_ROOT}/selection/selected_checkpoint_audit.json"
else
    CURRENT_STAGE=replica_train
    CACHE_ROOT="${V2_ROOT}/cache"
    TRAIN_CACHE="${CACHE_ROOT}/train_seed0/cache.pt"
    CALIBRATION_CACHES=(
        "${CACHE_ROOT}/calibration_seed0/cache.pt"
        "${CACHE_ROOT}/calibration_seed1/cache.pt"
        "${CACHE_ROOT}/calibration_seed2/cache.pt"
    )
    REPLICA_ROOT="${V2_ROOT}/formal_replicas/seed${SEED}"
    TRAIN_ROOT="${REPLICA_ROOT}/train"
    REPORT="${TRAIN_ROOT}/report.json"
    SELECTOR_STATE="${TRAIN_ROOT}/epoch_${EPOCH}_selector.pt"
    CHECKPOINT="${REPLICA_ROOT}/checkpoint.pth"
    MANIFEST="${REPLICA_ROOT}/checkpoint_manifest.json"
    AUDIT="${REPLICA_ROOT}/checkpoint_audit.json"
    mkdir -p "${TRAIN_ROOT}"
    if [[ -s "${REPORT}" ]] && "${ALGENGINE_PYTHON}" -c '
import json, sys
p=json.load(open(sys.argv[1])); epoch=int(float(sys.argv[7]))
ok=(p.get("status")=="PASS" and p.get("method")=="exact_group_grpo" and p.get("temperature")==float(sys.argv[2]) and p.get("learning_rate")==float(sys.argv[3]) and p.get("kl_weight")==float(sys.argv[4]) and p.get("train_seed")==int(sys.argv[5]) and p.get("reward_contract")==sys.argv[6] and p.get("epochs")==epoch and p.get("checkpoint_epochs")==[epoch])
raise SystemExit(0 if ok else 1)
' "${REPORT}" "${TEMPERATURE}" "${LEARNING_RATE}" "${KL_WEIGHT}" "${SEED}" "${REWARD_CONTRACT}" "${EPOCH}"; then
        echo "REUSE PASS corrected V2 formal replica seed ${SEED}"
    else
        CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
            "${SCRIPT_DIR}/train_grpo_selector_v2_cached.py" \
            --train-cache "${TRAIN_CACHE}" \
            --calibration-cache "${CALIBRATION_CACHES[0]}" \
            --calibration-cache "${CALIBRATION_CACHES[1]}" \
            --calibration-cache "${CALIBRATION_CACHES[2]}" \
            --output-dir "${TRAIN_ROOT}" \
            --temperature "${TEMPERATURE}" \
            --learning-rate "${LEARNING_RATE}" \
            --kl-weight "${KL_WEIGHT}" \
            --seed "${SEED}" --epochs "${EPOCH}" \
            --checkpoint-epochs "${EPOCH}" --device cuda
    fi

    CURRENT_STAGE=replica_materialize
    SELECTOR_SHA="$("${ALGENGINE_PYTHON}" -c '
import json, sys
p=json.load(open(sys.argv[1])); epoch=int(float(sys.argv[2])); rows=[r for r in p["checkpoints"] if int(r["epoch"])==epoch]
if len(rows)!=1: raise SystemExit("selected epoch state is missing or ambiguous")
print(rows[0]["selector_state_sha256"])
' "${REPORT}" "${EPOCH}")"
    if [[ -s "${MANIFEST}" && -f "${CHECKPOINT}" ]] && "${ALGENGINE_PYTHON}" -c '
import hashlib, json, pathlib, sys
p=json.load(open(sys.argv[1])); checkpoint=pathlib.Path(sys.argv[2])
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
ok=(p.get("status")=="PASS" and p.get("checkpoint_sha256")==digest and p.get("reward_implementation_sha256")==sys.argv[3])
raise SystemExit(0 if ok else 1)
' "${MANIFEST}" "${CHECKPOINT}" "${REWARD_SHA}"; then
        echo "REUSE PASS corrected V2 materialized replica seed ${SEED}"
    else
        "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v2_replica.py" \
            --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
            --selector-state "${SELECTOR_STATE}" \
            --expected-selector-sha256 "${SELECTOR_SHA}" \
            --train-seed "${SEED}" --temperature "${TEMPERATURE}" \
            --learning-rate "${LEARNING_RATE}" --kl-weight "${KL_WEIGHT}" \
            --epoch "${EPOCH}" \
            --expected-reward-contract "${REWARD_CONTRACT}" \
            --expected-reward-sha256 "${REWARD_SHA}" \
            --experiment-method e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_replica \
            --output "${CHECKPOINT}" --manifest "${MANIFEST}"
    fi
    EXPECTED_SHA256="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "${MANIFEST}")"
fi

CURRENT_STAGE=checkpoint_audit
[[ -f "${CHECKPOINT}" ]] || { echo "Missing formal checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]] || {
    echo "Formal checkpoint SHA256 mismatch" >&2
    exit 1
}
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --checkpoint "${CHECKPOINT}" --output "${AUDIT}"

CURRENT_STAGE=four_block_evaluation
export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="${TEMPERATURE}"
export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="${KL_WEIGHT}"
if [[ -s "${CURRENT_SUMMARY}" ]] && "${ALGENGINE_PYTHON}" -c '
import json, sys
p=json.load(open(sys.argv[1])); ok=p.get("status")=="PASS" and int(p.get("eval_seed",-1))==int(sys.argv[2]) and p.get("checkpoint_sha256")==sys.argv[3]
raise SystemExit(0 if ok else 1)
' "${CURRENT_SUMMARY}" "${SEED}" "${EXPECTED_SHA256}"; then
    echo "REUSE PASS corrected V2 four-block evaluation seed ${SEED}"
else
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
        "${CHECKPOINT}" "${EXPECTED_SHA256}" "${FORMAL_MODEL}" "${SEED}" \
        "selector-only corrected-reward exact-group GRPO V2; T=${TEMPERATURE}; lr=${LEARNING_RATE}; KL=${KL_WEIGHT}; epoch=${EPOCH}; selection=${SELECTION_MODE}; train/eval seed=${SEED}"
fi

CURRENT_STAGE=complete
trap - ERR
echo "PASS corrected DiffusionDrive selector GRPO V2 formal seed ${SEED}"
echo "summary: ${CURRENT_SUMMARY}"
echo "persistent_log: ${LOG_FILE}"
