#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: run_grpo_selector_v2_formal_seed_h100.sh SEED}"
if [[ ! "${SEED}" =~ ^[012]$ ]]; then
    echo "SEED must be 0, 1, or 2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
TEMPERATURE="1"
LEARNING_RATE="1e-3"
KL_WEIGHT="1e-3"
EPOCH="32"
V2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2"
SELECTION="${V2_ROOT}/selection/selection.json"
REFERENCE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
REFERENCE_SUMMARY="${REFERENCE_ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s${SEED}/summary.json"
FORMAL_ROOT="${V2_ROOT}/formal"
FORMAL_MODEL="e2e_diffusiondrive_grpo_selector_v2_s${SEED}"
CURRENT_SUMMARY="${FORMAL_ROOT}/formal_eval/${FORMAL_MODEL}/summary.json"
LOG_ROOT="${V2_ROOT}/formal/logs"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL V2 formal seed=${SEED} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

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

"${ALGENGINE_PYTHON}" -c '
import json, sys
selection = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
seed = int(sys.argv[3])
selected = selection.get("selected", {})
expected = {
    "temperature": 1.0,
    "learning_rate": 1e-3,
    "kl_weight": 1e-3,
    "epoch": 32,
    "train_seed": 0,
}
if selection.get("status") != "PASS":
    raise SystemExit("V2 selection gate did not pass")
for key, value in expected.items():
    if selected.get(key) != value:
        raise SystemExit(f"selected {key} drifted")
if reference.get("status") != "PASS":
    raise SystemExit("reference summary did not pass")
if int(reference.get("eval_seed", -1)) != seed:
    raise SystemExit("reference eval seed drifted")
if reference.get("checkpoint_sha256") != sys.argv[4]:
    raise SystemExit("reference checkpoint SHA drifted")
' "${SELECTION}" "${REFERENCE_SUMMARY}" "${SEED}" "${BASELINE_SHA256}"

if [[ "${SEED}" == "0" ]]; then
    CHECKPOINT="${V2_ROOT}/selection/selected_checkpoint.pth"
    EXPECTED_SHA256="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint_sha256"])' "${SELECTION}")"
    AUDIT="${V2_ROOT}/selection/selected_checkpoint_audit.json"
else
    CURRENT_STAGE=replica_train
    CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_diagnostic_v1/cache"
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
row = json.load(open(sys.argv[1]))
expected = ("PASS", "exact_group_grpo", 1.0, 1e-3, 1e-3, int(sys.argv[2]), 32, [32])
actual = (row.get("status"), row.get("method"), row.get("temperature"), row.get("learning_rate"), row.get("kl_weight"), row.get("train_seed"), row.get("epochs"), row.get("checkpoint_epochs"))
raise SystemExit(0 if actual == expected else 1)
' "${REPORT}" "${SEED}"; then
        echo "REUSE PASS V2 formal replica training seed ${SEED}"
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
row = json.load(open(sys.argv[1]))
matches = [item for item in row["checkpoints"] if int(item["epoch"]) == 32]
if len(matches) != 1:
    raise SystemExit("expected exactly one epoch-32 selector state")
print(matches[0]["selector_state_sha256"])
' "${REPORT}")"
    if [[ -s "${MANIFEST}" && -f "${CHECKPOINT}" ]] && "${ALGENGINE_PYTHON}" -c '
import hashlib, json, pathlib, sys
manifest = json.load(open(sys.argv[1]))
checkpoint = pathlib.Path(sys.argv[2])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
raise SystemExit(0 if manifest.get("status") == "PASS" and manifest.get("checkpoint_sha256") == digest else 1)
' "${MANIFEST}" "${CHECKPOINT}"; then
        echo "REUSE PASS V2 materialized replica seed ${SEED}"
    else
        "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v2_replica.py" \
            --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
            --selector-state "${SELECTOR_STATE}" \
            --expected-selector-sha256 "${SELECTOR_SHA}" \
            --train-seed "${SEED}" --output "${CHECKPOINT}" \
            --manifest "${MANIFEST}"
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
row = json.load(open(sys.argv[1]))
ok = row.get("status") == "PASS" and int(row.get("eval_seed", -1)) == int(sys.argv[2]) and row.get("checkpoint_sha256") == sys.argv[3]
raise SystemExit(0 if ok else 1)
' "${CURRENT_SUMMARY}" "${SEED}" "${EXPECTED_SHA256}"; then
    echo "REUSE PASS V2 four-block evaluation seed ${SEED}"
else
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
        "${CHECKPOINT}" "${EXPECTED_SHA256}" "${FORMAL_MODEL}" "${SEED}" \
        "selector-only exact-group GRPO V2; T=1; lr=1e-3; KL=1e-3; epoch=32; train/eval seed=${SEED}"
fi

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive selector GRPO V2 formal seed ${SEED}"
echo "summary: ${CURRENT_SUMMARY}"
echo "persistent_log: ${LOG_FILE}"
