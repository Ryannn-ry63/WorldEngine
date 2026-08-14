#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: run_grpo_selector_rollout_v1_formal_seed_h100.sh SEED}"
if [[ ! "${SEED}" =~ ^[012]$ ]]; then
    echo "SEED must be 0, 1, or 2" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
SELECTION="${ROOT}/development/selection.json"
TRIAL_ROOT="${ROOT}/development/trials"
REFERENCE_SUMMARY="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${SEED}/summary.json"
FORMAL_ROOT="${ROOT}/formal"
FORMAL_MODEL="e2e_diffusiondrive_grpo_selector_rollout_v1_s${SEED}"
METHOD=scene_conditioned_exact_group_grpo_rollout_v1
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_rollout_v1_finish.py"
LOG_DIR="${FORMAL_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
STATUS="${FORMAL_ROOT}/status_seed${SEED}.txt"
trap 'rc=$?; echo "FAIL stage=${CURRENT_STAGE} exit=${rc}" > "${STATUS}"; echo "FAIL rollout-v1 formal seed=${SEED} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for file in "${SELECTION}" "${REFERENCE_SUMMARY}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" "${AUDITOR}"; do
    [[ -f "${file}" ]] || { echo "Missing rollout-v1 formal input: ${file}" >&2; exit 1; }
done
"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage certification

readarray -t SELECTED_VALUES < <("${ALGENGINE_PYTHON}" -c '
import json, sys
row = json.load(open(sys.argv[1]))
if row.get("status") != "PASS" or row.get("method") != sys.argv[2]:
    raise SystemExit("invalid rollout-v1 selection")
selected = row["selected"]
for key in ("temperature", "kl_weight", "epoch"):
    print(selected[key])
' "${SELECTION}" "${METHOD}")
TEMPERATURE="${SELECTED_VALUES[0]}"
KL_WEIGHT="${SELECTED_VALUES[1]}"
EPOCH="${SELECTED_VALUES[2]}"

REPLICA_ROOT="${FORMAL_ROOT}/replicas/seed${SEED}"
CHECKPOINT="${REPLICA_ROOT}/checkpoint.pth"
MANIFEST="${REPLICA_ROOT}/checkpoint_manifest.json"
SELECTOR_STATE="${TRIAL_ROOT}/optimizer_seed${SEED}/epoch_${EPOCH}_scene_selector.pt"
[[ ! -e "${REPLICA_ROOT}" ]] || { echo "Immutable replica exists: ${REPLICA_ROOT}" >&2; exit 1; }
[[ -f "${SELECTOR_STATE}" ]] || { echo "Missing development replica state: ${SELECTOR_STATE}" >&2; exit 1; }
SELECTOR_SHA="$(sha256sum "${SELECTOR_STATE}" | awk '{print $1}')"
CURRENT_STAGE=replica_materialize
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --scene-selector-state "${SELECTOR_STATE}" \
    --expected-selector-sha256 "${SELECTOR_SHA}" \
    --expected-method "${METHOD}" \
    --release-name e2e_diffusiondrive_grpo_selector_rollout_v1 \
    --output "${CHECKPOINT}" --manifest "${MANIFEST}"

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
    "same V3 selector-only exact-group GRPO trained on base-policy closed-loop rollout states; train seed=${SEED}; eval seed=${SEED}"

CURRENT_STAGE=final_audit
"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage formal --seed "${SEED}"
CURRENT_STAGE=complete
echo "PASS" > "${STATUS}"
trap - ERR
echo "PASS DiffusionDrive selector rollout-v1 formal seed ${SEED}"
echo "summary: ${FORMAL_ROOT}/formal_eval/${FORMAL_MODEL}/summary.json"
echo "persistent_log: ${LOG_FILE}"
