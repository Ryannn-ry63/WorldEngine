#!/usr/bin/env bash
# One 8-H100 allocation: development -> certification -> formal seeds 0,1,2.

set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_rollout_v1_finish.py"
PIPELINE_ROOT="${ROOT}/finish_pipeline"
STATUS="${PIPELINE_ROOT}/status.txt"
LOG_DIR="${PIPELINE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CURRENT_STAGE=static_inputs
printf 'RUNNING attempt_id=%s code_sha=%s stage=%s started_utc=%s\n' \
    "${ATTEMPT_ID}" "${CODE_SHA}" "${CURRENT_STAGE}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
trap 'rc=$?; printf "FAIL attempt_id=%s code_sha=%s stage=%s exit=%s\n" "${ATTEMPT_ID}" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" > "${STATUS}"; echo "FAIL rollout-v1 finish attempt_id=${ATTEMPT_ID} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for file in "${AUDITOR}" \
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_development_h100.sh" \
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_certify_h100.sh" \
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_formal_seed_h100.sh"; do
    [[ -f "${file}" ]] || { echo "Missing finish-pipeline input: ${file}" >&2; exit 1; }
done

"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage inputs

CURRENT_STAGE=development
if "${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage development \
    >/dev/null 2>&1; then
    echo "SKIP completed and audited rollout-v1 development"
elif [[ -e "${ROOT}/development/selection.json" \
    || -e "${ROOT}/development/status.txt" \
    || -e "${ROOT}/development/trials" ]]; then
    echo "Partial or invalid immutable development output exists; refusing overwrite" >&2
    exit 1
else
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_development_h100.sh"
fi
"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage development

CURRENT_STAGE=certification
if "${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage certification \
    >/dev/null 2>&1; then
    echo "SKIP completed and audited one-shot rollout-v1 certification"
elif [[ -e "${ROOT}/certification" ]]; then
    echo "Partial or invalid immutable certification output exists; refusing overwrite" >&2
    exit 1
else
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_certify_h100.sh"
fi
"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage certification
"${ALGENGINE_PYTHON}" -c '
import json, sys
row = json.load(open(sys.argv[1]))
print("certification_scientific_gate=" + json.dumps(row["scientific_gate"], sort_keys=True))
' "${ROOT}/certification/report.json"

for seed in 0 1 2; do
    CURRENT_STAGE="formal_seed${seed}"
    model="e2e_diffusiondrive_grpo_selector_rollout_v1_s${seed}"
    summary="${ROOT}/formal/formal_eval/${model}/summary.json"
    if "${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
        --worldengine-root "${WORLDENGINE_ROOT}" --stage formal --seed "${seed}" \
        >/dev/null 2>&1; then
        echo "SKIP completed and audited rollout-v1 formal seed ${seed}"
        continue
    fi
    if [[ -e "${ROOT}/formal/replicas/seed${seed}" \
        || -e "${ROOT}/formal/formal_eval/${model}" \
        || -e "${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${model}" \
        || -e "${ROOT}/formal/status_seed${seed}.txt" ]]; then
        echo "Partial or invalid immutable formal seed ${seed} output exists; refusing overwrite" >&2
        echo "Expected complete summary: ${summary}" >&2
        exit 1
    fi
    "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_formal_seed_h100.sh" "${seed}"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
        --worldengine-root "${WORLDENGINE_ROOT}" --stage formal --seed "${seed}"
done

CURRENT_STAGE=complete_audit
"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage complete
CURRENT_STAGE=complete
printf 'PASS attempt_id=%s code_sha=%s completed_utc=%s\n' \
    "${ATTEMPT_ID}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUS}"
trap - ERR
echo "PASS DiffusionDrive selector rollout-v1 finish pipeline"
echo "status: ${STATUS}"
echo "persistent_log: ${LOG_FILE}"
