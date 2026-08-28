#!/usr/bin/env bash
# Resume-safe serial four-block evaluation for all three independently trained seeds.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_hopper_preflight 8

FORMAL_ROOT="${REASONER_FORMAL_ROOT:-${REASONER_EXPERIMENT_ROOT}/formal}"
MANIFEST="${1:-${FORMAL_ROOT}/train/relational_only_epoch64/formal_train_manifest.json}"
[[ -f "${MANIFEST}" ]] || { echo "Missing formal train manifest: ${MANIFEST}" >&2; exit 1; }

RUN_ROOT="${FORMAL_ROOT}/formal_eval_all"
mkdir -p "${RUN_ROOT}/logs"
LOG_FILE="${RUN_ROOT}/logs/$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS="${RUN_ROOT}/status.txt"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=validate_manifest
printf 'RUNNING stage=%s started_utc=%s\n' "${CURRENT_STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS}"; echo "FAIL all-seed evaluation stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

"${ALGENGINE_PYTHON}" - "${MANIFEST}" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]).expanduser().resolve(); row=json.load(path.open())
assert row["schema_version"]==1 and row["status"]=="PASS"
assert row["independent_training_seeds"]==[0,1,2]
assert set(row["models"])=={"0","1","2"}
assert row["label"]=="relational_only" and int(row["epoch"])==64
shas=[]
for seed in range(3):
    model=row["models"][str(seed)]; checkpoint=Path(model["checkpoint"]).resolve()
    assert checkpoint.is_file()
    digest=hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            digest.update(chunk)
    digest=digest.hexdigest()
    assert digest==model["checkpoint_sha256"] and digest not in shas
    shas.append(digest)
print("PASS immutable independent three-seed manifest")
PY

manifest_value() {
    local seed="$1" key="$2"
    "${ALGENGINE_PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]][sys.argv[3]])' \
        "${MANIFEST}" "${seed}" "${key}"
}

evaluation_ready() {
    local summary="$1" seed="$2" expected_sha="$3"
    [[ -f "${summary}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row["schema_version"]==1 and row["status"]=="PASS"
assert int(row["eval_seed"])==int(sys.argv[2])
assert row["checkpoint_sha256"]==sys.argv[3]
assert int(row["counts"]["openloop_navtest"])>=10000
for key in ("openloop_failures","closedloop_nr","closedloop_r"):
    assert int(row["counts"][key])>=250
' "${summary}" "${seed}" "${expected_sha}"
}

summaries=()
for seed in 0 1 2; do
    checkpoint="$(manifest_value "${seed}" checkpoint)"
    checkpoint_sha="$(manifest_value "${seed}" checkpoint_sha256)"
    model="e2e_diffusiondrive_trajectory_set_reasoner_relational_only_epoch64_s${seed}"
    result_root="${FORMAL_ROOT}/formal_eval/${model}"
    summary="${result_root}/summary.json"
    summaries+=("${summary}")
    CURRENT_STAGE="formal_eval_seed${seed}"
    if evaluation_ready "${summary}" "${seed}" "${checkpoint_sha}"; then
        echo "SKIP verified formal evaluation seed ${seed}: ${summary}"
        continue
    fi
    if [[ -e "${result_root}" ]]; then
        echo "Incomplete immutable evaluation root exists: ${result_root}" >&2
        echo "Refusing to overwrite it; inspect the persistent log before retrying." >&2
        exit 1
    fi
    "${SCRIPT_DIR}/run_trajectory_set_reasoner_formal_eval_8hopper.sh" \
        "${checkpoint}" "${checkpoint_sha}" "${model}" "${seed}" \
        "Relational trajectory-set reasoner; epoch64; independent train seed ${seed}; ungated"
    evaluation_ready "${summary}" "${seed}" "${checkpoint_sha}"
done

CURRENT_STAGE=collect
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/collect_trajectory_set_reasoner_formal_metrics.py" \
    --summary "${summaries[0]}" --summary "${summaries[1]}" --summary "${summaries[2]}" \
    --output "${FORMAL_ROOT}/relational_only_epoch64_candidate_metrics.json"

CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
trap - ERR
echo "PASS all three formal trajectory-set evaluations"
echo "metrics: ${FORMAL_ROOT}/relational_only_epoch64_candidate_metrics.json"
echo "persistent_log: ${LOG_FILE}"
