#!/usr/bin/env bash
# Stage II: only after Stage-I passes, run fixed A0/A1/A2 with equal budgets.
set -Eeuo pipefail

PROBE_REPORT="${1:?usage: $0 PROBE_GATE_JSON}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/interaction_selector_env.sh"
interaction_preflight
"${ALGENGINE_PYTHON}" - "${PROBE_REPORT}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["decision"]=="AUTHORIZE_A0_A1_A2" and row["all_gates_passed"]
assert row["development_or_certification_consumed"] is False
print("PASS train-only interaction probe authorization")
PY

RUN_ID="${INTERACTION_DEVELOPMENT_ID:-development_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${INTERACTION_EXPERIMENT_ROOT}/development/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "development root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/trials" "${ROOT}/logs"
LOG="${ROOT}/development.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=development_cache
trap 'rc=$?; echo "FAIL interaction development stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

for seed in 3 4 5; do
    interaction_extract_cache development "${seed}" 1254 \
        "${INTERACTION_TUNING_ROOT}/development/rare_common_union.yaml"
done

TRAINER="${SCRIPT_DIR}/train_trajectory_set_reasoner_grpo.py"
train_arm() {
    local gpu="$1" label="$2" architecture="$3"
    local output="${ROOT}/trials/${label}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
        "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed0/cache.pt" \
        --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed1/cache.pt" \
        --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed2/cache.pt" \
        --pair-manifest "${INTERACTION_TUNING_ROOT}/train/pairs.jsonl" \
        --rare-data-audit "${INTERACTION_TUNING_ROOT}/train/rare_data_audit.json" \
        --output-dir "${output}" --sampling-mode rare_balanced \
        --architecture "${architecture}" --ablation full \
        --method-name "interaction_selector_v1_${label}" --seed 0 \
        --temperature 1 --learning-rate 1e-4 --kl-weight 1e-3 \
        --epochs 16 --examples-per-cache-epoch 6339 \
        --checkpoint-epochs 16 --batch-size 64 --device cuda
}

CURRENT_STAGE=train_A0_A1_A2
pids=()
labels=(A0 A1 A2)
architectures=(scene_conditioned_v3 interaction_generic interaction_relation)
for gpu in 0 1 2; do
    (train_arm "${gpu}" "${labels[${gpu}]}" "${architectures[${gpu}]}") \
        > "${ROOT}/logs/${labels[${gpu}]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
[[ "${failed}" -eq 0 ]] || { echo "A0/A1/A2 training failed; inspect ${ROOT}/logs" >&2; exit 1; }

CURRENT_STAGE=budget_audit
"${ALGENGINE_PYTHON}" - "${ROOT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
expected={"A0":"scene_conditioned_v3","A1":"interaction_generic","A2":"interaction_relation"}
for label,architecture in expected.items():
    row=json.load(open(root/"trials"/label/"report.json"))
    assert row["status"]=="PASS" and row["selector_architecture"]==architecture
    assert row["epochs"]==16 and row["checkpoint_epochs"]==[16]
    assert row["temperature"]==1.0 and row["learning_rate"]==1e-4 and row["kl_weight"]==1e-3
    assert row["sampling"]["examples_per_cache_epoch"]==6339
    assert row["sampling"]["total_examples"]==304272
    assert row["sampling"]["total_optimizer_steps"]==4800
    assert row["sampling"]["class_examples"]=={"rare":152136,"common":152136}
print("PASS matched A0/A1/A2 16-epoch/4800-step budgets")
PY

CURRENT_STAGE=development_gate
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/evaluate_interaction_selector_development.py" \
    --cache "${INTERACTION_CACHE_ROOT}/tuning/development_seed3/cache.pt" \
    --cache "${INTERACTION_CACHE_ROOT}/tuning/development_seed4/cache.pt" \
    --cache "${INTERACTION_CACHE_ROOT}/tuning/development_seed5/cache.pt" \
    --state "A0=${ROOT}/trials/A0/epoch_16_scene_selector.pt" \
    --state "A1=${ROOT}/trials/A1/epoch_16_scene_selector.pt" \
    --state "A2=${ROOT}/trials/A2/epoch_16_scene_selector.pt" \
    --pair-manifest "${INTERACTION_TUNING_ROOT}/development/pairs.jsonl" \
    --rare-data-audit "${INTERACTION_TUNING_ROOT}/development/rare_data_audit.json" \
    --output "${ROOT}/development_gate.json" --batch-size 64 --device cuda

CURRENT_STAGE=complete
trap - ERR
echo "PASS fixed interaction A0/A1/A2 development study"
echo "gate: ${ROOT}/development_gate.json"
echo "log: ${LOG}"

