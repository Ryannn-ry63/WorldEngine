#!/usr/bin/env bash
# Five-arm architecture-first exact-GRPO study on homogeneous 8x H100/H200.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_hopper_preflight 8

RUN_ID="${REASONER_ARCHITECTURE_ID:-architecture_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${REASONER_EXPERIMENT_ROOT}/architecture/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "architecture root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/trials" "${ROOT}/logs"
LOG="${ROOT}/architecture.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL reasoner architecture stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

TRAINER="${SCRIPT_DIR}/train_trajectory_set_reasoner_grpo.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_trajectory_set_reasoner_grpo.py"
SELECTOR="${SCRIPT_DIR}/select_trajectory_set_reasoner_grpo.py"
CEILING="${SCRIPT_DIR}/summarize_selector_ceiling.py"
TRAIN_PAIR_ROOT="${REASONER_TUNING_ROOT}/train"
DEV_PAIR_ROOT="${REASONER_TUNING_ROOT}/development"

for path in \
    "${REASONER_CACHE_ROOT}/tuning/train_seed0/cache.pt" \
    "${REASONER_CACHE_ROOT}/tuning/train_seed1/cache.pt" \
    "${REASONER_CACHE_ROOT}/tuning/train_seed2/cache.pt" \
    "${REASONER_CACHE_ROOT}/tuning/development_seed3/cache.pt" \
    "${REASONER_CACHE_ROOT}/tuning/development_seed4/cache.pt" \
    "${REASONER_CACHE_ROOT}/tuning/development_seed5/cache.pt" \
    "${TRAIN_PAIR_ROOT}/pairs.jsonl" "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
    "${DEV_PAIR_ROOT}/pairs.jsonl" "${DEV_PAIR_ROOT}/rare_data_audit.json"; do
    [[ -f "${path}" ]] || { echo "missing architecture input: ${path}" >&2; exit 1; }
done

run_trial() {
    local gpu="$1" label="$2" architecture="$3" ablation="$4" num_set_layers="$5"
    local trial_root="${ROOT}/trials/${label}"
    mkdir -p "${trial_root}"
    local extra=(--architecture "${architecture}" --ablation "${ablation}")
    if [[ "${architecture}" == "scene_conditioned_v3" ]]; then
        extra+=(--num-set-layers "${num_set_layers}")
    fi
    if [[ "${label}" == "full" ]]; then extra+=(--formal-contract); fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${REASONER_CACHE_ROOT}/tuning/train_seed0/cache.pt" \
        --train-cache "${REASONER_CACHE_ROOT}/tuning/train_seed1/cache.pt" \
        --train-cache "${REASONER_CACHE_ROOT}/tuning/train_seed2/cache.pt" \
        --pair-manifest "${TRAIN_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
        --output-dir "${trial_root}" --sampling-mode rare_balanced --seed 0 \
        --temperature 1 --learning-rate 1e-4 --kl-weight 1e-3 \
        --epochs 64 --examples-per-cache-epoch 6339 \
        --checkpoint-epochs 16,32,64 --batch-size 64 --device cuda \
        "${extra[@]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --training-report "${trial_root}/report.json" \
        --cache "${REASONER_CACHE_ROOT}/tuning/development_seed3/cache.pt" \
        --cache "${REASONER_CACHE_ROOT}/tuning/development_seed4/cache.pt" \
        --cache "${REASONER_CACHE_ROOT}/tuning/development_seed5/cache.pt" \
        --split development --pair-manifest "${DEV_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${DEV_PAIR_ROOT}/rare_data_audit.json" \
        --output "${trial_root}/development_evaluation.json" \
        --device cuda --batch-size 64
}

CURRENT_STAGE=train_and_cache_development
labels=(capacity_v3 temporal_only relational_only no_scene_context full)
architectures=(scene_conditioned_v3 trajectory_set_reasoner trajectory_set_reasoner trajectory_set_reasoner trajectory_set_reasoner)
ablations=(full temporal_only relational_only no_scene_context full)
set_layers=(4 1 1 1 1)
pids=()
for gpu in 0 1 2 3 4; do
    (
        run_trial "${gpu}" "${labels[${gpu}]}" "${architectures[${gpu}]}" \
            "${ablations[${gpu}]}" "${set_layers[${gpu}]}"
    ) > "${ROOT}/logs/${labels[${gpu}]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
[[ "${failed}" -eq 0 ]] || { echo "architecture arm failed; inspect ${ROOT}/logs" >&2; exit 1; }

CURRENT_STAGE=budget_audit
"${ALGENGINE_PYTHON}" - "${ROOT}" "${labels[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for label in sys.argv[2:]:
    report=json.load(open(root/"trials"/label/"report.json"))
    evaluation=json.load(open(root/"trials"/label/"development_evaluation.json"))
    assert report["status"]==evaluation["status"]=="PASS"
    assert report["schema_version"]==4
    assert report["epochs"]==64 and report["checkpoint_epochs"]==[16,32,64]
    assert report["sampling"]["total_examples"]==1217088
    assert report["sampling"]["total_optimizer_steps"]==19200
    assert len(evaluation["checkpoints"])==3
print("PASS equal 64-epoch/19200-step architecture budgets")
PY

CURRENT_STAGE=development_shortlist
"${ALGENGINE_PYTHON}" "${SELECTOR}" \
    --trial-root "${ROOT}/trials" --expected-candidates 15 --shortlist-size 3 \
    --output "${ROOT}/shortlist.json"

CURRENT_STAGE=ceiling
ceiling_args=()
for label in "${labels[@]}"; do
    ceiling_args+=(--state "${label}=${ROOT}/trials/${label}/epoch_64_scene_selector.pt")
done
CUDA_VISIBLE_DEVICES=5 "${ALGENGINE_PYTHON}" "${CEILING}" \
    --cache "${REASONER_CACHE_ROOT}/tuning/development_seed3/cache.pt" \
    --cache "${REASONER_CACHE_ROOT}/tuning/development_seed4/cache.pt" \
    --cache "${REASONER_CACHE_ROOT}/tuning/development_seed5/cache.pt" \
    --split development --pair-manifest "${DEV_PAIR_ROOT}/pairs.jsonl" \
    --rare-data-audit "${DEV_PAIR_ROOT}/rare_data_audit.json" \
    --output "${ROOT}/selector_ceiling.json" --device cuda --batch-size 64 \
    "${ceiling_args[@]}"

CURRENT_STAGE=parameter_audit
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" "${ALGENGINE_PYTHON}" - "${ROOT}" "${labels[@]}" <<'PY'
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]); rows={}
for label in sys.argv[2:]:
    payload=torch.load(root/"trials"/label/"epoch_64_scene_selector.pt",map_location="cpu")
    rows[label]={"architecture":payload["selector_architecture"],"ablation":payload["ablation"],"trainable_parameters":sum(value.numel() for value in payload["scene_selector_state"].values())}
output={"schema_version":1,"status":"PASS","methods":rows}
(root/"parameter_counts.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
print(json.dumps(output,sort_keys=True))
PY

CURRENT_STAGE=complete
trap - ERR
echo "PASS trajectory-set architecture-first study"
echo "shortlist: ${ROOT}/shortlist.json"
echo "ceiling: ${ROOT}/selector_ceiling.json"
echo "log: ${LOG}"
echo "NOTE: final promotion still requires paired three-seed closed-loop evaluation."
