#!/usr/bin/env bash
# Fixed-budget 8-arm test for diffusion-set selector post-training.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"

reasoner_hopper_preflight 8

SOURCE_ROOT="${REASONER_SOURCE_WORLDENGINE_ROOT}"
CACHE_ROOT="${REASONER_CACHE_ROOT}/tuning"
PAIR_ROOT="${REASONER_TUNING_ROOT}"
V3_CHECKPOINT="${SOURCE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/sweep/trials/t1_lr1e-4_kl1e-3/epoch_64_scene_selector.pt"
V3_SHA256=d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133
MECHANISM_AUDIT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/diffusion_robust_group_v1/audit/development_mechanism.json"
TRAINER="${SCRIPT_DIR}/train_diffusion_robust_group_grpo.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_diffusion_robust_group_grpo.py"
SELECTOR="${SCRIPT_DIR}/select_diffusion_robust_group_development.py"
GRADIENT_AUDITOR="${SCRIPT_DIR}/audit_diffusion_set_gradient_conflict.py"

SEARCH_ID="${1:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
if ! [[ "${SEARCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid search id: ${SEARCH_ID}" >&2
    exit 2
fi
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/diffusion_set_selector_v1/search/${SEARCH_ID}"
[[ ! -e "${ROOT}" ]] || {
    echo "Refusing to reuse existing DRG search root: ${ROOT}" >&2
    exit 1
}
mkdir -p "${ROOT}/trials" "${ROOT}/logs"
LOG="${ROOT}/search.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL diffusion-set search stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

required=(
    "${ALGENGINE_PYTHON}"
    "${V3_CHECKPOINT}"
    "${MECHANISM_AUDIT}"
    "${TRAINER}"
    "${EVALUATOR}"
    "${SELECTOR}"
    "${GRADIENT_AUDITOR}"
    "${CACHE_ROOT}/train_seed0/cache.pt"
    "${CACHE_ROOT}/train_seed1/cache.pt"
    "${CACHE_ROOT}/train_seed2/cache.pt"
    "${CACHE_ROOT}/development_seed3/cache.pt"
    "${CACHE_ROOT}/development_seed4/cache.pt"
    "${CACHE_ROOT}/development_seed5/cache.pt"
    "${PAIR_ROOT}/train/pairs.jsonl"
    "${PAIR_ROOT}/train/rare_data_audit.json"
    "${PAIR_ROOT}/development/pairs.jsonl"
    "${PAIR_ROOT}/development/rare_data_audit.json"
)
for path in "${required[@]}"; do
    [[ -e "${path}" ]] || {
        echo "Missing diffusion-set dependency: ${path}" >&2
        exit 1
    }
done
[[ "$(sha256sum "${V3_CHECKPOINT}" | awk '{print $1}')" == "${V3_SHA256}" ]] || {
    echo "V3 anchor SHA256 mismatch" >&2
    exit 1
}
"${ALGENGINE_PYTHON}" - "${MECHANISM_AUDIT}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["decision"]=="AUTHORIZE_TOP1_DIFFUSION_ROBUST_GROUP_PROTOTYPE"
assert row["scientific_contract"]["reward_components_consumed"] is False
assert row["scientific_contract"]["certification_consumed"] is False
print("PASS diffusion-set mechanism authorization")
PY

CURRENT_STAGE=training_mechanism_audit
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
"${ALGENGINE_PYTHON}" "${GRADIENT_AUDITOR}" \
    --cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --cache "${CACHE_ROOT}/train_seed1/cache.pt" \
    --cache "${CACHE_ROOT}/train_seed2/cache.pt" \
    --expected-noise-seeds 0,1,2 \
    --pair-manifest "${PAIR_ROOT}/train/pairs.jsonl" \
    --rare-data-audit "${PAIR_ROOT}/train/rare_data_audit.json" \
    --v3-checkpoint "${V3_CHECKPOINT}" \
    --v3-checkpoint-sha256 "${V3_SHA256}" \
    --output "${ROOT}/training_mechanism_audit.json" \
    --tokens-per-stratum 16 \
    --seed 20260906 \
    --device cuda

labels=(
    repeat_grpo
    mean_soft
    mean_top1
    pure_softmin_top1
    bounded_relative_top1
    bounded_relative_soft
    bounded_relative_top1_shuffled
    bounded_absolute_top1
)
arms=(
    repeat_grpo
    mean_soft
    mean_top1
    pure_softmin_top1
    bounded_relative_top1
    bounded_relative_soft
    bounded_relative_top1_shuffled
    bounded_absolute_top1
)
taus=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

run_trial() {
    local gpu="$1"
    local label="$2"
    local arm="$3"
    local tau="$4"
    local trial="${ROOT}/trials/${label}"
    mkdir -p "${trial}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
        --pair-manifest "${PAIR_ROOT}/train/pairs.jsonl" \
        --rare-data-audit "${PAIR_ROOT}/train/rare_data_audit.json" \
        --v3-checkpoint "${V3_CHECKPOINT}" \
        --v3-checkpoint-sha256 "${V3_SHA256}" \
        --output-dir "${trial}" \
        --arm "${arm}" \
        --temperature 1 \
        --learning-rate 3e-5 \
        --kl-weight 1e-3 \
        --risk-temperature "${tau}" \
        --risk-mix 0.5 \
        --reward-scale 1 \
        --epochs 8 \
        --examples-per-epoch 6339 \
        --batch-size 64 \
        --checkpoint-epochs 8 \
        --anchor-logits-mode precompute \
        --require-full-coverage \
        --seed 0 \
        --device cuda

    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --checkpoint "${trial}/epoch_8_scene_selector.pt" \
        --cache "${CACHE_ROOT}/development_seed3/cache.pt" \
        --cache "${CACHE_ROOT}/development_seed4/cache.pt" \
        --cache "${CACHE_ROOT}/development_seed5/cache.pt" \
        --split development \
        --pair-manifest "${PAIR_ROOT}/development/pairs.jsonl" \
        --rare-data-audit "${PAIR_ROOT}/development/rare_data_audit.json" \
        --output "${trial}/development_evaluation.json" \
        --batch-size 64 \
        --bootstrap-repetitions 5000 \
        --seed 20260904 \
        --device cuda
}

CURRENT_STAGE=fixed_budget_trials
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    (
        run_trial \
            "${gpu}" \
            "${labels[${gpu}]}" \
            "${arms[${gpu}]}" \
            "${taus[${gpu}]}"
    ) > "${ROOT}/logs/${labels[${gpu}]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "One or more diffusion-set trials failed; inspect ${ROOT}/logs" >&2
    exit 1
fi

CURRENT_STAGE=development_gate
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
"${ALGENGINE_PYTHON}" "${SELECTOR}" \
    --trial-root "${ROOT}/trials" \
    --output "${ROOT}/development_gate.json" \
    --bootstrap-repetitions 10000 \
    --seed 20260905

CURRENT_STAGE=complete
trap - ERR
echo "PASS diffusion-set fixed-budget development search: ${ROOT}"
echo "gate: ${ROOT}/development_gate.json"
echo "logs: ${ROOT}/logs"
