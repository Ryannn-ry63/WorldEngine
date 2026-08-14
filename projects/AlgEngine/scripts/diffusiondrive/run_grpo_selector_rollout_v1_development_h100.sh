#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
CACHE_ROOT="${ROOT}/cache"
TRIAL_ROOT="${ROOT}/development/trials"
SELECTION="${ROOT}/development/selection.json"
STATUS="${ROOT}/development/status.txt"
LOG_ROOT="${ROOT}/development/logs"
METHOD=scene_conditioned_exact_group_grpo_rollout_v1
mkdir -p "${TRIAL_ROOT}" "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; echo "FAIL exit=${rc}" > "${STATUS}"; echo "FAIL rollout-v1 development line=${LINENO} command=${BASH_COMMAND}" >&2' ERR

for seed in {0..8}; do
    if (( seed <= 2 )); then split=train; elif (( seed <= 5 )); then split=development; else continue; fi
    [[ -f "${CACHE_ROOT}/${split}_seed${seed}/cache.pt" ]] || {
        echo "Missing rollout cache ${split}_seed${seed}" >&2; exit 1;
    }
done
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py"

pids=()
for optimizer_seed in 0 1 2; do
    output="${TRIAL_ROOT}/optimizer_seed${optimizer_seed}"
    [[ ! -e "${output}" ]] || { echo "Immutable trial exists: ${output}" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="${optimizer_seed}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
        --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
        --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed3/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed4/cache.pt" \
        --development-cache "${CACHE_ROOT}/development_seed5/cache.pt" \
        --output-dir "${output}" --temperature 1.0 \
        --learning-rate 1e-4 --kl-weight 0 --seed "${optimizer_seed}" \
        --epochs 16 --checkpoint-epochs 1,2,4,8,16 --batch-size 64 \
        --device cuda --ablation full --method-name "${METHOD}" \
        > "${output}.stdout.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "At least one rollout-v1 development optimizer trial failed" >&2
    exit 1
fi

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_grpo_selector_v3.py" \
    --trial-root "${TRIAL_ROOT}" --output "${SELECTION}" \
    --method-name "${METHOD}" --minimum-worst-seed-gain 0 \
    --minimum-delta-no-at-fault-collisions -0.005 \
    --minimum-delta-drivable-area-compliance -0.005 \
    --minimum-delta-time-to-collision -0.005
echo "PASS" > "${STATUS}"
trap - ERR
echo "PASS DiffusionDrive selector rollout-v1 development gate"
echo "selection: ${SELECTION}"
echo "persistent_log: ${LOG_FILE}"
