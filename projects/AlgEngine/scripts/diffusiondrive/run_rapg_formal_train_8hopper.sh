#!/usr/bin/env bash
# Compute-matched 3-seed B00/B01/B10/B11 formal training on the full locked pool.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8
RUN_ID="${RAPG_FORMAL_ID:-formal_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${RAPG_EXPERIMENT_ROOT}/formal/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "formal root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
LOCKED_WEIGHT="${RAPG_LOCKED_PREFERENCE_WEIGHT:-0.5}"

run_job() {
    local gpu="$1" label="$2" architecture="$3" objective="$4" weight="$5" seed="$6"
    local output="${ROOT}/${label}/seed${seed}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" \
        "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" --output-dir "${output}" \
        --architecture "${architecture}" --ablation relational_only \
        --objective "${objective}" --preference-weight "${weight}" \
        --data-split all --seed "${seed}" --epochs 16 --checkpoint-epochs 16 \
        --batch-size 64 --device cuda --formal-contract >"${output}/train.log" 2>&1
}

jobs=()
for seed in 0 1 2; do
    jobs+=("b00|trajectory_set_reasoner|official_pdm|0|${seed}")
    jobs+=("b01|trajectory_set_reasoner|official_pdm_plus_scalar_preference|${LOCKED_WEIGHT}|${seed}")
    jobs+=("b10|reference_anchored_preference_graph|official_pdm|0|${seed}")
    jobs+=("b11|reference_anchored_preference_graph|official_pdm_plus_scalar_preference|${LOCKED_WEIGHT}|${seed}")
done
for index in "${!jobs[@]}"; do
    IFS='|' read -r label architecture objective weight seed <<<"${jobs[index]}"
    run_job "$((index % 8))" "${label}" "${architecture}" "${objective}" "${weight}" "${seed}" &
    if (( index % 8 == 7 )); then wait; fi
done
wait
echo "PASS RAPG formal causal training: ${ROOT}"
