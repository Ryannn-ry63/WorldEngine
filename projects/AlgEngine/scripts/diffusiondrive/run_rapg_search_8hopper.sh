#!/usr/bin/env bash
# Fixed-16-epoch causal search: B00/B01/B10/B11 and capacity control.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8
RUN_ID="${RAPG_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${RAPG_EXPERIMENT_ROOT}/search/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "search root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)

run_arm() {
    local gpu="$1" label="$2" architecture="$3" objective="$4" weight="$5"
    mkdir -p "${ROOT}/${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" \
        "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --output-dir "${ROOT}/${label}" --architecture "${architecture}" \
        --ablation relational_only --objective "${objective}" \
        --preference-weight "${weight}" --data-split train --seed 0 \
        --epochs 16 --checkpoint-epochs 16 --batch-size 64 --device cuda \
        >"${ROOT}/${label}/train.log" 2>&1
}

arms=(
    "b00|trajectory_set_reasoner|official_pdm|0"
    "b10|reference_anchored_preference_graph|official_pdm|0"
    "unary_control|capacity_matched_unary|official_pdm|0"
    "b01_l025|trajectory_set_reasoner|official_pdm_plus_scalar_preference|0.25"
    "b01_l050|trajectory_set_reasoner|official_pdm_plus_scalar_preference|0.5"
    "b01_l100|trajectory_set_reasoner|official_pdm_plus_scalar_preference|1.0"
    "b11_l025|reference_anchored_preference_graph|official_pdm_plus_scalar_preference|0.25"
    "b11_l050|reference_anchored_preference_graph|official_pdm_plus_scalar_preference|0.5"
    "b11_l100|reference_anchored_preference_graph|official_pdm_plus_scalar_preference|1.0"
)
for index in "${!arms[@]}"; do
    IFS='|' read -r label architecture objective weight <<<"${arms[index]}"
    run_arm "$((index % 8))" "${label}" "${architecture}" "${objective}" "${weight}" &
    if (( index % 8 == 7 )); then wait; fi
done
wait
echo "PASS RAPG fixed-budget search: ${ROOT}"
