#!/usr/bin/env bash
set -Eeo pipefail

CACHE_NAME="${1:?usage: $0 train_seed0|calibration_seed0|calibration_seed1|calibration_seed2 [LIMIT]}"
LIMIT="${2:-0}"
if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "LIMIT must be a non-negative integer" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT="${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT:-8}"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_diagnostic_v1/cache"
OUTPUT_ROOT="${V2_PROGRESS_FIX_CACHE_ROOT:-${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/cache}"
BASELINE_SHA="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
TRAIN_FILTER_SHA="d0f469e6360b712619afc00ec933faeb257a14d33b6b5b3b44d17c30f2fc12b5"
CALIBRATION_FILTER_SHA="4e30c35d72eeedfe75ccf294463a88a2da7304f2a49f9ca062af0fdf0e5d910a"
RESCORER="${SCRIPT_DIR}/rescore_grpo_selector_v2_cache.py"
VALIDATOR="${SCRIPT_DIR}/validate_grpo_selector_pdm_progress_fix.py"
WORLD_SIZE="${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT}"

case "${CACHE_NAME}" in
    train_seed0)
        SOURCE_CACHE_SHA="79c7802ba73a547f259d3b86e4e1a9353b30b41700d7662d68244067af217a98"
        SOURCE_MANIFEST_SHA="e70d7a6e2c88fc0397b593ade1b3ae8606f6e0cbfc5f7a43df7e5282e093bf3c"
        SPLIT=train
        NOISE_SEED=0
        NUM_TOKENS=6339
        NAV_FILTER_SHA="${TRAIN_FILTER_SHA}"
        ;;
    calibration_seed0)
        SOURCE_CACHE_SHA="76a4ce7f5d1e826ce6dcc8fe1984e392d77b7352e513bd4437afd5426d039f39"
        SOURCE_MANIFEST_SHA="cecd0868f4b64ea1e185506aa695189f66cdb11fa81d342d11d9911a2c70d855"
        SPLIT=calibration
        NOISE_SEED=0
        NUM_TOKENS=1118
        NAV_FILTER_SHA="${CALIBRATION_FILTER_SHA}"
        ;;
    calibration_seed1)
        SOURCE_CACHE_SHA="6521e4ad826edd3dc36d5b8447c89ddd820a7877751d891c0ddd46037da2cd64"
        SOURCE_MANIFEST_SHA="88cd3eac050dbfa77e6b41b3c6d26c918b4f6425757a704a83d97a8043a78a0b"
        SPLIT=calibration
        NOISE_SEED=1
        NUM_TOKENS=1118
        NAV_FILTER_SHA="${CALIBRATION_FILTER_SHA}"
        ;;
    calibration_seed2)
        SOURCE_CACHE_SHA="461654db5c73d1d1dfb3c4a324a1453bfb551eadd44f1bcf995b5b27440b1c25"
        SOURCE_MANIFEST_SHA="6a24bb4e3765c0a40151df50922d9d6d51c2dd65701d301e2897daeae415d6c9"
        SPLIT=calibration
        NOISE_SEED=2
        NUM_TOKENS=1118
        NAV_FILTER_SHA="${CALIBRATION_FILTER_SHA}"
        ;;
    *)
        echo "Unknown cache: ${CACHE_NAME}" >&2
        exit 2
        ;;
esac

SOURCE_DIR="${SOURCE_ROOT}/${CACHE_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/${CACHE_NAME}"
SOURCE_CACHE="${SOURCE_DIR}/cache.pt"
SOURCE_MANIFEST="${SOURCE_DIR}/manifest.json"
REWARD_FILE="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
for path in "${SOURCE_CACHE}" "${SOURCE_MANIFEST}" "${RESCORER}" "${VALIDATOR}" "${REWARD_FILE}"; do
    [[ -f "${path}" ]] || { echo "Missing cache-rescore input: ${path}" >&2; exit 1; }
done

REWARD_SHA="$(sha256sum "${REWARD_FILE}" | awk '{print $1}')"
VALIDATOR_SHA="$(sha256sum "${VALIDATOR}" | awk '{print $1}')"
MANIFEST="${OUTPUT_DIR}/manifest.json"
if [[ -f "${MANIFEST}" ]] && "${ALGENGINE_PYTHON}" -c '
import json, sys
p = json.load(open(sys.argv[1]))
expected_rows = int(sys.argv[5]) or int(sys.argv[6])
ok = (
    p.get("status") == "PASS"
    and p.get("source_cache_sha256") == sys.argv[2]
    and p.get("reward_implementation_sha256") == sys.argv[3]
    and p.get("validator_sha256") == sys.argv[4]
    and p.get("num_tokens") == expected_rows
    and p.get("world_size") == int(sys.argv[7])
)
raise SystemExit(0 if ok else 1)
' "${MANIFEST}" "${SOURCE_CACHE_SHA}" "${REWARD_SHA}" "${VALIDATOR_SHA}" \
        "${LIMIT}" "${NUM_TOKENS}" "${WORLD_SIZE}"; then
    echo "REUSE PASS V2 progress-fixed cache ${CACHE_NAME}: ${MANIFEST}"
    exit 0
fi

mkdir -p "${OUTPUT_DIR}/parts" "${OUTPUT_DIR}/logs"
COMMON_ARGS=(
    --source-cache "${SOURCE_CACHE}"
    --source-manifest "${SOURCE_MANIFEST}"
    --expected-source-cache-sha256 "${SOURCE_CACHE_SHA}"
    --expected-source-manifest-sha256 "${SOURCE_MANIFEST_SHA}"
    --expected-checkpoint-sha256 "${BASELINE_SHA}"
    --expected-nav-filter-sha256 "${NAV_FILTER_SHA}"
    --expected-split "${SPLIT}"
    --expected-noise-seed "${NOISE_SEED}"
    --expected-num-tokens "${NUM_TOKENS}"
    --metric-cache-path "${NAVSIM_METRIC_CACHE_PATH_TRAIN}"
    --validator "${VALIDATOR}"
    --output-dir "${OUTPUT_DIR}"
    --world-size "${WORLD_SIZE}"
    --limit "${LIMIT}"
)

PIDS=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    CUDA_VISIBLE_DEVICES="${rank}" "${ALGENGINE_PYTHON}" "${RESCORER}" shard \
        "${COMMON_ARGS[@]}" --rank "${rank}" --device cuda:0 \
        > "${OUTPUT_DIR}/logs/rank${rank}.log" 2>&1 &
    PIDS+=("$!")
done
failed=0
for rank in "${!PIDS[@]}"; do
    if wait "${PIDS[${rank}]}"; then
        echo "PASS cache=${CACHE_NAME} rank=${rank}"
    else
        echo "FAIL cache=${CACHE_NAME} rank=${rank}; see ${OUTPUT_DIR}/logs/rank${rank}.log" >&2
        failed=1
    fi
done
[[ "${failed}" -eq 0 ]] || exit 1

"${ALGENGINE_PYTHON}" "${RESCORER}" merge "${COMMON_ARGS[@]}"
echo "PASS V2 progress-fixed cache ${CACHE_NAME}: ${MANIFEST}"
