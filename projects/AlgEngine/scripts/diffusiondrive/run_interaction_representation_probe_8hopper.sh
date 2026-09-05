#!/usr/bin/env bash
# Stage I: train-split frozen-track caches followed by the pre-registered probe.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/interaction_selector_env.sh"
interaction_preflight

RUN_ID="${INTERACTION_PROBE_ID:-probe_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${INTERACTION_EXPERIMENT_ROOT}/probe/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "probe root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
LOG="${ROOT}/probe.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=train_cache
trap 'rc=$?; echo "FAIL interaction probe stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

for seed in 0 1 2; do
    interaction_extract_cache train "${seed}" 10032 \
        "${INTERACTION_TUNING_ROOT}/train/rare_common_union.yaml"
done

CURRENT_STAGE=representation_probe
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/probe_interaction_representation.py" \
    --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed0/cache.pt" \
    --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed1/cache.pt" \
    --train-cache "${INTERACTION_CACHE_ROOT}/tuning/train_seed2/cache.pt" \
    --pair-manifest "${INTERACTION_TUNING_ROOT}/train/pairs.jsonl" \
    --rare-data-audit "${INTERACTION_TUNING_ROOT}/train/rare_data_audit.json" \
    --output "${ROOT}/probe_gate.json" --device cuda

CURRENT_STAGE=complete
trap - ERR
echo "PASS interaction representation probe"
echo "gate: ${ROOT}/probe_gate.json"
echo "log: ${LOG}"

