#!/usr/bin/env bash
# CPU-only aggregation/filtering after all three online collection lanes pass.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
export WORLDENGINE_ROOT ALGENGINE_ROOT
export PYTHONPATH="${ALGENGINE_ROOT}:${PYTHONPATH:-}"

ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
COLLECTION_ROOT="${ROOT}/collection/formal"
DATA_ROOT="${ROOT}/data"
RARE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/rare_data"
REAL_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
BUILDER="${ALGENGINE_ROOT}/scripts/diffusiondrive/build_grpo_selector_v3_rare_rollout_data.py"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_data_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL rare-rollout data preparation stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${BUILDER}" "${RARE_ROOT}/pairs.jsonl" "${RARE_ROOT}/rare_data_audit.json"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
for lane in 0 1 2; do
    [[ -f "${COLLECTION_ROOT}/lane${lane}/collection_audit.json" ]] || {
        echo "Missing completed collection lane ${lane}" >&2
        exit 1
    }
done
for seed in 0 1 2; do
    [[ -f "${REAL_CACHE_ROOT}/train_seed${seed}/cache.pt" ]] || {
        echo "Missing rare-original real cache seed ${seed}" >&2
        exit 1
    }
done
[[ -x "${ALGENGINE_PYTHON}" ]] || { echo "Missing AlgEngine Python" >&2; exit 1; }

if [[ -f "${DATA_ROOT}/manifest.json" ]]; then
    CURRENT_STAGE=reuse_audit
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,pathlib,sys
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
manifest_path=pathlib.Path(sys.argv[1]).resolve()
row=json.loads(manifest_path.read_text())
assert row["status"]=="PASS"
assert row["method"]=="diffusiondrive_v3_rare_rollout_mixture_v1"
for path_key,sha_key in (("synthetic_cache","synthetic_cache_sha256"),("hard_pool","hard_pool_sha256")):
    path=pathlib.Path(row[path_key])
    assert path.is_file()
    assert sha(path)==row[sha_key]
' "${DATA_ROOT}/manifest.json"
    trap - ERR
    echo "SKIP verified rare-rollout mixed data"
    echo "manifest: ${DATA_ROOT}/manifest.json"
    exit 0
fi
if [[ -e "${DATA_ROOT}" ]]; then
    echo "Partial immutable data root exists; inspect before retry: ${DATA_ROOT}" >&2
    exit 1
fi

CURRENT_STAGE=build_filtered_mixture
"${ALGENGINE_PYTHON}" "${BUILDER}" \
    --lane-root "${COLLECTION_ROOT}/lane0" \
    --lane-root "${COLLECTION_ROOT}/lane1" \
    --lane-root "${COLLECTION_ROOT}/lane2" \
    --lane-audit "${COLLECTION_ROOT}/lane0/collection_audit.json" \
    --lane-audit "${COLLECTION_ROOT}/lane1/collection_audit.json" \
    --lane-audit "${COLLECTION_ROOT}/lane2/collection_audit.json" \
    --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt" \
    --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt" \
    --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt" \
    --pair-manifest "${RARE_ROOT}/pairs.jsonl" \
    --rare-data-audit "${RARE_ROOT}/rare_data_audit.json" \
    --minimum-synthetic 1 \
    --output-dir "${DATA_ROOT}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive V3 rare-rollout mixed-data preparation"
echo "manifest: ${DATA_ROOT}/manifest.json"
echo "persistent_log: ${LOG_FILE}"
