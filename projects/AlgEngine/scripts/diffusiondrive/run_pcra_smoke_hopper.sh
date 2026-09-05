#!/usr/bin/env bash
# One-Hopper real-cache smoke for PCRA construction, training, and freeze audit.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 1

PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || {
    echo "frozen proposal32 SHA256 mismatch" >&2
    exit 1
}
RUN_ID="${PCRA_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v1/smoke/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "PCRA smoke root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_proposal_conditioned_regret_arbitrator.py" "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" --proposal-checkpoint "${PROPOSAL32}" --output-dir "${ROOT}" --arbiter-loss regret --use-decision-context --override-threshold 0.0 --data-split train --seed 0 --epochs 1 --examples-per-cache-epoch 4 --checkpoint-epochs 1 --batch-size 2 --smoke-limit-hard-pool 2 --device cuda >"${ROOT}/train.log" 2>&1

"${ALGENGINE_PYTHON}" - "${ROOT}/report.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
assert report["status"] == "PASS"
assert report["selector_architecture"] == "proposal_conditioned_regret_arbitration"
assert report["arbiter_loss"] == "regret"
assert report["override_threshold"] == 0.0
assert report["frozen_parameter_max_abs_delta"] == 0.0
assert report["proposal_regret_arbitration_diagnostics"]["proposal_frozen"] is True
checkpoint = pathlib.Path(report["checkpoints"][-1]["scene_selector_state"])
assert checkpoint.is_file()
print(f"PASS PCRA real-cache smoke audit: {path}")
PY
