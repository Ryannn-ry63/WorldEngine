#!/usr/bin/env bash
# Materialize one locked architecture checkpoint; no GPU required.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT}/projects/AlgEngine/scripts/diffusiondrive"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_setup_assets

ARCHITECTURE_ROOT="${1:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
LABEL="${2:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
EPOCH="${3:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
[[ "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid label" >&2; exit 2; }
[[ "${EPOCH}" =~ ^(16|32|64)$ ]] || { echo "epoch must be 16, 32, or 64" >&2; exit 2; }

TRIAL="${ARCHITECTURE_ROOT}/trials/${LABEL}"
STATE="${TRIAL}/epoch_${EPOCH}_scene_selector.pt"
REPORT="${TRIAL}/report.json"
[[ -f "${STATE}" && -f "${REPORT}" ]] || { echo "missing state/report under ${TRIAL}" >&2; exit 1; }
METHOD="$(${ALGENGINE_PYTHON} - "${STATE}" <<'PY'
import sys,torch
row=torch.load(sys.argv[1],map_location="cpu")
assert row["schema_version"]==4 and int(row["epoch"]) in (16,32,64)
print(row["method"])
PY
)"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
OUTPUT_ROOT="${ARCHITECTURE_ROOT}/materialized/${LABEL}_epoch${EPOCH}"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "materialization already exists: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_trajectory_set_reasoner.py" \
    --baseline "${REASONER_BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" --expected-method "${METHOD}" \
    --release-name "trajectory_set_reasoner_${LABEL}_epoch${EPOCH}" \
    --output "${OUTPUT_ROOT}/checkpoint.pth" \
    --manifest "${OUTPUT_ROOT}/checkpoint_manifest.json"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${REASONER_BASELINE}" --checkpoint "${OUTPUT_ROOT}/checkpoint.pth" \
    --output "${OUTPUT_ROOT}/checkpoint_audit.json"
echo "PASS materialized selector: ${OUTPUT_ROOT}/checkpoint.pth"
echo "manifest: ${OUTPUT_ROOT}/checkpoint_manifest.json"
