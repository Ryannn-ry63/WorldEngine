#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 POLICY_CKPT REFERENCE_CKPT DATA_TYPE POLICY_VERSION [ASSET_NAME] [MAX_UPDATES] [MAX_SCENES_PER_WORKER]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORLDENGINE_ROOT=${WORLDENGINE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
export WORLDENGINE_ROOT

POLICY_CKPT=$1
REFERENCE_CKPT=$2
DATA_TYPE=$3
POLICY_VERSION=$4
ASSET_NAME=${5:-$DATA_TYPE}
MAX_UPDATES=${6:-128}
MAX_SCENES_PER_WORKER=${7:-2}

if [ ! -f "$POLICY_CKPT" ]; then
  echo "ERROR: policy checkpoint not found: $POLICY_CKPT" >&2
  exit 2
fi
if [ ! -f "$REFERENCE_CKPT" ]; then
  echo "ERROR: reference checkpoint not found: $REFERENCE_CKPT" >&2
  exit 2
fi

ALGENGINE_ROOT="$WORLDENGINE_ROOT/projects/AlgEngine"
SIMENGINE_ROOT="$WORLDENGINE_ROOT/projects/SimEngine"
CONFIG="$ALGENGINE_ROOT/configs/navformer/e2e_diffusiondrive_grpo.py"
MODEL_NAME="e2e_diffusiondrive_grpo_${POLICY_VERSION}"
ROLLOUT_ROOT="$WORLDENGINE_ROOT/experiments/closed_loop_exps/$MODEL_NAME/${DATA_TYPE}_NR"
ROUND_ROOT="$WORLDENGINE_ROOT/experiments/grpo_rounds/$POLICY_VERSION"
MANIFEST="$ROUND_ROOT/rollout_manifest.pkl"
WORK_DIR="$ROUND_ROOT/training"

mkdir -p "$ROUND_ROOT"
export PYTHONPATH="$ALGENGINE_ROOT:$SIMENGINE_ROOT:${PYTHONPATH:-}"
export GRPO_REFERENCE_CKPT="$REFERENCE_CKPT"

echo "[1/3] Collecting versioned on-policy rollout: $POLICY_VERSION"
bash "$SIMENGINE_ROOT/scripts/run_ray_distributed_rollout.sh" \
  "$CONFIG" \
  "$POLICY_CKPT" \
  "$MODEL_NAME" \
  "$DATA_TYPE" \
  "$ASSET_NAME" \
  "$POLICY_VERSION" \
  "$MAX_SCENES_PER_WORKER"

echo "[2/3] Joining exact diffusion traces with same-state PDM rewards"
conda run --no-capture-output -n algengine \
  python "$ALGENGINE_ROOT/scripts/build_grpo_rollout_manifest.py" \
  --rollout-root "$ROLLOUT_ROOT" \
  --reward-root "$ROLLOUT_ROOT" \
  --policy-version "$POLICY_VERSION" \
  --output "$MANIFEST"

export GRPO_POLICY_CKPT="$POLICY_CKPT"
export GRPO_ROLLOUT_MANIFEST="$MANIFEST"
export GRPO_MAX_UPDATES="$MAX_UPDATES"

echo "[3/3] Running one generation-only GRPO pass (capped at $MAX_UPDATES updates)"
cd "$ALGENGINE_ROOT"
conda run --no-capture-output -n algengine \
  python scripts/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --no-validate

echo "Round complete"
echo "  manifest: $MANIFEST"
echo "  training: $WORK_DIR"
