#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:?usage: $0 seed SEED | all}"
WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_gate_conditioned_h100.sh"

case "${MODE}" in
    seed)
        SEED="${2:?usage: $0 seed SEED}"
        [[ "${#}" -eq 2 && "${SEED}" =~ ^[0-2]$ ]] || {
            echo "seed must be one of 0, 1, 2" >&2
            exit 2
        }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${RUNNER}" seed "${SEED}"
        ;;
    all)
        [[ "${#}" -eq 1 ]] || { echo "all takes no arguments" >&2; exit 2; }
        export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
        exec "${RUNNER}"
        ;;
    *)
        echo "usage: $0 seed SEED | all" >&2
        exit 2
        ;;
esac
