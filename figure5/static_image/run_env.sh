#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the comma-separated GPU(s) to use.}"

IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
MASTER_PORT="${MASTER_PORT:-29500}"

if ((NPROC_PER_NODE < 1)); then
  echo "NPROC_PER_NODE must be at least 1." >&2
  return 1
fi
