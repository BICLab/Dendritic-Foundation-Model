#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../run_env.sh"

CONFIG=${1:?Usage: dist_train.sh CONFIG [NPROC_PER_NODE] [extra args...]}
GPUS=${2:-$NPROC_PER_NODE}

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
    "$(dirname "$0")/train.py" "$CONFIG" --launcher pytorch "${@:3}"
