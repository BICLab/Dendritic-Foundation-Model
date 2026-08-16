#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../run_env.sh"

CONFIG=${1:?Usage: dist_test.sh CONFIG CHECKPOINT [NPROC_PER_NODE] [extra args...]}
CHECKPOINT=${2:?Usage: dist_test.sh CONFIG CHECKPOINT [NPROC_PER_NODE] [extra args...]}
GPUS=${3:-$NPROC_PER_NODE}

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
    "$(dirname "$0")/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"
