#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../run_env.sh"

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    $(dirname "$0")/train.py configs/upernet_metala_tiny_512x512_160k_ade20k_ms.py --launcher pytorch ${@:3}
