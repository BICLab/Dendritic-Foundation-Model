#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../run_env.sh"

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    $(dirname "$0")/train.py configs/mask_rcnn_sa_tiny_fpn_1x_coco.py --launcher pytorch ${@:3}
