#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the ImageNet validation directory.}"
: "${CHECKPOINT:?Set CHECKPOINT to the model checkpoint file.}"
BS=256

python validate.py \
--model ours_tiny \
--checkpoint="$CHECKPOINT" \
--data-dir=$DATA_PATH \
--batch-size $BS \
--input-size 3 224 224
