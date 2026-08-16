#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../run_env.sh"

: "${DATA_PATH:?Set DATA_PATH to the ImageNet dataset root.}"
MODEL=mlp_tiny_hybrid
BS=256
EXP="mlp_tiny_hybrid"
LR=2.5e-4
WD=0.05
WR_LR=0
DR=0.2
OPT="adamw"
MESA=0

TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" torchrun --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --master-port="$MASTER_PORT" train.py \
--mesa ${MESA} \
--input-size 3 224 224 \
--crop-pct=0.875 \
--opt ${OPT} \
--data_dir=$DATA_PATH \
--model $MODEL \
--amp \
--weight-decay ${WD} \
--drop-path ${DR} \
--batch-size $BS \
--tag ${EXP} \
--lr $LR \
--warmup-lr $WR_LR \
--warmup-epochs 20 \
--clip-grad 5 \
--use-multi-epochs-loader