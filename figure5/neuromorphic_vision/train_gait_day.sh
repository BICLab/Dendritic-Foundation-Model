#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the generated DVS Gait npy root.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU(s) to use.}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MODELS="${MODELS:-conv mlp softmax hgru1d hgru2d mamba1d mamba2d sparsela_v sparsela_m}"
mkdir -p "$OUTPUT_DIR"

for model_name in $MODELS; do
  model="V3_tem_attn_${model_name}_tiny"
  run_dir="$OUTPUT_DIR/${model}_T36_gait_day"
  mkdir -p "$run_dir"
  python3 main_finetune.py \
    --dataset gait-day \
    --batch_size 32 \
    --batch_size_val 16 \
    --lr 6e-4 \
    --min_lr 1e-5 \
    --weight_decay 6e-2 \
    --time_steps 36 \
    --warmup_epochs 10 \
    --epochs 200 \
    --model "$model" \
    --data_path "$DATA_PATH" \
    --output_dir "$run_dir" \
    --log_dir "$run_dir" \
    --model_mode tem_attn \
    --dist_eval \
    2>&1 | tee "$run_dir/train.log"
done
