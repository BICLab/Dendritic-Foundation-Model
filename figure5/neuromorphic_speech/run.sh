#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the generated SSC dataset root.}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU(s) to use.}"

OUTPUT_DIR="${OUTPUT_DIR:-ssc_0.3m}"
BLOCK="${BLOCK:-sparsela_m}"
SEEDS="${SEEDS:-42 43 44 45 46}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
mkdir -p "$OUTPUT_DIR"

pids=()
for seed in $SEEDS; do
  command=(python ssc_train.py --data-path "$DATA_PATH" --block "$BLOCK" --seed "$seed")
  log_file="$OUTPUT_DIR/${BLOCK}_${seed}.out"
  if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
    "${command[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
  else
    "${command[@]}" 2>&1 | tee "$log_file"
  fi
done

if ((${#pids[@]})); then
  wait "${pids[@]}"
fi
