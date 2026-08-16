#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU(s) to use.}"

OUTPUT_DIR="${OUTPUT_DIR:-logs}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
mkdir -p "$OUTPUT_DIR"

command=(python -m zoology.launch ./zoology/model_sp.py --outdir "$OUTPUT_DIR")
if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  "${command[@]}" >"$OUTPUT_DIR/sparsela.log" 2>&1 &
  echo "Started MQAR sweep with PID $!"
else
  "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/sparsela.log"
fi