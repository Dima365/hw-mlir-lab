#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-build/resnet-blocks-pipeline}"
APP="${APP:-$OUT_DIR/app}"

./pipelines/resnet_blocks_pipeline.sh demo/resnet_blocks.mlir "$OUT_DIR"

./pipelines/compile_pipeline.sh \
  "$OUT_DIR/resnet_blocks.ll" \
  demo/resnet_blocks_main.c \
  "$APP" \
  "$OUT_DIR"

PYTHON="$PYTHON" ./pipelines/cocotb_pipeline.sh "$APP" accelerator

