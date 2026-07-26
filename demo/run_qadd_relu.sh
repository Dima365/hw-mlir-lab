#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-build/qadd-relu-pipeline}"
APP="${APP:-$OUT_DIR/app}"

./pipelines/qadd_relu_pipeline.sh

./pipelines/compile_pipeline.sh \
  "$OUT_DIR/qadd_relu.ll" \
  demo/qadd_relu_main.c \
  "$APP" \
  "$OUT_DIR"

PYTHON="$PYTHON" ./pipelines/cocotb_pipeline.sh "$APP" qadd_relu
