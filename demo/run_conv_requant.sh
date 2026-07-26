#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-build/conv-requant-pipeline}"
APP="${APP:-$OUT_DIR/app}"

./pipelines/conv_requant_pipeline.sh

./pipelines/compile_pipeline.sh \
  "$OUT_DIR/conv_requant.ll" \
  demo/conv_requant_main.c \
  "$APP" \
  "$OUT_DIR"

PYTHON="$PYTHON" ./pipelines/cocotb_pipeline.sh "$APP" conv_requant
