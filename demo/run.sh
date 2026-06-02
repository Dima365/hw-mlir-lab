#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-build/mlir-pipeline}"
APP="${APP:-$OUT_DIR/app}"

./pipelines/mlir_pipeline.sh

./pipelines/compile_pipeline.sh "$OUT_DIR/08_llvm.ll" demo/main.c "$APP" "$OUT_DIR"

PYTHON="$PYTHON" ./pipelines/cocotb_pipeline.sh "$APP"
