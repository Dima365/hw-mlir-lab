#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-build/epilogue-pipeline}"
APP="${APP:-$OUT_DIR/app}"

./pipelines/epilogue_pipeline.sh

./pipelines/compile_pipeline.sh "$OUT_DIR/epilogue.ll" demo/epilogue_main.c "$APP" "$OUT_DIR"

PYTHON="$PYTHON" ./pipelines/cocotb_pipeline.sh "$APP" epilogue
