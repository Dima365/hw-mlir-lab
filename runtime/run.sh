#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CC="${CC:-clang}"
LLC="${LLC:-/home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc}"
OUT_DIR="${OUT_DIR:-build/mlir-pipeline}"
APP="${APP:-$OUT_DIR/app}"
SOCK="${SOCK:-/tmp/systolic_cocotb.sock}"
MLIR_RUNNER_UTILS_DIR="${MLIR_RUNNER_UTILS_DIR:-/home/mandzhiev/workspace/llvm/llvm-project/build/lib}"

./run_pipeline.sh

"$LLC" \
  -filetype=obj \
  "$OUT_DIR/08_llvm.ll" \
  -o "$OUT_DIR/mlir_program.o"

"$CC" \
  "$OUT_DIR/mlir_program.o" \
  "$OUT_DIR/systolic_runtime.o" \
  runtime/main.c \
  -L"$MLIR_RUNNER_UTILS_DIR" \
  -Wl,-rpath,"$MLIR_RUNNER_UTILS_DIR" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -o "$APP"

rm -f "$SOCK"

"$PYTHON" simulator/verilator.py &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$APP"
