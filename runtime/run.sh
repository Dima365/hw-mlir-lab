#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CC="${CC:-clang}"
LLC="${LLC:-/home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc}"
OUT_DIR="${OUT_DIR:-build/mlir-pipeline}"
APP="${APP:-$OUT_DIR/app}"
SOCK="${SOCK:-/tmp/systolic_cocotb.sock}"
READY="${READY:-/tmp/systolic_cocotb.ready}"
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

rm -f "$SOCK" "$READY"

"$PYTHON" simulator/verilator.py &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 200); do
  if [ -e "$READY" ]; then
    break
  fi
  sleep 0.01
done

if [ ! -e "$READY" ]; then
  echo "runtime: server ready file was not created: $READY" >&2
  exit 1
fi

"$APP"
