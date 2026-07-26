#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OPT+x}" ]; then
  if [ -x standalone/build/bin/standalone-opt ]; then
    OPT="standalone/build/bin/standalone-opt"
  else
    echo "qadd_relu_pipeline: standalone-opt not found at standalone/build/bin/standalone-opt" >&2
    echo "Run: make standalone" >&2
    exit 1
  fi
fi
if [ -z "${MLIR_TRANSLATE+x}" ]; then
  if [ -x /opt/llvm/bin/mlir-translate ]; then
    MLIR_TRANSLATE="/opt/llvm/bin/mlir-translate"
  else
    echo "qadd_relu_pipeline: mlir-translate not found at /opt/llvm/bin/mlir-translate" >&2
    echo "Run this pipeline through Docker, for example: make demo-qadd-relu" >&2
    exit 1
  fi
fi
INPUT="${1:-demo/qadd_relu.mlir}"
OUT_DIR="${2:-build/qadd-relu-pipeline}"

mkdir -p "$OUT_DIR"

echo "[1/3] lower QADD_RELU: $INPUT -> $OUT_DIR/01_call.mlir"
"$OPT" \
  --lower-qadd-relu-to-func-call \
  "$INPUT" \
  > "$OUT_DIR/01_call.mlir"

echo "[2/3] llvm lowering: $OUT_DIR/01_call.mlir -> $OUT_DIR/02_llvm.mlir"
"$OPT" \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  "$OUT_DIR/01_call.mlir" \
  > "$OUT_DIR/02_llvm.mlir"

echo "[3/3] llvm translate: $OUT_DIR/02_llvm.mlir -> $OUT_DIR/qadd_relu.ll"
"$MLIR_TRANSLATE" \
  --mlir-to-llvmir \
  "$OUT_DIR/02_llvm.mlir" \
  > "$OUT_DIR/qadd_relu.ll"

cat <<EOF
Done.
  lowered call:  $OUT_DIR/01_call.mlir
  llvm dialect:  $OUT_DIR/02_llvm.mlir
  llvm ir:       $OUT_DIR/qadd_relu.ll
EOF
