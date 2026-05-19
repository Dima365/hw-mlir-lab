#!/usr/bin/env bash
set -euo pipefail

OPT="${OPT:-standalone/build/bin/standalone-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-/home/mandzhiev/workspace/llvm/llvm-project/build/bin/mlir-translate}"
CC="${CC:-cc}"
INPUT="${1:-matmul.mlir}"
OUT_DIR="${2:-build/mlir-pipeline}"
TRANSFORM_DIR="${TRANSFORM_DIR:-transforms}"
TILE_TRANSFORM="$TRANSFORM_DIR/matmul_tile.mlir"
PAD_TRANSFORM="$TRANSFORM_DIR/matmul_pad.mlir"
RUNTIME_SRC="${RUNTIME_SRC:-simulator/interface.c}"

mkdir -p "$OUT_DIR"

echo "[1/8] tile: $INPUT -> $OUT_DIR/01_tiled.mlir"
"$OPT" \
  --transform-preload-library="transform-library-paths=$TILE_TRANSFORM" \
  --transform-interpreter \
  --canonicalize \
  "$INPUT" \
  > "$OUT_DIR/01_tiled.mlir"

echo "[2/8] pad: $OUT_DIR/01_tiled.mlir -> $OUT_DIR/02_padded.mlir"
"$OPT" \
  --transform-preload-library="transform-library-paths=$PAD_TRANSFORM" \
  --transform-interpreter \
  --canonicalize \
  "$OUT_DIR/01_tiled.mlir" \
  > "$OUT_DIR/02_padded.mlir"

echo "[3/8] bufferize: $OUT_DIR/02_padded.mlir -> $OUT_DIR/03_bufferized.mlir"
"$OPT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --canonicalize \
  --cse \
  "$OUT_DIR/02_padded.mlir" \
  > "$OUT_DIR/03_bufferized.mlir"

echo "[4/8] systolic conversion: $OUT_DIR/03_bufferized.mlir -> $OUT_DIR/04_systolic_memref.mlir"
"$OPT" \
  --convert-linalg-matmul-to-systolic \
  "$OUT_DIR/03_bufferized.mlir" \
  > "$OUT_DIR/04_systolic_memref.mlir"

echo "[5/8] systolic call lowering: $OUT_DIR/04_systolic_memref.mlir -> $OUT_DIR/05_systolic_call.mlir"
"$OPT" \
  --lower-systolic-to-func-call \
  "$OUT_DIR/04_systolic_memref.mlir" \
  > "$OUT_DIR/05_systolic_call.mlir"

echo "[6/8] llvm lowering: $OUT_DIR/05_systolic_call.mlir -> $OUT_DIR/06_llvm.mlir"
"$OPT" \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  "$OUT_DIR/05_systolic_call.mlir" \
  > "$OUT_DIR/06_llvm.mlir"

echo "[7/8] llvm translate: $OUT_DIR/06_llvm.mlir -> $OUT_DIR/07_llvm.ll"
"$MLIR_TRANSLATE" \
  --mlir-to-llvmir \
  "$OUT_DIR/06_llvm.mlir" \
  > "$OUT_DIR/07_llvm.ll"

echo "[8/8] runtime object: $RUNTIME_SRC -> $OUT_DIR/systolic_runtime.o"
"$CC" \
  -Wall \
  -Wextra \
  -c \
  "$RUNTIME_SRC" \
  -o "$OUT_DIR/systolic_runtime.o"

cat <<EOF
Done.
  tiled:             $OUT_DIR/01_tiled.mlir
  padded:            $OUT_DIR/02_padded.mlir
  bufferized:        $OUT_DIR/03_bufferized.mlir
  systolic memref:   $OUT_DIR/04_systolic_memref.mlir
  systolic call:     $OUT_DIR/05_systolic_call.mlir
  llvm dialect:      $OUT_DIR/06_llvm.mlir
  llvm ir:           $OUT_DIR/07_llvm.ll
  runtime object:    $OUT_DIR/systolic_runtime.o
EOF
