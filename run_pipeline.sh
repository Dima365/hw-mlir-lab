#!/usr/bin/env bash
set -euo pipefail

OPT="${OPT:-standalone/build/bin/standalone-opt}"
INPUT="${1:-matmul.mlir}"
OUT_DIR="${2:-build/mlir-pipeline}"
TRANSFORM_DIR="${TRANSFORM_DIR:-transforms}"
TILE_TRANSFORM="$TRANSFORM_DIR/matmul_tile.mlir"
PAD_TRANSFORM="$TRANSFORM_DIR/matmul_pad.mlir"

mkdir -p "$OUT_DIR"

echo "[1/5] tile: $INPUT -> $OUT_DIR/01_tiled.mlir"
"$OPT" \
  --transform-preload-library="transform-library-paths=$TILE_TRANSFORM" \
  --transform-interpreter \
  --canonicalize \
  "$INPUT" \
  > "$OUT_DIR/01_tiled.mlir"

echo "[2/5] pad: $OUT_DIR/01_tiled.mlir -> $OUT_DIR/02_padded.mlir"
"$OPT" \
  --transform-preload-library="transform-library-paths=$PAD_TRANSFORM" \
  --transform-interpreter \
  --canonicalize \
  "$OUT_DIR/01_tiled.mlir" \
  > "$OUT_DIR/02_padded.mlir"

echo "[3/5] bufferize: $OUT_DIR/02_padded.mlir -> $OUT_DIR/03_bufferized.mlir"
"$OPT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --canonicalize \
  --cse \
  "$OUT_DIR/02_padded.mlir" \
  > "$OUT_DIR/03_bufferized.mlir"

echo "[4/5] systolic conversion: $OUT_DIR/03_bufferized.mlir -> $OUT_DIR/04_systolic_memref.mlir"
"$OPT" \
  --convert-linalg-matmul-to-systolic \
  "$OUT_DIR/03_bufferized.mlir" \
  > "$OUT_DIR/04_systolic_memref.mlir"

echo "[5/5] systolic call lowering: $OUT_DIR/04_systolic_memref.mlir -> $OUT_DIR/05_systolic_call.mlir"
"$OPT" \
  --lower-systolic-to-func-call \
  "$OUT_DIR/04_systolic_memref.mlir" \
  > "$OUT_DIR/05_systolic_call.mlir"

cat <<EOF
Done.
  tiled:             $OUT_DIR/01_tiled.mlir
  padded:            $OUT_DIR/02_padded.mlir
  bufferized:        $OUT_DIR/03_bufferized.mlir
  systolic memref:   $OUT_DIR/04_systolic_memref.mlir
  systolic call:     $OUT_DIR/05_systolic_call.mlir
EOF
