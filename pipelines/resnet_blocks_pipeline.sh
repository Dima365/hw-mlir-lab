#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OPT+x}" ]; then
  if [ -x standalone/build/bin/standalone-opt ]; then
    OPT="standalone/build/bin/standalone-opt"
  else
    echo "resnet_blocks_pipeline: standalone-opt not found" >&2
    echo "Run: make standalone" >&2
    exit 1
  fi
fi
MLIR_TRANSLATE="${MLIR_TRANSLATE:-/opt/llvm/bin/mlir-translate}"
INPUT="${1:-demo/resnet_blocks.mlir}"
OUT_DIR="${2:-build/resnet-blocks-pipeline}"
TRANSFORM_DIR="${TRANSFORM_DIR:-transforms}"
MATMUL_TILE_TRANSFORM="$TRANSFORM_DIR/u8s8_matmul_tile.mlir"

mkdir -p "$OUT_DIR"

echo "[1/9] Linalg quantized Conv -> im2col -> normalized padded matmul"
"$OPT" \
  --transform-preload-library="transform-library-paths=$TRANSFORM_DIR/conv_im2col.mlir" \
  --transform-interpreter \
  --prepare-u8s8-matmul \
  "$INPUT" \
  -o "$OUT_DIR/01_linalg_matmul.mlir"

echo "[2/9] fuse output epilogues and tile matmul reductions to microkernels"
"$OPT" --fuse-matmul-epilogues \
  "$OUT_DIR/01_linalg_matmul.mlir" -o "$OUT_DIR/01_fused_epilogues.mlir"
"$OPT" \
  --transform-preload-library="transform-library-paths=$MATMUL_TILE_TRANSFORM" \
  --transform-interpreter \
  "$OUT_DIR/01_fused_epilogues.mlir" \
  -o "$OUT_DIR/02_linalg_tiles.mlir"

echo "[3/9] convert fused epilogue tiles to hardware buffers"
"$OPT" \
  --lower-conv-requant-to-tiles \
  --lower-qadd-relu-to-tiles \
  "$OUT_DIR/02_linalg_tiles.mlir" \
  -o "$OUT_DIR/03_tensor_tiles.mlir"

echo "[4/9] bufferize tensor boundaries"
"$OPT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --canonicalize \
  --cse \
  "$OUT_DIR/03_tensor_tiles.mlir" \
  -o "$OUT_DIR/04_bufferized.mlir"

echo "[5/9] convert Linalg u8s8 microkernels to accelerator operations"
"$OPT" \
  --convert-linalg-u8s8-matmul-to-systolic \
  "$OUT_DIR/04_bufferized.mlir" \
  -o "$OUT_DIR/05_accelerator_tiles.mlir"

echo "[6/9] create C entry wrappers"
"$OPT" \
  --create-c-interface-entry-wrappers \
  "$OUT_DIR/05_accelerator_tiles.mlir" \
  -o "$OUT_DIR/06_wrapped.mlir"

echo "[7/9] lower accelerator operations to runtime calls"
"$OPT" \
  --lower-systolic-u8s8-to-func-call \
  --lower-conv-requant-to-func-call \
  --lower-qadd-relu-to-func-call \
  "$OUT_DIR/06_wrapped.mlir" \
  -o "$OUT_DIR/07_calls.mlir"

echo "[8/9] lower remaining dialects to LLVM dialect"
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
  "$OUT_DIR/07_calls.mlir" \
  -o "$OUT_DIR/08_llvm.mlir"

echo "[9/9] translate LLVM dialect to LLVM IR"
"$MLIR_TRANSLATE" \
  --mlir-to-llvmir \
  "$OUT_DIR/08_llvm.mlir" \
  -o "$OUT_DIR/resnet_blocks.ll"

echo "Done: $OUT_DIR/resnet_blocks.ll"
