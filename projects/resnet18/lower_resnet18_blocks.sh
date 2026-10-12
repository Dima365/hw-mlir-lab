#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPT="${OPT:-$REPO_ROOT/standalone/build/bin/standalone-opt}"
INPUT="${1:-$SCRIPT_DIR/resnet18_quantized_generic.mlir}"
OUT_DIR="${2:-$REPO_ROOT/build/resnet18-block-lowering}"
CONV_TRANSFORMS="$SCRIPT_DIR/conv_requant_match.mlir"
QADD_TRANSFORMS="$SCRIPT_DIR/qadd_relu.mlir"
MATMUL_TILE_TRANSFORM="$REPO_ROOT/transforms/u8s8_matmul_tile.mlir"

if [ ! -x "$OPT" ]; then
  echo "lower_resnet18_blocks: standalone-opt not found at $OPT" >&2
  echo "Run: make standalone" >&2
  exit 1
fi
if [ ! -f "$INPUT" ]; then
  echo "lower_resnet18_blocks: input not found at $INPUT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[1/8] ONNX Conv QDQ -> Linalg quantized Conv and CONV_REQUANT"
"$OPT" \
  --allow-unregistered-dialect \
  --transform-preload-library="transform-library-paths=$CONV_TRANSFORMS" \
  --transform-interpreter \
  "$INPUT" \
  -o "$OUT_DIR/01_integer_conv.mlir"

echo "[2/8] ONNX residual Add QDQ -> standalone.qadd_relu"
"$OPT" \
  --allow-unregistered-dialect \
  --transform-preload-library="transform-library-paths=$QADD_TRANSFORMS" \
  --transform-interpreter \
  "$OUT_DIR/01_integer_conv.mlir" \
  -o "$OUT_DIR/02_integer_residual.mlir"

echo "[3/8] Linalg quantized Conv -> im2col -> normalized padded matmul"
"$OPT" \
  --allow-unregistered-dialect \
  --transform-preload-library="transform-library-paths=$REPO_ROOT/transforms/conv_im2col.mlir" \
  --transform-interpreter \
  --prepare-u8s8-matmul \
  "$OUT_DIR/02_integer_residual.mlir" \
  -o "$OUT_DIR/03_linalg_matmul.mlir"

echo "[4/8] Fuse output epilogues and tile matmul reductions"
"$OPT" --allow-unregistered-dialect --fuse-matmul-epilogues \
  "$OUT_DIR/03_linalg_matmul.mlir" -o "$OUT_DIR/03_fused_epilogues.mlir"
"$OPT" \
  --allow-unregistered-dialect \
  --transform-preload-library="transform-library-paths=$MATMUL_TILE_TRANSFORM" \
  --transform-interpreter \
  "$OUT_DIR/03_fused_epilogues.mlir" \
  -o "$OUT_DIR/04_linalg_tiles.mlir"

echo "[5/8] Fused epilogue tiles -> hardware buffers"
"$OPT" \
  --allow-unregistered-dialect \
  --lower-conv-requant-to-tiles \
  --lower-qadd-relu-to-tiles \
  "$OUT_DIR/04_linalg_tiles.mlir" \
  -o "$OUT_DIR/05_tensor_tiles.mlir"

echo "[6/8] Bufferize known tensor operations"
"$OPT" \
  --allow-unregistered-dialect \
  --one-shot-bufferize="allow-unknown-ops bufferize-function-boundaries" \
  --canonicalize \
  --cse \
  "$OUT_DIR/05_tensor_tiles.mlir" \
  -o "$OUT_DIR/06_bufferized.mlir"

echo "[7/8] Linalg u8s8 microkernels -> accelerator operations"
"$OPT" \
  --allow-unregistered-dialect \
  --convert-linalg-u8s8-matmul-to-systolic \
  "$OUT_DIR/06_bufferized.mlir" \
  -o "$OUT_DIR/07_accelerator_tiles.mlir"

echo "[8/8] Accelerator operations -> simulator runtime calls"
"$OPT" \
  --allow-unregistered-dialect \
  --lower-systolic-u8s8-to-func-call \
  --lower-conv-requant-to-func-call \
  --lower-qadd-relu-to-func-call \
  "$OUT_DIR/07_accelerator_tiles.mlir" \
  -o "$OUT_DIR/08_runtime_calls.mlir"

echo "Done."
echo "  tensor residual graph: $OUT_DIR/02_integer_residual.mlir"
echo "  logical Linalg matmul:  $OUT_DIR/03_linalg_matmul.mlir"
echo "  Linalg tiles:           $OUT_DIR/04_linalg_tiles.mlir"
echo "  accelerator tiles:     $OUT_DIR/07_accelerator_tiles.mlir"
echo "  runtime calls:          $OUT_DIR/08_runtime_calls.mlir"
