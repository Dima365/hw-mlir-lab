#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-torch-mlir}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DOCKER_IMAGE="${DOCKER_IMAGE:-onnxmlir/onnx-mlir-dev:latest}"
ONNX_MLIR_BIN="${ONNX_MLIR_BIN:-/workdir/onnx-mlir/build/Debug/bin/onnx-mlir}"
ONNX_MLIR_OPT_BIN="${ONNX_MLIR_OPT_BIN:-/workdir/onnx-mlir/build/Debug/bin/onnx-mlir-opt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found" >&2
  echo "Install Miniconda or make conda available in PATH, then rerun this script." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/6] Creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "[1/6] Conda env already exists: $ENV_NAME"
fi

echo "[2/6] Installing Python dependencies"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install --pre torch-mlir torchvision onnx \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

echo "[3/6] Pulling ONNX-MLIR Docker image"
docker pull "$DOCKER_IMAGE"

echo "[4/6] Exporting quantized ResNet18 to ONNX"
conda run -n "$ENV_NAME" python resnet18_to_mlir.py

echo "[5/6] Converting ONNX to ONNX dialect MLIR"
docker run --rm \
  -v "$PWD":/work \
  "$DOCKER_IMAGE" \
  "$ONNX_MLIR_BIN" \
  --EmitONNXIR /work/resnet18_quantized.onnx \
  -o /work/resnet18_quantized_onnxir

echo "[6/6] Printing generic ONNX MLIR for standalone-opt"
docker run --rm \
  -v "$PWD":/work \
  "$DOCKER_IMAGE" \
  "$ONNX_MLIR_OPT_BIN" \
  /work/resnet18_quantized_onnxir.onnx.mlir \
  --mlir-print-op-generic \
  -o /work/resnet18_quantized_onnxir_generic.mlir

echo "Done:"
echo "  $SCRIPT_DIR/resnet18_quantized.onnx"
echo "  $SCRIPT_DIR/resnet18_quantized_onnxir.onnx.mlir"
echo "  $SCRIPT_DIR/resnet18_quantized_onnxir_generic.mlir"
