#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-standalone/build}"
LLVM_DIR="${LLVM_DIR:-/opt/llvm/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-/opt/llvm/lib/cmake/mlir}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  cmake -S standalone -B "$BUILD_DIR" -G Ninja \
    -DLLVM_DIR="$LLVM_DIR" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
fi

cmake --build "$BUILD_DIR"
