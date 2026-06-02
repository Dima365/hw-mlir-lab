#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <llvm-ir> <main.c> <app> <object-dir>" >&2
  exit 1
fi

LLVM_IR="$1"
MAIN_C="$2"
APP="$3"
OBJECT_DIR="$4"

CC="${CC:-clang}"
LLC="${LLC:-/home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc}"
MLIR_RUNNER_UTILS_DIR="${MLIR_RUNNER_UTILS_DIR:-/home/mandzhiev/workspace/llvm/llvm-project/build/lib}"
RUNTIME_SRC="${RUNTIME_SRC:-interface/interface.c}"

mkdir -p "$OBJECT_DIR"

"$CC" \
  -Wall \
  -Wextra \
  -c \
  "$RUNTIME_SRC" \
  -o "$OBJECT_DIR/interface.o"

"$LLC" \
  -filetype=obj \
  "$LLVM_IR" \
  -o "$OBJECT_DIR/mlir_program.o"

"$CC" \
  "$OBJECT_DIR/mlir_program.o" \
  "$OBJECT_DIR/interface.o" \
  "$MAIN_C" \
  -L"$MLIR_RUNNER_UTILS_DIR" \
  -Wl,-rpath,"$MLIR_RUNNER_UTILS_DIR" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -lm \
  -o "$APP"
