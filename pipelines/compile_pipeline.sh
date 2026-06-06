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

if [ -z "${CC+x}" ]; then
  if [ -x /opt/llvm/bin/clang ]; then
    CC="/opt/llvm/bin/clang"
  elif [ -x /home/mandzhiev/workspace/llvm/llvm-project/build/bin/clang ]; then
    CC="/home/mandzhiev/workspace/llvm/llvm-project/build/bin/clang"
  else
    CC="clang"
  fi
fi
if [ -z "${LLC+x}" ]; then
  if [ -x /opt/llvm/bin/llc ]; then
    LLC="/opt/llvm/bin/llc"
  elif [ -x /home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc ]; then
    LLC="/home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc"
  elif command -v llc >/dev/null 2>&1; then
    LLC="llc"
  else
    echo "compile_pipeline: llc not found" >&2
    exit 1
  fi
fi
if [ -z "${MLIR_RUNNER_UTILS_DIR+x}" ]; then
  if [ -d /opt/llvm/lib ]; then
    MLIR_RUNNER_UTILS_DIR="/opt/llvm/lib"
  elif [ -d /home/mandzhiev/workspace/llvm/llvm-project/build/lib ]; then
    MLIR_RUNNER_UTILS_DIR="/home/mandzhiev/workspace/llvm/llvm-project/build/lib"
  elif command -v llvm-config >/dev/null 2>&1; then
    MLIR_RUNNER_UTILS_DIR="$(llvm-config --libdir)"
  else
    echo "compile_pipeline: MLIR runner utils lib directory not found" >&2
    exit 1
  fi
fi
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
