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
  else
    echo "compile_pipeline: clang not found at /opt/llvm/bin/clang" >&2
    echo "Run this pipeline through Docker, for example: make demo" >&2
    exit 1
  fi
fi
if [ -z "${LLC+x}" ]; then
  if [ -x /opt/llvm/bin/llc ]; then
    LLC="/opt/llvm/bin/llc"
  else
    echo "compile_pipeline: llc not found at /opt/llvm/bin/llc" >&2
    echo "Run this pipeline through Docker, for example: make demo" >&2
    exit 1
  fi
fi
if [ -z "${LD_LLD+x}" ]; then
  if [ -x /opt/llvm/bin/ld.lld ]; then
    LD_LLD="/opt/llvm/bin/ld.lld"
  else
    echo "compile_pipeline: ld.lld not found at /opt/llvm/bin/ld.lld" >&2
    echo "Run: make docker-build" >&2
    exit 1
  fi
fi
if [ -z "${MLIR_RUNNER_UTILS_DIR+x}" ]; then
  if [ -d /opt/llvm/lib ]; then
    MLIR_RUNNER_UTILS_DIR="/opt/llvm/lib"
  else
    echo "compile_pipeline: MLIR runner utils lib directory not found at /opt/llvm/lib" >&2
    echo "Run this pipeline through Docker, for example: make demo" >&2
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
  -fuse-ld="$LD_LLD" \
  "$OBJECT_DIR/mlir_program.o" \
  "$OBJECT_DIR/interface.o" \
  "$MAIN_C" \
  -L"$MLIR_RUNNER_UTILS_DIR" \
  -Wl,-rpath,"$MLIR_RUNNER_UTILS_DIR" \
  -lmlir_c_runner_utils \
  -lmlir_runner_utils \
  -lm \
  -o "$APP"
