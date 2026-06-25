#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <app> [ip]" >&2
  exit 1
fi

APP="$1"
IP="${2:-matmul}"
REPO_ROOT="$(pwd)"
COCOTB_TEST_DIR="${COCOTB_TEST_DIR:-tests/cocotb}"

if ! cocotb-config --makefiles >/dev/null 2>&1; then
  echo "cocotb pipeline: cocotb-config is not usable." >&2
  echo "Run this pipeline through Docker, for example: make demo" >&2
  exit 1
fi

APP="$APP" REPO_ROOT="$REPO_ROOT" IP="$IP" make -C "$COCOTB_TEST_DIR" IP="$IP"
