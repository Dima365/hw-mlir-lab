#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <app>" >&2
  exit 1
fi

APP="$1"
REPO_ROOT="$(pwd)"
COCOTB_TEST_DIR="${COCOTB_TEST_DIR:-tests/cocotb/systolic_array_demo}"

if ! cocotb-config --makefiles >/dev/null 2>&1; then
  echo "cocotb pipeline: cocotb-config is not usable." >&2
  echo "Install or repair cocotb, for example: python3 -m pip install cocotb" >&2
  exit 1
fi

APP="$APP" REPO_ROOT="$REPO_ROOT" make -C "$COCOTB_TEST_DIR"
