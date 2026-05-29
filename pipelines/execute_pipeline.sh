#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <app>" >&2
  exit 1
fi

APP="$1"
PYTHON="${PYTHON:-python3}"
SOCK="${SOCK:-/tmp/systolic_cocotb.sock}"
READY="${READY:-/tmp/systolic_cocotb.ready}"

rm -f "$SOCK" "$READY"

"$PYTHON" simulator/verilator.py &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 200); do
  if [ -e "$READY" ]; then
    break
  fi
  sleep 0.01
done

if [ ! -e "$READY" ]; then
  echo "runtime: server ready file was not created: $READY" >&2
  exit 1
fi

"$APP"
