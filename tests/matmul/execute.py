import os
import subprocess
import sys
import time
from pathlib import Path

from .util import ROOT, run


SOCK = Path("/tmp/systolic_cocotb.sock")
READY = Path("/tmp/systolic_cocotb.ready")
PYTHON = os.environ.get("PYTHON", sys.executable)


def wait_for_server_ready(timeout_s: float = 2.0) -> None:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if READY.exists():
      return
    time.sleep(0.01)
  raise RuntimeError(f"server ready file did not appear: {READY}")


def run_app_with_server(app: Path) -> subprocess.CompletedProcess:
  for path in (SOCK, READY):
    try:
      path.unlink()
    except FileNotFoundError:
      pass

  server = subprocess.Popen(
      [PYTHON, "simulator/verilator.py"],
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  try:
    wait_for_server_ready()
    return run([str(app)], timeout=30)
  finally:
    server.terminate()
    try:
      server.communicate(timeout=2)
    except subprocess.TimeoutExpired:
      server.kill()
      server.communicate()
