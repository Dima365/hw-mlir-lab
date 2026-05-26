import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd, *, cwd=ROOT, timeout=120):
  completed = subprocess.run(
      cmd,
      cwd=cwd,
      timeout=timeout,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  if completed.returncode != 0:
    command = " ".join(str(arg) for arg in cmd)
    raise RuntimeError(
        f"command failed ({completed.returncode}): {command}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
  return completed
