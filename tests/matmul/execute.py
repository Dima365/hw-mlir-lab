import subprocess
from pathlib import Path

from .util import run


def run_app_with_server(app: Path) -> subprocess.CompletedProcess:
  return run(["./pipelines/cocotb_pipeline.sh", str(app)], timeout=60)
