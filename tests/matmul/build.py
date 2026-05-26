import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .cases import MatmulTest
from .templates import (
    expected_matmul,
    generate_main_c,
    generate_matmul_mlir,
    matrix,
)
from .util import ROOT, run


BUILD_ROOT = ROOT / "build" / "tests"
CC = os.environ.get("CC", "clang")
LLC = os.environ.get(
    "LLC", "/home/mandzhiev/workspace/llvm/llvm-project/build/bin/llc"
)
MLIR_RUNNER_UTILS_DIR = os.environ.get(
    "MLIR_RUNNER_UTILS_DIR",
    "/home/mandzhiev/workspace/llvm/llvm-project/build/lib",
)


@dataclass(frozen=True)
class BuiltTest:
  test: MatmulTest
  test_dir: Path
  pipeline_dir: Path
  app: Path


def build_test(test: MatmulTest) -> BuiltTest:
  test_dir = BUILD_ROOT / test.name
  pipeline_dir = test_dir / "pipeline"
  if test_dir.exists():
    shutil.rmtree(test_dir)
  test_dir.mkdir(parents=True)

  a = matrix(test.m, test.k, test.a_fn)
  b = matrix(test.k, test.n, test.b_fn)
  expected = expected_matmul(test.m, test.k, test.n, a, b)

  input_mlir = test_dir / "input.mlir"
  main_c = test_dir / "main.c"
  mlir_obj = test_dir / "mlir_program.o"
  app = test_dir / "app"

  input_mlir.write_text(generate_matmul_mlir(test.m, test.k, test.n))
  main_c.write_text(generate_main_c(test.m, test.k, test.n, a, b, expected))

  run(["./run_pipeline.sh", str(input_mlir), str(pipeline_dir)], timeout=180)
  run([LLC, "-filetype=obj", str(pipeline_dir / "08_llvm.ll"), "-o",
       str(mlir_obj)])
  run([
      CC,
      str(mlir_obj),
      str(pipeline_dir / "systolic_runtime.o"),
      str(main_c),
      f"-L{MLIR_RUNNER_UTILS_DIR}",
      f"-Wl,-rpath,{MLIR_RUNNER_UTILS_DIR}",
      "-lmlir_c_runner_utils",
      "-lmlir_runner_utils",
      "-lm",
      "-o",
      str(app),
  ])

  return BuiltTest(test=test, test_dir=test_dir, pipeline_dir=pipeline_dir,
                   app=app)
