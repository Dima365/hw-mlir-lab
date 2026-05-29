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
  app = test_dir / "app"

  input_mlir.write_text(generate_matmul_mlir(test.m, test.k, test.n))
  main_c.write_text(generate_main_c(test.m, test.k, test.n, a, b, expected))

  run(["./pipelines/mlir_pipeline.sh", str(input_mlir), str(pipeline_dir)],
      timeout=180)
  run([
      "./pipelines/compile_pipeline.sh",
      str(pipeline_dir / "08_llvm.ll"),
      str(main_c),
      str(app),
      str(pipeline_dir),
  ])

  return BuiltTest(test=test, test_dir=test_dir, pipeline_dir=pipeline_dir,
                   app=app)
