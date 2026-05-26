from .build import build_test
from .cases import MatmulTest
from .execute import run_app_with_server


def run_test(test: MatmulTest) -> None:
  built = build_test(test)
  completed = run_app_with_server(built.app)
  if "PASS" not in completed.stdout:
    raise RuntimeError(f"test app did not report PASS:\n{completed.stdout}")


def run_tests(tests: list[MatmulTest]) -> int:
  failed = 0
  for test in tests:
    try:
      run_test(test)
      print(f"PASS {test.name}")
    except Exception as exc:
      failed += 1
      print(f"FAIL {test.name}: {exc}")

  passed = len(tests) - failed
  print(f"{passed} passed, {failed} failed")
  return 1 if failed else 0
