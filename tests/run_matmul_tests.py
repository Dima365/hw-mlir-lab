#!/usr/bin/env python3
from matmul.cases import TESTS
from matmul.runner import run_tests


if __name__ == "__main__":
  raise SystemExit(run_tests(TESTS))
