from dataclasses import dataclass

from .templates import Pattern


@dataclass(frozen=True)
class MatmulTest:
  name: str
  m: int
  k: int
  n: int
  a_fn: Pattern
  b_fn: Pattern


def ones(i: int, j: int) -> int:
  return 1


def row_col(i: int, j: int) -> int:
  return (i + 2 * j) % 7 - 3


def small_mod(i: int, j: int) -> int:
  return (3 * i - j) % 5 - 2


TESTS = [
    MatmulTest("8x16x8_ones", 8, 16, 8, ones, ones),
    MatmulTest("8x16x8_pattern", 8, 16, 8, row_col, small_mod),
    MatmulTest("4x8x16_ones", 4, 8, 16, ones, ones),
    MatmulTest("5x9x6_pattern", 5, 9, 6, row_col, small_mod),
]
