from typing import Callable


Pattern = Callable[[int, int], int]


def generate_matmul_mlir(m: int, k: int, n: int) -> str:
  return f"""module {{
  func.func @matmul(
    %arg0: tensor<{m}x{k}xi8>,
    %arg1: tensor<{k}x{n}xi8>
  ) -> tensor<{m}x{n}xi32> {{
    %empty = tensor.empty() : tensor<{m}x{n}xi32>
    %c0 = arith.constant 0 : i32
    %zero = linalg.fill ins(%c0 : i32)
      outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
      outs(%zero : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    return %result : tensor<{m}x{n}xi32>
  }}
}}
"""


def matrix(rows: int, cols: int, fn: Pattern) -> list[int]:
  return [fn(i, j) for i in range(rows) for j in range(cols)]


def expected_matmul(m: int, k: int, n: int, a: list[int],
                    b: list[int]) -> list[int]:
  result = []
  for i in range(m):
    for j in range(n):
      acc = 0
      for kk in range(k):
        acc += a[i * k + kk] * b[kk * n + j]
      result.append(acc)
  return result


def c_int_list(values: list[int]) -> str:
  return ", ".join(str(value) for value in values)


def generate_main_c(m: int, k: int, n: int, a: list[int], b: list[int],
                    expected: list[int]) -> str:
  return f"""#include <stdint.h>
#include <stdio.h>

typedef struct {{
  int8_t *allocated;
  int8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
}} MemRef2DI8;

typedef struct {{
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
}} MemRef2DI32;

extern void _mlir_ciface_matmul_entry(MemRef2DI8 *a, MemRef2DI8 *b,
                                      MemRef2DI32 *c);

int main(void) {{
  int8_t a[{m * k}] = {{ {c_int_list(a)} }};
  int8_t b[{k * n}] = {{ {c_int_list(b)} }};
  int32_t c[{m * n}] = {{0}};
  int32_t expected[{m * n}] = {{ {c_int_list(expected)} }};

  MemRef2DI8 a_ref = {{a, a, 0, {{{m}, {k}}}, {{{k}, 1}}}};
  MemRef2DI8 b_ref = {{b, b, 0, {{{k}, {n}}}, {{{n}, 1}}}};
  MemRef2DI32 c_ref = {{c, c, 0, {{{m}, {n}}}, {{{n}, 1}}}};

  _mlir_ciface_matmul_entry(&a_ref, &b_ref, &c_ref);

  for (int64_t i = 0; i < {m * n}; ++i) {{
    if (c[i] != expected[i]) {{
      printf("FAIL at %ld: got %d expected %d\\n", (long)i, c[i], expected[i]);
      return 1;
    }}
  }}

  printf("PASS\\n");
  return 0;
}}
"""
