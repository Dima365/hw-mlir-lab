from typing import Callable


Pattern = Callable[[int, int], float]


def generate_matmul_mlir(m: int, k: int, n: int) -> str:
  return f"""module {{
  func.func @matmul(
    %arg0: tensor<{m}x{k}xf32>,
    %arg1: tensor<{k}x{n}xf32>
  ) -> tensor<{m}x{n}xf32> {{
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %zero = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
      outs(%zero : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %result : tensor<{m}x{n}xf32>
  }}
}}
"""


def matrix(rows: int, cols: int, fn: Pattern) -> list[float]:
  return [fn(i, j) for i in range(rows) for j in range(cols)]


def expected_matmul(m: int, k: int, n: int, a: list[float],
                    b: list[float]) -> list[float]:
  result = []
  for i in range(m):
    for j in range(n):
      acc = 0.0
      for kk in range(k):
        acc += a[i * k + kk] * b[kk * n + j]
      result.append(acc)
  return result


def c_float_literal(value: float) -> str:
  text = f"{value:.8g}"
  if "." not in text and "e" not in text and "E" not in text:
    text += ".0"
  return text + "f"


def c_float_list(values: list[float]) -> str:
  return ", ".join(c_float_literal(value) for value in values)


def generate_main_c(m: int, k: int, n: int, a: list[float], b: list[float],
                    expected: list[float]) -> str:
  return f"""#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {{
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
}} MemRef2DF32;

extern void _mlir_ciface_matmul_entry(MemRef2DF32 *a, MemRef2DF32 *b,
                                      MemRef2DF32 *c);

int main(void) {{
  float a[{m * k}] = {{ {c_float_list(a)} }};
  float b[{k * n}] = {{ {c_float_list(b)} }};
  float c[{m * n}] = {{0}};
  float expected[{m * n}] = {{ {c_float_list(expected)} }};

  MemRef2DF32 a_ref = {{a, a, 0, {{{m}, {k}}}, {{{k}, 1}}}};
  MemRef2DF32 b_ref = {{b, b, 0, {{{k}, {n}}}, {{{n}, 1}}}};
  MemRef2DF32 c_ref = {{c, c, 0, {{{m}, {n}}}, {{{n}, 1}}}};

  _mlir_ciface_matmul_entry(&a_ref, &b_ref, &c_ref);

  for (int64_t i = 0; i < {m * n}; ++i) {{
    float diff = fabsf(c[i] - expected[i]);
    if (diff > 1.0e-4f) {{
      printf("FAIL at %ld: got %.8g expected %.8g\\n", (long)i, c[i],
             expected[i]);
      return 1;
    }}
  }}

  printf("PASS\\n");
  return 0;
}}
"""
