#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint8_t *allocated;
  uint8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DU8;

extern void _mlir_ciface_qadd_relu_entry(
    MemRef2DU8 *lhs, MemRef2DU8 *rhs, MemRef2DU8 *out);

static int64_t round_shift_rne(int64_t value, int32_t shift) {
  if (shift == 0)
    return value;

  uint64_t magnitude = value < 0 ? (uint64_t)(-value) : (uint64_t)value;
  uint64_t quotient = magnitude >> shift;
  uint64_t remainder = magnitude & (((uint64_t)1 << shift) - 1);
  uint64_t half = (uint64_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return value < 0 ? -(int64_t)quotient : (int64_t)quotient;
}

static uint8_t qadd_relu(
    uint8_t lhs, uint8_t rhs, int32_t lhs_multiplier,
    int32_t rhs_multiplier, int32_t shift, int32_t lhs_zero_point,
    int32_t rhs_zero_point, int32_t output_zero_point,
    int32_t relu_enable) {
  int64_t wide =
      ((int64_t)lhs - lhs_zero_point) * lhs_multiplier +
      ((int64_t)rhs - rhs_zero_point) * rhs_multiplier;
  int64_t centered = round_shift_rne(wide, shift);
  if (relu_enable && centered < 0)
    centered = 0;

  int64_t quantized = centered + output_zero_point;
  if (quantized < 0)
    quantized = 0;
  if (quantized > UINT8_MAX)
    quantized = UINT8_MAX;
  return (uint8_t)quantized;
}

int main(void) {
  enum { N = 8, ELEMS = N * N };
  const int32_t lhs_multiplier = 1715842648;
  const int32_t rhs_multiplier = 1328001668;
  const int32_t shift = 30;
  const int32_t lhs_zero_point = 68;
  const int32_t rhs_zero_point = 65;
  const int32_t output_zero_point = 0;
  const int32_t relu_enable = 1;

  uint8_t lhs[ELEMS];
  uint8_t rhs[ELEMS];
  uint8_t out[ELEMS] = {0};
  for (int i = 0; i < ELEMS; ++i) {
    lhs[i] = (uint8_t)((17 * i + 11) & 0xFF);
    rhs[i] = (uint8_t)((29 * i + 7) & 0xFF);
  }

  MemRef2DU8 lhs_ref = {lhs, lhs, 0, {N, N}, {N, 1}};
  MemRef2DU8 rhs_ref = {rhs, rhs, 0, {N, N}, {N, 1}};
  MemRef2DU8 out_ref = {out, out, 0, {N, N}, {N, 1}};

  _mlir_ciface_qadd_relu_entry(&lhs_ref, &rhs_ref, &out_ref);

  for (int i = 0; i < ELEMS; ++i) {
    uint8_t expected = qadd_relu(
        lhs[i], rhs[i], lhs_multiplier, rhs_multiplier, shift,
        lhs_zero_point, rhs_zero_point, output_zero_point, relu_enable);
    if (out[i] != expected) {
      printf("FAIL at %d: got %u expected %u\n", i, (unsigned)out[i],
             (unsigned)expected);
      return 1;
    }
  }

  printf("PASS\n");
  return 0;
}
