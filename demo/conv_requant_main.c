#include <stdint.h>
#include <stdio.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DI32;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1DI32;

typedef struct {
  uint8_t *allocated;
  uint8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DU8;

extern void _mlir_ciface_conv_requant_entry(
    MemRef2DI32 *acc, MemRef1DI32 *multiplier, MemRef1DI32 *shift,
    MemRef2DU8 *out);

// Signed round-to-nearest, ties-to-even division by 2^shift.
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

static uint8_t conv_requantize(int32_t acc, int32_t multiplier, int32_t shift,
                               int32_t output_zero_point,
                               int32_t relu_enable) {
  int64_t scaled =
      round_shift_rne((int64_t)acc * (int64_t)multiplier, shift);
  if (relu_enable && scaled < 0)
    scaled = 0;

  int64_t quantized = scaled + output_zero_point;
  if (quantized < 0)
    quantized = 0;
  if (quantized > UINT8_MAX)
    quantized = UINT8_MAX;
  return (uint8_t)quantized;
}

int main(void) {
  enum { N = 8, ELEMS = N * N };
  const int32_t output_zero_point = 17;
  const int32_t relu_enable = 1;

  int32_t acc[ELEMS];
  int32_t multiplier[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t shift[N] = {3, 3, 3, 3, 3, 3, 3, 3};
  uint8_t out[ELEMS] = {0};

  for (int row = 0; row < N; ++row)
    for (int channel = 0; channel < N; ++channel)
      acc[row * N + channel] = (row - 3) * 16 + 5;

  MemRef2DI32 acc_ref = {acc, acc, 0, {N, N}, {N, 1}};
  MemRef1DI32 multiplier_ref = {
      multiplier, multiplier, 0, {N}, {1}};
  MemRef1DI32 shift_ref = {shift, shift, 0, {N}, {1}};
  MemRef2DU8 out_ref = {out, out, 0, {N, N}, {N, 1}};

  _mlir_ciface_conv_requant_entry(
      &acc_ref, &multiplier_ref, &shift_ref, &out_ref);

  for (int idx = 0; idx < ELEMS; ++idx) {
    int channel = idx % N;
    uint8_t expected =
        conv_requantize(acc[idx], multiplier[channel], shift[channel],
                        output_zero_point, relu_enable);
    if (out[idx] != expected) {
      printf("FAIL at %d: got %u expected %u\n", idx, (unsigned)out[idx],
             (unsigned)expected);
      return 1;
    }
  }

  printf("PASS\n");
  return 0;
}
