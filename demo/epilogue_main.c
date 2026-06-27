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
  int8_t *allocated;
  int8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DI8;

extern void _mlir_ciface_requant_entry(MemRef2DI32 *acc, MemRef2DI8 *out);

// Golden: per-tensor requantize + ReLU, round half up.
// Must match the attributes in demo/epilogue.mlir and the RTL/driver.
static int32_t requantize(int32_t c, int32_t mult, int32_t shift, int32_t zp) {
  int64_t prod = (int64_t)c * (int64_t)mult;
  int64_t round_add = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
  int64_t q = ((prod + round_add) >> shift) + zp;
  if (q < 0)
    q = 0;
  if (q > 127)
    q = 127;
  return (int32_t)q;
}

int main(void) {
  enum { N = 8, ELEMS = N * N };
  const int32_t mult = 12897, shift = 20, zero_point = 0;

  int32_t acc[ELEMS];
  int8_t out[ELEMS] = {0};
  for (int i = 0; i < ELEMS; ++i)
    acc[i] = (i - 32) * 200;

  MemRef2DI32 acc_ref = {acc, acc, 0, {N, N}, {N, 1}};
  MemRef2DI8 out_ref = {out, out, 0, {N, N}, {N, 1}};

  _mlir_ciface_requant_entry(&acc_ref, &out_ref);

  for (int i = 0; i < ELEMS; ++i) {
    int32_t expected = requantize(acc[i], mult, shift, zero_point);
    if (out[i] != expected) {
      printf("FAIL at %d: got %d expected %d\n", i, out[i], expected);
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}
