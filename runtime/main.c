#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DF32;

extern void _mlir_ciface_matmul_entry(MemRef2DF32 *a, MemRef2DF32 *b,
                                      MemRef2DF32 *c);

static float memref_get(const MemRef2DF32 *memref, int64_t i, int64_t j) {
  return memref->aligned[memref->offset + i * memref->strides[0] +
                         j * memref->strides[1]];
}

int main(void) {
  float a[8 * 16];
  float b[16 * 8];
  float c_storage[8 * 8];

  for (int i = 0; i < 8 * 16; ++i)
    a[i] = 1.0f;

  for (int i = 0; i < 16 * 8; ++i)
    b[i] = 1.0f;

  MemRef2DF32 a_ref = {a, a, 0, {8, 16}, {16, 1}};
  MemRef2DF32 b_ref = {b, b, 0, {16, 8}, {8, 1}};
  MemRef2DF32 c_ref = {c_storage, c_storage, 0, {8, 8}, {8, 1}};

  _mlir_ciface_matmul_entry(&a_ref, &b_ref, &c_ref);

  for (int64_t i = 0; i < c_ref.sizes[0]; ++i) {
    for (int64_t j = 0; j < c_ref.sizes[1]; ++j)
      printf("%6.1f ", memref_get(&c_ref, i, j));
    printf("\n");
  }

  return 0;
}
