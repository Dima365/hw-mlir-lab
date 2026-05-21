#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DF32;

extern MemRef2DF32 matmul(float *a_allocated, float *a_aligned,
                          int64_t a_offset, int64_t a_size0,
                          int64_t a_size1, int64_t a_stride0,
                          int64_t a_stride1, float *b_allocated,
                          float *b_aligned, int64_t b_offset,
                          int64_t b_size0, int64_t b_size1,
                          int64_t b_stride0, int64_t b_stride1);

static float memref_get(const MemRef2DF32 *memref, int64_t i, int64_t j) {
  return memref->aligned[memref->offset + i * memref->strides[0] +
                         j * memref->strides[1]];
}

int main(void) {
  float a[8 * 16];
  float b[16 * 8];

  for (int i = 0; i < 8 * 16; ++i)
    a[i] = 1.0f;

  for (int i = 0; i < 16 * 8; ++i)
    b[i] = 1.0f;

  MemRef2DF32 c = matmul(a, a, 0, 8, 16, 16, 1, b, b, 0, 16, 8, 8, 1);

  for (int64_t i = 0; i < c.sizes[0]; ++i) {
    for (int64_t j = 0; j < c.sizes[1]; ++j)
      printf("%6.1f ", memref_get(&c, i, j));
    printf("\n");
  }

  return 0;
}
