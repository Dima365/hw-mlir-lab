#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  uint8_t *allocated;
  uint8_t *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} MemRef4DU8;

extern void _mlir_ciface_identity_block_entry(MemRef4DU8 *input,
                                               MemRef4DU8 *output);
extern void _mlir_ciface_downsample_block_entry(MemRef4DU8 *input,
                                                 MemRef4DU8 *output);

static const uint8_t expected_identity[128] = {
    100, 100, 100, 101, 100, 100, 103, 106, 100, 103, 106, 100, 105,
    108, 100, 100, 103, 106, 109, 100, 108, 100, 100, 100, 100, 100,
    100, 101, 100, 100, 101, 106, 100, 100, 100, 102, 100, 101, 104,
    107, 101, 104, 107, 100, 106, 100, 100, 100, 104, 107, 110, 100,
    109, 100, 100, 100, 100, 100, 100, 102, 100, 100, 102, 107, 100,
    100, 100, 103, 100, 102, 105, 108, 102, 105, 108, 100, 107, 100,
    100, 100, 105, 108, 100, 100, 110, 100, 100, 100, 100, 100, 100,
    103, 100, 100, 103, 108, 100, 100, 101, 104, 100, 103, 106, 109,
    103, 106, 100, 100, 108, 100, 100, 100, 106, 109, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 104, 100, 101, 104, 109};

static const uint8_t expected_downsample[64] = {
    100, 100, 100, 101, 100, 100, 100, 101, 100, 100, 100,
    101, 100, 100, 100, 101, 100, 100, 100, 101, 100, 100,
    100, 101, 100, 100, 100, 101, 100, 100, 100, 101, 100,
    100, 100, 101, 100, 100, 100, 101, 100, 100, 100, 101,
    100, 100, 100, 101, 100, 100, 100, 101, 100, 100, 100,
    101, 100, 100, 100, 101, 100, 100, 100, 101};

static MemRef4DU8 make_ref(uint8_t *data, int64_t channels, int64_t height,
                           int64_t width) {
  MemRef4DU8 ref = {
      .allocated = data,
      .aligned = data,
      .offset = 0,
      .sizes = {1, channels, height, width},
      .strides = {channels * height * width, height * width, width, 1},
  };
  return ref;
}

static int check_output(const char *name, const uint8_t *actual,
                        const uint8_t *expected, int64_t size) {
  for (int64_t index = 0; index < size; ++index) {
    if (actual[index] == expected[index])
      continue;
    fprintf(stderr, "%s mismatch at %ld: got %u, expected %u\n", name,
            index, actual[index], expected[index]);
    return 1;
  }
  return 0;
}

int main(void) {
  uint8_t input[128];
  for (int64_t channel = 0; channel < 8; ++channel) {
    for (int64_t row = 0; row < 4; ++row) {
      for (int64_t column = 0; column < 4; ++column) {
        int64_t index = channel * 16 + row * 4 + column;
        input[index] = (uint8_t)(100 +
            ((channel * 11 + row * 5 + column * 3 + 2) % 21) - 10);
      }
    }
  }

  uint8_t identity_output[128] = {0};
  uint8_t downsample_output[64] = {0};
  MemRef4DU8 input_ref = make_ref(input, 8, 4, 4);
  MemRef4DU8 identity_ref = make_ref(identity_output, 8, 4, 4);
  MemRef4DU8 downsample_ref = make_ref(downsample_output, 16, 2, 2);

  _mlir_ciface_identity_block_entry(&input_ref, &identity_ref);
  _mlir_ciface_downsample_block_entry(&input_ref, &downsample_ref);

  if (check_output("identity", identity_output, expected_identity, 128) ||
      check_output("downsample", downsample_output, expected_downsample, 64))
    return 1;

  puts("PASS identity_block");
  puts("PASS downsample_block");
  return 0;
}
