#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "accel_opcodes.h"

#define SYSTOLIC_SOCKET_PATH "/tmp/systolic_cocotb.sock"
#define ACCEL_MAGIC 0x54535953u // "SYST" little-endian

struct RequestHeader {
  uint32_t magic;
  uint32_t opcode;
  uint32_t param_bytes;
  uint32_t in_bytes;
  uint32_t out_bytes;
};

typedef struct {
  int8_t *allocated;
  int8_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DI8;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DI32;

static int systolic_fd = -1;

static void die(const char *msg) {
  perror(msg);
  abort();
}

static void write_all(int fd, const void *buf, size_t size) {
  const char *p = (const char *)buf;
  while (size > 0) {
    ssize_t n = write(fd, p, size);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      die("write");
    }
    p += n;
    size -= (size_t)n;
  }
}

static void read_all(int fd, void *buf, size_t size) {
  char *p = (char *)buf;
  while (size > 0) {
    ssize_t n = read(fd, p, size);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      die("read");
    }
    if (n == 0) {
      fprintf(stderr, "systolic runtime: python server closed connection\n");
      abort();
    }
    p += n;
    size -= (size_t)n;
  }
}

static int connect_to_server(void) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0)
    die("socket");

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SYSTOLIC_SOCKET_PATH, sizeof(addr.sun_path) - 1);

  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    die("connect");

  return fd;
}

static int get_connection(void) {
  if (systolic_fd < 0)
    systolic_fd = connect_to_server();
  return systolic_fd;
}

// Generic transport to the simulator: one call for any IP block.
// Protocol: header -> params -> in -> (read back) out.
static void accel_invoke(uint32_t opcode, const void *in, size_t in_bytes,
                         void *out, size_t out_bytes, const void *params,
                         size_t param_bytes) {
  int fd = get_connection();

  struct RequestHeader header = {
      .magic = ACCEL_MAGIC,
      .opcode = opcode,
      .param_bytes = (uint32_t)param_bytes,
      .in_bytes = (uint32_t)in_bytes,
      .out_bytes = (uint32_t)out_bytes,
  };

  write_all(fd, &header, sizeof(header));
  if (param_bytes)
    write_all(fd, params, param_bytes);
  if (in_bytes)
    write_all(fd, in, in_bytes);
  if (out_bytes)
    read_all(fd, out, out_bytes);
}

static void check_8x8_i8_memref(const char *name, const MemRef2DI8 *memref) {
  if (memref->sizes[0] != 8 || memref->sizes[1] != 8) {
    fprintf(stderr, "systolic runtime: %s must be 8x8, got %ldx%ld\n", name,
            (long)memref->sizes[0], (long)memref->sizes[1]);
    abort();
  }
}

static void check_8x8_i32_memref(const char *name, const MemRef2DI32 *memref) {
  if (memref->sizes[0] != 8 || memref->sizes[1] != 8) {
    fprintf(stderr, "systolic runtime: %s must be 8x8, got %ldx%ld\n", name,
            (long)memref->sizes[0], (long)memref->sizes[1]);
    abort();
  }
}

static void pack_i8_8x8(const MemRef2DI8 *src, int8_t dst[64]) {
  int8_t *base = src->aligned + src->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      dst[i * 8 + j] = base[i * src->strides[0] + j * src->strides[1]];
}

static void pack_i32_8x8(const MemRef2DI32 *src, int32_t dst[64]) {
  int32_t *base = src->aligned + src->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      dst[i * 8 + j] = base[i * src->strides[0] + j * src->strides[1]];
}

static void unpack_i32_8x8(const int32_t src[64], MemRef2DI32 *dst) {
  int32_t *base = dst->aligned + dst->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      base[i * dst->strides[0] + j * dst->strides[1]] = src[i * 8 + j];
}

static void unpack_i8_8x8(const int8_t src[64], MemRef2DI8 *dst) {
  int8_t *base = dst->aligned + dst->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      base[i * dst->strides[0] + j * dst->strides[1]] = src[i * 8 + j];
}

void systolic_matmul_8x8(
    int8_t *a_allocated, int8_t *a_aligned, int64_t a_offset, int64_t a_size0,
    int64_t a_size1, int64_t a_stride0, int64_t a_stride1,
    int8_t *b_allocated, int8_t *b_aligned, int64_t b_offset, int64_t b_size0,
    int64_t b_size1, int64_t b_stride0, int64_t b_stride1,
    int32_t *c_allocated, int32_t *c_aligned, int64_t c_offset,
    int64_t c_size0,
    int64_t c_size1, int64_t c_stride0, int64_t c_stride1) {
  MemRef2DI8 a = {a_allocated, a_aligned, a_offset,
                  {a_size0, a_size1},
                  {a_stride0, a_stride1}};
  MemRef2DI8 b = {b_allocated, b_aligned, b_offset,
                  {b_size0, b_size1},
                  {b_stride0, b_stride1}};
  MemRef2DI32 c = {c_allocated, c_aligned, c_offset,
                   {c_size0, c_size1},
                   {c_stride0, c_stride1}};

  check_8x8_i8_memref("lhs", &a);
  check_8x8_i8_memref("rhs", &b);
  check_8x8_i32_memref("acc", &c);

  int8_t a_buf[64];
  int8_t b_buf[64];
  int32_t c_buf[64];
  pack_i8_8x8(&a, a_buf);
  pack_i8_8x8(&b, b_buf);
  pack_i32_8x8(&c, c_buf);

  // input = a(i8x64) || b(i8x64) || c_in(i32x64), output = c(i32x64)
  uint8_t in[sizeof(a_buf) + sizeof(b_buf) + sizeof(c_buf)];
  memcpy(in, a_buf, sizeof(a_buf));
  memcpy(in + sizeof(a_buf), b_buf, sizeof(b_buf));
  memcpy(in + sizeof(a_buf) + sizeof(b_buf), c_buf, sizeof(c_buf));

  accel_invoke(OP_MATMUL, in, sizeof(in), c_buf, sizeof(c_buf), NULL, 0);
  unpack_i32_8x8(c_buf, &c);
}

void epilogue_8x8(
    int32_t *acc_allocated, int32_t *acc_aligned, int64_t acc_offset,
    int64_t acc_size0, int64_t acc_size1, int64_t acc_stride0,
    int64_t acc_stride1,
    int8_t *out_allocated, int8_t *out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0,
    int64_t out_stride1,
    int32_t mult, int32_t shift, int32_t zero_point) {
  MemRef2DI32 acc = {acc_allocated, acc_aligned, acc_offset,
                     {acc_size0, acc_size1},
                     {acc_stride0, acc_stride1}};
  MemRef2DI8 out = {out_allocated, out_aligned, out_offset,
                    {out_size0, out_size1},
                    {out_stride0, out_stride1}};

  check_8x8_i32_memref("acc", &acc);
  check_8x8_i8_memref("out", &out);

  int32_t acc_buf[64];
  int8_t out_buf[64];
  pack_i32_8x8(&acc, acc_buf);

  // params = mult, shift, zero_point (3 x i32)
  int32_t params[3] = {mult, shift, zero_point};

  accel_invoke(OP_EPILOGUE, acc_buf, sizeof(acc_buf), out_buf, sizeof(out_buf),
               params, sizeof(params));
  unpack_i8_8x8(out_buf, &out);
}
