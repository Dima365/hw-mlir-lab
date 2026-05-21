#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define SYSTOLIC_SOCKET_PATH "/tmp/systolic_cocotb.sock"
#define SYSTOLIC_OPCODE_MATMUL_8X8 1u

struct RequestHeader {
  uint32_t magic;
  uint32_t opcode;
  uint32_t payload_bytes;
};

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DF32;

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
  for (int attempt = 0; attempt < 100; ++attempt) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
      die("socket");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SYSTOLIC_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
      return fd;

    int saved_errno = errno;
    close(fd);

    if (saved_errno == ENOENT || saved_errno == ECONNREFUSED) {
      usleep(10000);
      continue;
    }

    errno = saved_errno;
    die("connect");
  }

  fprintf(stderr, "systolic runtime: timed out waiting for server socket %s\n",
          SYSTOLIC_SOCKET_PATH);
  abort();
}

static int get_connection(void) {
  if (systolic_fd < 0)
    systolic_fd = connect_to_server();
  return systolic_fd;
}

static void check_8x8_memref(const char *name, const MemRef2DF32 *memref) {
  if (memref->sizes[0] != 8 || memref->sizes[1] != 8) {
    fprintf(stderr, "systolic runtime: %s must be 8x8, got %ldx%ld\n", name,
            (long)memref->sizes[0], (long)memref->sizes[1]);
    abort();
  }
}

static void pack_8x8(const MemRef2DF32 *src, float dst[64]) {
  float *base = src->aligned + src->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      dst[i * 8 + j] = base[i * src->strides[0] + j * src->strides[1]];
}

static void unpack_8x8(const float src[64], MemRef2DF32 *dst) {
  float *base = dst->aligned + dst->offset;
  for (int64_t i = 0; i < 8; ++i)
    for (int64_t j = 0; j < 8; ++j)
      base[i * dst->strides[0] + j * dst->strides[1]] = src[i * 8 + j];
}

void systolic_matmul_8x8(
    float *a_allocated, float *a_aligned, int64_t a_offset, int64_t a_size0,
    int64_t a_size1, int64_t a_stride0, int64_t a_stride1,
    float *b_allocated, float *b_aligned, int64_t b_offset, int64_t b_size0,
    int64_t b_size1, int64_t b_stride0, int64_t b_stride1,
    float *c_allocated, float *c_aligned, int64_t c_offset, int64_t c_size0,
    int64_t c_size1, int64_t c_stride0, int64_t c_stride1) {
  MemRef2DF32 a = {a_allocated, a_aligned, a_offset,
                   {a_size0, a_size1},
                   {a_stride0, a_stride1}};
  MemRef2DF32 b = {b_allocated, b_aligned, b_offset,
                   {b_size0, b_size1},
                   {b_stride0, b_stride1}};
  MemRef2DF32 c = {c_allocated, c_aligned, c_offset,
                   {c_size0, c_size1},
                   {c_stride0, c_stride1}};

  check_8x8_memref("lhs", &a);
  check_8x8_memref("rhs", &b);
  check_8x8_memref("acc", &c);

  float a_buf[64];
  float b_buf[64];
  float c_buf[64];
  pack_8x8(&a, a_buf);
  pack_8x8(&b, b_buf);
  pack_8x8(&c, c_buf);

  int fd = get_connection();

  struct RequestHeader header;
  header.magic = 0x54535953u; // "SYST" little-endian
  header.opcode = SYSTOLIC_OPCODE_MATMUL_8X8;
  header.payload_bytes = 3u * 64u * sizeof(float);

  write_all(fd, &header, sizeof(header));
  write_all(fd, a_buf, sizeof(a_buf));
  write_all(fd, b_buf, sizeof(b_buf));
  write_all(fd, c_buf, sizeof(c_buf));

  read_all(fd, c_buf, sizeof(c_buf));
  unpack_8x8(c_buf, &c);
}
