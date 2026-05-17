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

void systolic_matmul_8x8(float *a, float *b, float *c) {
  int fd = get_connection();

  struct RequestHeader header;
  header.magic = 0x54535953u; // "SYST" little-endian
  header.opcode = SYSTOLIC_OPCODE_MATMUL_8X8;
  header.payload_bytes = 3u * 64u * sizeof(float);

  write_all(fd, &header, sizeof(header));
  write_all(fd, a, 64u * sizeof(float));
  write_all(fd, b, 64u * sizeof(float));
  write_all(fd, c, 64u * sizeof(float));

  read_all(fd, c, 64u * sizeof(float));
}
