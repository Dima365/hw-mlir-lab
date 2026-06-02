import socket
import struct
import os
from pathlib import Path

SOCK = "/tmp/systolic_cocotb.sock"
READY = "/tmp/systolic_cocotb.ready"
MAGIC = 0x54535953
OP_MATMUL_8X8 = 1

def read_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise EOFError("client closed")
        data.extend(chunk)
    return bytes(data)

for path in (SOCK, READY):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(1)
Path(READY).touch()

conn, _ = srv.accept()
while True:
    hdr = read_exact(conn, 12)
    magic, opcode, payload_bytes = struct.unpack("<III", hdr)
    assert magic == MAGIC
    assert opcode == OP_MATMUL_8X8
    assert payload_bytes == 2 * 64 + 64 * 4

    a = struct.unpack("<64b", read_exact(conn, 64))
    b = struct.unpack("<64b", read_exact(conn, 64))
    c = list(struct.unpack("<64i", read_exact(conn, 64 * 4)))

    # Тут позже вызываешь cocotb/Verilator.
    for i in range(8):
        for j in range(8):
            acc = c[i * 8 + j]
            for k in range(8):
                acc += a[i * 8 + k] * b[k * 8 + j]
            c[i * 8 + j] = acc

    conn.sendall(struct.pack("<64i", *c))
