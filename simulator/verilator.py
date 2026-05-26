import socket
import struct
import array
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
    assert payload_bytes == 3 * 64 * 4

    a = array.array("f")
    b = array.array("f")
    c = array.array("f")
    a.frombytes(read_exact(conn, 64 * 4))
    b.frombytes(read_exact(conn, 64 * 4))
    c.frombytes(read_exact(conn, 64 * 4))

    # Тут позже вызываешь cocotb/Verilator.
    for i in range(8):
        for j in range(8):
            acc = c[i * 8 + j]
            for k in range(8):
                acc += a[i * 8 + k] * b[k * 8 + j]
            c[i * 8 + j] = acc

    conn.sendall(c.tobytes())
