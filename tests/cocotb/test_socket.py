"""Generic cocotb harness.

Serves accel requests over a unix socket and routes each to the IP driver
selected by opcode (drivers/registry.py). The single DUT in this simulation is
chosen by the Makefile (TOPLEVEL from the manifest, via IP=<name>); the matching
driver drives it. The harness itself only moves bytes and knows nothing
IP-specific.

Wire protocol (mirrors interface/interface.c):
  header  = <5 x uint32 LE>: magic, opcode, param_bytes, in_bytes, out_bytes
  payload = params (param_bytes) ++ data_in (in_bytes)
  reply   = out_bytes
"""
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import cocotb
from cocotb.clock import Clock

from drivers import registry

SOCK = Path(os.environ.get("SOCK", "/tmp/systolic_cocotb.sock"))
READY = Path(os.environ.get("READY", "/tmp/systolic_cocotb.ready"))
REPO_ROOT = Path(os.environ["REPO_ROOT"])
APP = os.environ["APP"]
IP = os.environ.get("IP", "matmul")

MAGIC = 0x54535953
HEADER = struct.Struct("<5I")  # magic, opcode, param_bytes, in_bytes, out_bytes


async def poll_until_ready(fn, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = fn()
        if result is not None:
            return result
        time.sleep(0.001)
    raise TimeoutError("timed out waiting for socket event")


def accept_nonblocking(server):
    try:
        conn, _ = server.accept()
        conn.setblocking(False)
        return conn
    except BlockingIOError:
        return None


def recv_nonblocking(conn, size, state):
    try:
        chunk = conn.recv(size - len(state))
    except BlockingIOError:
        return None
    if not chunk:
        raise EOFError("client closed socket")
    state.extend(chunk)
    if len(state) == size:
        data = bytes(state)
        state.clear()
        return data
    return None


async def read_exact(conn, size):
    if size == 0:
        return b""
    state = bytearray()
    return await poll_until_ready(lambda: recv_nonblocking(conn, size, state))


@cocotb.test()
async def test_socket(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await registry.BY_NAME[IP].setup(dut)

    for path in (SOCK, READY):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setblocking(False)
    server.bind(str(SOCK))
    server.listen(1)
    READY.touch()

    app = subprocess.Popen(
        [APP],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    conn = None
    try:
        conn = await poll_until_ready(lambda: accept_nonblocking(server))

        while app.poll() is None:
            header = await read_exact(conn, HEADER.size)
            magic, opcode, param_bytes, in_bytes, out_bytes = HEADER.unpack(header)
            assert magic == MAGIC, f"bad magic {magic:#x}"

            driver = registry.BY_OPCODE[opcode]
            params = await read_exact(conn, param_bytes)
            data_in = await read_exact(conn, in_bytes)

            result = await driver.handle(dut, params, data_in)
            assert len(result) == out_bytes, (
                f"driver returned {len(result)} bytes, header expects {out_bytes}"
            )
            conn.sendall(result)

    except EOFError:
        pass
    finally:
        stdout, stderr = app.communicate(timeout=2)
        assert app.returncode == 0, (
            f"app failed with code {app.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
        if conn is not None:
            conn.close()
        server.close()
        if app.poll() is None:
            app.terminate()
            app.wait(timeout=2)
        for path in (SOCK, READY):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
