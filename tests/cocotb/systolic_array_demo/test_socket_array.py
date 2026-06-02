import os
import socket
import struct
import subprocess
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


SOCK = Path(os.environ.get("SOCK", "/tmp/systolic_cocotb.sock"))
READY = Path(os.environ.get("READY", "/tmp/systolic_cocotb.ready"))
REPO_ROOT = Path(os.environ["REPO_ROOT"])
APP = os.environ["APP"]

MAGIC = 0x54535953
OP_MATMUL_8X8 = 1
PAYLOAD_BYTES = 2 * 64 + 64 * 4


async def poll_until_ready(fn, timeout_ns=2_000_000_000):
  elapsed = 0
  step = 1_000_000
  while elapsed < timeout_ns:
    result = fn()
    if result is not None:
      return result
    await Timer(step, units="ns")
    elapsed += step
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
  state = bytearray()
  return await poll_until_ready(lambda: recv_nonblocking(conn, size, state))


def pack_i8(values):
  word = 0
  for idx, value in enumerate(values):
    word |= (value & 0xFF) << (idx * 8)
  return word


def unpack_i32(word):
  values = []
  for idx in range(64):
    raw = (word >> (idx * 32)) & 0xFFFFFFFF
    if raw & 0x80000000:
      raw -= 0x100000000
    values.append(raw)
  return values


async def run_array(dut, a, b):
  dut.a_flat.value = pack_i8(a)
  dut.b_flat.value = pack_i8(b)
  dut.start.value = 1
  await RisingEdge(dut.clk)
  dut.start.value = 0

  for _ in range(32):
    await RisingEdge(dut.clk)
    if int(dut.done.value) == 1:
      return unpack_i32(int(dut.c_flat.value))

  raise TimeoutError("DUT did not assert done")


async def reset_dut(dut):
  dut.rst.value = 1
  dut.start.value = 0
  dut.a_flat.value = 0
  dut.b_flat.value = 0
  await RisingEdge(dut.clk)
  await RisingEdge(dut.clk)
  dut.rst.value = 0
  await RisingEdge(dut.clk)


@cocotb.test()
async def test_socket_array(dut):
  cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
  await reset_dut(dut)

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
      header = await read_exact(conn, 12)
      magic, opcode, payload_bytes = struct.unpack("<III", header)
      assert magic == MAGIC
      assert opcode == OP_MATMUL_8X8
      assert payload_bytes == PAYLOAD_BYTES

      a = struct.unpack("<64b", await read_exact(conn, 64))
      b = struct.unpack("<64b", await read_exact(conn, 64))
      c_in = struct.unpack("<64i", await read_exact(conn, 64 * 4))

      partial = await run_array(dut, a, b)
      c = [c_in[idx] + partial[idx] for idx in range(64)]
      conn.sendall(struct.pack("<64i", *c))

  except EOFError:
    pass
  finally:
    stdout, stderr = app.communicate(timeout=2)
    assert app.returncode == 0, (
        f"app failed with code {app.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
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
