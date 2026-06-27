"""Matmul IP driver: drives the systolic array (ip/systolic_array_demo/array.sv)."""
import struct

from cocotb.triggers import ReadOnly, ReadWrite, RisingEdge

from .base import IPDriver

N = 8
ELEMS = N * N


def _pack_i8(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (idx * 8)
    return word


def _pack_i32(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFFFFFFFF) << (idx * 32)
    return word


def _unpack_i32(word):
    values = []
    for idx in range(ELEMS):
        raw = (word >> (idx * 32)) & 0xFFFFFFFF
        if raw & 0x80000000:
            raw -= 0x100000000
        values.append(raw)
    return values


class MatmulDriver(IPDriver):
    NAME = "matmul"

    async def setup(self, dut):
        dut.rst.value = 1
        dut.start.value = 0
        dut.a_flat.value = 0
        dut.b_flat.value = 0
        dut.c_in_flat.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        # in = a(i8 x 64) || b(i8 x 64) || c_in(i32 x 64); out = c(i32 x 64)
        a = struct.unpack("<64b", data_in[0:64])
        b = struct.unpack("<64b", data_in[64:128])
        c_in = struct.unpack("<64i", data_in[128:384])

        await ReadWrite()
        dut.a_flat.value = _pack_i8(a)
        dut.b_flat.value = _pack_i8(b)
        dut.c_in_flat.value = _pack_i32(c_in)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        await ReadWrite()
        dut.start.value = 0

        for _ in range(32):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.done.value) == 1:
                c = _unpack_i32(int(dut.c_out_flat.value))
                return struct.pack("<64i", *c)

        raise TimeoutError("DUT did not assert done")
