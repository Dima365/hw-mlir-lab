"""Epilogue IP driver: drives the requantize+ReLU block (ip/epilogue_demo/epilogue.sv)."""
import struct

from cocotb.triggers import ReadOnly, ReadWrite, RisingEdge, FallingEdge

from .base import IPDriver

N = 8
ELEMS = N * N
OUT_MAX = 127
MASK32 = 0xFFFFFFFF


def _pack_i32(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & MASK32) << (idx * 32)
    return word


def _unpack_i8(word):
    values = []
    for idx in range(ELEMS):
        raw = (word >> (idx * 8)) & 0xFF
        if raw & 0x80:
            raw -= 0x100
        values.append(raw)
    return values


def requantize(c, mult, shift, zero_point):
    """Python golden: per-tensor requantize + ReLU, round half up.

    Mirrors epilogue.sv exactly (Python >> is arithmetic floor, like SV >>>).
    """
    round_add = (1 << (shift - 1)) if shift > 0 else 0
    q = ((c * mult + round_add) >> shift) + zero_point
    return max(0, min(OUT_MAX, q))


class EpilogueDriver(IPDriver):
    NAME = "epilogue"

    async def setup(self, dut):
        dut.rst.value = 1
        dut.start.value = 0
        dut.c_in_flat.value = 0
        dut.mult.value = 0
        dut.shift.value = 0
        dut.zero_point.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        # params = mult, shift, zero_point (3 x i32); in = c_in (i32 x 64)
        mult, shift, zero_point = struct.unpack("<3i", params)
        c_in = struct.unpack("<64i", data_in)

        await ReadWrite()
        dut.c_in_flat.value = _pack_i32(c_in)
        dut.mult.value = mult & MASK32
        dut.shift.value = shift
        dut.zero_point.value = zero_point & MASK32
        dut.start.value = 1

        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            out = _unpack_i8(int(dut.out_flat.value))
            await FallingEdge(dut.clk)
            await ReadWrite()
            dut.start.value = 0
            return struct.pack("<64b", *out)

        raise TimeoutError("epilogue DUT did not assert done")

    def smoke_cases(self):
        # (mult, shift, zero_point, c_in[64])
        scenarios = [
            # scale ~= 0.0123: mix of negatives (-> ReLU 0) and positives
            (12897, 20, 0, [(i - 32) * 200 for i in range(ELEMS)]),
            # scale == 1.0 (mult=2^20, shift=20): out == clamp(c_in, 0, 127)
            (1 << 20, 20, 0, [(i * 5) - 40 for i in range(ELEMS)]),
        ]
        cases = []
        for mult, shift, zero_point, c_in in scenarios:
            params = struct.pack("<3i", mult, shift, zero_point)
            data_in = struct.pack("<64i", *c_in)
            expected = struct.pack(
                "<64b", *[requantize(c, mult, shift, zero_point) for c in c_in]
            )
            cases.append((params, data_in, expected))
        return cases
