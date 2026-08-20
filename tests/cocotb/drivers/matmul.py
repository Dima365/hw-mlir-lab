"""Matmul IP driver: drives the systolic array (ip/systolic_array_demo/array.sv)."""
import struct

from cocotb.triggers import ReadOnly, ReadWrite, RisingEdge

from tests.golden.quantized_matmul import matmul_tile

from .base import IPDriver

N = 8
ELEMS = N * N
INPUT_BYTES = 2 * ELEMS + 4 * ELEMS
PARAM_BYTES = 2 * 4


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
        dut.a_is_unsigned.value = 0
        dut.a_zero_point.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        # params = a_is_unsigned, a_zero_point (2 x i32). Empty parameters retain
        # compatibility with the original signed-i8 protocol.
        if len(params) == 0:
            a_is_unsigned, a_zero_point = 0, 0
        elif len(params) == PARAM_BYTES:
            a_is_unsigned, a_zero_point = struct.unpack("<2i", params)
        else:
            raise ValueError(
                f"matmul params are {len(params)} bytes; expected 0 or 8"
            )

        if len(data_in) != INPUT_BYTES:
            raise ValueError(
                f"matmul input is {len(data_in)} bytes; expected {INPUT_BYTES}"
            )
        if a_is_unsigned not in (0, 1):
            raise ValueError("matmul a_is_unsigned must be 0 or 1")
        if a_zero_point < 0 or a_zero_point > 255:
            raise ValueError("matmul a_zero_point must be uint8")
        if not a_is_unsigned and a_zero_point != 0:
            raise ValueError(
                "signed matmul requires a_zero_point to be zero"
            )

        # in = a(bits x 64) || b(i8 x 64) || c_in(i32 x 64)
        a = data_in[0:64]
        b = struct.unpack("<64b", data_in[64:128])
        c_in = struct.unpack("<64i", data_in[128:384])

        await ReadWrite()
        dut.a_flat.value = _pack_i8(a)
        dut.b_flat.value = _pack_i8(b)
        dut.c_in_flat.value = _pack_i32(c_in)
        dut.a_is_unsigned.value = a_is_unsigned
        dut.a_zero_point.value = a_zero_point
        dut.start.value = 1
        await RisingEdge(dut.clk)
        await ReadWrite()
        dut.start.value = 0
        # The RTL command contract captures every input on start.
        dut.a_flat.value = 0
        dut.b_flat.value = 0
        dut.c_in_flat.value = 0
        dut.a_is_unsigned.value = 0
        dut.a_zero_point.value = 0

        for _ in range(32):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.done.value) == 1:
                c = _unpack_i32(int(dut.c_out_flat.value))
                return struct.pack("<64i", *c)

        raise TimeoutError("DUT did not assert done")

    def smoke_cases(self):
        cases = []

        # Existing signed-i8 behavior, including a non-zero accumulator.
        signed_a = [((3 * row - 5 * kk) % 17) - 8
                    for row in range(N) for kk in range(N)]
        signed_b = [((7 * kk + 2 * col) % 13) - 6
                    for kk in range(N) for col in range(N)]
        c_in = [11 * row - 3 * col
                for row in range(N) for col in range(N)]
        cases.append(self._case(signed_a, signed_b, c_in, 0, 0))

        # Quantized activations exercise values on both sides of the signed-i8
        # boundary and a non-zero activation zero point.
        quantized_a = [
            (37 * row + 53 * kk + 19) & 0xFF
            for row in range(N) for kk in range(N)
        ]
        quantized_b = [((5 * kk - 3 * col) % 15) - 7
                       for kk in range(N) for col in range(N)]
        cases.append(self._case(
            quantized_a, quantized_b, [0] * ELEMS, 1, 57
        ))

        # A padding tile filled with the zero point contributes exactly zero.
        padding_a = [173] * ELEMS
        extreme_b = [-128 if idx & 1 else 127 for idx in range(ELEMS)]
        padding_c = [idx - 32 for idx in range(ELEMS)]
        cases.append(self._case(padding_a, extreme_b, padding_c, 1, 173))

        # Boundary values verify that the centered activation really has nine
        # bits: one case reaches +255 and the other reaches -255.
        boundary_a = [0 if idx & 1 else 255 for idx in range(ELEMS)]
        boundary_b = [-128 if idx & 1 else 127 for idx in range(ELEMS)]
        cases.append(self._case(
            boundary_a, boundary_b, [0] * ELEMS, 1, 0
        ))
        cases.append(self._case(
            boundary_a, boundary_b, [0] * ELEMS, 1, 255
        ))

        # The second K tile starts from the first tile's accumulator, matching
        # how convolution reductions are chained through c_in.
        first_a = [(row * N + kk) & 0xFF
                   for row in range(N) for kk in range(N)]
        first_b = [1 if (kk + col) & 1 else -1
                   for kk in range(N) for col in range(N)]
        first_c = [0] * ELEMS
        first_result = matmul_tile(
            first_a,
            first_b,
            first_c,
            a_is_unsigned=True,
            a_zero_point=31,
        )
        cases.append(self._case(first_a, first_b, first_c, 1, 31))

        second_a = [(255 - 9 * row - 7 * kk) & 0xFF
                    for row in range(N) for kk in range(N)]
        second_b = [((kk - col) % 9) - 4
                    for kk in range(N) for col in range(N)]
        cases.append(self._case(
            second_a, second_b, first_result, 1, 31
        ))

        return cases

    @staticmethod
    def _case(a, b, c_in, a_is_unsigned, a_zero_point):
        if a_is_unsigned:
            a_bytes = struct.pack("<64B", *a)
        else:
            a_bytes = struct.pack("<64b", *a)
        data_in = a_bytes + struct.pack("<64b", *b) + struct.pack(
            "<64i", *c_in
        )
        expected = matmul_tile(
            a,
            b,
            c_in,
            a_is_unsigned=bool(a_is_unsigned),
            a_zero_point=a_zero_point,
        )
        return (
            struct.pack("<2i", a_is_unsigned, a_zero_point),
            data_in,
            struct.pack("<64i", *expected),
        )
