"""Driver and golden model for the configurable quantized epilogue."""
import struct

from cocotb.triggers import FallingEdge, ReadOnly, ReadWrite, RisingEdge

from .base import IPDriver

N = 8
ELEMS = N * N
MASK32 = 0xFFFFFFFF


def _pack_i32(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & MASK32) << (idx * 32)
    return word


def _pack_u8(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (idx * 8)
    return word


def _round_shift_rne(value, shift):
    """Signed round-to-nearest, ties-to-even division by 2**shift."""
    if shift == 0:
        return value

    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if value < 0 else quotient


def epilogue_element(
    acc,
    bias,
    residual,
    mult,
    residual_mult,
    shift,
    residual_zero_point,
    output_zero_point,
    bias_enable,
    add_enable,
    relu_enable,
    output_signed,
):
    main_value = acc + (bias if bias_enable else 0)
    combined = main_value * mult
    if add_enable:
        combined += (residual - residual_zero_point) * residual_mult

    centered = _round_shift_rne(combined, shift)
    if relu_enable:
        centered = max(centered, 0)
    q = centered + output_zero_point

    if output_signed:
        return max(-128, min(127, q)) & 0xFF
    return max(0, min(255, q))


def requantize(c, mult, shift, zero_point):
    """Legacy signed-i8 requantize+ReLU contract."""
    return epilogue_element(
        c, 0, 0, mult, 0, shift, 0, zero_point,
        False, False, True, True,
    )


class EpilogueDriver(IPDriver):
    NAME = "epilogue"

    async def setup(self, dut):
        dut.rst.value = 1
        dut.start.value = 0
        dut.c_in_flat.value = 0
        dut.bias_in_flat.value = 0
        dut.residual_in_flat.value = 0
        dut.mult.value = 0
        dut.residual_mult.value = 0
        dut.shift.value = 0
        dut.residual_zero_point.value = 0
        dut.zero_point.value = 0
        dut.bias_enable.value = 0
        dut.add_enable.value = 0
        dut.relu_enable.value = 1
        dut.output_signed.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        if len(params) == 3 * 4:
            # Backwards-compatible runtime protocol:
            # params = mult, shift, output_zero_point; data = acc[64].
            mult, shift, output_zero_point = struct.unpack("<3i", params)
            residual_mult = residual_zero_point = 0
            bias_enable = add_enable = 0
            relu_enable = output_signed = 1
            acc = struct.unpack("<64i", data_in)
            bias = (0,) * ELEMS
            residual = (0,) * ELEMS
        elif len(params) == 9 * 4:
            # Extended protocol:
            # params = main_mult, residual_mult, shift, residual_zp, output_zp,
            #          bias_enable, add_enable, relu_enable, output_signed
            (
                mult,
                residual_mult,
                shift,
                residual_zero_point,
                output_zero_point,
                bias_enable,
                add_enable,
                relu_enable,
                output_signed,
            ) = struct.unpack("<9i", params)
            expected_size = ELEMS * 4 + ELEMS * 4 + ELEMS
            if len(data_in) != expected_size:
                raise ValueError(
                    f"extended epilogue input is {len(data_in)} bytes, "
                    f"expected {expected_size}"
                )
            acc = struct.unpack_from("<64i", data_in, 0)
            bias = struct.unpack_from("<64i", data_in, ELEMS * 4)
            residual = tuple(data_in[2 * ELEMS * 4:])
        else:
            raise ValueError(
                f"epilogue params are {len(params)} bytes; expected 12 or 36"
            )

        await ReadWrite()
        dut.c_in_flat.value = _pack_i32(acc)
        dut.bias_in_flat.value = _pack_i32(bias)
        dut.residual_in_flat.value = _pack_u8(residual)
        dut.mult.value = mult & MASK32
        dut.residual_mult.value = residual_mult & MASK32
        dut.shift.value = shift
        dut.residual_zero_point.value = residual_zero_point & MASK32
        dut.zero_point.value = output_zero_point & MASK32
        dut.bias_enable.value = bias_enable
        dut.add_enable.value = add_enable
        dut.relu_enable.value = relu_enable
        dut.output_signed.value = output_signed
        dut.start.value = 1

        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            out = int(dut.out_flat.value).to_bytes(ELEMS, "little")
            await FallingEdge(dut.clk)
            await ReadWrite()
            dut.start.value = 0
            return out

        raise TimeoutError("epilogue DUT did not assert done")

    def smoke_cases(self):
        cases = []

        # Legacy requantize+ReLU mode used by standalone.requantize.
        mult, shift, output_zero_point = 12897, 20, 0
        acc = [(i - 32) * 200 for i in range(ELEMS)]
        cases.append((
            struct.pack("<3i", mult, shift, output_zero_point),
            struct.pack("<64i", *acc),
            bytes(requantize(c, mult, shift, output_zero_point) for c in acc),
        ))

        # Unsigned requantize without ReLU: zero is represented by zp=128.
        config = (1 << 16, 0, 16, 0, 128, 0, 0, 0, 0)
        acc = [(i * 7) - 220 for i in range(ELEMS)]
        bias = [0] * ELEMS
        residual = [0] * ELEMS
        cases.append(self._extended_case(config, acc, bias, residual))

        # Signed output without ReLU, including positive and negative ties.
        config = (1, 0, 1, 0, 0, 0, 0, 0, 1)
        tie_values = [1, 3, 5, -1, -3, -5]
        acc = [tie_values[i % len(tie_values)] for i in range(ELEMS)]
        cases.append(self._extended_case(
            config, acc, [0] * ELEMS, [0] * ELEMS
        ))

        # Full bias + residual add + ReLU path, with both scales equal to one.
        config = (1 << 16, 1 << 16, 16, 100, 0, 1, 1, 1, 0)
        acc = [(i * 4) - 100 for i in range(ELEMS)]
        bias = [(i % 5) - 2 for i in range(ELEMS)]
        residual = [(70 + i * 3) & 0xFF for i in range(ELEMS)]
        cases.append(self._extended_case(config, acc, bias, residual))

        return cases

    @staticmethod
    def _extended_case(config, acc, bias, residual):
        (
            mult,
            residual_mult,
            shift,
            residual_zero_point,
            output_zero_point,
            bias_enable,
            add_enable,
            relu_enable,
            output_signed,
        ) = config
        expected = bytes(
            epilogue_element(
                acc[i], bias[i], residual[i], mult, residual_mult, shift,
                residual_zero_point, output_zero_point, bias_enable,
                add_enable, relu_enable, output_signed,
            )
            for i in range(ELEMS)
        )
        data_in = (
            struct.pack("<64i", *acc)
            + struct.pack("<64i", *bias)
            + bytes(residual)
        )
        return struct.pack("<9i", *config), data_in, expected
