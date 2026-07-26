"""Driver and golden model for quantized residual addition."""
import struct

from cocotb.triggers import FallingEdge, ReadOnly, ReadWrite, RisingEdge

from .base import IPDriver

N = 8
ELEMS = N * N
MASK32 = 0xFFFFFFFF
MAX_MULTIPLIER = 0x7FFFFFFF
SHIFT_BITS = 6


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


def common_fixed_point_params(lhs_real, rhs_real, max_shift=31):
    """Approximate two non-negative scale ratios with one binary point."""
    if lhs_real < 0 or rhs_real < 0:
        raise ValueError("real multipliers must be non-negative")

    for shift in range(max_shift, -1, -1):
        lhs_multiplier = round(lhs_real * (1 << shift))
        rhs_multiplier = round(rhs_real * (1 << shift))
        if (
            0 <= lhs_multiplier <= MAX_MULTIPLIER
            and 0 <= rhs_multiplier <= MAX_MULTIPLIER
        ):
            step = 1.0 / (1 << shift)
            error_bound = 0.5 * step
            lhs_error = abs(lhs_real - lhs_multiplier * step)
            rhs_error = abs(rhs_real - rhs_multiplier * step)
            tolerance = 1e-15
            if (
                lhs_error > error_bound + tolerance
                or rhs_error > error_bound + tolerance
            ):
                raise ArithmeticError("fixed-point approximation is invalid")
            return lhs_multiplier, rhs_multiplier, shift

    raise ValueError("scale ratios do not fit signed i32 multipliers")


def qadd_relu_element(
    lhs,
    rhs,
    lhs_multiplier,
    rhs_multiplier,
    shift,
    lhs_zero_point,
    rhs_zero_point,
    output_zero_point,
    relu_enable,
):
    wide = (
        (lhs - lhs_zero_point) * lhs_multiplier
        + (rhs - rhs_zero_point) * rhs_multiplier
    )
    centered = _round_shift_rne(wide, shift)
    if relu_enable:
        centered = max(centered, 0)
    return max(0, min(255, centered + output_zero_point))


class QAddReluDriver(IPDriver):
    NAME = "qadd_relu"

    async def setup(self, dut):
        dut.rst.value = 1
        dut.start.value = 0
        dut.lhs_flat.value = 0
        dut.rhs_flat.value = 0
        dut.lhs_multiplier.value = 0
        dut.rhs_multiplier.value = 0
        dut.shift.value = 0
        dut.lhs_zero_point.value = 0
        dut.rhs_zero_point.value = 0
        dut.output_zero_point.value = 0
        dut.relu_enable.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        # params = lhs_mult, rhs_mult, shift, lhs_zp, rhs_zp, output_zp,
        #          relu_enable (7 x i32)
        if len(params) != 7 * 4:
            raise ValueError(
                f"qadd_relu params are {len(params)} bytes; expected 28"
            )
        if len(data_in) != 2 * ELEMS:
            raise ValueError(
                f"qadd_relu input is {len(data_in)} bytes; expected 128"
            )

        (
            lhs_multiplier,
            rhs_multiplier,
            shift,
            lhs_zero_point,
            rhs_zero_point,
            output_zero_point,
            relu_enable,
        ) = struct.unpack("<7i", params)
        lhs = data_in[:ELEMS]
        rhs = data_in[ELEMS:]

        if lhs_multiplier < 0 or rhs_multiplier < 0:
            raise ValueError("qadd_relu multipliers must be non-negative")
        if shift < 0 or shift >= (1 << SHIFT_BITS):
            raise ValueError("qadd_relu shift must be in [0, 63]")
        for name, value in (
            ("lhs_zero_point", lhs_zero_point),
            ("rhs_zero_point", rhs_zero_point),
            ("output_zero_point", output_zero_point),
        ):
            if value < 0 or value > 255:
                raise ValueError(f"qadd_relu {name} must be uint8")
        if relu_enable not in (0, 1):
            raise ValueError("qadd_relu relu_enable must be 0 or 1")

        await ReadWrite()
        dut.lhs_flat.value = _pack_u8(lhs)
        dut.rhs_flat.value = _pack_u8(rhs)
        dut.lhs_multiplier.value = lhs_multiplier & MASK32
        dut.rhs_multiplier.value = rhs_multiplier & MASK32
        dut.shift.value = shift
        dut.lhs_zero_point.value = lhs_zero_point
        dut.rhs_zero_point.value = rhs_zero_point
        dut.output_zero_point.value = output_zero_point
        dut.relu_enable.value = relu_enable
        dut.start.value = 1

        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            out = int(dut.out_flat.value).to_bytes(ELEMS, "little")
            await FallingEdge(dut.clk)
            await ReadWrite()
            dut.start.value = 0
            return out

        raise TimeoutError("qadd_relu DUT did not assert done")

    def smoke_cases(self):
        cases = []

        # Positive and negative ties with ReLU disabled.
        config = (1, 1, 1, 128, 128, 128, 0)
        centered = [1, 3, 5, -1, -3, -5, 7, -7]
        lhs = [128 + centered[i % len(centered)] for i in range(ELEMS)]
        rhs = [128] * ELEMS
        cases.append(self._case(config, lhs, rhs))

        # Both saturation limits with different input zero points and scales.
        config = (4, 7, 0, 120, 80, 128, 0)
        lhs_values = [0, 255, 120, 121, 119, 200, 40, 250]
        rhs_values = [0, 255, 80, 79, 81, 180, 20, 240]
        lhs = [lhs_values[i % len(lhs_values)] for i in range(ELEMS)]
        rhs = [rhs_values[i % len(rhs_values)] for i in range(ELEMS)]
        cases.append(self._case(config, lhs, rhs))

        # Negative sums are clamped in the centered domain before output zp.
        config = (1 << 16, 1 << 16, 16, 100, 150, 23, 1)
        lhs = [(70 + 5 * i) & 0xFF for i in range(ELEMS)]
        rhs = [(110 + 3 * i) & 0xFF for i in range(ELEMS)]
        cases.append(self._case(config, lhs, rhs))

        # Quantization domains from the layer4/0 residual Add in ResNet18.
        lhs_scale = 0.0473271385
        rhs_scale = 0.0366295353
        output_scale = 0.0296164267
        lhs_multiplier, rhs_multiplier, shift = common_fixed_point_params(
            lhs_scale / output_scale,
            rhs_scale / output_scale,
        )
        config = (
            lhs_multiplier,
            rhs_multiplier,
            shift,
            68,
            65,
            0,
            1,
        )
        lhs = [(17 * i + 11) & 0xFF for i in range(ELEMS)]
        rhs = [(29 * i + 7) & 0xFF for i in range(ELEMS)]
        cases.append(self._case(config, lhs, rhs))

        return cases

    @staticmethod
    def _case(config, lhs, rhs):
        (
            lhs_multiplier,
            rhs_multiplier,
            shift,
            lhs_zero_point,
            rhs_zero_point,
            output_zero_point,
            relu_enable,
        ) = config
        expected = bytes(
            qadd_relu_element(
                lhs[i],
                rhs[i],
                lhs_multiplier,
                rhs_multiplier,
                shift,
                lhs_zero_point,
                rhs_zero_point,
                output_zero_point,
                relu_enable,
            )
            for i in range(ELEMS)
        )
        return (
            struct.pack("<7i", *config),
            bytes(lhs) + bytes(rhs),
            expected,
        )
