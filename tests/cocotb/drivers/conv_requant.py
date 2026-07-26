"""Driver and golden model for per-channel convolution requantization."""
import struct

from cocotb.triggers import FallingEdge, ReadOnly, ReadWrite, RisingEdge

from .base import IPDriver

N = 8
ELEMS = N * N
MASK32 = 0xFFFFFFFF
SHIFT_BITS = 6


def _pack_unsigned(values, width):
    word = 0
    mask = (1 << width) - 1
    for idx, value in enumerate(values):
        word |= (value & mask) << (idx * width)
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


def fixed_point_params(real_multiplier, max_shift=31):
    """Return the highest-precision signed-i32 multiplier and shift."""
    if real_multiplier < 0:
        raise ValueError("real multiplier must be non-negative")

    for shift in range(max_shift, -1, -1):
        multiplier = round(real_multiplier * (1 << shift))
        if 0 <= multiplier <= 0x7FFFFFFF:
            return multiplier, shift
    raise ValueError(f"real multiplier {real_multiplier} does not fit signed i32")


def conv_requant_element(
    acc,
    multiplier,
    shift,
    output_zero_point,
    relu_enable,
):
    scaled = _round_shift_rne(acc * multiplier, shift)
    if relu_enable:
        scaled = max(scaled, 0)
    return max(0, min(255, scaled + output_zero_point))


class ConvRequantDriver(IPDriver):
    NAME = "conv_requant"

    async def setup(self, dut):
        dut.rst.value = 1
        dut.start.value = 0
        dut.c_in_flat.value = 0
        dut.multiplier_flat.value = 0
        dut.shift_flat.value = 0
        dut.output_zero_point.value = 0
        dut.relu_enable.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        # params = multiplier[8], shift[8], output_zp, relu_enable (18 x i32)
        if len(params) != 18 * 4:
            raise ValueError(
                f"conv_requant params are {len(params)} bytes; expected 72"
            )
        if len(data_in) != ELEMS * 4:
            raise ValueError(
                f"conv_requant input is {len(data_in)} bytes; expected 256"
            )

        unpacked = struct.unpack("<18i", params)
        multipliers = unpacked[0:N]
        shifts = unpacked[N:2 * N]
        output_zero_point, relu_enable = unpacked[2 * N:]
        acc = struct.unpack("<64i", data_in)

        if any(shift < 0 or shift >= (1 << SHIFT_BITS) for shift in shifts):
            raise ValueError("conv_requant shifts must be in [0, 63]")
        if output_zero_point < 0 or output_zero_point > 255:
            raise ValueError("conv_requant output zero point must be uint8")
        if relu_enable not in (0, 1):
            raise ValueError("conv_requant relu_enable must be 0 or 1")

        await ReadWrite()
        dut.c_in_flat.value = _pack_unsigned(acc, 32)
        dut.multiplier_flat.value = _pack_unsigned(multipliers, 32)
        dut.shift_flat.value = _pack_unsigned(shifts, SHIFT_BITS)
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

        raise TimeoutError("conv_requant DUT did not assert done")

    def smoke_cases(self):
        cases = []

        # Column-specific scaling with ReLU. Equal values in each row must
        # produce a different result in every output-channel column.
        multipliers = [1, 2, 3, 4, 5, 6, 7, 8]
        shifts = [3] * N
        acc = [(row - 3) * 16 + 5 for row in range(N) for _ in range(N)]
        cases.append(self._case(
            multipliers, shifts, 0, 1, acc
        ))

        # Per-channel shifts, non-zero output zero point, and no ReLU preserve
        # representable negative values around the uint8 zero code.
        multipliers = [1, 1, 3, 3, 5, 5, 7, 7]
        shifts = [1, 2, 1, 2, 1, 2, 1, 2]
        tie_values = [1, 3, 5, -1, -3, -5, 255, -255]
        acc = [tie_values[(row + col) % len(tie_values)]
               for row in range(N) for col in range(N)]
        cases.append(self._case(
            multipliers, shifts, 128, 0, acc
        ))

        # Real per-channel scale ratios exercise high-precision multipliers and
        # both saturation limits.
        activation_scale = 0.0374455191
        weight_scales = [
            0.00140720594,
            0.000349023234,
            1.1920929e-7,
            0.000380833168,
            1.1920929e-7,
            0.000891569536,
            0.000914381293,
            1.1920929e-7,
        ]
        output_scale = 0.0286055468
        fixed = [
            fixed_point_params(
                activation_scale * weight_scale / output_scale
            )
            for weight_scale in weight_scales
        ]
        multipliers = [item[0] for item in fixed]
        shifts = [item[1] for item in fixed]
        acc = [
            ((row * N + col) - 24) * 50000
            for row in range(N) for col in range(N)
        ]
        cases.append(self._case(
            multipliers, shifts, 0, 1, acc
        ))

        return cases

    @staticmethod
    def _case(
        multipliers,
        shifts,
        output_zero_point,
        relu_enable,
        acc,
    ):
        params = struct.pack(
            "<18i",
            *multipliers,
            *shifts,
            output_zero_point,
            relu_enable,
        )
        data_in = struct.pack("<64i", *acc)
        expected = bytes(
            conv_requant_element(
                value,
                multipliers[idx % N],
                shifts[idx % N],
                output_zero_point,
                relu_enable,
            )
            for idx, value in enumerate(acc)
        )
        return params, data_in, expected
