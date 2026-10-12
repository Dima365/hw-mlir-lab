"""Golden model for per-output-channel convolution requantization."""

MAX_MULTIPLIER = (1 << 31) - 1


def round_shift_rne(value, shift):
    """Signed round-to-nearest, ties-to-even division by ``2**shift``."""
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
        if not 0 <= multiplier <= MAX_MULTIPLIER:
            continue

        step = 1.0 / (1 << shift)
        error_bound = 0.5 * step
        approximation_error = abs(real_multiplier - multiplier * step)
        if approximation_error > error_bound + 1e-15:
            raise ArithmeticError("fixed-point approximation is invalid")
        return multiplier, shift

    raise ValueError(
        f"real multiplier {real_multiplier} does not fit signed i32"
    )


def conv_requant_element(
    acc,
    multiplier,
    shift,
    output_zero_point,
    relu_enable,
):
    scaled = round_shift_rne(acc * multiplier, shift)
    if relu_enable:
        scaled = max(scaled, 0)
    return max(0, min(255, scaled + output_zero_point))


def conv_requant_tensor(
    acc,
    multipliers,
    shifts,
    output_zero_point,
    relu_enable,
):
    """Requantize an NCHW accumulator using one pair per channel."""
    channels = len(acc[0])
    if len(multipliers) != channels or len(shifts) != channels:
        raise ValueError("requant parameters must match the channel count")

    return [
        [
            [
                [
                    conv_requant_element(
                        value,
                        multipliers[channel],
                        shifts[channel],
                        output_zero_point,
                        relu_enable,
                    )
                    for value in row
                ]
                for row in channel_data
            ]
            for channel, channel_data in enumerate(batch)
        ]
        for batch in acc
    ]
