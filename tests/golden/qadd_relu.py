"""Golden model for per-tensor quantized residual addition."""

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


def common_fixed_point_params(lhs_real, rhs_real, max_shift=31):
    """Approximate two non-negative scale ratios with one binary point."""
    if lhs_real < 0 or rhs_real < 0:
        raise ValueError("real multipliers must be non-negative")

    for shift in range(max_shift, -1, -1):
        lhs_multiplier = round(lhs_real * (1 << shift))
        rhs_multiplier = round(rhs_real * (1 << shift))
        if not (
            0 <= lhs_multiplier <= MAX_MULTIPLIER
            and 0 <= rhs_multiplier <= MAX_MULTIPLIER
        ):
            continue

        step = 1.0 / (1 << shift)
        error_bound = 0.5 * step
        tolerance = 1e-15
        if abs(lhs_real - lhs_multiplier * step) > error_bound + tolerance:
            raise ArithmeticError("lhs fixed-point approximation is invalid")
        if abs(rhs_real - rhs_multiplier * step) > error_bound + tolerance:
            raise ArithmeticError("rhs fixed-point approximation is invalid")
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
    centered = round_shift_rne(wide, shift)
    if relu_enable:
        centered = max(centered, 0)
    return max(0, min(255, centered + output_zero_point))


def qdq_add_relu_element(
    lhs,
    rhs,
    lhs_scale,
    rhs_scale,
    output_scale,
    lhs_zero_point,
    rhs_zero_point,
    output_zero_point,
    relu_enable,
):
    real_sum = (
        (lhs - lhs_zero_point) * lhs_scale
        + (rhs - rhs_zero_point) * rhs_scale
    )
    if relu_enable:
        real_sum = max(real_sum, 0.0)
    quantized = round(real_sum / output_scale) + output_zero_point
    return max(0, min(255, quantized))
