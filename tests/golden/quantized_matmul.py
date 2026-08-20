"""Golden model for one integer 8x8 matrix-accumulate tile."""


def wrap_i32(value):
    """Return the two's-complement i32 interpretation of ``value``."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def matmul_tile(a, b, c_in, *, size=8, a_is_unsigned=False,
                a_zero_point=0):
    """Compute ``C_in + centered(A) * B`` with i32 accumulation."""
    elems = size * size
    if len(a) != elems or len(b) != elems or len(c_in) != elems:
        raise ValueError(f"all matrix tiles must contain {elems} elements")
    if a_is_unsigned:
        if not 0 <= a_zero_point <= 255:
            raise ValueError("a_zero_point must be uint8")
        if any(value < 0 or value > 255 for value in a):
            raise ValueError("unsigned A values must be uint8")
    else:
        if a_zero_point != 0:
            raise ValueError("signed A requires a_zero_point to be zero")
        if any(value < -128 or value > 127 for value in a):
            raise ValueError("signed A values must be int8")
    if any(value < -128 or value > 127 for value in b):
        raise ValueError("B values must be int8")
    if any(value < -(1 << 31) or value >= (1 << 31) for value in c_in):
        raise ValueError("C input values must be int32")

    result = []
    for row in range(size):
        for col in range(size):
            acc = c_in[row * size + col]
            for kk in range(size):
                a_value = a[row * size + kk]
                if a_is_unsigned:
                    a_value -= a_zero_point
                acc = wrap_i32(acc + a_value * b[kk * size + col])
            result.append(acc)
    return result
