"""Golden model for ``standalone.conv_accumulate``."""

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _shape(value, name):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        raise ValueError(f"{name} dimensions must be non-empty")

    child_shape = _shape(value[0], name)
    for child in value[1:]:
        if _shape(child, name) != child_shape:
            raise ValueError(f"{name} must be rectangular")
    return (len(value),) + child_shape


def _validate_integer_tensor(value, name, minimum, maximum):
    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(f"{name} values must be integers")
        if item < minimum or item > maximum:
            raise ValueError(
                f"{name} value {item} is outside [{minimum}, {maximum}]"
            )

    visit(value)


def _pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    result = tuple(value)
    if any(not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive integers")
    return result


def conv_accumulate(
    input_q,
    weight_q,
    bias_q,
    input_zero_point,
    strides=(1, 1),
    pads=(0, 0, 0, 0),
    dilations=(1, 1),
    group=1,
):
    """Compute an NCHW/OIHW uint8-by-int8 convolution into i32.

    Out-of-bounds activation values introduced by padding are equal to
    ``input_zero_point``. They therefore contribute zero after centering.
    """

    input_shape = _shape(input_q, "input_q")
    weight_shape = _shape(weight_q, "weight_q")
    bias_shape = _shape(bias_q, "bias_q")
    if len(input_shape) != 4:
        raise ValueError("input_q must be rank-4 NCHW")
    if len(weight_shape) != 4:
        raise ValueError("weight_q must be rank-4 OIHW")
    if len(bias_shape) != 1:
        raise ValueError("bias_q must be rank-1")
    if group != 1:
        raise ValueError("only group=1 is supported")
    if (
        not isinstance(input_zero_point, int)
        or not 0 <= input_zero_point <= 255
    ):
        raise ValueError("input_zero_point must be in [0, 255]")

    stride_h, stride_w = _pair(strides, "strides")
    dilation_h, dilation_w = _pair(dilations, "dilations")
    if len(pads) != 4 or any(
        not isinstance(item, int) or item < 0 for item in pads
    ):
        raise ValueError("pads must contain four non-negative integers")
    pad_top, pad_left, pad_bottom, pad_right = pads

    _validate_integer_tensor(input_q, "input_q", 0, 255)
    _validate_integer_tensor(weight_q, "weight_q", -128, 127)
    _validate_integer_tensor(bias_q, "bias_q", INT32_MIN, INT32_MAX)

    batch, input_channels, input_height, input_width = input_shape
    output_channels, weight_channels, kernel_height, kernel_width = weight_shape
    if weight_channels != input_channels:
        raise ValueError("weight input channels must match input_q channels")
    if bias_shape[0] != output_channels:
        raise ValueError("bias_q length must match weight output channels")

    effective_kernel_height = dilation_h * (kernel_height - 1) + 1
    effective_kernel_width = dilation_w * (kernel_width - 1) + 1
    height_numerator = (
        input_height + pad_top + pad_bottom - effective_kernel_height
    )
    width_numerator = input_width + pad_left + pad_right - effective_kernel_width
    if height_numerator < 0 or width_numerator < 0:
        raise ValueError("kernel is larger than the padded input")
    output_height = height_numerator // stride_h + 1
    output_width = width_numerator // stride_w + 1

    output = []
    for n in range(batch):
        output_batch = []
        for oc in range(output_channels):
            output_channel = []
            for oh in range(output_height):
                output_row = []
                for ow in range(output_width):
                    accumulator = bias_q[oc]
                    for ic in range(input_channels):
                        for kh in range(kernel_height):
                            ih = oh * stride_h - pad_top + kh * dilation_h
                            if ih < 0 or ih >= input_height:
                                continue
                            for kw in range(kernel_width):
                                iw = ow * stride_w - pad_left + kw * dilation_w
                                if iw < 0 or iw >= input_width:
                                    continue
                                centered = (
                                    input_q[n][ic][ih][iw] - input_zero_point
                                )
                                accumulator += (
                                    centered * weight_q[oc][ic][kh][kw]
                                )
                                if not INT32_MIN <= accumulator <= INT32_MAX:
                                    raise OverflowError(
                                        "convolution accumulator exceeds i32"
                                    )
                    output_row.append(accumulator)
                output_channel.append(output_row)
            output_batch.append(output_channel)
        output.append(output_batch)
    return output
