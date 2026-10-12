"""Golden model for the 8x8x8 integer Conv tile schedule."""

from tests.golden.quantized_matmul import matmul_tile


TILE = 8


def conv_accumulate_tiled(
    input_q,
    weight_q,
    bias_q,
    input_zero_point,
    strides=(1, 1),
    pads=(0, 0, 0, 0),
    dilations=(1, 1),
):
    """Execute Conv as 8 positions x 8 channels x 8 reduction tiles."""
    batch = len(input_q)
    input_channels = len(input_q[0])
    input_height = len(input_q[0][0])
    input_width = len(input_q[0][0][0])
    output_channels = len(weight_q)
    kernel_height = len(weight_q[0][0])
    kernel_width = len(weight_q[0][0][0])
    stride_h, stride_w = strides
    pad_top, pad_left, pad_bottom, pad_right = pads
    dilation_h, dilation_w = dilations

    effective_kernel_h = dilation_h * (kernel_height - 1) + 1
    effective_kernel_w = dilation_w * (kernel_width - 1) + 1
    output_height = (
        input_height + pad_top + pad_bottom - effective_kernel_h
    ) // stride_h + 1
    output_width = (
        input_width + pad_left + pad_right - effective_kernel_w
    ) // stride_w + 1
    spatial_size = output_height * output_width
    reduction_size = input_channels * kernel_height * kernel_width

    output = [
        [
            [[0 for _ in range(output_width)] for _ in range(output_height)]
            for _ in range(output_channels)
        ]
        for _ in range(batch)
    ]

    for batch_index in range(batch):
        for position_base in range(0, spatial_size, TILE):
            for output_channel_base in range(0, output_channels, TILE):
                acc_tile = [
                    bias_q[output_channel_base + column]
                    if output_channel_base + column < output_channels
                    else 0
                    for _row in range(TILE)
                    for column in range(TILE)
                ]

                for reduction_base in range(0, reduction_size, TILE):
                    lhs_tile = [input_zero_point] * (TILE * TILE)
                    rhs_tile = [0] * (TILE * TILE)

                    for row in range(TILE):
                        position = position_base + row
                        if position >= spatial_size:
                            continue
                        output_row, output_column = divmod(
                            position, output_width
                        )

                        for reduction_offset in range(TILE):
                            reduction = reduction_base + reduction_offset
                            if reduction >= reduction_size:
                                continue
                            input_channel, kernel_remainder = divmod(
                                reduction, kernel_height * kernel_width
                            )
                            kernel_row, kernel_column = divmod(
                                kernel_remainder, kernel_width
                            )
                            input_row = (
                                output_row * stride_h
                                - pad_top
                                + kernel_row * dilation_h
                            )
                            input_column = (
                                output_column * stride_w
                                - pad_left
                                + kernel_column * dilation_w
                            )
                            if (
                                0 <= input_row < input_height
                                and 0 <= input_column < input_width
                            ):
                                lhs_tile[row * TILE + reduction_offset] = (
                                    input_q[batch_index][input_channel]
                                    [input_row][input_column]
                                )

                    for reduction_offset in range(TILE):
                        reduction = reduction_base + reduction_offset
                        if reduction >= reduction_size:
                            continue
                        input_channel, kernel_remainder = divmod(
                            reduction, kernel_height * kernel_width
                        )
                        kernel_row, kernel_column = divmod(
                            kernel_remainder, kernel_width
                        )
                        for column in range(TILE):
                            output_channel = output_channel_base + column
                            if output_channel < output_channels:
                                rhs_tile[reduction_offset * TILE + column] = (
                                    weight_q[output_channel][input_channel]
                                    [kernel_row][kernel_column]
                                )

                    acc_tile = matmul_tile(
                        lhs_tile,
                        rhs_tile,
                        acc_tile,
                        a_is_unsigned=True,
                        a_zero_point=input_zero_point,
                    )

                for row in range(TILE):
                    position = position_base + row
                    if position >= spatial_size:
                        continue
                    output_row, output_column = divmod(
                        position, output_width
                    )
                    for column in range(TILE):
                        output_channel = output_channel_base + column
                        if output_channel < output_channels:
                            output[batch_index][output_channel][output_row][
                                output_column
                            ] = acc_tile[row * TILE + column]

    return output
