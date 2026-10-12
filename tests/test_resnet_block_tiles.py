import unittest

from tests.golden.conv_accumulate import conv_accumulate
from tests.golden.conv_requant import conv_requant_tensor
from tests.golden.conv_tiles import conv_accumulate_tiled
from tests.golden.qadd_relu import qadd_relu_element


def make_input(channels, height, width, zero_point, seed=0):
    return [
        [
            [
                [
                    zero_point
                    + ((channel * 11 + row * 5 + column * 3 + seed) % 21)
                    - 10
                    for column in range(width)
                ]
                for row in range(height)
            ]
            for channel in range(channels)
        ]
    ]


def make_weights(
    output_channels,
    input_channels,
    kernel_height,
    kernel_width,
    seed,
):
    return [
        [
            [
                [
                    (
                        output_channel * 3
                        + input_channel * 5
                        + kernel_row * 7
                        + kernel_column
                        + seed
                    )
                    % 5
                    - 2
                    for kernel_column in range(kernel_width)
                ]
                for kernel_row in range(kernel_height)
            ]
            for input_channel in range(input_channels)
        ]
        for output_channel in range(output_channels)
    ]


def make_bias(channels, seed):
    return [((channel * 13 + seed) % 17) - 8 for channel in range(channels)]


def qadd_tensors(lhs, rhs, config):
    (
        lhs_multiplier,
        rhs_multiplier,
        shift,
        lhs_zero_point,
        rhs_zero_point,
        output_zero_point,
        relu_enable,
    ) = config
    return [
        [
            [
                [
                    qadd_relu_element(
                        lhs[batch][channel][row][column],
                        rhs[batch][channel][row][column],
                        lhs_multiplier,
                        rhs_multiplier,
                        shift,
                        lhs_zero_point,
                        rhs_zero_point,
                        output_zero_point,
                        relu_enable,
                    )
                    for column in range(len(lhs[batch][channel][row]))
                ]
                for row in range(len(lhs[batch][channel]))
            ]
            for channel in range(len(lhs[batch]))
        ]
        for batch in range(len(lhs))
    ]


def flatten(tensor):
    if isinstance(tensor, list):
        return [value for child in tensor for value in flatten(child)]
    return [tensor]


class ResNetBlockTileTest(unittest.TestCase):
    def assert_conv_matches(
        self,
        input_q,
        weight_q,
        bias_q,
        input_zero_point,
        *,
        strides=(1, 1),
        pads=(0, 0, 0, 0),
        dilations=(1, 1),
    ):
        direct = conv_accumulate(
            input_q,
            weight_q,
            bias_q,
            input_zero_point,
            strides=strides,
            pads=pads,
            dilations=dilations,
        )
        tiled = conv_accumulate_tiled(
            input_q,
            weight_q,
            bias_q,
            input_zero_point,
            strides=strides,
            pads=pads,
            dilations=dilations,
        )
        self.assertEqual(tiled, direct)
        return tiled

    def test_identity_residual_block(self):
        input_zero_point = 100
        input_q = make_input(8, 4, 4, input_zero_point)

        conv1_acc = self.assert_conv_matches(
            input_q,
            make_weights(8, 8, 3, 3, seed=1),
            make_bias(8, seed=2),
            input_zero_point,
            pads=(1, 1, 1, 1),
        )
        conv1_q = conv_requant_tensor(
            conv1_acc, [1] * 8, [4] * 8, 96, 1
        )

        conv2_acc = self.assert_conv_matches(
            conv1_q,
            make_weights(8, 8, 3, 3, seed=3),
            make_bias(8, seed=4),
            96,
            pads=(1, 1, 1, 1),
        )
        conv2_q = conv_requant_tensor(
            conv2_acc, [1] * 8, [6] * 8, 110, 0
        )

        result = qadd_tensors(
            conv2_q,
            input_q,
            (1, 1, 1, 110, input_zero_point, 90, 1),
        )
        self.assertEqual(len(result[0]), 8)
        self.assertGreater(len(set(flatten(result))), 1)

    def test_downsample_residual_block(self):
        input_zero_point = 101
        input_q = make_input(8, 4, 4, input_zero_point, seed=5)

        conv1_acc = self.assert_conv_matches(
            input_q,
            make_weights(16, 8, 3, 3, seed=6),
            make_bias(16, seed=7),
            input_zero_point,
            strides=(2, 2),
            pads=(1, 1, 1, 1),
        )
        conv1_q = conv_requant_tensor(
            conv1_acc, [1] * 16, [4] * 16, 96, 1
        )

        conv2_acc = self.assert_conv_matches(
            conv1_q,
            make_weights(16, 16, 3, 3, seed=8),
            make_bias(16, seed=9),
            96,
            pads=(1, 1, 1, 1),
        )
        conv2_q = conv_requant_tensor(
            conv2_acc, [1] * 16, [7] * 16, 120, 0
        )

        downsample_acc = self.assert_conv_matches(
            input_q,
            make_weights(16, 8, 1, 1, seed=10),
            make_bias(16, seed=11),
            input_zero_point,
            strides=(2, 2),
        )
        downsample_q = conv_requant_tensor(
            downsample_acc, [1] * 16, [4] * 16, 115, 0
        )

        result = qadd_tensors(
            conv2_q,
            downsample_q,
            (1, 1, 1, 120, 115, 93, 1),
        )
        self.assertEqual(
            (len(result), len(result[0]), len(result[0][0]),
             len(result[0][0][0])),
            (1, 16, 2, 2),
        )
        self.assertGreater(len(set(flatten(result))), 1)


if __name__ == "__main__":
    unittest.main()
