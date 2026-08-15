import unittest

from tests.golden.conv_accumulate import conv_accumulate


class ConvAccumulateTest(unittest.TestCase):
    def test_one_by_one_with_zero_point_bias_and_two_channels(self):
        result = conv_accumulate(
            input_q=[[[[5, 7], [3, 9]]]],
            weight_q=[[[[2]]], [[[-1]]]],
            bias_q=[1, -2],
            input_zero_point=5,
        )

        self.assertEqual(
            result,
            [[[[1, 5], [-3, 9]], [[-2, -4], [0, -6]]]],
        )

    def test_padding_uses_input_zero_point(self):
        result = conv_accumulate(
            input_q=[[[[11, 12], [13, 14]]]],
            weight_q=[[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]],
            bias_q=[5],
            input_zero_point=10,
            pads=(1, 1, 1, 1),
        )

        self.assertEqual(result, [[[[15, 15], [15, 15]]]])

    def test_stride_and_dilation(self):
        result = conv_accumulate(
            input_q=[
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ],
            weight_q=[[[[1, 0], [0, 1]]]],
            bias_q=[0],
            input_zero_point=0,
            strides=(2, 2),
            dilations=(2, 2),
        )

        self.assertEqual(result, [[[[14, 18], [34, 38]]]])

    def test_i32_overflow_is_rejected(self):
        with self.assertRaises(OverflowError):
            conv_accumulate(
                input_q=[[[[1]]]],
                weight_q=[[[[1]]]],
                bias_q=[(1 << 31) - 1],
                input_zero_point=0,
            )

    def test_grouped_convolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "group=1"):
            conv_accumulate(
                input_q=[[[[1]]]],
                weight_q=[[[[1]]]],
                bias_q=[0],
                input_zero_point=0,
                group=2,
            )

    def test_channel_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "channels"):
            conv_accumulate(
                input_q=[[[[1]], [[2]]]],
                weight_q=[[[[1]]]],
                bias_q=[0],
                input_zero_point=0,
            )


if __name__ == "__main__":
    unittest.main()
