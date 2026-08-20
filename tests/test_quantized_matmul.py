import unittest

from tests.golden.quantized_matmul import matmul_tile


N = 8
ELEMS = N * N


class QuantizedMatmulTest(unittest.TestCase):
    def test_signed_mode_preserves_existing_matmul(self):
        result = matmul_tile(
            [1] * ELEMS,
            [2] * ELEMS,
            [3] * ELEMS,
        )
        self.assertEqual(result, [19] * ELEMS)

    def test_unsigned_mode_supports_full_centered_range(self):
        positive = matmul_tile(
            [255] * ELEMS,
            [1] * ELEMS,
            [0] * ELEMS,
            a_is_unsigned=True,
            a_zero_point=0,
        )
        negative = matmul_tile(
            [0] * ELEMS,
            [1] * ELEMS,
            [0] * ELEMS,
            a_is_unsigned=True,
            a_zero_point=255,
        )
        self.assertEqual(positive, [8 * 255] * ELEMS)
        self.assertEqual(negative, [-8 * 255] * ELEMS)

    def test_zero_point_padding_preserves_accumulator(self):
        c_in = [idx - 32 for idx in range(ELEMS)]
        result = matmul_tile(
            [57] * ELEMS,
            [-128 if idx & 1 else 127 for idx in range(ELEMS)],
            c_in,
            a_is_unsigned=True,
            a_zero_point=57,
        )
        self.assertEqual(result, c_in)

    def test_second_k_tile_accumulates_first_result(self):
        first = matmul_tile(
            [32] * ELEMS,
            [1] * ELEMS,
            [0] * ELEMS,
            a_is_unsigned=True,
            a_zero_point=31,
        )
        second = matmul_tile(
            [33] * ELEMS,
            [1] * ELEMS,
            first,
            a_is_unsigned=True,
            a_zero_point=31,
        )
        self.assertEqual(first, [8] * ELEMS)
        self.assertEqual(second, [24] * ELEMS)


if __name__ == "__main__":
    unittest.main()
