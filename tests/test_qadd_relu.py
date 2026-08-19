import unittest

from tests.golden.qadd_relu import (
    common_fixed_point_params,
    qadd_relu_element,
    qdq_add_relu_element,
    round_shift_rne,
)


RESNET18_RESIDUAL_DOMAINS = (
    ("layer1.0", 0.046455312520265579, 0.028605546802282333,
     0.034476079046726227, 75, 0, 0, 1446829610, 890906763, 30),
    ("layer1.1", 0.065839782357215881, 0.034476079046726227,
     0.037044569849967957, 82, 0, 0, 1908374919, 999293774, 30),
    ("layer2.0", 0.043153878301382065, 0.033979002386331558,
     0.025826521217823029, 58, 68, 0, 1794129516, 1412682556, 30),
    ("layer2.1", 0.044221054762601852, 0.025826521217823029,
     0.032176367938518524, 70, 0, 0, 1475679172, 861844197, 30),
    ("layer3.0", 0.05117332935333252, 0.014961435459554195,
     0.027315333485603333, 47, 84, 0, 2011578736, 588120918, 30),
    ("layer3.1", 0.046813614666461945, 0.027315333485603333,
     0.026840999722480774, 77, 0, 0, 1872722198, 1092716974, 30),
    ("layer4.0", 0.047327138483524323, 0.036629535257816315,
     0.029616426676511765, 68, 65, 0, 1715842649, 1328001667, 30),
    ("layer4.1", 0.2509911060333252, 0.029616426676511765,
     0.17645132541656494, 42, 0, 0, 1527331389, 180221916, 30),
)


class QAddReluTest(unittest.TestCase):
    def test_signed_round_shift_uses_rne(self):
        self.assertEqual(round_shift_rne(5, 1), 2)
        self.assertEqual(round_shift_rne(7, 1), 4)
        self.assertEqual(round_shift_rne(-5, 1), -2)
        self.assertEqual(round_shift_rne(-7, 1), -4)

    def test_resnet18_domains_match_qdq_exhaustively(self):
        for case in RESNET18_RESIDUAL_DOMAINS:
            (
                name,
                lhs_scale,
                rhs_scale,
                output_scale,
                lhs_zero_point,
                rhs_zero_point,
                output_zero_point,
                expected_lhs_multiplier,
                expected_rhs_multiplier,
                expected_shift,
            ) = case
            fixed_point = common_fixed_point_params(
                lhs_scale / output_scale,
                rhs_scale / output_scale,
            )
            self.assertEqual(
                fixed_point,
                (
                    expected_lhs_multiplier,
                    expected_rhs_multiplier,
                    expected_shift,
                ),
                name,
            )

            lhs_multiplier, rhs_multiplier, shift = fixed_point
            for lhs in range(256):
                for rhs in range(256):
                    reference = qdq_add_relu_element(
                        lhs,
                        rhs,
                        lhs_scale,
                        rhs_scale,
                        output_scale,
                        lhs_zero_point,
                        rhs_zero_point,
                        output_zero_point,
                        True,
                    )
                    actual = qadd_relu_element(
                        lhs,
                        rhs,
                        lhs_multiplier,
                        rhs_multiplier,
                        shift,
                        lhs_zero_point,
                        rhs_zero_point,
                        output_zero_point,
                        True,
                    )
                    self.assertEqual(actual, reference, (name, lhs, rhs))


if __name__ == "__main__":
    unittest.main()
