"""Stable probability receipts supplement the unchanged reference equations."""
import math
import unittest

from context_time_travel.mirostat2 import choose


class MirostatReceiptTests(unittest.TestCase):
    def test_draw_zero_underflow_has_finite_log_probability_and_zero_probability(self):
        result = choose([-745., 0., 0.], b'\x07', temperature=1., draw53=0,
                        mu=2000., tau=1., eta=.1)
        self.assertEqual(result['token'], 0)
        self.assertTrue(math.isfinite(result['selected_log_probability']))
        self.assertLess(result['selected_log_probability'], -745.)
        self.assertEqual(result['selected_probability'], 0.)
        self.assertEqual(result['surprise_bits'], -result['selected_log_probability']/math.log(2.))

    def test_finite_negative_mu_forces_best_and_proposes_unclamped_feedback(self):
        result = choose([0., 1., -2.], b'\x07', temperature=.8, draw53=0,
                        mu=-2.5, tau=4., eta=.05)
        self.assertEqual(result['token'], 1)
        self.assertEqual(result['selected_log_probability'], 0.)
        self.assertEqual(result['selected_probability'], 1.)
        self.assertEqual(result['surprise_bits'], 0.)
        self.assertEqual(result['next_mu'], -2.3)


if __name__ == '__main__':
    unittest.main()
