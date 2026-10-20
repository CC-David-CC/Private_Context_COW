"""Independent composed Adaptive-P reference checks, without model inference."""
import copy
import math
import unittest

from context_time_travel import adaptive_p


class AdaptivePPipelineReferenceTests(unittest.TestCase):
    def choose(self, logits, mask, **changes):
        from context_time_travel.adaptive_p_pipeline import choose_v7
        policy = dict(temperature=.8, min_p=.05, top_k=0, top_p=1.,
                      adaptive_target=.55, adaptive_decay=.9)
        policy.update(changes)
        return choose_v7(logits, mask, adaptive_p.initial_state(.55, .9),
                         draw53=1 << 52, **policy)

    def test_equal_probabilities_keep_uniform_cdf_and_update_original_probability(self):
        state = adaptive_p.initial_state(.55, .9)
        result = self.choose([0., 0., 0., 0.], b'\x0f')
        self.assertEqual(result['token'], 2)
        self.assertEqual(result['kept_ids'], [0, 1, 2, 3])
        self.assertEqual(result['original_probability'], .25)
        self.assertEqual(result['draw_count'], 1)
        self.assertEqual(result['feedback_state'], dict(state,
            weighted_sum=.25 + .9 * state['weighted_sum'],
            total_weight=1. + .9 * state['total_weight']))

    def test_top_p_probabilities_are_normalized_before_adaptation(self):
        # Top-K retains IDs 0,1. Top-P=.7 retains both: the first has mass 2/3.
        result = self.choose([math.log(2.), 0., -10., -20.], b'\x0f',
                             temperature=1., min_p=0., top_k=2, top_p=.7)
        self.assertEqual(result['kept_ids'], [0, 1])
        expected = (2. / 3.) if result['token'] == 0 else (1. / 3.)
        self.assertEqual(result['original_probability'], expected)
        self.assertEqual(result['top_k_kept_count'], 2)

    def test_dynamic_zero_temperature_still_consumes_one_adaptive_draw(self):
        result = self.choose([3., 2., 1.], b'\x07', temperature=.5,
                             top_k=1, dynatemp_range=.5, dynatemp_exponent=1.)
        self.assertEqual(result['effective_temperature'], 0.)
        self.assertEqual(result['token'], 0)
        self.assertEqual(result['kept_ids'], [0])
        self.assertEqual(result['original_probability'], 1.)
        self.assertEqual(result['draw_count'], 1)

    def test_combined_filters_and_dry_do_not_mutate_inputs_or_feedback(self):
        from context_time_travel.adaptive_p_pipeline import choose_v7
        logits = [1., 1., 3., -20.]
        state = adaptive_p.initial_state(.55, .9)
        frozen = copy.deepcopy((logits, state))
        result = choose_v7(logits, b'\x0f', state, draw53=1 << 52,
            temperature=.7, min_p=.05, top_k=3, top_p=.95, typical_p=.95,
            dynatemp_range=.1, dynatemp_exponent=1.,
            dry_multiplier=4., dry_base=2., dry_allowed_length=1,
            dry_penalty_last_n=8, dry_history=(1, 2, 1), dry_breakers=(),
            adaptive_target=.55, adaptive_decay=.9)
        self.assertNotIn(3, result['kept_ids'])
        self.assertGreater(result['original_probability'], 0.)
        self.assertEqual((logits, state), frozen)

    def test_incompatible_terminal_filters_and_invalid_inputs_fail(self):
        from context_time_travel.adaptive_p_pipeline import choose_v7
        state = adaptive_p.initial_state(.55, .9)
        valid = dict(temperature=.8, min_p=.05, top_k=0, top_p=1.,
                     adaptive_target=.55, adaptive_decay=.9)
        for changes in ({'mirostat': 2, 'min_p': 0.}, {'xtc_probability': .12, 'xtc_threshold': .15},
                        {'adaptive_target': -1}, {'temperature': 0.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                choose_v7([1., 0.], b'\x03', state, draw53=0, **dict(valid, **changes))
        for scores, mask in (([math.inf, 0.], b'\x02'), ([1., 0.], b'\x00'),
                             ([1., 0.], b'\xff')):
            with self.subTest(scores=scores, mask=mask), self.assertRaises(ValueError):
                choose_v7(scores, mask, state, draw53=0, **valid)


if __name__ == '__main__':
    unittest.main()
