"""Independent CPU regressions for the optional XTC sampling stage."""
import copy
import unittest

from context_time_travel import min_p
from context_time_travel.xtc_pipeline import choose_v5


def policy(**overrides):
    result = dict(temperature=1.0, min_p=0.0, top_k=0, top_p=1.0,
                  repetition_penalty=1.0, dry_multiplier=0.0,
                  dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=256,
                  typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
                  xtc_probability=0.0, xtc_threshold=0.1)
    result.update(overrides)
    return result


def mask_for(size, allowed=None):
    allowed = set(range(size)) if allowed is None else set(allowed)
    return bytes(sum((1 << (token & 7)) for token in range(byte * 8, min(size, byte * 8 + 8))
                     if token in allowed) for byte in range((size + 7) // 8))


class XtcGenerationTests(unittest.TestCase):
    def test_policy_pair_defaults_and_protocol_admission(self):
        normalized = min_p.validate_policy(policy(xtc_probability=0.4))
        self.assertEqual(normalized["xtc_probability"], 0.4)
        self.assertEqual(normalized["xtc_threshold"], 0.1)
        self.assertEqual(min_p.SAMPLING_PROTOCOL_V5, "full-vocab-xtc-min-p-u53-v5")
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V5, normalized))
        self.assertFalse(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V4, normalized))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V4,
                                              min_p.validate_policy(policy())))

    def test_xct_disabled_preserves_v4_token_and_pipeline_metadata(self):
        logits = [2.0, 1.0, -0.5, -3.0]
        mask = mask_for(len(logits))
        settings = policy(temperature=0.8, min_p=0.05, top_k=3, top_p=0.9,
                          typical_p=0.95, dynatemp_range=0.2)
        old_settings = {key: value for key, value in settings.items()
                        if key not in ("repetition_penalty", "xtc_probability", "xtc_threshold")}
        old = min_p.choose(logits, mask, draw53=1 << 52, **old_settings)
        got = choose_v5(logits, mask, draw53=1 << 52, **settings)
        for key, value in old.items():
            self.assertEqual(got[key], value, key)
        self.assertEqual(got["draw_count"], 1)
        self.assertEqual(got["activation_draws"], 0)
        self.assertFalse(got["activated"])
        self.assertEqual(got["removed_count"], 0)

    def test_activation_draw_precedes_categorical_draw_and_removes_high_mass_prefix(self):
        logits = [3.0, 2.0, 1.0, 0.0]
        original = list(logits)
        got = choose_v5(logits, mask_for(4), draw53=0, draw53_second=0,
                        **policy(xtc_probability=1.0, xtc_threshold=0.1))
        self.assertEqual(got["draw_count"], 2)
        self.assertEqual(got["activation_draws"], 1)
        self.assertTrue(got["activated"])
        self.assertEqual(got["pre_xtc_count"], 4)
        self.assertEqual(got["removed_count"], 1)
        self.assertIn(got["token"], (1, 2, 3))
        self.assertEqual(logits, original)
        self.assertEqual(got["draw53_second"], 0)

    def test_nonactivation_still_uses_second_categorical_draw(self):
        got = choose_v5([2.0, 1.0, 0.0], mask_for(3), draw53=(1 << 53) - 1,
                        draw53_second=0,
                        **policy(xtc_probability=0.01, xtc_threshold=0.1))
        self.assertEqual(got["draw_count"], 2)
        self.assertEqual(got["activation_draws"], 1)
        self.assertFalse(got["activated"])
        self.assertEqual(got["removed_count"], 0)
        self.assertEqual(got["token"], 0)

    def test_activation_boundary_is_inclusive_and_categorical_uses_second_draw(self):
        half = (1 << 52)
        common = dict(draw53_second=0,
                      **policy(xtc_probability=0.5, xtc_threshold=0.1))
        boundary = choose_v5([3.0, 2.0, 1.0], mask_for(3), draw53=half, **common)
        above = choose_v5([3.0, 2.0, 1.0], mask_for(3), draw53=half + 1, **common)
        self.assertTrue(boundary["activated"])
        self.assertFalse(above["activated"])
        self.assertEqual(boundary["token"], 1)
        self.assertEqual(above["token"], 0)

    def test_dry_adjustment_is_in_pipeline_and_history_is_immutable(self):
        history = (2, 2)
        breakers = ()
        original_history = tuple(history)
        original_breakers = tuple(breakers)
        settings = policy(dry_multiplier=0.8, dry_base=1.75,
                          dry_allowed_length=1, dry_penalty_last_n=10,
                          xtc_probability=1.0, xtc_threshold=0.1)
        got = choose_v5([4.0, 3.0, 2.0, 1.0], mask_for(4), draw53=0,
                        draw53_second=0, dry_history=history,
                        dry_breakers=breakers, **settings)
        self.assertEqual(got["selection_protocol"], min_p.SAMPLING_PROTOCOL_V5)
        self.assertEqual(history, original_history)
        self.assertEqual(breakers, original_breakers)

    def test_second_draw_required_only_when_active_support_needs_activation(self):
        with self.assertRaisesRegex(ValueError, "second draw"):
            choose_v5([2.0, 1.0], mask_for(2), draw53=0,
                      **policy(xtc_probability=1.0, xtc_threshold=0.1))
        disabled = choose_v5([2.0, 1.0], mask_for(2), draw53=0,
                             **policy(xtc_probability=0.0))
        over_threshold = choose_v5([2.0, 1.0], mask_for(2), draw53=0,
                                   **policy(xtc_probability=1.0, xtc_threshold=0.50001))
        singleton = choose_v5([2.0, 1.0], mask_for(2, (1,)), draw53=0,
                              **policy(xtc_probability=1.0, xtc_threshold=0.1))
        for result in (disabled, over_threshold, singleton):
            self.assertEqual(result["draw_count"], 1)
            self.assertEqual(result["activation_draws"], 0)

    def test_greedy_and_effective_zero_temperature_do_not_consume_activation_draw(self):
        greedy = choose_v5([1.0, 4.0, 4.0], mask_for(3), draw53=0,
                           **policy(temperature=0.0, xtc_probability=1.0))
        self.assertEqual(greedy["token"], 1)
        self.assertEqual(greedy["draw_count"], 0)
        self.assertEqual(greedy["activation_draws"], 0)
        singleton = choose_v5([0.0, 0.0], mask_for(2, (1,)), draw53=9,
                              **policy(xtc_probability=1.0))
        self.assertEqual(singleton["draw_count"], 1)
        self.assertEqual(singleton["activation_draws"], 0)
        dynamic_zero = choose_v5([1000.0, 0.0, -1.0], mask_for(3), draw53=0,
            **policy(temperature=1.0, typical_p=0.95, dynatemp_range=1.0,
                    xtc_probability=1.0, xtc_threshold=0.1))
        self.assertEqual(dynamic_zero["effective_temperature"], 0.0)
        self.assertEqual(dynamic_zero["draw_count"], 1)
        self.assertEqual(dynamic_zero["activation_draws"], 0)

    def test_effective_zero_fallback_preserves_active_dry_history(self):
        logits = [1000.0, 0.0, 1000.5]
        mask = mask_for(3)
        history = (2, 2)
        dry_settings = dict(dry_multiplier=100.0, dry_base=2.0,
                            dry_allowed_length=1, dry_penalty_last_n=10,
                            dry_history=history, dry_breakers=())
        base = policy(temperature=1.0, typical_p=0.95, dynatemp_range=1.0,
                      dry_multiplier=dry_settings["dry_multiplier"],
                      dry_base=dry_settings["dry_base"],
                      dry_allowed_length=dry_settings["dry_allowed_length"],
                      dry_penalty_last_n=dry_settings["dry_penalty_last_n"])
        draw = (1 << 53) - 1
        expected = min_p.choose(logits, mask, draw53=draw,
            **{key: value for key, value in base.items()
               if key not in ("repetition_penalty", "xtc_probability", "xtc_threshold")},
            dry_history=history, dry_breakers=())
        self.assertEqual(expected["token"], 0)
        no_history = min_p.choose(logits, mask, draw53=draw,
            **{key: value for key, value in base.items()
               if key not in ("repetition_penalty", "xtc_probability", "xtc_threshold")},
            dry_history=(), dry_breakers=())
        self.assertEqual(no_history["token"], 2)
        self.assertNotEqual(expected["token"], no_history["token"])
        for second in (0, 1 << 53):
            with self.subTest(draw53_second=second):
                got = choose_v5(logits, mask, draw53=draw,
                    draw53_second=second, xtc_probability=1.0, **{
                        key: value for key, value in base.items()
                        if key not in ("xtc_probability", "xtc_threshold")},
                    dry_history=history, dry_breakers=())
                self.assertEqual(got["token"], expected["token"])
                self.assertEqual(got["effective_temperature"], 0.0)
                self.assertEqual(got["draw_count"], 1)
                self.assertEqual(got["activation_draws"], 0)

    def test_native_no_second_draw_sentinel_is_accepted_and_echoed_when_unused(self):
        sentinel = 1 << 53
        inactive = choose_v5([2.0, 1.0], mask_for(2), draw53=0,
                             draw53_second=sentinel, **policy(xtc_probability=0.0))
        singleton = choose_v5([2.0, 1.0], mask_for(2, (1,)), draw53=0,
                              draw53_second=sentinel,
                              **policy(xtc_probability=1.0, xtc_threshold=0.1))
        self.assertEqual(inactive["draw53_second"], sentinel)
        self.assertEqual(singleton["draw53_second"], sentinel)
        with self.assertRaisesRegex(ValueError, "second draw"):
            choose_v5([2.0, 1.0], mask_for(2), draw53=0,
                      draw53_second=sentinel,
                      **policy(xtc_probability=1.0, xtc_threshold=0.1))

    def test_threshold_zero_keeps_exactly_one_candidate_even_with_extreme_gap(self):
        got = choose_v5([10000.0, 0.0, -10000.0], mask_for(3), draw53=0,
                        draw53_second=0,
                        **policy(xtc_probability=1.0, xtc_threshold=0.0))
        self.assertEqual(got["removed_count"], 2)
        self.assertEqual(got["kept_count"], 1)
        self.assertEqual(got["token"], 2)

    def test_activation_only_sees_grammar_legal_support_and_inputs_are_unchanged(self):
        logits = [4.0, 3.0, 2.0, 1.0]
        original = copy.deepcopy(logits)
        got = choose_v5(logits, mask_for(4, (1, 3)), draw53=0,
                        draw53_second=(1 << 53) - 1,
                        **policy(xtc_probability=1.0, xtc_threshold=0.1))
        self.assertEqual(got["legal_count"], 2)
        self.assertEqual(got["pre_xtc_count"], 2)
        self.assertEqual(got["removed_count"], 1)
        self.assertEqual(got["token"], 3)
        self.assertEqual(logits, original)

    def test_invalid_optional_pair_values_reject_booleans_nan_and_ranges(self):
        for settings in ({"xtc_probability": True}, {"xtc_threshold": False},
                         {"xtc_probability": float("nan")},
                         {"xtc_probability": 1.01}, {"xtc_threshold": -0.01}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                min_p.validate_policy(policy(**settings))


if __name__ == "__main__":
    unittest.main()
