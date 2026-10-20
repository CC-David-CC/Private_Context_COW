"""Generation-policy contract for the independent Adaptive-P reference."""
from __future__ import annotations

import math
import unittest

from context_time_travel import min_p


class AdaptivePGenerationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(temperature=0.7, min_p=0.05, top_k=0,
                         top_p=1.0, repetition_penalty=1.0)

    def test_legacy_policy_shape_is_byte_compatible_and_v7_accepts_old_sampler(self):
        result = min_p.validate_policy(self.base)
        self.assertEqual(set(result), set(self.base))
        self.assertNotIn("adaptive_target", result)
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V7, result))

    def test_any_adaptive_field_normalizes_the_complete_group(self):
        target_only = min_p.validate_policy({**self.base, "adaptive_target": 0.55})
        self.assertEqual(target_only["adaptive_target"], 0.55)
        self.assertEqual(target_only["adaptive_decay"], 0.90)
        decay_only = min_p.validate_policy({**self.base, "adaptive_decay": 0.0})
        self.assertEqual(decay_only["adaptive_target"], -1.0)
        self.assertEqual(decay_only["adaptive_decay"], 0.0)

    def test_active_target_requires_v7_but_allows_preceding_filters(self):
        policy = min_p.validate_policy({
            **self.base, "adaptive_target": 0.55, "adaptive_decay": 0.9,
            "dry_multiplier": 0.3, "typical_p": 0.92,
            "dynatemp_range": 0.2, "dynatemp_exponent": 1.2,
        })
        self.assertTrue(min_p.adaptive_enabled(policy))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V7, policy))
        self.assertFalse(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, policy))
        self.assertEqual(min_p.sampling_policy(policy)["adaptive_target"], 0.55)

    def test_disabled_target_preserves_old_protocols_and_explicit_parameters(self):
        policy = min_p.validate_policy({**self.base, "adaptive_target": -0.1,
                                        "adaptive_decay": 0.99})
        self.assertFalse(min_p.adaptive_enabled(policy))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, policy))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V7, policy))
        self.assertEqual((policy["adaptive_target"], policy["adaptive_decay"]), (-0.1, 0.99))

    def test_active_adaptive_rejects_zero_temperature_mirostat_and_active_xtc(self):
        for change in (
            {"temperature": 0.0},
            {"mirostat": 2, "min_p": 0.0},
            {"xtc_probability": 0.2, "xtc_threshold": 0.1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                min_p.validate_policy({**self.base, "adaptive_target": 0.5, **change})

    def test_target_and_decay_reject_bool_nonfinite_and_out_of_range(self):
        for change in (
            {"adaptive_target": True}, {"adaptive_target": math.nan},
            {"adaptive_target": math.inf}, {"adaptive_target": 1.00001},
            {"adaptive_decay": True}, {"adaptive_decay": math.nan},
            {"adaptive_decay": math.inf}, {"adaptive_decay": -0.001},
            {"adaptive_decay": 0.991},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                min_p.validate_policy({**self.base, **change})

    def test_old_protocol_support_remains_for_mirostat2_and_v7_supports_it_too(self):
        policy = min_p.validate_policy({**self.base, "temperature": 0.8,
                                        "min_p": 0.0, "top_p": 1.0,
                                        "mirostat": 2})
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, policy))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V7, policy))


if __name__ == "__main__":
    unittest.main()
