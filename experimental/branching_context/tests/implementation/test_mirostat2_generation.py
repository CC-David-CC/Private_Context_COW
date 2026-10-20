"""Independent CPU tests for the Mirostat-v2 selection reference."""
from __future__ import annotations

import copy
import json
import math
import unittest

from context_time_travel.mirostat2 import choose, start_state, validate_state
from context_time_travel import min_p


def mask_for(vocab_size, legal_ids):
    result = bytearray((vocab_size + 7) // 8)
    for token_id in legal_ids:
        result[token_id >> 3] |= 1 << (token_id & 7)
    return bytes(result)


class Mirostat2ReferenceTests(unittest.TestCase):
    def test_hand_calculation_renormalizes_after_surprise_truncation(self):
        # At T=1, logits [0,-1,-2] have original surprisals about
        # [.63, 2.03, 3.50] bits. mu=2.1 retains IDs 0 and 1. Their
        # renormalized probabilities are 1/(1+e^-1) and e^-1/(1+e^-1).
        state = start_state(2.0)
        self.assertEqual(state, {"protocol": "mirostat2-state-v1", "version": 1, "mu": 4.0})
        result = choose([0.0, -1.0, -2.0], mask_for(3, [0, 1, 2]),
                        temperature=1.0, draw53=(1 << 53) - 1,
                        mu=2.1, tau=2.0, eta=0.5)
        self.assertEqual(result["token"], 1)
        self.assertEqual(result["kept_ids"], [0, 1])
        self.assertEqual(result["kept_count"], 2)
        self.assertAlmostEqual(result["surprise_bits"],
                               math.log2(1.0 + math.e), places=14)
        self.assertAlmostEqual(result["next_mu"],
                               2.1 - 0.5 * (math.log2(1.0 + math.e) - 2.0), places=14)
        self.assertEqual(state["mu"], 4.0, "selection must not mutate persisted state")

    def test_temperature_and_mask_are_applied_before_mirostat_truncation(self):
        result = choose([100.0, 0.0, -2.0], mask_for(3, [1, 2]),
                        temperature=0.5, draw53=0, mu=10.0, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 1)
        self.assertEqual(result["legal_ids"], [1, 2])
        self.assertEqual(result["kept_ids"], [1, 2])

    def test_forces_one_candidate_when_mu_excludes_every_legal_token(self):
        result = choose([0.0, -1.0], mask_for(2, [0, 1]), temperature=0.8,
                        draw53=(1 << 53) - 1, mu=-1.0, tau=4.0, eta=0.05)
        self.assertEqual(result["kept_ids"], [0])
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["surprise_bits"], 0.0)

    def test_ties_use_token_id_order_and_draw_is_external(self):
        result = choose([0.0, 0.0, 0.0], mask_for(3, [0, 1, 2]), temperature=1.0,
                        draw53=0, mu=10.0, tau=4.0, eta=0.0)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["kept_ids"], [0, 1, 2])
        self.assertAlmostEqual(result["surprise_bits"], math.log2(3.0), places=14)
        self.assertEqual(result["next_mu"], 10.0)
        self.assertEqual(result["draw53"], 0)

    def test_single_legal_token_has_zero_surprise(self):
        result = choose([1.0, 200.0, -2.0], mask_for(3, [0]), temperature=0.8,
                        draw53=123, mu=3.0, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["kept_ids"], [0])
        self.assertEqual(result["surprise_bits"], 0.0)
        self.assertAlmostEqual(result["next_mu"], 3.2)

    def test_underflowing_legal_weight_keeps_finite_log_domain_surprise(self):
        result = choose([0.0, -2000.0], mask_for(2, [0, 1]), temperature=1.0,
                        draw53=(1 << 53) - 1, mu=4000.0, tau=4.0, eta=0.05)
        self.assertEqual(result["kept_ids"], [0, 1])
        self.assertEqual(result["token"], 0)
        self.assertTrue(math.isfinite(result["surprise_bits"]))
        self.assertTrue(math.isfinite(result["next_mu"]))

    def test_cdf_boundary_advances_to_next_token_id(self):
        result = choose([0.0, 0.0], mask_for(2, [0, 1]), temperature=1.0,
                        draw53=1 << 52, mu=8.0, tau=4.0, eta=0.0)
        self.assertEqual(result["token"], 1)

    def test_restored_mu_state_is_detached_and_fork_local(self):
        original = start_state(4.0)
        encoded = json.dumps(original, sort_keys=True)
        restored = validate_state(json.loads(encoded))
        self.assertEqual(restored, original)
        fork_a = copy.deepcopy(restored)
        fork_b = copy.deepcopy(restored)
        a = choose([0.0, -1.0], mask_for(2, [0, 1]), temperature=0.8,
                   draw53=0, mu=fork_a["mu"], tau=4.0, eta=0.05)
        fork_a["mu"] = a["next_mu"]
        self.assertEqual(fork_b["mu"], 8.0)
        self.assertEqual(restored["mu"], 8.0)
        self.assertEqual(validate_state(fork_a), fork_a)

    def test_input_vectors_and_mask_are_never_mutated(self):
        logits = [3.0, -2.0, 1.0]
        mask = mask_for(3, [0, 2])
        old_logits, old_mask = logits.copy(), bytes(mask)
        choose(logits, mask, temperature=0.8, draw53=4, mu=8.0, tau=4.0, eta=0.05)
        self.assertEqual(logits, old_logits)
        self.assertEqual(mask, old_mask)

    def test_invalid_parameters_and_draws_fail_closed(self):
        logits, mask = [0.0, 1.0], mask_for(2, [0, 1])
        valid = dict(temperature=0.8, draw53=0, mu=8.0, tau=4.0, eta=0.05)
        for field, values in {
            "temperature": [0.0, -1.0, math.inf, math.nan, True],
            "mu": [math.inf, math.nan, True],
            "tau": [0.0, -1.0, math.inf, True],
            "eta": [-0.1, math.inf, math.nan, True],
            "draw53": [-1, 1 << 53, True, 0.0],
        }.items():
            for value in values:
                args = dict(valid)
                args[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    choose(logits, mask, **args)

    def test_bad_support_and_nonfinite_logits_fail_closed(self):
        with self.assertRaises(ValueError):
            choose([], b"", temperature=0.8, draw53=0, mu=8.0, tau=4.0, eta=0.05)
        with self.assertRaises(ValueError):
            choose([0.0, 1.0], b"\0", temperature=0.8,
                   draw53=0, mu=8.0, tau=4.0, eta=0.05)
        with self.assertRaises(ValueError):
            choose([0.0, math.nan], mask_for(2, [0]), temperature=0.8,
                   draw53=0, mu=8.0, tau=4.0, eta=0.05)
        with self.assertRaises(ValueError):
            choose([0.0, 1.0], mask_for(2, []), temperature=0.8,
                   draw53=0, mu=8.0, tau=4.0, eta=0.05)

    def test_state_schema_rejects_unknown_fields_and_nonfinite_mu(self):
        for altered in [
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": math.nan},
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": True},
            {"protocol": "mirostat2-state-v1", "version": True, "mu": 8.0},
            {"protocol": "mirostat2-state-v1", "version": 1, "mu": 8.0, "extra": 0},
        ]:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                validate_state(altered)

    def test_nonfinite_feedback_update_is_rejected_not_clamped(self):
        with self.assertRaises(ValueError):
            choose([0.0, -1.0], mask_for(2, [0, 1]), temperature=1.0,
                   draw53=0, mu=1.0e308, tau=1.0, eta=1.7e308)


class Mirostat2StabilityTests(unittest.TestCase):
    def test_common_offset_is_removed_before_temperature_scaling(self):
        result = choose([1.0e308, 1.0e308], mask_for(2, [0, 1]), temperature=0.5, draw53=1 << 52, mu=8.0, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 1)
        self.assertEqual(result["kept_ids"], [0, 1])
        self.assertAlmostEqual(result["surprise_bits"], 1.0)

    def test_float32_frontier_common_offset_survives_tiny_temperature(self):
        result = choose([3.0e38, 3.0e38], mask_for(2, [0, 1]), temperature=1.0e-300, draw53=0, mu=8.0, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["kept_ids"], [0, 1])
        self.assertAlmostEqual(result["surprise_bits"], 1.0)

    def test_masked_extreme_logit_is_not_temperature_scaled(self):
        result = choose([0.0, 3.0e38], mask_for(2, [0]), temperature=1.0e-300, draw53=123, mu=8.0, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["legal_ids"], [0])
        self.assertEqual(result["surprise_bits"], 0.0)

    def test_negative_infinite_shift_underflow_is_zero_weight_never_selected(self):
        result = choose([3.0e38, -3.0e38], mask_for(2, [0, 1]), temperature=1.0e-300, draw53=(1 << 53) - 1, mu=1.0e308, tau=4.0, eta=0.05)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["kept_ids"], [0])
        self.assertEqual(result["surprise_bits"], 0.0)


class Mirostat2GenerationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.legacy = dict(temperature=0.8, min_p=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)

    def test_old_policy_shape_is_unchanged_and_v6_accepts_it(self):
        normalized = min_p.validate_policy(self.legacy)
        self.assertEqual(set(normalized), set(self.legacy))
        self.assertNotIn("mirostat", normalized)
        self.assertEqual(min_p.SAMPLING_PROTOCOL_V6, "full-vocab-mirostat2-u53-v6")
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, normalized))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V5, normalized))

    def test_any_group_field_normalizes_the_entire_mirostat_group(self):
        normalized = min_p.validate_policy({**self.legacy, "mirostat": 2})
        self.assertEqual(normalized["mirostat"], 2)
        self.assertEqual(normalized["mirostat_tau"], 5.0)
        self.assertEqual(normalized["mirostat_eta"], 0.1)
        partial = min_p.validate_policy({**self.legacy, "mirostat_eta": 0.0})
        self.assertEqual((partial["mirostat"], partial["mirostat_tau"], partial["mirostat_eta"]), (0, 5.0, 0.0))

    def test_active_mirostat_preset_is_v6_only_and_keeps_dry(self):
        settings = {**self.legacy, "mirostat": 2, "mirostat_tau": 4.0, "mirostat_eta": 0.05, "dry_multiplier": 0.25}
        normalized = min_p.validate_policy(settings)
        self.assertEqual(normalized["dry_multiplier"], 0.25)
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, normalized))
        self.assertFalse(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V5, normalized))
        self.assertEqual(min_p.sampling_policy(normalized)["mirostat"], 2)

    def test_explicit_mode_zero_is_compatible_with_legacy_protocols(self):
        normalized = min_p.validate_policy({**self.legacy, "mirostat": 0})
        self.assertEqual(normalized["mirostat"], 0)
        self.assertEqual(normalized["mirostat_tau"], 5.0)
        self.assertEqual(normalized["mirostat_eta"], 0.1)
        self.assertTrue(min_p.supports_policy(min_p.MIN_P_PROTOCOL, normalized))
        self.assertTrue(min_p.supports_policy(min_p.SAMPLING_PROTOCOL_V6, normalized))

    def test_mode_two_rejects_conflicting_probability_truncators(self):
        conflicts = [{"min_p": 0.01}, {"top_k": 1}, {"top_p": 0.9}, {"typical_p": 0.95}, {"dynatemp_range": 0.1}, {"xtc_probability": 0.01}, {"temperature": 0.0}]
        for change in conflicts:
            with self.subTest(change=change), self.assertRaises(ValueError):
                min_p.validate_policy({**self.legacy, "mirostat": 2, **change})

    def test_mirostat_group_rejects_invalid_modes_and_numbers(self):
        invalid = [{"mirostat": True}, {"mirostat": 1}, {"mirostat": 2.0}, {"mirostat": -1}, {"mirostat_tau": 0.0}, {"mirostat_tau": -1.0}, {"mirostat_tau": 1.0e308}, {"mirostat_tau": math.nan}, {"mirostat_tau": True}, {"mirostat_eta": -0.01}, {"mirostat_eta": math.inf}, {"mirostat_eta": True}]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                min_p.validate_policy({**self.legacy, **change})
