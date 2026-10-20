"""Pure Adaptive-P contract tests; no model, server, or native imports."""

import copy
import math
import unittest

from context_time_travel.adaptive_p import (
    PROTOCOL,
    STATE_VERSION,
    accept,
    choose,
    initial_state,
    validate_state,
)


def packed_mask(vocab_size, allowed):
    raw = bytearray((vocab_size + 7) // 8)
    for token in allowed:
        raw[token // 8] |= 1 << (token % 8)
    return bytes(raw)


class AdaptivePInitializationTests(unittest.TestCase):
    def test_cold_start_ema_prior_equals_target(self):
        state = initial_state(0.55, 0.90)
        self.assertEqual(state["version"], STATE_VERSION)
        self.assertAlmostEqual(state["weighted_sum"] / state["total_weight"], 0.55)
        self.assertAlmostEqual(state["weighted_sum"], 5.5)
        self.assertAlmostEqual(state["total_weight"], 10.0)

    def test_validates_target_decay_and_state_without_bool_coercion(self):
        for target, decay in (
            (True, 0.9), (float("nan"), 0.9), (float("inf"), 0.9),
            (1.01, 0.9), (0.5, True), (0.5, float("nan")),
            (0.5, -0.01), (0.5, 0.991),
        ):
            with self.subTest(target=target, decay=decay), self.assertRaises(ValueError):
                initial_state(target, decay)

        for bad_state in (
            {},
            {"version": STATE_VERSION, "weighted_sum": float("nan"), "total_weight": 1.0},
            {"version": STATE_VERSION, "weighted_sum": 0.5, "total_weight": 0.0},
            {"version": STATE_VERSION, "weighted_sum": True, "total_weight": 1.0},
            {"version": STATE_VERSION, "weighted_sum": 0.5, "total_weight": 1.0, "pending": 1},
        ):
            with self.subTest(state=bad_state), self.assertRaises(ValueError):
                validate_state(bad_state)


class AdaptivePSelectionTests(unittest.TestCase):
    def test_adaptive_transform_selects_near_adapted_target_but_reports_original_p(self):
        # This state yields adapted_target = 2*0.55 - 0.90/1.0 = 0.20.
        state = {"version": STATE_VERSION, "weighted_sum": 0.90, "total_weight": 1.0}
        original = copy.deepcopy(state)
        result = choose(
            [0.0, math.log(0.25)], packed_mask(2, (0, 1)), state,
            adaptive_target=0.55, adaptive_decay=0.90, draw53=1 << 52,
        )
        self.assertEqual(result["token"], 1)
        self.assertAlmostEqual(result["original_probability"], 0.2)
        self.assertAlmostEqual(result["adapted_target"], 0.2)
        self.assertEqual(state, original)

    def test_accept_updates_from_original_probability_and_is_detached(self):
        state = initial_state(0.55, 0.90)
        result = choose(
            [0.0, math.log(0.25)], packed_mask(2, (0, 1)), state,
            adaptive_target=0.55, adaptive_decay=0.90, draw53=0,
        )
        before = copy.deepcopy(state)
        updated = accept(state, result["proposal"], result["token"])
        self.assertAlmostEqual(updated["weighted_sum"], result["original_probability"] + 0.90 * before["weighted_sum"])
        self.assertAlmostEqual(updated["total_weight"], 1.0 + 0.90 * before["total_weight"])
        self.assertEqual(state, before)
        self.assertIsNot(updated, state)

    def test_wrong_accepted_token_rejects_without_mutating_state_or_proposal(self):
        state = initial_state(0.55, 0.90)
        result = choose(
            [0.0, -1.0], packed_mask(2, (0, 1)), state,
            adaptive_target=0.55, adaptive_decay=0.90, draw53=0,
        )
        before_state = copy.deepcopy(state)
        before_proposal = copy.deepcopy(result["proposal"])
        wrong = 1 - result["token"]
        with self.assertRaises(ValueError):
            accept(state, result["proposal"], wrong)
        self.assertEqual(state, before_state)
        self.assertEqual(result["proposal"], before_proposal)

    def test_disabled_target_is_ordinary_distribution_and_preserves_state(self):
        state = initial_state(0.55, 0.90)
        result = choose(
            [0.0, 0.0, -4.0], packed_mask(3, (0, 1)), state,
            adaptive_target=-1.0, adaptive_decay=0.90, draw53=0,
        )
        self.assertEqual(result["token"], 0)
        self.assertFalse(result["proposal"]["feedback_enabled"])
        self.assertEqual(accept(state, result["proposal"], result["token"]), state)

    def test_masked_scores_do_not_enter_softmax_or_adaptive_feedback(self):
        state = initial_state(0.55, 0.90)
        result = choose(
            [0.0, -0.2, 100.0], packed_mask(3, (0, 1)), state,
            adaptive_target=0.55, adaptive_decay=0.90, draw53=0,
        )
        self.assertEqual(result["legal_count"], 2)
        self.assertAlmostEqual(result["original_probability"], 1.0 / (1.0 + math.exp(-0.2)))

    def test_equal_scores_tie_selection_follows_lowest_id_order(self):
        state = initial_state(0.55, 0.90)
        first = choose([0.0, 0.0], packed_mask(2, (0, 1)), state,
                       adaptive_target=0.55, adaptive_decay=0.90, draw53=0)
        last_draw = choose([0.0, 0.0], packed_mask(2, (0, 1)), state,
                           adaptive_target=0.55, adaptive_decay=0.90, draw53=(1 << 53) - 1)
        self.assertEqual(first["token"], 0)
        self.assertEqual(last_draw["token"], 1)

    def test_invalid_logits_mask_draw_and_proposal_fail_closed(self):
        state = initial_state(0.55, 0.90)
        kwargs = dict(state=state, adaptive_target=0.55, adaptive_decay=0.90, draw53=0)
        for logits, mask, changes in (
            ([0.0], b"", {}),
            ([0.0], b"\x00", {}),
            ([float("nan")], b"\x01", {}),
            ([float("inf")], b"\x01", {}),
            ([0.0], b"\x01", {"draw53": True}),
            ([0.0], b"\x01", {"adaptive_target": float("inf")}),
            ([0.0], b"\x01", {"adaptive_decay": float("nan")}),
        ):
            with self.subTest(logits=logits, mask=mask, changes=changes), self.assertRaises(ValueError):
                choose(logits, mask, **(kwargs | changes))

        with self.assertRaises(TypeError):
            choose([0.0], b"\x01", state, adaptive_target=0.55,
                   adaptive_decay=0.90, draw53=0, proposal_extra=True)

    def test_separate_branch_states_accept_independently(self):
        base = initial_state(0.55, 0.90)
        branch_a, branch_b = copy.deepcopy(base), copy.deepcopy(base)
        proposal_a = choose([0.0, -1.0], b"\x03", branch_a,
                            adaptive_target=0.55, adaptive_decay=0.90, draw53=0)
        proposal_b = choose([-1.0, 0.0], b"\x03", branch_b,
                            adaptive_target=0.55, adaptive_decay=0.90, draw53=0)
        after_a = accept(branch_a, proposal_a["proposal"], proposal_a["token"])
        after_b = accept(branch_b, proposal_b["proposal"], proposal_b["token"])
        self.assertEqual(base, initial_state(0.55, 0.90))
        self.assertNotEqual(after_a, after_b)


if __name__ == "__main__":
    unittest.main()
