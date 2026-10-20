"""Tests for the independent, byte-breaker DRY reference implementation."""
import math
import random
import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from context_time_travel.dry import penalties


TOKENS = {
    1: b"a",
    2: b"b",
    3: b"c",
    4: b"\n",
    5: b":",
    6: b'"',
    7: b"*",
    8: b"x",
    9: b"c*",
    10: b"d",
    11: b"y",
    12: b"z",
}


def token_bytes(token):
    return TOKENS[token]


def penalty_map(history, **options):
    return penalties(history, token_bytes=token_bytes, **options)


class DryReferenceTests(unittest.TestCase):
    def test_repeat_maps_to_following_token_and_uses_allowed_exponent(self):
        # The suffix "ab" previously occurs before candidate c.
        self.assertEqual(
            penalty_map([1, 2, 3, 1, 2], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=2, dry_penalty_last_n=256),
            {3: 2.0},
        )

    def test_longest_match_wins_instead_of_summing_matches(self):
        # Candidate x follows matches of lengths 3 and 4. A distinct leading
        # token prevents the shorter occurrence from extending to length 5.
        result = penalty_map([8, 2, 3, 10, 8, 11, 1, 2, 3, 10, 8, 12, 1, 2, 3, 10],
                             dry_multiplier=2.0, dry_base=1.75,
                             dry_allowed_length=1, dry_penalty_last_n=256)
        self.assertEqual(result, {8: 10.71875})

    def test_overlapping_match_is_allowed(self):
        self.assertEqual(
            penalty_map([1, 1, 1], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=2, dry_penalty_last_n=256),
            {1: 2.0},
        )

    def test_latest_breaker_caps_only_current_suffix_not_prior_search(self):
        # The current suffix is "ab" after newline. The earlier "ab" before
        # the breaker remains an eligible source occurrence for candidate c.
        self.assertEqual(
            penalty_map([1, 2, 3, 4, 1, 2], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=2, dry_penalty_last_n=256),
            {3: 2.0},
        )

    def test_candidates_containing_any_fixed_breaker_are_exempt(self):
        for candidate in (4, 5, 6, 7, 9):
            tokens = dict(TOKENS)
            tokens[candidate] = {4: b"\n", 5: b":", 6: b'"', 7: b"*", 9: b"c*"}[candidate]
            def lookup(token, table=tokens):
                return table[token]
            result = penalties([1, 2, candidate, 1, 2], dry_multiplier=2.0,
                               dry_base=1.75, dry_allowed_length=2,
                               dry_penalty_last_n=256, token_bytes=lookup)
            self.assertNotIn(candidate, result)

    def test_window_clips_old_candidates_but_includes_context_tokens(self):
        history = [1, 2, 3, 1, 2]
        self.assertEqual(
            penalty_map(history, dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=1, dry_penalty_last_n=3),
            {},
        )
        self.assertEqual(
            penalty_map(history, dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=1, dry_penalty_last_n=4),
            {3: 2.0},
        )

    def test_zero_allowed_length_qualifies_each_following_history_token(self):
        self.assertEqual(
            penalty_map([1, 2, 3], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=0, dry_penalty_last_n=256),
            {2: 2.0, 3: 2.0},
        )
        self.assertEqual(
            penalty_map([1, 2, 1, 2], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=0, dry_penalty_last_n=256),
            {1: 6.125, 2: 2.0},
        )

    def test_matches_independent_exhaustive_small_history_reference(self):
        from context_time_travel.dry import FLOAT32_MAX, penalties_from_ids

        rng = random.Random(0xD12)
        breaker_ids = {4, 7}
        for _ in range(300):
            history = [rng.randrange(1, 9) for _ in range(rng.randrange(0, 18))]
            last_n = rng.randrange(0, 20)
            allowed = rng.randrange(0, 6)
            multiplier = rng.choice((0.0, 0.25, 1.0, 2.0))
            base = rng.choice((1.0, 1.25, 1.75))
            window = history[-last_n:] if last_n else []
            lengths = {}
            if multiplier != 0.0 and last_n:
                latest = max((i for i, token in enumerate(window) if token in breaker_ids),
                             default=-1)
                current_suffix = len(window) - latest - 1
                for candidate_index in range(1, len(window)):
                    candidate = window[candidate_index]
                    if candidate in breaker_ids:
                        continue
                    if allowed == 0:
                        lengths[candidate] = max(lengths.get(candidate, 0), 0)
                    matching = [
                        length
                        for length in range(max(1, allowed), min(candidate_index, current_suffix) + 1)
                        if window[candidate_index - length:candidate_index] == window[-length:]
                    ]
                    if matching:
                        lengths[candidate] = max(lengths.get(candidate, 0), max(matching))
            expected = {
                token: min(multiplier * base ** (length - allowed), FLOAT32_MAX)
                for token, length in lengths.items()
            }
            actual = penalties_from_ids(
                history,
                breaker_ids,
                dry_multiplier=multiplier,
                dry_base=base,
                dry_allowed_length=allowed,
                dry_penalty_last_n=last_n,
            )
            self.assertEqual(actual, expected, (history, last_n, allowed))

    def test_zero_multiplier_or_last_n_disables_penalties(self):
        self.assertEqual(
            penalty_map([1, 2, 3, 1, 2], dry_multiplier=0.0, dry_base=1.75,
                        dry_allowed_length=1, dry_penalty_last_n=256),
            {},
        )
        self.assertEqual(
            penalty_map([1, 2, 3, 1, 2], dry_multiplier=2.0, dry_base=1.75,
                        dry_allowed_length=1, dry_penalty_last_n=0),
            {},
        )

    def test_overflow_saturates_to_finite_binary32_max(self):
        result = penalty_map([1, 2, 3, 1, 2], dry_multiplier=1e308,
                             dry_base=1e308, dry_allowed_length=1,
                             dry_penalty_last_n=256)
        self.assertEqual(result[3], 3.4028234663852886e38)
        self.assertTrue(math.isfinite(result[3]))

    def test_history_is_not_mutated_and_callback_must_return_bytes(self):
        history = [1, 2, 3, 1, 2]
        before = list(history)
        penalty_map(history, dry_multiplier=2.0, dry_base=1.75,
                    dry_allowed_length=2, dry_penalty_last_n=256)
        self.assertEqual(history, before)
        with self.assertRaises(TypeError):
            penalties([1, 2], dry_multiplier=1.0, token_bytes=lambda token: "a")
        from context_time_travel.dry import penalties_from_ids
        with self.assertRaises(ValueError):
            penalties_from_ids([1, 2], {True}, dry_multiplier=1.0)

    def test_id_wrapper_matches_byte_wrapper_and_only_reads_last_n_window(self):
        from context_time_travel.dry import penalties_from_ids
        history = [1, 2, 3, 1, 2]
        seen = []
        def tracked(token):
            seen.append(token)
            return token_bytes(token)
        byte_result = penalties(history, dry_multiplier=2.0, dry_base=1.75,
                                dry_allowed_length=2, dry_penalty_last_n=3,
                                token_bytes=tracked)
        self.assertEqual(seen, [3, 1, 2])
        id_result = penalties_from_ids(history[-3:], (), dry_multiplier=2.0,
                                       dry_base=1.75, dry_allowed_length=2,
                                       dry_penalty_last_n=3)
        self.assertEqual(byte_result, id_result)
        with_breaker = penalties_from_ids([1, 2, 3, 4, 1, 2], {4},
                                          dry_multiplier=2.0, dry_base=1.75,
                                          dry_allowed_length=2, dry_penalty_last_n=256)
        self.assertEqual(with_breaker, {3: 2.0})

    def test_rejects_invalid_parameter_types_ranges_and_nonfinite_values(self):
        valid = dict(dry_multiplier=2.0, dry_base=1.75,
                     dry_allowed_length=2, dry_penalty_last_n=256)
        invalid = (
            {"dry_multiplier": True},
            {"dry_multiplier": float("nan")},
            {"dry_multiplier": float("inf")},
            {"dry_multiplier": -1.0},
            {"dry_base": 0.99},
            {"dry_base": float("nan")},
            {"dry_allowed_length": True},
            {"dry_allowed_length": -1},
            {"dry_allowed_length": 2**32},
            {"dry_penalty_last_n": False},
            {"dry_penalty_last_n": -1},
            {"dry_penalty_last_n": 2**32},
        )
        for mutation in invalid:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                penalty_map([1, 2], **(valid | mutation))


if __name__ == "__main__":
    unittest.main()
