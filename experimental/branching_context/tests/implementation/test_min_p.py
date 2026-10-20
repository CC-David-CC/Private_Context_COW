"""Pure Min-P reference and HMAC draw tests; no backend/model imports."""
import hashlib
import hmac
import math
import random
import unittest

from context_time_travel.min_p import (
    MIN_P_PROTOCOL,
    choose,
    draw53,
    validate_policy,
)


def packed_mask(vocab_size, allowed):
    raw = bytearray((vocab_size + 7) // 8)
    for token in allowed:
        raw[token // 8] |= 1 << (token % 8)
    return bytes(raw)


def independent_draw(stream_key, cursor):
    key = bytes.fromhex(stream_key)
    digest = hmac.new(key, b"ctt-u53-v1:" + cursor.to_bytes(16, "big"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") >> 11


def independent_choose(logits, mask, temperature, min_p, draw):
    legal = [token for token in range(len(logits))
             if mask[token // 8] & (1 << (token % 8))]
    maximum = max(float(logits[token]) for token in legal)
    if min_p == 0:
        kept = legal
    else:
        boundary = maximum + math.log(min_p)
        kept = [token for token in legal if float(logits[token]) >= boundary]
    weights = []
    total = 0.0
    for token in kept:
        weight = math.exp((float(logits[token]) - maximum) / temperature)
        weights.append((token, weight))
        total += weight
    target = draw / (1 << 53) * total
    cumulative = 0.0
    for token, weight in weights:
        if weight > 0:
            cumulative += weight
            if cumulative > target:
                return token, len(legal), len(kept), total
    return next(token for token, weight in reversed(weights) if weight > 0), len(legal), len(kept), total


class PolicyTests(unittest.TestCase):
    def test_explicit_supported_policy_is_canonical(self):
        self.assertEqual(
            validate_policy({
                "temperature": 0.70,
                "min_p": 0.05,
                "top_k": 0,
                "top_p": 1,
                "repetition_penalty": 1,
            }),
            {"temperature": 0.70, "min_p": 0.05, "top_k": 0,
             "top_p": 1.0, "repetition_penalty": 1.0},
        )

    def test_rejects_invalid_ranges_types_missing_and_extra_fields(self):
        good = {"temperature": 0.7, "min_p": 0.05, "top_k": 0,
                "top_p": 1, "repetition_penalty": 1}
        invalid = (
            {**good, "temperature": -1}, {**good, "temperature": float("inf")},
            {**good, "temperature": True}, {**good, "min_p": -0.1},
            {**good, "min_p": float("nan")}, {**good, "top_k": False},
            {**good, "top_k": -1}, {**good, "top_p": 0},
            {**good, "repetition_penalty": 1.1},
            {key: value for key, value in good.items() if key != "min_p"},
            {**good, "adaptive": False},
        )
        for policy in invalid:
            with self.subTest(policy=policy), self.assertRaises((TypeError, ValueError)):
                validate_policy(policy)


class DrawTests(unittest.TestCase):
    def test_matches_independent_hmac_u53_and_streams_are_independent(self):
        key_a = hashlib.sha256(b"branch-A").hexdigest()
        key_b = hashlib.sha256(b"branch-B").hexdigest()
        for cursor in (0, 1, 17, (1 << 128) - 2):
            self.assertEqual(draw53(key_a, cursor), independent_draw(key_a, cursor))
        self.assertNotEqual(draw53(key_a, 7), draw53(key_b, 7))

    def test_draw_rejects_noncanonical_key_and_out_of_range_cursor(self):
        key = "a" * 64
        for bad_key in ("A" * 64, "0" * 63, "g" * 64, b"0" * 32):
            with self.subTest(key=bad_key), self.assertRaises((TypeError, ValueError)):
                draw53(bad_key, 0)
        for cursor in (-1, 1 << 128, True, 1.0):
            with self.subTest(cursor=cursor), self.assertRaises((TypeError, ValueError)):
                draw53(key, cursor)


class ChooseTests(unittest.TestCase):
    def test_full_support_masked_categorical_and_result_receipt(self):
        logits = [0.0] * 130
        mask = packed_mask(len(logits), (1, 64, 129))
        before_logits, before_mask = logits[:], mask
        result = choose(logits, mask, temperature=1.0, min_p=0.0, draw53=0)
        self.assertEqual(result, {
            "selection_protocol": MIN_P_PROTOCOL,
            "token": 1,
            "logits_count": 130,
            "mask_bytes": 17,
            "legal_count": 3,
            "kept_count": 3,
            "draw53": 0,
            "total_weight": 3.0,
        })
        self.assertEqual(logits, before_logits)
        self.assertEqual(mask, before_mask)

    def test_inclusive_min_p_cutoff_is_applied_before_temperature(self):
        cutoff = math.log(0.05)
        logits = [0.0, cutoff, -3.0, 10.0]
        mask = packed_mask(4, (0, 1, 2))
        result = choose(logits, mask, temperature=0.01, min_p=0.05, draw53=0)
        self.assertEqual(result["legal_count"], 3)
        self.assertEqual(result["kept_count"], 2)
        self.assertEqual(result["token"], 0)

    def test_min_p_zero_keeps_support_but_underflowed_weight_is_never_selected(self):
        result = choose([0.0, -1000.0], packed_mask(2, (0, 1)),
                        temperature=1.0, min_p=0.0, draw53=(1 << 53) - 1)
        self.assertEqual(result["kept_count"], 2)
        self.assertEqual(result["token"], 0)
        self.assertEqual(result["total_weight"], 1.0)

    def test_last_positive_token_fallback_when_float_cdf_rounds_to_total(self):
        # A huge legal support makes (u * total) round to total in binary64.
        logits = [0.0] * 4097
        result = choose(logits, packed_mask(len(logits), range(len(logits))),
                        temperature=1.0, min_p=0.0, draw53=(1 << 53) - 1)
        self.assertEqual(result["token"], len(logits) - 1)

    def test_seeded_varying_full_vocab_vectors_match_independent_reference(self):
        rng = random.Random(20261016)
        for case in range(100):
            vocab_size = rng.randint(1, 160)
            logits = [rng.uniform(-24.0, 24.0) for _ in range(vocab_size)]
            legal = sorted(rng.sample(range(vocab_size), rng.randint(1, vocab_size)))
            mask = packed_mask(vocab_size, legal)
            temperature = rng.choice((0.07, 0.25, 0.70, 1.0, 2.5))
            min_p = rng.choice((0.0, 1e-8, 0.05, 0.5, 1.0))
            draw = rng.choice((0, 1, (1 << 52), (1 << 53) - 1,
                               rng.randrange(1 << 53)))
            expected = independent_choose(logits, mask, temperature, min_p, draw)
            actual = choose(logits, mask, temperature=temperature,
                            min_p=min_p, draw53=draw)
            self.assertEqual(
                (actual["token"], actual["legal_count"], actual["kept_count"], actual["total_weight"]),
                expected,
                f"case {case}",
            )

    def test_rejects_bad_masks_empty_support_and_any_nonfinite_raw_logit(self):
        with self.assertRaises(ValueError):
            choose([0.0], b"", temperature=1.0, min_p=0.0, draw53=0)
        with self.assertRaises(ValueError):
            choose([0.0], b"\x02", temperature=1.0, min_p=0.0, draw53=0)
        with self.assertRaises(ValueError):
            choose([0.0], b"\x00", temperature=1.0, min_p=0.0, draw53=0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(logit=bad), self.assertRaises(ValueError):
                choose([0.0, bad], packed_mask(2, (0,)),
                       temperature=1.0, min_p=0.0, draw53=0)

    def test_rejects_boolean_numeric_inputs_and_does_not_mutate_sequences(self):
        for kwargs in (
            {"temperature": True, "min_p": 0.0, "draw53": 0},
            {"temperature": 1.0, "min_p": False, "draw53": 0},
            {"temperature": 1.0, "min_p": 0.0, "draw53": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                choose([0.0], b"\x01", **kwargs)
        for temperature, min_p, draw in ((-0.1, 0.0, 0), (float("nan"), 0.0, 0),
                                         (1.0, 1.01, 0), (1.0, 0.0, 1 << 53)):
            with self.subTest(temperature=temperature, min_p=min_p, draw=draw), self.assertRaises(ValueError):
                choose([0.0], b"\x01", temperature=temperature, min_p=min_p, draw53=draw)


if __name__ == "__main__":
    unittest.main()
