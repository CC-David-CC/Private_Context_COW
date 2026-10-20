import random
import unittest

from context_time_travel.text_stops import TextStopMatcher, normalize_stop_sequences


def _whole_stream_reference(pieces, stops):
    """Brute-force full byte stream oracle, independent of incremental state."""
    data = b"".join(pieces)
    encoded = [value.encode("utf-8") for value in stops]
    candidates = []
    for order, marker in enumerate(encoded):
        for start in range(len(data) - len(marker) + 1):
            if data[start:start + len(marker)] == marker:
                end = start + len(marker)
                candidates.append((end, -len(marker), order, start, marker))
    if not candidates:
        return data, False, None
    _end, _length, _order, start, marker = min(candidates)
    return data[:start], True, marker.decode("utf-8")


class TextStopMatcherTests(unittest.TestCase):
    def test_marker_inside_one_token_consumes_marker_and_hides_token_tail(self):
        matcher = TextStopMatcher(["<END>"])
        result = matcher.feed(b"answer<END>hidden")
        self.assertEqual(result.safe_bytes, b"answer")
        self.assertTrue(result.stopped)
        self.assertEqual(result.matched_stop, "<END>")
        self.assertEqual(matcher.pending, b"")
        with self.assertRaises(ValueError):
            matcher.feed(b"must-not-appear")

    def test_match_crossing_two_and_three_token_pieces(self):
        matcher = TextStopMatcher(["ababcd"])
        self.assertEqual(matcher.feed(b"preaba").safe_bytes, b"pre")
        self.assertEqual(matcher.pending, b"aba")
        self.assertEqual(matcher.feed(b"b").safe_bytes, b"")
        result = matcher.feed(b"cdtail")
        self.assertEqual((result.safe_bytes, result.stopped), (b"", True))
        self.assertEqual(result.matched_stop, "ababcd")

    def test_first_completion_position_then_longest(self):
        matcher = TextStopMatcher(["abc", "bc"])
        result = matcher.feed(b"zabc-tail")
        # `abc` and `bc` end together; the longer marker wins.
        self.assertEqual((result.safe_bytes, result.matched_stop), (b"z", "abc"))

        matcher = TextStopMatcher(["xy", "abc"])
        # `xy` completes at byte 2 although `abc` begins earlier and completes later.
        result = matcher.feed(b"axyaabc")
        self.assertEqual((result.safe_bytes, result.matched_stop), (b"a", "xy"))

    def test_longest_marker_wins_same_end_with_overlapping_starts(self):
        matcher = TextStopMatcher(["bc", "abc"])
        result = matcher.feed(b"abc-tail")
        self.assertEqual((result.safe_bytes, result.matched_stop), (b"", "abc"))

    def test_split_multibyte_utf8_marker_is_matched_as_bytes(self):
        marker = "\U0001f6d1"
        raw = marker.encode("utf-8")
        matcher = TextStopMatcher([marker])
        self.assertEqual(matcher.feed(b"ok" + raw[:1]).safe_bytes, b"ok")
        self.assertEqual(matcher.pending, raw[:1])
        self.assertEqual(matcher.feed(raw[1:3]).safe_bytes, b"")
        result = matcher.feed(raw[3:] + b"after")
        self.assertEqual((result.safe_bytes, result.stopped, result.matched_stop), (b"", True, marker))

    def test_unrelated_incomplete_utf8_before_marker_is_preserved_for_decoder(self):
        matcher = TextStopMatcher(["<STOP>"])
        self.assertEqual(matcher.feed(b"\xe2").safe_bytes, b"\xe2")
        result = matcher.feed(b"<STOP>tail")
        self.assertEqual((result.safe_bytes, result.stopped), (b"", True))

    def test_eos_flushes_unmatched_ambiguous_prefix(self):
        matcher = TextStopMatcher(["<STOP>"])
        self.assertEqual(matcher.feed(b"visible<ST").safe_bytes, b"visible")
        self.assertEqual(matcher.pending, b"<ST")
        result = matcher.feed(b"", final=True)
        self.assertEqual((result.safe_bytes, result.stopped), (b"<ST", False))
        self.assertEqual(matcher.pending, b"")
        self.assertTrue(matcher.finished)
        with self.assertRaises(ValueError):
            matcher.feed(b"later")

    def test_pause_restore_preserves_ambiguous_prefix_and_matches_continuation(self):
        first = TextStopMatcher(["<STOP>"])
        initial = first.feed(b"hello<ST")
        self.assertEqual(initial.safe_bytes, b"hello")
        resumed = TextStopMatcher.restore(first.export_state())
        result = resumed.feed(b"OP>hidden")
        self.assertEqual((result.safe_bytes, result.stopped), (b"", True))
        self.assertEqual(result.matched_stop, "<STOP>")

    def test_default_empty_stop_set_is_byte_identical_passthrough(self):
        matcher = TextStopMatcher()
        pieces = (b"\xf0\x9f", b"\x92", b"\xa9")
        emitted = b"".join(matcher.feed(piece).safe_bytes for piece in pieces)
        emitted += matcher.feed(b"", final=True).safe_bytes
        self.assertEqual(emitted, b"".join(pieces))
        self.assertFalse(matcher.stopped)

    def test_limits_and_malformed_stop_state_rejected(self):
        for values in (None, "x", [""], ["same", "same"], ["\ud800"]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_stop_sequences(values)
        with self.assertRaises(ValueError):
            normalize_stop_sequences(["x"] * 17)
        with self.assertRaises(ValueError):
            normalize_stop_sequences(["x" * 257])
        matcher = TextStopMatcher(["abc"])
        state = matcher.export_state()
        state["pending_hex"] = "ff"  # not a prefix of any configured marker
        with self.assertRaises(ValueError):
            TextStopMatcher.restore(state)
        oversized = matcher.export_state()
        oversized["pending_hex"] = "61" * 100000
        with self.assertRaises(ValueError):
            TextStopMatcher.restore(oversized)

    def test_seeded_chunk_partitions_match_full_stream_reference(self):
        rng = random.Random(734921)
        alphabet = b"abc<>xyz"
        for _ in range(300):
            stops = ["ab", "abc", "<x", "bcx"]
            data = bytes(rng.choice(alphabet) for _ in range(rng.randrange(1, 40)))
            cuts = sorted({0, len(data), *(rng.randrange(len(data) + 1) for _ in range(5))})
            pieces = [data[left:right] for left, right in zip(cuts, cuts[1:])]
            expected = _whole_stream_reference(pieces, stops)
            matcher = TextStopMatcher(stops)
            output = bytearray()
            stopped = False
            matched = None
            for piece in pieces:
                result = matcher.feed(piece)
                output.extend(result.safe_bytes)
                if result.stopped:
                    stopped, matched = True, result.matched_stop
                    break
            if not stopped:
                result = matcher.feed(b"", final=True)
                output.extend(result.safe_bytes)
            self.assertEqual((bytes(output), stopped, matched), expected)


if __name__ == "__main__":
    unittest.main()
