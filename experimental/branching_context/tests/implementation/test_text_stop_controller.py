import copy
import unittest

from context_time_travel.sampled_controller import PlainSamplingSession, start
from context_time_travel.store import IntegrityError, InvalidOperation


class ByteTokenizer:
    pieces = (b"x", b"<ST", b"OP>hidden", b"\xe2", b"z", b"")
    def token_bytes(self, token): return self.pieces[token]
    def is_eog(self, token): return token == 5


def policy(vocab):
    return {"temperature": 0.0, "min_p": 0.05, "top_k": 0, "top_p": 1.0,
            "repetition_penalty": 1.0}


def take(session, state, token):
    chosen = session.choose(bytes([0x1f]), lambda *_args, **_kwargs: token)
    assert chosen == token
    return session.consume(state, token, session._token_bytes(token), session._is_eog(token))


class TextStopControllerTests(unittest.TestCase):
    def test_stop_state_survives_pause_and_hides_marker_and_token_tail(self):
        tok = ByteTokenizer()
        state = start(policy(len(tok.pieces)), "stop-controller", (), len(tok.pieces), [5],
                      text_stops=["<STOP>"])
        self.assertEqual(state["schema"], 5)
        session = PlainSamplingSession.restore(state, (), len(tok.pieces), tok.token_bytes, tok.is_eog)
        state = take(session, state, 0)
        state = take(session, state, 1)
        self.assertEqual(state["text_stop"]["pending_hex"], b"<ST".hex())
        self.assertEqual(state["output_pending_hex"], "")
        restored = PlainSamplingSession.restore(state, (0, 1), len(tok.pieces), tok.token_bytes, tok.is_eog)
        state = take(restored, state, 2)
        self.assertEqual(state["status"], "stopped")
        self.assertTrue(state["text_stop"]["stopped"])
        self.assertEqual(state["text_stop"]["matched_stop"], "<STOP>")
        self.assertEqual(state["accepted_in_scope"], 3)  # full token was consumed
        self.assertEqual(state["seen"], {"0": 1, "1": 1, "2": 1})
        # This field is incremental UTF-8 decoder carry, not accumulated text.
        self.assertEqual(state["output_pending_hex"], "")
        with self.assertRaises(IntegrityError):
            PlainSamplingSession.restore(state, (0, 1, 2, 3), len(tok.pieces), tok.token_bytes, tok.is_eog)

    def test_restore_rejects_stop_matcher_state_not_replayed_from_history(self):
        tok = ByteTokenizer()
        state = start(policy(len(tok.pieces)), "stop-controller", (), len(tok.pieces), [5],
                      text_stops=["<STOP>"])
        state["text_stop"]["pending_hex"] = b"<ST".hex()  # valid prefix, but not bound by empty history
        with self.assertRaises(IntegrityError):
            PlainSamplingSession.restore(state, (), len(tok.pieces), tok.token_bytes, tok.is_eog)

    def test_empty_stops_preserve_legacy_schema(self):
        state = start(policy(6), "no-stop", (), 6, [5])
        self.assertEqual(state["schema"], 3)
        self.assertNotIn("text_stop", state)

    def test_configured_eos_id_finishes_stop_matcher_without_eog_callback(self):
        tok = ByteTokenizer()
        state = start(policy(len(tok.pieces)), "eos-id-stop", (), len(tok.pieces), [5],
                      text_stops=["<STOP>"])
        no_eog = lambda _token: False
        session = PlainSamplingSession.restore(state, (), len(tok.pieces), tok.token_bytes, no_eog)
        self.assertEqual(session.choose(bytes([0x3f]), lambda *_args, **_kwargs: 5), 5)
        state = session.consume(state, 5, tok.token_bytes(5), False)
        self.assertEqual(state["status"], "stopped")
        self.assertTrue(state["text_stop"]["finished"])
        restored = PlainSamplingSession.restore(state, (5,), len(tok.pieces), tok.token_bytes, no_eog)
        self.assertTrue(restored.is_stopped())

    def test_invalid_falsy_stop_configurations_are_not_silently_disabled(self):
        for invalid in (False, 0, "", None):
            with self.subTest(invalid=invalid), self.assertRaises((InvalidOperation, ValueError)):
                start(policy(6), "invalid-stop-config", (), 6, [5], text_stops=invalid)

    def test_incomplete_utf8_before_stop_survives_pause_and_terminal_flush(self):
        tok = ByteTokenizer()
        state = start(policy(len(tok.pieces)), "utf8-stop", (), len(tok.pieces), [5],
                      text_stops=["<STOP>"])
        session = PlainSamplingSession.restore(state, (), len(tok.pieces), tok.token_bytes, tok.is_eog)
        state = take(session, state, 3)  # an ordinary token with an incomplete UTF-8 prefix
        self.assertEqual(state["output_pending_hex"], "e2")
        state = take(session, state, 1)
        self.assertEqual(state["output_pending_hex"], "e2")
        resumed = PlainSamplingSession.restore(state, (3, 1), len(tok.pieces), tok.token_bytes, tok.is_eog)
        state = take(resumed, state, 2)
        self.assertEqual(state["output_pending_hex"], "e2")
        self.assertTrue(state["text_stop"]["stopped"])
        self.assertEqual(state["text_stop"]["matched_stop"], "<STOP>")
        # The presentation decoder performs the same replacement flush at terminal stop.
        import codecs
        decoder = codecs.getincrementaldecoder("utf8")(errors="replace")
        decoder.setstate((bytes.fromhex(state["output_pending_hex"]), 0))
        self.assertEqual(decoder.decode(b"", final=True), "\ufffd")


if __name__ == "__main__":
    unittest.main()
