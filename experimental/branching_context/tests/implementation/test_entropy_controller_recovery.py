"""Focused CPU checks for entropy samplers at grammar/controller boundaries."""
import copy
import math
import threading
import unittest

from context_time_travel import min_p, sampled_controller, typical
from context_time_travel.store import IntegrityError, InvalidOperation
import test_grammar_first as grammar_fixtures


BASE = dict(temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0)


def mask(vocab_size, allowed):
    bits = bytearray((vocab_size + 7) // 8)
    for token in allowed:
        bits[token >> 3] |= 1 << (token & 7)
    return bytes(bits)


class EntropyControllerRecoveryTests(unittest.TestCase):
    def grammar(self, *, sampling=None):
        factory = grammar_fixtures.GrammarFirstTests()
        engine = factory.make_engine({
            "edit": {"format": "gbnf", "source": 'root ::= "AB" | "AC"'},
        })
        state = engine.start("edit", "recovery-test", (), engine.vocab_size,
                             sampling=sampling)
        return engine, state

    def test_grammar_mask_and_parser_resume_keep_typical_dynatemp_inside_legal_support(self):
        policy = dict(BASE, typical_p=0.9, dynatemp_range=0.35,
                      dynatemp_exponent=1.0)
        engine, state = self.grammar(sampling=policy)
        session = engine.restore(state, ())
        logits = [-100.0] * engine.vocab_size
        logits[ord("A")] = 1.0
        logits[ord("Z")] = 1e6  # the grammar mask must exclude this maximum
        first = session.choose(logits)
        self.assertEqual(first, ord("A"))
        state = session.consume(state, first, b"A", False)
        self.assertEqual(state["next_draw"], 1)

        # Resume from canonical partial history. The replayed parser permits B/C,
        # while the high-scoring unrelated token remains unavailable.
        resumed = engine.restore(state, (ord("A"),))
        resumed.configure_generation(dict(BASE, typical_p=0.75,
                                           dynatemp_range=0.25,
                                           dynatemp_exponent=1.0))
        resumed_state = resumed.current_controller
        resumed_logits = [-100.0] * engine.vocab_size
        resumed_logits[ord("B")] = 2.0
        resumed_logits[ord("C")] = 1.0
        resumed_logits[ord("Z")] = 1e6
        selected = resumed.choose(resumed_logits)
        self.assertIn(selected, (ord("B"), ord("C")))
        state = resumed.consume(resumed_state, selected, bytes((selected,)), False)
        self.assertEqual(state["accepted_in_scope"], 2)
        self.assertTrue(resumed.is_complete())

    def test_forked_grammar_sessions_detach_draw_cursor_and_generation_settings(self):
        initial = dict(BASE, typical_p=0.8)
        engine, state = self.grammar(sampling=initial)
        left = engine.restore(copy.deepcopy(state), ())
        right = engine.restore(copy.deepcopy(state), ())
        left.configure_generation(dict(BASE, typical_p=0.45,
                                       dynatemp_range=0.2,
                                       dynatemp_exponent=1.5))

        self.assertEqual(right.current_controller, state)
        self.assertEqual(left.current_controller["policy"]["typical_p"], 0.45)
        self.assertNotIn("dynatemp_range", right.current_controller["policy"])
        for session in (left, right):
            before = session.current_controller
            logits = [-10.0] * engine.vocab_size
            logits[ord("A")] = 3.0
            token = session.choose(logits)
            session.consume(before, token, b"A", False)
        self.assertEqual(left.current_controller["next_draw"], 1)
        self.assertEqual(right.current_controller["next_draw"], 1)
        self.assertEqual(state["next_draw"], 0)
        self.assertEqual(state["policy"]["typical_p"], 0.8)

    def test_grammar_session_rejects_foreign_thread_before_state_mutation(self):
        engine, state = self.grammar(sampling=dict(BASE, typical_p=0.9))
        session = engine.restore(state, ())
        failures = []

        def call_from_other_thread():
            try:
                session.choose([0.0] * engine.vocab_size)
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=call_from_other_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], InvalidOperation)
        self.assertEqual(session.current_controller, state)

    def test_settings_cannot_change_between_selection_and_consumption(self):
        engine, state = self.grammar(sampling=dict(BASE, typical_p=0.9))
        session = engine.restore(state, ())
        logits = [-5.0] * engine.vocab_size
        logits[ord("A")] = 2.0
        selected = session.choose(logits)
        with self.assertRaises(InvalidOperation):
            session.configure_generation(dict(BASE, typical_p=0.5,
                                              dynatemp_range=0.2))
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller
        self.assertEqual(state["next_draw"], 0)
        updated = session.consume(state, selected, b"A", False)
        self.assertEqual(updated["next_draw"], 1)
        self.assertEqual(updated["policy"]["typical_p"], 0.9)

    def test_selector_exception_leaves_no_pending_controller_or_draw_advance(self):
        tokenizer = grammar_fixtures.TinyByteTokenizer()
        vocab = len(tokenizer.tokens)
        state = sampled_controller.start(dict(BASE, typical_p=0.9,
                                              dynatemp_range=0.2),
                                         "selector-error", (), vocab,
                                         [grammar_fixtures.TinyByteTokenizer.EOS])
        session = sampled_controller.PlainSamplingSession.restore(
            state, (), vocab, tokenizer.token_bytes,
            lambda token: token == grammar_fixtures.TinyByteTokenizer.EOS)
        before = session.current_controller
        with self.assertRaisesRegex(RuntimeError, "selector exploded"):
            session.choose(mask(vocab, (ord("A"), ord("B"))),
                           lambda *args, **kwargs: (_ for _ in ()).throw(
                               RuntimeError("selector exploded")))
        self.assertEqual(session.current_controller, before)
        self.assertEqual(session.current_controller["next_draw"], 0)

    def test_positive_base_temperature_with_zero_effective_temperature_consumes_one_draw(self):
        tokenizer = grammar_fixtures.TinyByteTokenizer()
        vocab = len(tokenizer.tokens)
        policy = dict(BASE, temperature=0.7, typical_p=1.0,
                      dynatemp_range=0.7, dynatemp_exponent=1.0)
        state = sampled_controller.start(policy, "zero-effective", (), vocab,
                                         [grammar_fixtures.TinyByteTokenizer.EOS])
        session = sampled_controller.PlainSamplingSession.restore(
            state, (), vocab, tokenizer.token_bytes,
            lambda token: token == grammar_fixtures.TinyByteTokenizer.EOS)
        logits = [0.0] * vocab
        logits[ord("A")] = 0.0
        logits[ord("B")] = -1000.0

        def select(actual_mask, temperature, min_keep, draw, **filters):
            result = min_p.choose(logits, actual_mask, temperature=temperature,
                                  min_p=min_keep, draw53=draw, **filters)
            self.assertEqual(result["effective_temperature"], 0.0)
            return result["token"]

        token = session.choose(mask(vocab, (ord("A"), ord("B"))), select)
        self.assertEqual(token, ord("A"))
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller  # a pending selection is not publishable
        committed = session._consume_selected(token, b"A", False)
        self.assertIsNone(committed)
        self.assertEqual(session.current_controller["next_draw"], 1)
        self.assertEqual(state["next_draw"], 0)  # session owns a detached copy

    def test_inactive_entropy_defaults_preserve_exact_selection_receipt(self):
        logits = [0.3, 1.1, -0.5, 0.7]
        legal = bytes((0b00001111,))
        common = dict(temperature=0.7, min_p=0.05, draw53=123456)
        legacy = min_p.choose(logits, legal, **common)
        explicit_inactive = min_p.choose(
            logits, legal, **common, typical_p=1.0,
            dynatemp_range=0.0, dynatemp_exponent=1.0)
        self.assertEqual(explicit_inactive, legacy)

    def test_typical_may_drop_original_max_and_tiny_temperature_keeps_nonempty_support(self):
        # Probabilities .60/.30/.10: entropy-nearest token 1 alone exceeds p=.25,
        # so the original maximum is removed before DynaTemp computes entropy.
        logits = [math.log(0.60), math.log(0.30), math.log(0.10)]
        retained, details = typical.retain_typical(logits, (0, 1, 2), 0.25)
        self.assertEqual(retained, [1])
        result = min_p.choose(
            logits, b"\x07", temperature=1e-200, min_p=0.0,
            draw53=0, typical_p=0.25, dynatemp_range=1e-200,
            dynatemp_exponent=1.0)
        self.assertEqual(result["typical_kept_count"], 1)
        self.assertEqual(result["token"], 1)
        self.assertGreaterEqual(result["kept_count"], 1)
        self.assertEqual(details["kept_count"], 1)


if __name__ == "__main__":
    unittest.main()
