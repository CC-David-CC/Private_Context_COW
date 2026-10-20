"""Bounded thinking uses canonical token progress, never a second transcript."""
import copy
import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError, InvalidOperation
from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class ThinkingTokenizer(Tokenizer):
    EOS = 256
    CLOSE = 257
    OTHER = 258

    def __init__(self):
        super().__init__([BYTE_TO_UNICODE[b] for b in range(256)] +
                         ['<|eos|>', '</think>', '<|im_start|>'], [],
                         [1] * 256 + [3, 4, 3],
                         special_ids={'eos_token_id': self.EOS})


class ThinkingGrammarTests(unittest.TestCase):
    def engine(self, budget=2, final=None):
        return GrammarEngine(ThinkingTokenizer(), {'answer': dict(format='thinking', source=dict(
            schema=1, max_tokens=budget,
            final=final or dict(format='gbnf', source='root ::= "A" | "B"')))}, [256])

    def start(self, engine):
        state = engine.start('answer', 'thinking-test', [], engine.vocab_size)
        return engine.restore(state, [])

    def consume(self, session, token):
        # The real generation loop commits its already-owned selection here;
        # sampled proposals cannot export a controller until consumed.
        session._consume_selected(token, session.engine.tokenizer.token_bytes(token), token == 256)

    def choose(self, session, preferred):
        logits = [0.] * session.engine.vocab_size
        logits[preferred] = 100.
        return session.choose(logits)

    def test_budget_masks_reasoning_then_only_closing_then_final_schema(self):
        session = self.start(self.engine())
        for token in (ord('x'), ord('y')):
            self.assertEqual(self.choose(session, token), token)
            self.consume(session, token)
        self.assertEqual(self.choose(session, ord('z')), 257)
        self.consume(session, 257)
        self.assertEqual(self.choose(session, ord('z')), ord('A'))
        self.consume(session, ord('B'))
        self.assertTrue(session.is_complete())
        self.consume(session, 256)
        self.assertTrue(session.is_stopped())

    def test_natural_early_close_and_zero_budget(self):
        for budget in (0, 128):
            with self.subTest(budget=budget):
                session = self.start(self.engine(budget))
                self.assertEqual(self.choose(session, 257), 257)
                self.consume(session, 257)
                self.consume(session, ord('A'))
                self.consume(session, 256)
                self.assertEqual(session.current_controller['accepted_in_scope'], 3)

    def test_fork_resume_reconstructs_budget_and_preserves_parent_controller(self):
        engine = self.engine()
        parent = self.start(engine)
        self.consume(parent, ord('x'))
        frozen = parent.current_controller
        original = copy.deepcopy(frozen)
        left = engine.restore(frozen, [ord('x')])
        right = engine.restore(frozen, [ord('x')])
        self.consume(left, ord('y'))
        self.consume(left, 257)
        self.consume(left, ord('A'))
        self.consume(right, 257)
        self.consume(right, ord('B'))
        self.assertEqual(frozen, original)
        self.assertEqual(parent.current_controller, original)
        restored = engine.restore(left.current_controller, [ord('x'), ord('y'), 257, ord('A')])
        self.assertEqual(restored.current_controller, left.current_controller)
        self.assertEqual(self.choose(restored, 256), 256)

    def test_wrong_budget_identity_early_eos_and_unrelated_control_reject(self):
        engine = self.engine()
        session = self.start(engine)
        for token in (256, 258):
            with self.subTest(token=token), self.assertRaises(IntegrityError):
                self.consume(session, token)
        with self.assertRaises(IntegrityError):
            self.engine(3).restore(session.current_controller, [])

    def test_utf8_budget_cannot_strand_a_partial_codepoint(self):
        session = self.start(self.engine(2))
        self.consume(session, 0xc3)
        self.assertNotEqual(self.choose(session, 257), 257)
        self.assertEqual(self.choose(session, 0xa9), 0xa9)
        self.consume(session, 0xa9)
        self.assertEqual(self.choose(session, 257), 257)
        one = self.start(self.engine(1))
        self.assertNotEqual(self.choose(one, 0xc3), 0xc3)

    def test_json_final_constraints_and_invalid_definitions(self):
        schema = dict(type='object', properties={'tool': dict(const='read')},
                      required=['tool'], additionalProperties=False)
        session = self.start(self.engine(0, dict(format='json_schema', source=schema)))
        self.consume(session, 257)
        for token in b'{"tool":"read"}':
            self.consume(session, token)
        self.consume(session, 256)
        self.assertTrue(session.is_stopped())
        for budget in (-1, True, 1.5, 32769):
            with self.subTest(budget=budget), self.assertRaises(InvalidOperation):
                self.engine(budget)

    def test_min_p_thoughts_have_independent_draw_progress_and_exact_resume(self):
        engine = self.engine(4)
        policy = dict(temperature=.7, min_p=.05, top_k=2, top_p=.95, repetition_penalty=1.)
        observed = set()
        for seed in range(8):
            state = engine.start('answer', 'thought-seed-' + str(seed), [], engine.vocab_size, sampling=policy)
            session = engine.restore(state, [])
            logits = [-100.] * engine.vocab_size
            logits[ord('x')] = logits[ord('y')] = 1.
            token = session.choose(logits)
            self.consume(session, token)
            history = [token]
            observed.add(token)
            saved = session.current_controller
            resumed = engine.restore(saved, history)
            self.assertEqual(session.choose(logits), resumed.choose(logits))
            self.assertGreater(saved['next_draw'], 0)
            self.assertEqual(state['next_draw'], 0, 'starting state must remain immutable')
        self.assertEqual(observed, {ord('x'), ord('y')})

    def test_min_p_policy_and_rng_do_not_reset_at_closing_boundary(self):
        engine = self.engine(1)
        policy = dict(temperature=.7, min_p=.05, top_k=60, top_p=.95, repetition_penalty=1.)
        state = engine.start('answer', 'sampled-closure', [], engine.vocab_size, sampling=policy)
        session = engine.restore(state, [])
        history = []
        for preferred in (ord('x'), 257, ord('A'), 256):
            previous = session.current_controller
            token = self.choose(session, preferred)
            self.assertEqual(token, preferred)
            self.consume(session, token)
            history.append(token)
            current = session.current_controller
            self.assertEqual(current['policy'], previous['policy'])
            self.assertGreaterEqual(current['next_draw'], previous['next_draw'])
            resumed = engine.restore(current, history)
            self.assertEqual(resumed.current_controller, current)
            self.assertEqual(resumed.thinking_progress(), session.thinking_progress())


if __name__ == '__main__':
    unittest.main()
