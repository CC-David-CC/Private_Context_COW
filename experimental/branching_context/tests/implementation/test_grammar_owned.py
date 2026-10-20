"""Request-owned grammar transitions preserve the public detached controller."""
from concurrent.futures import ThreadPoolExecutor
import copy
import unittest
from unittest import mock

from test_grammar_first import TinyByteTokenizer
from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError, InvalidOperation


class GrammarOwnedTests(unittest.TestCase):
    def setUp(self):
        self.engine = GrammarEngine(TinyByteTokenizer(), {
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "AA"'},
        }, eos_tokens=[TinyByteTokenizer.EOS])
        self.initial = self.engine.start('EDIT', 'owned', (), self.engine.vocab_size)

    def session(self):
        return self.engine.restore(self.initial, ())

    def advance(self, session, token):
        return session._consume_selected(token, self.engine.tokenizer.token_bytes(token),
                                         token == TinyByteTokenizer.EOS)

    def test_owned_and_checked_public_transitions_produce_identical_controllers(self):
        checked, owned = self.session(), self.session()
        current = copy.deepcopy(self.initial)
        for token in [ord('A'), ord('A'), TinyByteTokenizer.EOS]:
            current = checked.consume(current, token, self.engine.tokenizer.token_bytes(token),
                                      token == TinyByteTokenizer.EOS)
            self.assertIsNone(self.advance(owned, token))
            self.assertEqual(owned.current_controller, current)
        self.assertTrue(owned.is_stopped())
        self.assertEqual(self.initial['accepted_in_scope'], 0)

    def test_owned_transition_does_not_copy_or_validate_vocabulary_per_token(self):
        session = self.session()
        with mock.patch('context_time_travel.grammar_first.validate_controller',
                        side_effect=AssertionError('per-token full validation')):
            with mock.patch('context_time_travel.grammar_first.copy.deepcopy',
                            side_effect=AssertionError('per-token deep copy')):
                self.advance(session, ord('A'))
                self.advance(session, ord('A'))
        self.assertEqual(session.current_controller['accepted_in_scope'], 2)

    def test_exported_controller_cannot_mutate_session_or_sibling(self):
        session, sibling = self.session(), self.session()
        exported = session.current_controller
        exported['policy']['allowed'].clear()
        exported['constraint']['agent_state'] = 'OTHER'
        exported['seen']['65'] = 100
        self.advance(session, ord('A'))
        self.assertEqual(session.current_controller['seen'], {'65': 1})
        self.assertEqual(sibling.current_controller, self.initial)
        self.assertEqual(session.current_controller['constraint']['agent_state'], 'EDIT')

    def test_foreign_thread_cannot_advance_or_read_owned_session(self):
        session = self.session()
        with ThreadPoolExecutor(max_workers=1) as pool:
            for call in (lambda: self.advance(session, ord('A')),
                         lambda: session.current_controller,
                         session.is_complete):
                with self.assertRaises(InvalidOperation):
                    pool.submit(call).result()
        self.assertEqual(session.current_controller, self.initial)
        self.advance(session, ord('A'))
        self.assertEqual(session.current_controller['accepted_in_scope'], 1)

    def test_public_stale_controller_rejected_before_matcher_changes(self):
        session = self.session()
        self.advance(session, ord('A'))
        with self.assertRaises(IntegrityError):
            session.consume(self.initial, ord('A'), b'A', False)
        self.advance(session, ord('A'))
        self.assertTrue(session.is_complete())

    def test_owned_utf8_partial_controller_restores_without_losing_pending_byte(self):
        engine = GrammarEngine(TinyByteTokenizer(), {
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "\u00e9"'},
        }, eos_tokens=[TinyByteTokenizer.EOS])
        initial = engine.start('EDIT', 'utf8-owned', [ord('Z')], engine.vocab_size)
        session = engine.restore(initial, [ord('Z')])
        session._consume_selected(0xC3, b'\xc3', False)
        partial = session.current_controller
        self.assertEqual(partial['output_pending_hex'], 'c3')
        self.assertEqual(partial['accepted_in_scope'], 1)
        sibling = engine.restore(partial, [ord('Z'), 0xC3])
        resumed = engine.restore(partial, [ord('Z'), 0xC3])
        resumed._consume_selected(0xA9, b'\xa9', False)
        self.assertEqual(resumed.current_controller['output_pending_hex'], '')
        self.assertTrue(resumed.is_complete())
        self.assertEqual(sibling.current_controller, partial)
        resumed._consume_selected(TinyByteTokenizer.EOS,
                                  engine.tokenizer.token_bytes(TinyByteTokenizer.EOS), True)
        self.assertTrue(resumed.is_stopped())
        self.assertEqual(partial['output_pending_hex'], 'c3')

    def test_invalid_owned_token_preserves_state_and_valid_owner_can_continue(self):
        session = self.session()
        for token, piece, eos in [(ord('B'), b'B', False), (ord('A'), b'B', False),
                                  (ord('A'), b'A', True), (True, b'A', False)]:
            with self.assertRaises((IntegrityError, InvalidOperation)):
                session._consume_selected(token, piece, eos)
            self.assertEqual(session.current_controller, self.initial)
        self.advance(session, ord('A'))
        self.advance(session, ord('A'))
        self.advance(session, TinyByteTokenizer.EOS)
        with self.assertRaises(InvalidOperation):
            self.advance(session, ord('A'))


if __name__ == '__main__':
    unittest.main()
