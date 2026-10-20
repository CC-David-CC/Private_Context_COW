"""Final-answer views derive from committed canonical tokens, not text tags."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from context_time_travel.store import Store, InvalidOperation
from serve.cow_grammar import GrammarRegistry
from serve.cow_text_adapter import CowTextAdapter
from test_thinking_grammar import ThinkingTokenizer


class ThinkingResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'store')
        self.tok = ThinkingTokenizer()
        config = self.root / 'registry.json'
        config.write_text(json.dumps(dict(schema=1, states=dict(
            THINK=dict(format='thinking', source=dict(schema=1, max_tokens=512,
                final=dict(format='gbnf', source='root ::= "A" | "{\\"text\\":\\"literal </think>\\"}"'))),
            PLAIN=dict(format='gbnf', source='root ::= "A"')))))
        self.registry = GrammarRegistry.load(config, self.tok, [256])
        self.adapter = CowTextAdapter(SimpleNamespace(coordinator=SimpleNamespace(store=self.store)),
            self.tok, lambda *_a, **_k: '', lambda *_a, **_k: (), grammar_registry=self.registry)

    def session(self, state='THINK'):
        prefix = list(b'user literal </think>')
        controller = self.registry.start(state, 'fixture', prefix, len(self.tok.tokens))
        return self.registry.restore(controller, prefix), prefix

    def consume(self, session, tokens, more):
        for token in more:
            session._consume_selected(token, self.tok.token_bytes(token), token == 256)
            tokens.append(token)

    def save(self, session, tokens):
        return self.store.create_root('fixture-model', tokens, session.current_controller)

    def test_literal_tag_in_thought_cannot_move_final_boundary(self):
        session, tokens = self.session()
        self.consume(session, tokens, list(b'first </think> still thinking') + [257, ord('A'), 256])
        sid = self.save(session, tokens)
        before = copy.deepcopy(self.store.get_snapshot(sid))
        view = self.adapter.thinking_result(sid)
        self.assertEqual(view['snapshot'], sid)
        self.assertTrue(view['complete'])
        self.assertEqual(view['reasoning_text'], 'first </think> still thinking')
        self.assertEqual(view['final_text'], 'A')
        self.assertEqual(view['progress']['close_reason'], 'natural')
        self.assertEqual(self.store.get_snapshot(sid), before)

    def test_partial_thought_and_partial_final_are_never_executable(self):
        session, tokens = self.session()
        self.consume(session, tokens, list(b'checking'))
        view = self.adapter.thinking_result(self.save(session, tokens))
        self.assertFalse(view['complete'])
        self.assertIsNone(view['final_text'])
        self.consume(session, tokens, [257, ord('A')])
        view = self.adapter.thinking_result(self.save(session, tokens))
        self.assertFalse(view['complete'])
        self.assertIsNone(view['final_text'], 'valid-looking output without consumed EOS is incomplete')

    def test_literal_closing_text_in_json_answer_is_preserved(self):
        session, tokens = self.session()
        text = '{"text":"literal </think>"}'
        # The final grammar represents this literal with the registered token;
        # the second CLOSE belongs to JSON, not the reasoning boundary.
        self.consume(session, tokens, [257, *b'{"text":"literal ', 257, *b'"}', 256])
        view = self.adapter.thinking_result(self.save(session, tokens))
        self.assertEqual(view['final_text'], text)
        self.assertEqual(view['reasoning_text'], '')

    def test_non_thinking_scope_adds_no_view_and_wrong_registry_fails(self):
        session, tokens = self.session('PLAIN')
        self.consume(session, tokens, [ord('A'), 256])
        self.assertIsNone(self.adapter.thinking_result(self.save(session, tokens)))
        session, tokens = self.session()
        sid = self.save(session, tokens)
        self.adapter.grammar_registry = None
        with self.assertRaises(InvalidOperation):
            self.adapter.thinking_result(sid)


    def test_text_stop_never_masquerades_as_a_consumed_final_eos(self):
        for before_close in (True, False):
            with self.subTest(before_close=before_close):
                prefix = list(b'prompt')
                state = self.registry.engine.start('THINK', 'stop-fixture', prefix,
                    len(self.tok.tokens), text_stops=('x' if before_close else 'A',))
                session = self.registry.restore(state, prefix)
                tokens = list(prefix)
                for token in ([ord('x')] if before_close else [257, ord('A')]):
                    logits = [-100.] * len(self.tok.tokens)
                    logits[token] = 100.
                    self.assertEqual(session.choose(logits), token)
                    self.consume(session, tokens, [token])
                self.assertEqual(session.current_controller['status'], 'stopped')
                result = self.adapter.thinking_result(self.save(session, tokens))
                self.assertFalse(result['complete'])
                self.assertIsNone(result['final_text'])


if __name__ == '__main__':
    unittest.main()
