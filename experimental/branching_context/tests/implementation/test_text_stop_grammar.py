import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError, InvalidOperation
from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class TinyTokenizer(Tokenizer):
    EOS = 256
    def __init__(self):
        tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ['<|eos|>']
        super().__init__(tokens, [], [1] * 256 + [3], special_ids={'eos_token_id': self.EOS})


class GrammarTextStopTests(unittest.TestCase):
    def test_stop_consumes_full_grammar_token_and_does_not_claim_grammar_complete(self):
        tokenizer = TinyTokenizer()
        engine = GrammarEngine(tokenizer,
            {'READ': {'format': 'gbnf', 'source': 'root ::= "ABC"'}},
            eos_tokens=[TinyTokenizer.EOS])
        state = engine.start('READ', 'stream-stop', [], len(tokenizer.tokens),
                             text_stops=['B'])
        self.assertEqual(state['schema'], 5)  # new greedy grammar scope becomes sampled only here
        session = engine.restore(state, [])
        for char in 'AB':
            token = ord(char)
            selected = session.choose([10.0 if i == token else 0.0 for i in range(len(tokenizer.tokens))])
            self.assertEqual(selected, token)
            state = session.consume(state, token, tokenizer.token_bytes(token), False)
        self.assertTrue(session.is_stopped())
        self.assertFalse(session.is_complete())
        self.assertEqual(state['accepted_in_scope'], 2)
        self.assertEqual(state['text_stop']['matched_stop'], 'B')
        self.assertEqual(state['output_pending_hex'], '')
        restored = engine.restore(state, [ord('A'), ord('B')])
        self.assertTrue(restored.is_stopped())
        with self.assertRaises(IntegrityError):
            engine.restore(state, [ord('A'), ord('B'), ord('C')])

    def test_configured_grammar_controller_rejects_altered_stop_replay(self):
        tokenizer = TinyTokenizer()
        engine = GrammarEngine(tokenizer,
            {'READ': {'format': 'gbnf', 'source': 'root ::= "AB"'}},
            eos_tokens=[TinyTokenizer.EOS])
        state = engine.start('READ', 'stream-stop', [], len(tokenizer.tokens), text_stops=['B'])
        session = engine.restore(state, [])
        token = ord('A')
        self.assertEqual(session.choose([1.0 if i == token else 0.0 for i in range(len(tokenizer.tokens))]), token)
        state = session.consume(state, token, tokenizer.token_bytes(token), False)
        altered = dict(state)
        altered['text_stop'] = dict(state['text_stop'], pending_hex='42')
        with self.assertRaises(IntegrityError):
            engine.restore(altered, [ord('A')])

    def test_invalid_falsy_stop_configurations_are_not_silently_disabled(self):
        tokenizer = TinyTokenizer()
        engine = GrammarEngine(tokenizer,
            {'READ': {'format': 'gbnf', 'source': 'root ::= "A"'}},
            eos_tokens=[TinyTokenizer.EOS])
        for invalid in (False, 0, '', None):
            with self.subTest(invalid=invalid), self.assertRaises((InvalidOperation, ValueError)):
                engine.start('READ', 'invalid-stop-config', [], len(tokenizer.tokens),
                             text_stops=invalid)


if __name__ == '__main__':
    unittest.main()
