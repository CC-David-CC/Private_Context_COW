"""CPU tests for the opt-in llguidance-backed grammar-first engine."""
import copy
import builtins
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

from context_time_travel import controller
from context_time_travel.grammar_first import GrammarEngine, validate_controller
from context_time_travel.store import IntegrityError, InvalidOperation
from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class TinyByteTokenizer(Tokenizer):
    """A real Strata Tokenizer with one token for each byte and explicit controls."""

    EOS = 256
    IM_START = 257
    TOOL_CALL = 258

    def __init__(self):
        tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + ['<|eos|>', '<|im_start|>', '<tool_call>']
        token_types = [1] * 256 + [3, 3, 4]
        super().__init__(tokens, [], token_types,
                         special_ids={'eos_token_id': self.EOS,
                                      'im_start_token_id': self.IM_START,
                                      'tool_call_token_id': self.TOOL_CALL})


class GrammarFirstTests(unittest.TestCase):
    def test_edit_operation_enum_excludes_unlisted_operations(self):
        engine = self.make_engine({'patch': {'format': 'gbnf', 'source':
            'root ::= "{\\"op\\":" op "}"\nop ::= "\\"replace\\"" | "\\"insert\\""'}})
        for operation in ('replace', 'insert'):
            state, session = self.start(engine, 'patch')
            self.consume_text(session, state, '{"op":"' + operation + '"}')
            self.assertTrue(session.is_complete())
        state, session = self.start(engine, 'patch')
        with self.assertRaises(IntegrityError):
            self.consume_text(session, state, '{"op":"delete"}')

    def test_restore_checks_repetition_and_pending_decoder_against_canonical_tokens(self):
        engine = self.make_engine()
        state, _ = self.start(engine, history=[ord('Z')])
        for field, value in [('seen', {str(ord('Z')): 2}), ('output_pending_hex', 'c3'), ('stop_tail', [65])]:
            altered = copy.deepcopy(state)
            altered[field] = value
            with self.assertRaises(IntegrityError):
                engine.restore(altered, [ord('Z')])

    def test_boolean_token_id_is_rejected(self):
        engine = self.make_engine()
        state, _ = self.start(engine)
        state['policy']['allowed'][1] = True
        with self.assertRaises(IntegrityError):
            validate_controller(state)

    def make_engine(self, states=None, tokenizer=None, eos=None):
        tokenizer = tokenizer or TinyByteTokenizer()
        states = states or {
            'edit': {'format': 'gbnf', 'source': 'root ::= "AA" | "AB"'},
            'answer': {'format': 'gbnf', 'source': 'root ::= "A"'},
        }
        return GrammarEngine(tokenizer, states,
                             eos_tokens=[TinyByteTokenizer.EOS] if eos is None else eos)

    def start(self, engine, state='edit', history=()):
        controller_state = engine.start(state, 'grammar-test-stream', list(history),
                                         len(engine.tokenizer.tokens))
        return controller_state, engine.restore(controller_state, list(history))

    def consume_text(self, session, state, text):
        for token in session.engine.tokenizer.encode(text, parse_special=False):
            state = session.consume(state, token, session.engine.tokenizer.token_bytes(token), False)
        return state

    def test_real_gbnf_mask_preserves_raw_argmax_and_ties_without_rng_draws(self):
        engine = self.make_engine()
        state, session = self.start(engine)
        logits = [0.0] * engine.vocab_size
        logits[ord('Z')] = 1000.0  # illegal raw maximum
        logits[ord('A')] = 5.0
        selected = session.choose(logits)
        self.assertEqual(selected, ord('A'))
        state = session.consume(state, selected, engine.tokenizer.token_bytes(selected), False)
        self.assertEqual(state['next_draw'], 0)

        logits = [0.0] * engine.vocab_size
        logits[ord('A')] = logits[ord('B')] = 9.0
        self.assertEqual(session.choose(logits), ord('A'), 'equal legal logits choose lowest token id')
        self.assertEqual(session.current_controller['next_draw'], 0)

    def test_accepting_is_not_stopped_and_explicit_eos_stops(self):
        engine = self.make_engine({'answer': {'format': 'gbnf', 'source': 'root ::= "A"'}})
        state, session = self.start(engine, 'answer')
        state = self.consume_text(session, state, 'A')
        self.assertTrue(session.is_complete())
        self.assertFalse(session.is_stopped())
        logits = [0.0] * engine.vocab_size
        logits[TinyByteTokenizer.EOS] = 10.0
        eos = session.choose(logits)
        self.assertEqual(eos, TinyByteTokenizer.EOS)
        state = session.consume(state, eos, engine.tokenizer.token_bytes(eos), True)
        self.assertTrue(session.is_complete())
        self.assertTrue(session.is_stopped())
        self.assertEqual(state['accepted_in_scope'], 2)

    def test_premature_eos_and_non_eos_end_of_generation_are_rejected(self):
        engine = self.make_engine({'answer': {'format': 'gbnf', 'source': 'root ::= "A"'}})
        state, session = self.start(engine, 'answer')
        with self.assertRaises(IntegrityError):
            session.consume(state, TinyByteTokenizer.EOS,
                            engine.tokenizer.token_bytes(TinyByteTokenizer.EOS), True)
        with self.assertRaises(IntegrityError):
            session.consume(state, ord('A'), b'A', True)
        self.assertFalse(session.is_stopped())

    def test_nonfinite_logits_are_rejected_even_for_masked_tokens(self):
        engine = self.make_engine()
        _, session = self.start(engine)
        logits = [0.0] * engine.vocab_size
        logits[ord('Z')] = float('nan')
        with self.assertRaises(IntegrityError):
            session.choose(logits)

    def test_special_control_tokens_are_not_grammar_emissions(self):
        engine = self.make_engine({'control': {'format': 'gbnf',
                                               'source': 'root ::= "<|im_start|>"'}})
        _, session = self.start(engine, 'control')
        logits = [0.0] * engine.vocab_size
        logits[TinyByteTokenizer.IM_START] = 100.0
        selected = session.choose(logits)
        self.assertNotEqual(selected, TinyByteTokenizer.IM_START)
        with self.assertRaises(IntegrityError):
            session.consume(session.current_controller, TinyByteTokenizer.IM_START,
                            engine.tokenizer.token_bytes(TinyByteTokenizer.IM_START), False)

    def test_user_defined_literal_token_remains_legal_inside_constrained_string(self):
        engine = self.make_engine({'edit': {'format': 'gbnf',
                                             'source': 'root ::= "A<tool_call>"'}})
        state, session = self.start(engine, 'edit')
        state = session.consume(state, ord('A'), b'A', False)
        logits = [0.0] * engine.vocab_size
        logits[TinyByteTokenizer.TOOL_CALL] = 8.0
        self.assertEqual(session.choose(logits), TinyByteTokenizer.TOOL_CALL)
        state = session.consume(state, TinyByteTokenizer.TOOL_CALL,
                                engine.tokenizer.token_bytes(TinyByteTokenizer.TOOL_CALL), False)
        self.assertTrue(session.is_complete())
        self.assertEqual(state['accepted_in_scope'], 2)

    def test_json_schema_format_enforces_the_host_tool_contract(self):
        # Owner requested JSON Schema beside GBNF on 2026-10-17. This replaces
        # only the former delivery-scope assertion that all schemas reject.
        schema = {
            'type': 'object',
            'properties': {
                'tool': {'type': 'string', 'enum': ['edit']},
                'args': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                         'required': ['path'], 'additionalProperties': False},
            },
            'required': ['tool', 'args'],
            'additionalProperties': False,
        }
        engine = self.make_engine({'tool': {'format': 'json_schema', 'source': schema}})
        state, session = self.start(engine, 'tool')
        self.consume_text(session, state, '{"args":{"path":"src/main.py"},"tool":"edit"}')
        self.assertTrue(session.is_complete())
        state, session = self.start(engine, 'tool')
        with self.assertRaises(IntegrityError):
            self.consume_text(session, state, '{"args":{"path":"src/main.py"},"tool":"delete"}')

    def test_static_read_edit_test_finish_gbnf_examples_compile_and_accept_outputs(self):
        grammar_dir = Path(__file__).resolve().parents[2] / 'examples' / 'agent_grammars'
        states = {
            name: {'format': 'gbnf', 'source': (grammar_dir / (name + '.gbnf')).read_text(encoding='utf-8')}
            for name in ('read', 'edit', 'test', 'finish')
        }
        engine = self.make_engine(states)
        outputs = {
            'read': '{"tool":"read_file","path":"src/main.py"}',
            'edit': r'{"tool":"propose_patch","path":"src/main.py","patch":"line 1\nUnicode é 🚀, quote \" and backslash \\; surrogate \uD83D\uDE80"}',
            'test': '{"tool":"run_tests","suite":"unit"}',
            'finish': '{"tool":"finish","status":"verified"}',
        }
        self.assertEqual(json.loads(outputs['edit'])['patch'],
                         'line 1\nUnicode é 🚀, quote " and backslash \\; surrogate 🚀')
        for name, output in outputs.items():
            with self.subTest(agent_state=name):
                state, session = self.start(engine, name)
                state = self.consume_text(session, state, output)
                self.assertTrue(session.is_complete())
                self.assertFalse(session.is_stopped())

    def test_schema2_restore_rejects_policy_changes_and_tokens_after_eos(self):
        engine = self.make_engine({'answer': {'format': 'gbnf', 'source': 'root ::= "A"'}})
        state, session = self.start(engine, 'answer')
        state = self.consume_text(session, state, 'A')
        eos_logits = [0.0] * engine.vocab_size
        eos_logits[TinyByteTokenizer.EOS] = 10.0
        state = session.consume(state, session.choose(eos_logits),
                                engine._adapter.tokens[TinyByteTokenizer.EOS], True)
        repeated = copy.deepcopy(state)
        repeated['accepted_in_scope'] += 1
        with self.assertRaises(IntegrityError):
            engine.restore(repeated, [ord('A'), TinyByteTokenizer.EOS, TinyByteTokenizer.EOS])
        for mutate in (
            lambda value: value['policy'].__setitem__('repetition_penalty', 2.0),
            lambda value: value['policy'].__setitem__('stop_sequences', [[ord('A')]]),
            lambda value: value.__setitem__('next_draw', 1),
            lambda value: value.__setitem__('grammar_state', 'forged'),
        ):
            altered = copy.deepcopy(state)
            mutate(altered)
            with self.assertRaises(IntegrityError):
                validate_controller(altered)

    def test_tokenizer_mutation_after_compile_does_not_change_matcher_identity(self):
        tokenizer = TinyByteTokenizer()
        engine = self.make_engine({'answer': {'format': 'gbnf', 'source': 'root ::= "A"'}},
                                  tokenizer=tokenizer)
        tokenizer.tokens[ord('A')] = 'Z'
        state, session = self.start(engine, 'answer')
        logits = [0.0] * engine.vocab_size
        logits[ord('A')] = 10.0
        self.assertEqual(session.choose(logits), ord('A'))
        state = session.consume(state, ord('A'), engine._adapter.tokens[ord('A')], False)
        self.assertTrue(session.is_complete())

    def test_utf8_byte_split_restore_replays_only_generation_scope(self):
        engine = self.make_engine({'answer': {'format': 'gbnf', 'source': 'root ::= "é"'}})
        prefix = [ord('Z')]
        state, session = self.start(engine, 'answer', prefix)
        self.assertEqual(state['constraint']['scope_start'], 1)
        self.assertEqual(engine.tokenizer.encode('é', parse_special=False), [0xC3, 0xA9])
        state = session.consume(state, 0xC3, engine.tokenizer.token_bytes(0xC3), False)
        self.assertEqual(state['output_pending_hex'], 'c3')
        session = engine.restore(state, prefix + [0xC3])
        state = session.consume(state, 0xA9, engine.tokenizer.token_bytes(0xA9), False)
        self.assertTrue(session.is_complete())
        self.assertEqual(state['accepted_in_scope'], 2)
        self.assertEqual(state['output_pending_hex'], '')

    def test_sibling_matchers_restore_independently_from_same_controller(self):
        engine = self.make_engine({'fork': {'format': 'gbnf', 'source': 'root ::= "AX" | "BY"'}})
        state, _ = self.start(engine, 'fork')
        left, right = engine.restore(state, []), engine.restore(state, [])
        left_state = left.consume(state, ord('A'), b'A', False)
        right_state = right.consume(state, ord('B'), b'B', False)
        self.assertEqual(left.choose([0.0] * ord('X') + [1.0] + [0.0] * (engine.vocab_size - ord('X') - 1)), ord('X'))
        self.assertEqual(right.choose([0.0] * ord('Y') + [1.0] + [0.0] * (engine.vocab_size - ord('Y') - 1)), ord('Y'))
        self.assertEqual(state['accepted_in_scope'], 0)
        self.assertEqual(left_state['accepted_in_scope'], 1)
        self.assertEqual(right_state['accepted_in_scope'], 1)

    def test_restore_rejects_wrong_scope_length_and_descriptor_identity(self):
        engine = self.make_engine()
        state, _ = self.start(engine)
        with self.assertRaises(IntegrityError):
            engine.restore(state, [ord('A')])
        for field, value in [('engine_id', '0' * 64), ('spec_sha256', '0' * 64),
                             ('tokenizer_sha256', '0' * 64), ('scope_start', True),
                             ('agent_state', 'unknown')]:
            with self.subTest(field=field):
                tampered = copy.deepcopy(state)
                tampered['constraint'][field] = value
                with self.assertRaises((IntegrityError, InvalidOperation)):
                    engine.restore(tampered, [])
        truncated = copy.deepcopy(state)
        truncated['policy']['allowed'] = [0, 1]
        with self.assertRaises(IntegrityError):
            engine.restore(truncated, [])

    def test_schema2_structural_validation_does_not_load_llguidance(self):
        legacy = controller.start({'temperature': 0.0, 'eos_tokens': [TinyByteTokenizer.EOS]},
                                  'test', [], 258)
        state = dict(legacy, schema=2, constraint={
            'schema': 1, 'engine_id': 'a' * 64, 'llguidance_version': '1.9.1',
            'tokenizer_sha256': 'b' * 64, 'spec_sha256': 'c' * 64,
            'agent_state': 'answer', 'scope_start': 0,
        })
        original = sys.modules.pop('llguidance', None)
        try:
            real_import = builtins.__import__
            def no_llguidance(name, *args, **kwargs):
                if name == 'llguidance' or name.startswith('llguidance.'):
                    raise AssertionError('structural validation imported parser dependency')
                return real_import(name, *args, **kwargs)
            with mock.patch('builtins.__import__', side_effect=no_llguidance):
                detached = validate_controller(state)
            self.assertEqual(detached, state)
            self.assertIsNot(detached, state)
        finally:
            if original is not None:
                sys.modules['llguidance'] = original

    def test_invalid_grammar_and_unsupported_json_schema_fail_closed(self):
        with self.assertRaises((InvalidOperation, IntegrityError)):
            self.make_engine({'bad': {'format': 'gbnf', 'source': 'this is not valid grammar'}})
        with self.assertRaises((InvalidOperation, IntegrityError)):
            self.make_engine({'bad': {'format': 'json_schema',
                                      'source': {'type': 'object', 'unsupportedKeyword': True}}})

    def test_dependency_failure_and_parser_input_limit_fail_closed(self):
        from context_time_travel import grammar_first
        real_import = builtins.__import__
        def no_llguidance(name, *args, **kwargs):
            if name == 'llguidance' or name.startswith('llguidance.'):
                raise ModuleNotFoundError('blocked test dependency')
            return real_import(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=no_llguidance):
            with self.assertRaises((InvalidOperation, RuntimeError)):
                GrammarEngine(TinyByteTokenizer(), {'answer': {'format': 'gbnf', 'source': 'root ::= "A"'}},
                              [TinyByteTokenizer.EOS])
        too_large = 'A' * (grammar_first.MAX_GRAMMAR_SOURCE_BYTES + 1)
        with self.assertRaises(InvalidOperation):
            self.make_engine({'huge': {'format': 'gbnf', 'source': too_large}})


if __name__ == '__main__':
    unittest.main()
