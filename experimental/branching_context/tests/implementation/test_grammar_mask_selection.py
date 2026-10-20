"""CPU protocol/controller checks; native execution is qualified separately."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from context_time_travel import strata_backend as sb
from context_time_travel.store import IntegrityError, InvalidOperation
import test_grammar_first as grammar_fixtures
import test_state_request as request_fixtures

PROTOCOL = 'finite-mask-argmax-v1'
TinyByteTokenizer = grammar_fixtures.TinyByteTokenizer


class GrammarMaskTests(unittest.TestCase):
    def session(self):
        factory = grammar_fixtures.GrammarFirstTests()
        engine = factory.make_engine()
        state = engine.start('edit', 'mask', (), engine.vocab_size)
        return engine, engine.restore(state, ())

    def test_masked_selection_matches_host_argmax_at_each_consumed_boundary(self):
        engine, session = self.session()
        for token in [ord('A'), ord('B'), TinyByteTokenizer.EOS]:
            logits = [-2.] * engine.vocab_size
            logits[token] = 5.
            logits[ord('Z')] = 1000.  # illegal maximum
            expected = session.choose(logits)
            def select(mask):
                allowed = [t for t in range(len(logits)) if mask[t >> 3] & (1 << (t & 7))]
                return max(allowed, key=lambda t: (logits[t], -t))
            actual = session._choose_masked(select)
            self.assertEqual(actual, expected)
            self.assertEqual(actual, token)
            session._consume_selected(actual, engine.tokenizer.token_bytes(actual), actual in engine.eos_tokens)

    def test_mask_removes_control_premature_eos_and_padding_but_retains_literal_tags(self):
        engine, session = self.session()
        # Matchers may round their bitset extent up. Extra vocabulary padding
        # must never become a native token or widen the transmitted mask.
        session._matcher = mock.Mock(compute_bitmask=lambda: b'\xff' * (((engine.vocab_size + 7) // 8) + 5),
                                     is_error=lambda: False, is_accepting=lambda: False)
        captured = []
        session._choose_masked(lambda mask: captured.append(mask) or ord('A'))
        mask = captured[0]
        self.assertEqual(len(mask), (engine.vocab_size + 7) // 8)
        for token in [TinyByteTokenizer.EOS, TinyByteTokenizer.IM_START]:
            self.assertFalse(mask[token >> 3] & (1 << (token & 7)))
        self.assertTrue(mask[TinyByteTokenizer.TOOL_CALL >> 3] & (1 << (TinyByteTokenizer.TOOL_CALL & 7)))
        self.assertEqual(mask[-1] >> (engine.vocab_size % 8), 0)

    def test_illegal_or_noninteger_native_result_cannot_advance_parser(self):
        engine, session = self.session()
        before = session.current_controller
        for token in [True, -1, engine.vocab_size, ord('Z'), 1.0, TinyByteTokenizer.EOS]:
            with self.assertRaises(IntegrityError):
                session._choose_masked(lambda mask: token)
            self.assertEqual(session.current_controller, before)
        self.assertEqual(session._choose_masked(lambda mask: ord('A')), ord('A'))

    def test_bad_or_empty_parser_mask_fails_before_selector(self):
        engine, session = self.session()
        selector = mock.Mock(return_value=ord('A'))
        for mask in [b'', 'bad', bytes((engine.vocab_size + 7) // 8)]:
            session._matcher = mock.Mock(compute_bitmask=lambda: mask,
                                         is_error=lambda: False, is_accepting=lambda: False)
            with self.assertRaises(IntegrityError):
                session._choose_masked(selector)
        selector.assert_not_called()


class MaskSelectionBackendTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.backend = object.__new__(sb.StrataBackend)
        self.backend.profile = {'execution': {'grammar_selection_protocol': PROTOCOL}}
        self.backend.vocab_size = 9
        self.backend._directory = Path(temp.name)
        self.backend._sequence = 0
        self.backend.stats = {}
        self.backend.process = SimpleNamespace(pid=12345)
        self.lease = sb._Lease(self.backend, bound=True, consumed=4,
                              state='ACTIVE', _owner_thread=threading.get_ident())
        self.backend._active = self.lease
        self.response = dict(status='ok', selection_protocol=PROTOCOL,
                             consumed=4, token=8, logits_count=9, mask_bytes=2)
        self.requests = []
        def request(command):
            fields = command.split()
            self.assertEqual(fields[:2], ['select_mask', '4'])
            self.requests.append(Path(fields[2]).read_bytes())
            return dict(self.response)
        self.backend._request = request

    def test_success_verifies_receipt_without_forward_or_lease_mutation_and_removes_file(self):
        self.assertEqual(self.backend.choose_legal(self.lease, b'\x00\x01'), 8)
        self.assertEqual(self.requests, [b'\x00\x01'])
        self.assertEqual(self.lease.consumed, 4)
        self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_wrong_receipt_rejected_and_mask_file_removed(self):
        for name, value in [('selection_protocol', 'wrong'), ('consumed', 5),
                            ('token', True), ('token', 0), ('token', 9),
                            ('logits_count', 8), ('mask_bytes', 3)]:
            original = dict(self.response)
            self.response[name] = value
            with self.assertRaises(IntegrityError):
                self.backend.choose_legal(self.lease, b'\x00\x01')
            self.assertEqual(list(self.backend._directory.iterdir()), [])
            self.assertEqual(self.lease.consumed, 4)
            self.response = original

    def test_invalid_mask_or_missing_capability_rejected_before_native_call(self):
        for mask in [b'', b'\x00', b'\x00\x02', b'\x00\x00', 'bad']:
            with self.assertRaises(InvalidOperation):
                self.backend.choose_legal(self.lease, mask)
        self.backend.profile['execution'].clear()
        with self.assertRaises(InvalidOperation):
            self.backend.choose_legal(self.lease, b'\x00\x01')
        self.assertEqual(self.requests, [])

    def test_foreign_thread_rejected_before_file_or_command(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(InvalidOperation):
                pool.submit(self.backend.choose_legal, self.lease, b'\x00\x01').result()
        self.assertEqual(self.requests, [])
        self.assertEqual(list(self.backend._directory.iterdir()), [])

    def test_ready_must_match_pinned_capability(self):
        for ready in ({}, {'grammar_selection_protocol': 'wrong'}):
            with self.assertRaises(IntegrityError):
                sb._validate_grammar_selection_capability(self.backend.profile, ready)
        sb._validate_grammar_selection_capability(self.backend.profile, {'grammar_selection_protocol': PROTOCOL})
        sb._validate_grammar_selection_capability({'execution': {}}, {})


class MaskSelectionRequestTests(unittest.TestCase):
    def setUp(self):
        request_fixtures.StateRequestTests.setUp(self)
        self.model.profile['execution']['grammar_selection_protocol'] = PROTOCOL
        self.masks = []
        def select(lease, mask):
            self.masks.append(mask)
            return next(t for t in range(2) if mask[t >> 3] & (1 << (t & 7)))
        self.backend.choose_legal = select

    def test_native_selection_avoids_full_frontier_read_per_output(self):
        original = self.backend.logits
        with mock.patch.object(self.backend, 'logits', wraps=original) as reads:
            result = self.model.run_state_request(self.sid, (), 16, engine=self.engine,
                                                 state='edit', stream='mask')
        self.assertEqual(result['tokens'], [0, 0, 1])
        self.assertEqual(len(self.masks), 3)
        self.assertLessEqual(reads.call_count, 2)
        self.assertEqual(self.backend.captures, 1)
        self.assertEqual(self.model.controller(result['snapshot'])['accepted_in_scope'], 3)

    def test_unsupported_selection_protocol_fails_before_materialization(self):
        self.model.profile['execution']['grammar_selection_protocol'] = 'unknown'
        with self.assertRaises(InvalidOperation):
            self.model.run_state_request(self.sid, (), 16, engine=self.engine,
                                         state='edit', stream='mask')
        self.assertEqual(self.backend.restores, 0)
        self.assertEqual(self.masks, [])

    def test_invalid_native_choice_does_not_publish_and_parent_can_be_reused(self):
        before = set(self.model.store.objects.iterdir())
        original = self.backend.choose_legal
        self.backend.choose_legal = lambda lease, mask: 1  # premature EOS
        with self.assertRaises(IntegrityError):
            self.model.run_state_request(self.sid, (), 16, engine=self.engine,
                                         state='edit', stream='bad')
        self.assertEqual(set(self.model.store.objects.iterdir()), before)
        self.assertEqual(self.backend.decode_calls, 0)
        self.backend.choose_legal = original
        result = self.model.run_state_request(self.sid, (), 16, engine=self.engine,
                                             state='edit', stream='good')
        self.assertEqual(result['tokens'], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
