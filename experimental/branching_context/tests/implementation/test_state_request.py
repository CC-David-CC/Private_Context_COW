"""Resident grammar request control tests; synthetic backend, real grammar parser."""
from concurrent.futures import CancelledError
import unittest

import test_resident_request_publication as fixtures
from context_time_travel import controller
from context_time_travel.store import IntegrityError, InvalidOperation


class StateRequestTests(unittest.TestCase):
    def setUp(self):
        fixtures.ResidentRequestPublicationTests.setUp(self)
        from context_time_travel.grammar_first import GrammarEngine
        from strata_tokenizer import Tokenizer
        tokenizer = Tokenizer(['A', '<eos>'], [], [1, 3], special_ids={'eos': 1})
        self.engine = GrammarEngine(tokenizer, {
            'edit': {'format': 'gbnf', 'source': 'root ::= "AA"'},
            'answer': {'format': 'gbnf', 'source': 'root ::= "A"'},
        }, eos_tokens=[1])
        self.backend.piece = tokenizer.token_bytes
        self.backend.validate_grammar_engine = lambda engine: self.assertEqual(
            [engine.tokenizer.token_bytes(token) for token in range(2)],
            [tokenizer.token_bytes(token) for token in range(2)])
        self.model.profile['execution']['selected_token_protocol'] = 'consumed-selected-receipt-v1'
        self.backend.decode_calls = 0
        def decode(lease, tokens):
            self.assertEqual(len(tokens), 1)
            self.backend.last_execution_receipt = fixtures.event(
                'normal-verifier-decode', self.backend.count, tokens)
            self.backend.count += len(tokens)
            self.backend.stats['consumed_tokens'] += len(tokens)
            self.backend.decode_calls += 1
        self.backend.consume_generated = decode

    def test_state_request_masks_before_greedy_and_captures_once(self):
        before = set(self.model.store.objects.iterdir())
        observations = []
        def progress(token):
            observations.append((token, self.backend.count))
            self.assertEqual(set(self.model.store.objects.iterdir()), before)
        result = self.model.run_state_request(self.sid, [], 8, engine=self.engine,
                                             state='edit', stream='host', on_token=progress)
        self.assertEqual(result['tokens'], [0, 0, 1])
        self.assertTrue(result['grammar_complete'])
        self.assertTrue(result['stopped'])
        self.assertEqual(observations, [(0, 3), (0, 4), (1, 5)])
        self.assertEqual((self.backend.restores, self.backend.captures), (1, 1))
        self.assertEqual(self.backend.decode_calls, 3)
        self.assertEqual(len(set(self.model.store.objects.iterdir()) - before), 1)
        decoder = self.model.controller(result['snapshot'])
        self.assertEqual((decoder['schema'], decoder['next_draw'], decoder['accepted_in_scope']), (2, 0, 3))
        self.assertEqual(self.model.history(result['snapshot']), (0, 1, 0, 0, 1))
        self.assertEqual(len(self.model.cache[result['snapshot']].execution_schedule), 4)

    def test_partial_scope_resumes_without_resetting_or_duplicating_prefix(self):
        first = self.model.run_state_request(self.sid, [], 1, engine=self.engine,
                                             state='edit', stream='original')
        self.assertFalse(first['grammar_complete'])
        second = self.model.run_state_request(first['snapshot'], [], 8, engine=self.engine,
                                              state='edit', stream='ignored', resume=True)
        self.assertEqual(second['tokens'], [0, 1])
        decoder = self.model.controller(second['snapshot'])
        self.assertEqual(decoder['stream_key'], self.model.controller(first['snapshot'])['stream_key'])
        self.assertEqual(decoder['accepted_in_scope'], 3)
        self.assertEqual(self.backend.stats['consumed_tokens'], 3)

    def test_host_switch_is_explicit_and_does_not_modify_parent(self):
        first = self.model.run_state_request(self.sid, [], 1, engine=self.engine,
                                             state='edit', stream='a')
        original = self.model.snapshot_bytes(first['snapshot'])
        with self.assertRaises((IntegrityError, InvalidOperation)):
            self.model.run_state_request(first['snapshot'], [], 2, engine=self.engine,
                                         state='answer', stream='b', resume=True)
        second = self.model.run_state_request(first['snapshot'], [], 3, engine=self.engine,
                                              state='answer', stream='b')
        self.assertEqual(second['tokens'], [0, 1])
        self.assertEqual(self.model.snapshot_bytes(first['snapshot']), original)

    def test_cancelled_progress_leaves_no_published_child_and_parent_reusable(self):
        before = set(self.model.store.objects.iterdir())
        cancel = []
        with self.assertRaises(CancelledError):
            self.model.run_state_request(self.sid, [], 8, engine=self.engine,
                state='edit', stream='cancel', on_token=lambda t: cancel.append(t),
                cancelled=lambda: bool(cancel))
        self.assertEqual(cancel, [0])
        self.assertEqual(set(self.model.store.objects.iterdir()), before)
        self.assertEqual(self.backend.captures, 0)
        recovered = self.model.run_state_request(self.sid, [], 3, engine=self.engine,
                                                 state='answer', stream='recover')
        self.assertEqual(recovered['tokens'], [0, 1])

    def test_invalid_state_and_resume_suffix_rejected_before_restore(self):
        for kwargs in [dict(state='unknown'), dict(state='edit', resume=True)]:
            with self.assertRaises((IntegrityError, InvalidOperation)):
                self.model.run_state_request(self.sid, [0], 2, engine=self.engine,
                                             stream='bad', **kwargs)
        self.assertEqual(self.backend.restores, 0)

    def test_missing_selected_token_capability_rejected_before_work(self):
        self.model.profile['execution'].pop('selected_token_protocol')
        with self.assertRaises(InvalidOperation):
            self.model.run_state_request(self.sid, [], 2, engine=self.engine, state='edit', stream='unsupported')
        self.assertEqual(self.backend.restores, 0)
        self.assertEqual(self.backend.decode_calls, 0)

    def test_generic_paths_cannot_bypass_a_saved_constraint(self):
        active = self.engine.start('edit', 'host', self.model.history(self.sid), 2)
        snapshot = self.model._controller_transition(self.sid, active)
        with self.assertRaises((IntegrityError, InvalidOperation)):
            controller.choose((1., 2.), active)
        with self.assertRaises((IntegrityError, InvalidOperation)):
            controller.consume(active, 1, b'', True)
        with self.assertRaises((IntegrityError, InvalidOperation)):
            self.model.run_greedy_request(snapshot, [], 2, stream='bypass')
        with self.assertRaises((IntegrityError, InvalidOperation)):
            self.model.run_greedy_request(snapshot, [0], 2, stream='bypass-with-input')
        self.assertEqual(self.backend.restores, 0)

    def test_forked_branch_can_enter_test_while_sibling_keeps_edit_progress(self):
        from context_time_travel.grammar_first import GrammarEngine
        engine = GrammarEngine(self.engine.tokenizer, {
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "AA"'},
            'TEST': {'format': 'gbnf', 'source': 'root ::= "A"'},
        }, eos_tokens=[1])
        partial = self.model.run_state_request(self.sid, [], 1, engine=engine,
                                               state='EDIT', stream='same-start')
        a = self.model.store.fork('BranchA', partial['snapshot'])
        b = self.model.store.fork('BranchB', partial['snapshot'])
        original_b = self.model.snapshot_bytes(b.tip)
        edit_a = self.model.run_state_request(a.tip, [], 3, engine=engine,
                                              state='EDIT', stream='ignored', resume=True)
        test_a = self.model.run_state_request(edit_a['snapshot'], [], 1, engine=engine,
                                              state='TEST', stream='host-a-test')
        self.model.store.move(a.name, a.revision, test_a['snapshot'], 'a-advance-test')
        self.assertEqual(self.model.store.head(b.name), b)
        self.assertEqual(self.model.snapshot_bytes(b.tip), original_b)
        controller_b = self.model.controller(b.tip)
        self.assertEqual((controller_b['constraint']['agent_state'], controller_b['accepted_in_scope']), ('EDIT', 1))
        controller_a = self.model.controller(test_a['snapshot'])
        self.assertEqual((controller_a['constraint']['agent_state'], controller_a['accepted_in_scope']), ('TEST', 1))
        self.assertNotEqual(controller_a['seen'], controller_b['seen'])
        edit_b = self.model.run_state_request(b.tip, [], 3, engine=engine,
                                              state='EDIT', stream='ignored', resume=True)
        self.assertEqual(edit_b['tokens'], [0, 1])
        self.assertEqual(self.model.controller(edit_b['snapshot'])['constraint']['agent_state'], 'EDIT')
        self.assertEqual(self.model.controller(test_a['snapshot']), controller_a)


if __name__ == '__main__':
    unittest.main()
