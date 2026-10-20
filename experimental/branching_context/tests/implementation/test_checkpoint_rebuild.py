"""Explicit cache-miss reconstruction controls; synthetic CPU backend only.

These tests do not certify numerical equivalence to an evicted computation.
Reconstruction has a new measured batch arrangement; strict restore stays strict.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from context_time_travel import controller
from context_time_travel.execution_receipt import tokens_digest, validate_operator_schedule
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation, Store, _json_bytes


class RebuildBackend:
    vocab_size = 32

    def __init__(self):
        self.stats = dict(forward_tokens=0, consumed_tokens=0, prefill_tokens=0)
        self.count = 0
        self.tokens = ()
        self.closed = False
        self.bad_work = False
        self.bad_receipt = False
        self.nonfinite = False
        self.partial_failure = False
        self.lock = threading.Lock()
        self.consume_entered = threading.Event()
        self.consume_release = threading.Event()
        self.consume_release.set()
        self.second_lease_attempted = threading.Event()
        self.lease_attempts = 0

    @contextmanager
    def lease(self):
        self.lease_attempts += 1
        if self.lease_attempts == 2:
            self.second_lease_attempted.set()
        with self.lock:
            self.count = 0
            self.tokens = ()
            yield self

    def work_counters(self):
        return dict(self.stats)

    def consume(self, lease, ids):
        self.consume_entered.set()
        if not self.consume_release.wait(5):
            raise TimeoutError('CPU fixture release was not signalled')
        self.tokens = tuple(ids)
        self.count = len(ids)
        self.stats['forward_tokens'] += 0 if self.bad_work else len(ids)
        self.stats['consumed_tokens'] = self.stats['forward_tokens']
        self.stats['prefill_tokens'] += len(ids)
        self.last_execution_receipt = dict(operator='batched-prefill', position=0,
            count=len(ids), tokens_sha256=tokens_digest(ids),
            chunks=[[start, min(4, len(ids)-start)] for start in range(0, len(ids), 4)])
        if self.bad_receipt:
            self.last_execution_receipt['chunks'][0][0] = 1
        if self.partial_failure:
            raise OSError('synthetic backend failed after consuming input')

    def export(self, lease):
        return json.dumps(self.tokens).encode('ascii')

    def logits(self, lease):
        if self.nonfinite:
            return (float('nan'),) * self.vocab_size
        return tuple(float(sum(self.tokens)+index) for index in range(self.vocab_size))

    def position(self, lease):
        return self.count-1

    def close(self):
        self.closed = True


class CheckpointRebuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.backend = RebuildBackend()
        self.model = BranchingModel.__new__(BranchingModel)
        self.model.backend = self.backend
        self.model.profile = dict(model_contract_id='model', execution_profile_id='profile',
            execution=dict(resume_contract='identical-native-operators-v1',
                generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1', prefill_chunk=4,
                capacity=64, eos_tokens=[31]))
        self.model.store = Store(temporary.name)
        self.model.cache = {}
        self.model.checkpoint_directory = Path(temporary.name)/'checkpoints'
        self.model.checkpoint_directory.mkdir()
        self.model._allocation_fault = lambda *args: None
        self.root = self.model.store.create_root('model', (1, 2, 3, 4, 5), {'status': 'idle'})
        state = controller.start(dict(temperature=0, eos_tokens=[31]),
                                 'saved-stream', (1, 2, 3, 4, 5), self.backend.vocab_size)
        state = controller.consume_many(state, ((6, b'six', False), (7, b'seven', False)))
        self.snapshot = self.model.store.append(self.root, (6, 7), state)
        self.model.store.fork('main', self.snapshot)

    def canonical_bytes(self):
        root = self.model.store.root
        paths = list((root/'objects').glob('*.json')) + list((root/'references').glob('*.json'))
        return {str(path.relative_to(root)): path.read_bytes() for path in paths}

    def test_missing_materialization_rebuilds_exact_inputs_and_keeps_controller(self):
        before = self.canonical_bytes()
        state = self.model.controller(self.snapshot)
        receipt = self.model.rebuild_materialization(self.snapshot)
        self.assertEqual(receipt['source_kind'], 'recomputed-history')
        self.assertEqual(receipt['snapshot'], self.snapshot)
        self.assertEqual(receipt['forward_tokens'], 7)
        self.assertEqual(receipt['consumed'], 7)
        self.assertEqual(receipt['positions'], 'absolute-zero-based')
        self.assertIs(receipt['durable'], False)
        self.assertIs(receipt['prior_computation_preserved'], False)
        self.assertEqual(self.backend.tokens, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(self.model.controller(self.snapshot), state)
        self.assertEqual(self.canonical_bytes(), before)
        saved = self.model.cache[self.snapshot]
        self.assertEqual(saved.consumed, 7)
        self.assertEqual(saved.input_identity, self.model._input_identity(self.backend.tokens))
        self.assertEqual(receipt['schedule_sha256'],
            validate_operator_schedule(saved.execution_schedule, self.backend.tokens, 4))
        self.assertEqual(receipt['controller_sha256'], hashlib.sha256(_json_bytes(state)).hexdigest())
        self.assertEqual(list(self.model.checkpoint_directory.iterdir()), [])

    def test_idle_root_can_rebuild_without_starting_generation(self):
        receipt = self.model.rebuild_materialization(self.root)
        self.assertEqual(receipt['forward_tokens'], 5)
        self.assertEqual(self.model.controller(self.root), {'status': 'idle'})

    def test_existing_checkpoint_is_not_bypassed_even_if_its_bytes_are_corrupt(self):
        (self.model.checkpoint_directory/(self.snapshot+'.json')).write_bytes(b'corrupt')
        with self.assertRaises(InvalidOperation):
            self.model.rebuild_materialization(self.snapshot)
        self.assertEqual(self.backend.stats['forward_tokens'], 0)
        self.assertNotIn(self.snapshot, self.model.cache)

    def test_existing_hot_materialization_is_not_recomputed(self):
        self.model.cache[self.snapshot] = object()
        with self.assertRaises(InvalidOperation):
            self.model.rebuild_materialization(self.snapshot)
        self.assertEqual(self.backend.stats['forward_tokens'], 0)

    def test_dangling_checkpoint_entry_cannot_be_bypassed(self):
        path = self.model.checkpoint_directory/(self.snapshot+'.json')
        path.symlink_to(self.model.checkpoint_directory/'absent-manifest')
        with self.assertRaises(InvalidOperation):
            self.model.rebuild_materialization(self.snapshot)
        self.assertEqual(self.backend.stats['forward_tokens'], 0)
        self.assertNotIn(self.snapshot, self.model.cache)

    def test_unknown_or_boolean_policy_fields_fail_before_work(self):
        state = self.model.controller(self.snapshot)
        for field, value in (('future_sampler', 1), ('temperature', False),
                             ('repetition_penalty', True), ('rejection_threshold', True),
                             ('allowed', [0, True]+list(range(2, 32))), ('eos_tokens', [True])):
            with self.subTest(field=field):
                changed = dict(state, policy=dict(state['policy'], **{field:value}))
                snapshot = self.model.store.append(self.snapshot, (), changed)
                with self.assertRaises(InvalidOperation):
                    self.model.rebuild_materialization(snapshot)
                self.assertEqual(self.backend.stats['forward_tokens'], 0)

    def test_concurrent_rebuild_rechecks_after_acquiring_native_lease(self):
        successes, errors = [], []
        self.backend.consume_release.clear()
        def rebuild():
            try:
                successes.append(self.model.rebuild_materialization(self.snapshot))
            except BaseException as error:
                errors.append(error)
        first = threading.Thread(target=rebuild)
        second = threading.Thread(target=rebuild)
        first.start()
        try:
            self.assertTrue(self.backend.consume_entered.wait(2))
            second.start()
            self.assertTrue(self.backend.second_lease_attempted.wait(2))
        finally:
            self.backend.consume_release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InvalidOperation)
        self.assertEqual(self.backend.stats['forward_tokens'], 7)
        self.assertIn(self.snapshot, self.model.cache)
        self.assertFalse(self.backend.closed, 'rejected second caller must not destroy the first image')

    def test_strict_restore_still_rejects_a_missing_checkpoint_without_work(self):
        with self.assertRaises(IntegrityError):
            self.model.restore_checkpoint(self.snapshot)
        self.assertEqual(self.backend.stats['forward_tokens'], 0)
        self.assertNotIn(self.snapshot, self.model.cache)

    def test_wrong_model_and_non_greedy_controller_fail_before_work(self):
        wrong = self.model.store.create_root('another-model', (1, 2), {'status': 'idle'})
        state = controller.start(dict(temperature=1), 'other-stream', (1, 2, 3, 4, 5), 32)
        sampled = self.model.store.append(self.root, (), state)
        for snapshot in (wrong, sampled):
            with self.subTest(snapshot=snapshot), self.assertRaises((IntegrityError, InvalidOperation)):
                self.model.rebuild_materialization(snapshot)
        self.assertEqual(self.backend.stats['forward_tokens'], 0)

    def test_incomplete_work_receipts_or_frontier_fail_closed(self):
        for attribute in ('bad_work', 'bad_receipt', 'nonfinite', 'partial_failure'):
            with self.subTest(attribute=attribute):
                self.backend = RebuildBackend()
                setattr(self.backend, attribute, True)
                self.model.backend = self.backend
                self.model.cache.clear()
                before = self.canonical_bytes()
                with self.assertRaises((IntegrityError, OSError)):
                    self.model.rebuild_materialization(self.snapshot)
                self.assertNotIn(self.snapshot, self.model.cache)
                self.assertTrue(self.backend.closed)
                self.assertEqual(self.canonical_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
