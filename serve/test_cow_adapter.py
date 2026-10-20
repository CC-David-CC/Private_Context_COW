"""Supplemental adapter tests with a real Store and a synthetic coordinator."""
from concurrent.futures import Future
import hashlib
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest

from serve.cow_adapter import CowAdapter
from serve.cow_prefix import RenderedPrefixMismatch
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.store import Conflict, IntegrityError, Store
from context_time_travel.store import _json_bytes


class SyntheticCoordinator:
    """Store-backed coordinator stub; does no model or GPU work."""
    # _intent delegates to this helper. Reuse the real implementation so the
    # fixture exercises its checksum/locking checks rather than bypassing them.
    _intent_locked = ContextCoordinator._intent_locked

    def __init__(self, directory):
        self.store = Store(directory)
        self.submissions = []
        self.operations = []
        self._condition = threading.Condition(threading.RLock())
        self._intents = self.store.root / 'pool-operations'
        self._intents.mkdir(mode=0o700)
        self._requests = {}
        self.request = SimpleNamespace(events=queue.SimpleQueue(), snapshot=Future(), durable=Future())

    def request_source_head(self, branch, revision, operation):
        return ContextCoordinator.request_source_head(self, branch, revision, operation)

    def request_source_state(self, branch, revision, operation):
        return ContextCoordinator.request_source_state(self, branch, revision, operation)

    def submit(self, branch, ids, max_steps, operation, **kwargs):
        call = (branch, tuple(ids), max_steps, operation, kwargs)
        self.submissions.append(call)
        signature = hashlib.sha256(_json_bytes(dict(branch=branch, ids=list(ids), max_steps=max_steps,
            kind='request', expected_revision=kwargs.get('expected_revision'), principal=None))).hexdigest()
        previous = self._requests.get(operation)
        if previous is not None:
            if previous[0] != signature:
                raise ValueError('Operation ID reused for different input')
            return previous[1]
        head = self.store.head(branch)
        ContextCoordinator._intent(self, operation, signature, head)
        self._requests[operation] = (signature, self.request)
        return self.request

    def create(self, branch, ids, operation):
        self.operations.append(('create', branch, tuple(ids), operation))
        snapshot = self.store.create_root('model', ids, {'status': 'idle'})
        self.store.fork(branch, snapshot)
        return self.request

    def fork(self, name, snapshot):
        self.operations.append(('fork', name, snapshot))
        return self.store.fork(name, snapshot)

    def rewind(self, branch, revision, snapshot, operation):
        self.operations.append(('rewind', branch, revision, snapshot, operation))
        return self.store.move(branch, revision, snapshot, operation)


class CowServingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.coordinator = SyntheticCoordinator(self.tmp.name)
        self.adapter = CowAdapter(self.coordinator)
        self.root = self.coordinator.store.create_root('model', (10, 11), {'status': 'idle'})
        self.initial = self.coordinator.store.fork('main', self.root)

    def test_branch_catalog_uses_canonical_store_and_never_submits_work(self):
        second = self.coordinator.store.fork('other', self.root)
        unrelated = self.coordinator.store.create_root('model', [99], {'status': 'idle'})
        third = self.coordinator.store.fork('separate', unrelated)
        self.assertEqual(self.adapter.branches(), (self.initial, second, third))
        self.assertEqual(self.adapter.branches(self.root), (self.initial, second))
        self.assertEqual(self.coordinator.submissions, [])
        self.assertEqual(self.coordinator.operations, [])

    def test_generate_submits_exact_new_tokens_and_returns_coordinator_request(self):
        cancelled = lambda: False
        request = self.adapter.generate('main', (10, 11, 12, 13), 0, 'op-1', 32,
                                        settings={'temperature': 0}, cancelled=cancelled)
        self.assertIs(request, self.coordinator.request)
        self.assertEqual(self.coordinator.submissions, [
            ('main', (12, 13), 32, 'op-1',
             {'expected_revision': 0, 'cancelled': cancelled})])
        self.assertFalse(self.coordinator.submissions[0][4]['cancelled']())

    def test_generate_rejects_rendered_prefix_mismatch_before_submit(self):
        with self.assertRaises(RenderedPrefixMismatch):
            self.adapter.generate('main', (10, 99, 12), 0, 'op-mismatch', 8)
        self.assertEqual(self.coordinator.submissions, [])
        self.assertEqual(self.coordinator.store.head('main'), self.initial)

    def test_completed_retry_uses_original_source_and_returns_same_request(self):
        first = self.adapter.generate('main', (10, 11, 12, 13), 0, 'retry-op', 32)
        advanced = self.coordinator.store.append(self.root, (12, 13), {'status': 'idle'})
        self.coordinator.store.move('main', 0, advanced, 'retry-op')

        retried = self.adapter.generate('main', (10, 11, 12, 13), 0, 'retry-op', 32)
        self.assertIs(retried, first)
        self.assertEqual(self.coordinator.submissions[-1][1], (12, 13))
        self.assertEqual(self.coordinator.store.head('main').tip, advanced)

    def test_completed_retry_rejects_changed_input_without_canonical_mutation(self):
        self.adapter.generate('main', (10, 11, 12, 13), 0, 'retry-op', 32)
        advanced = self.coordinator.store.append(self.root, (12, 13), {'status': 'idle'})
        moved = self.coordinator.store.move('main', 0, advanced, 'retry-op')

        with self.assertRaisesRegex(ValueError, 'Operation ID reused'):
            self.adapter.generate('main', (10, 11, 99), 0, 'retry-op', 32)
        self.assertEqual(self.coordinator.store.head('main'), moved)
        self.assertEqual(len(self.coordinator.submissions), 2)

    def test_generate_rejects_unsupported_sampling_before_submit(self):
        for settings in ({'temperature': -0.2}, {'temperature': 0, 'top_p': 0},
                         {'temperature': 0, 'tools': []}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.adapter.generate('main', (10, 11, 12), 0, 'op-unsupported', 8,
                                          settings=settings)
        self.assertEqual(self.coordinator.submissions, [])

    def test_stale_revision_is_rejected_without_submission(self):
        alternate = self.coordinator.store.append(self.root, (99,), {'status': 'idle'})
        moved = self.coordinator.store.move('main', 0, alternate, 'external-move')
        with self.assertRaises(Conflict):
            self.adapter.generate('main', (10, 11, 99, 3), 0, 'op-stale', 8)
        self.assertEqual(self.coordinator.submissions, [])
        self.assertEqual(self.coordinator.store.head('main'), moved)

    def _bind_intent(self, operation, head):
        ContextCoordinator._intent(self.coordinator, operation, 'bound-signature', head)

    def test_source_head_uses_uncommitted_intent_after_branch_moves(self):
        self._bind_intent('pending-op', self.initial)
        alternate = self.coordinator.store.append(self.root, (99,), {'status': 'idle'})
        self.coordinator.store.move('main', 0, alternate, 'external-move')

        self.assertEqual(self.coordinator.request_source_head('main', 0, 'pending-op'), self.initial)

    def test_source_head_rejects_intent_bound_to_different_branch_or_revision(self):
        self._bind_intent('bound-op', self.initial)
        with self.assertRaises(Conflict):
            self.coordinator.request_source_head('other', 0, 'bound-op')
        with self.assertRaises(Conflict):
            self.coordinator.request_source_head('main', 1, 'bound-op')

    def test_source_head_rejects_tampered_intent_checksum(self):
        self._bind_intent('tampered-op', self.initial)
        path = self.coordinator._intents / 'tampered-op.json'
        envelope = path.read_text(encoding='utf-8')
        path.write_text(envelope.replace('bound-signature', 'altered-signature'), encoding='utf-8')

        with self.assertRaisesRegex(IntegrityError, 'Operation intent is corrupt'):
            self.coordinator.request_source_head('main', 0, 'tampered-op')

    def test_head_and_history_read_only_from_canonical_store(self):
        self.assertEqual(self.adapter.head('main'), self.initial)
        self.assertEqual(self.adapter.history('main'), (10, 11))

    def test_source_state_reads_history_and_controller_from_same_immutable_source(self):
        head, history, controller = self.adapter.source_state('main', 0, 'read-source')
        self.assertEqual(head, self.initial)
        self.assertEqual(history, (10, 11))
        self.assertEqual(controller, {'status': 'idle'})

    def test_create_fork_and_rewind_dispatch_to_coordinator_store_operations(self):
        created = self.adapter.create('new-branch', (20, 21), 'create-op')
        self.assertIs(created, self.coordinator.request)
        self.assertEqual(self.coordinator.store.lineage(self.coordinator.store.head('new-branch').tip)[1], [20, 21])
        self.assertEqual(self.coordinator.operations[-1], ('create', 'new-branch', (20, 21), 'create-op'))

        forked = self.adapter.fork('sibling', self.root)
        self.assertEqual(forked.tip, self.root)
        self.assertEqual(self.coordinator.operations[-1], ('fork', 'sibling', self.root))

        rewound = self.adapter.rewind('sibling', 0, self.root, 'rewind-op')
        self.assertEqual(rewound.tip, self.root)
        self.assertEqual(self.coordinator.operations[-1], ('rewind', 'sibling', 0, self.root, 'rewind-op'))


if __name__ == '__main__':
    unittest.main()
