"""CPU contract for idempotent G4 evaluation and winner adoption operations."""
from __future__ import annotations

import hashlib
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import tempfile
from unittest.mock import patch

from context_time_travel.store import Conflict, IntegrityError, InvalidOperation, NotFound, Store
from context_time_travel.coordinator import ContextCoordinator


def _record(snapshot, root, score):
    return dict(
        kind='evaluation', schema=1, snapshot_id=snapshot, task_root=root,
        evaluator=dict(name='receipt-fixture', version='1', source_sha256=hashlib.sha256(b'eval').hexdigest()),
        fixtures={'expected': hashlib.sha256(b'PASS').hexdigest()},
        metric=dict(name='score', order='maximize'), seed=3,
        environment={'runtime': 'CPU receipt fixture'}, effect_policy={'mode': 'none'},
        outcomes=dict(mandatory={'valid': 'PASS'}, score=score, unresolved=[]),
        resources=dict(wall_seconds=0.0, cpu_seconds=0.0, peak_rss_bytes=0), artifacts={})


def _spec(root, source='eval'):
    return dict(task_root=root,
        evaluator=dict(name='receipt-fixture',version='1',source=source),
        metric=dict(name='score',order='maximize'), seed=3,
        mandatory=['valid'], timeout_seconds=5)


class EvaluationOperationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.root = self.store.create_root('contract', [1], {'status': 'idle'})
        self.short = self.store.append(self.root, [2], {'status': 'idle'})
        self.long = self.store.append(self.root, [3, 4], {'status': 'idle'})
        self.head = self.store.fork('main', self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_once(self, op, report, specification, fixtures, compute):
        method = getattr(self.store, 'evaluate_once', None)
        self.assertTrue(callable(method), 'Store must expose a durable evaluation operation receipt')
        return method(op, report['snapshot_id'], specification, fixtures, compute)

    def test_evaluation_operation_retry_returns_report_without_rerunning(self):
        report = _record(self.short, self.root, 2)
        calls = []

        def compute():
            calls.append('run')
            return report

        first = self.run_once('eval-once', report, _spec(self.root), {'expected':b'PASS'}, compute)
        second = self.run_once('eval-once', report, _spec(self.root), {'expected':b'PASS'}, compute)
        self.assertEqual(first, second)
        self.assertEqual(first['record'], self.store.get_evaluation(first['evaluation_id']))
        self.assertEqual(calls, ['run'])
        with self.assertRaisesRegex(InvalidOperation, 'different input'):
            self.run_once('eval-once', report, _spec(self.root,'other'), {'expected':b'PASS'}, compute)
        with self.assertRaisesRegex(InvalidOperation, 'different input'):
            self.run_once('eval-once', report, _spec(self.root), {'expected':b'FAIL'}, compute)
        self.assertEqual(calls, ['run'])

    def test_committed_operation_survives_lost_ack_and_different_endpoint_collision(self):
        report = _record(self.short, self.root, 2)
        calls = []

        def fault(step):
            if step == 'after_evaluation_operation_commit':
                raise OSError('simulated lost acknowledgement')

        store = Store(self.temp.name, fault_hook=fault)
        def compute():
            calls.append('run')
            return report

        with self.assertRaisesRegex(OSError, 'lost acknowledgement'):
            store.evaluate_once('shared-op', self.short, _spec(self.root), {'expected':b'PASS'}, compute)
        # Reopen to prove the receipt, not process memory, is authoritative.
        reopened = Store(self.temp.name)
        replay = reopened.evaluate_once('shared-op', self.short, _spec(self.root), {'expected':b'PASS'},
                                        lambda: calls.append('unexpected rerun'))
        self.assertEqual(replay['record']['snapshot_id'], self.short)
        self.assertEqual(calls, ['run'])
        with self.assertRaises(InvalidOperation):
            reopened.move('main', self.head.revision, self.root, 'shared-op')
        self.assertEqual(reopened.head('main'), self.head)

    def test_concurrent_same_id_calls_execute_one_evaluator(self):
        report = _record(self.short, self.root, 2)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def compute():
            calls.append('run')
            entered.set()
            self.assertTrue(release.wait(5))
            return report

        method = getattr(self.store, 'evaluate_once', None)
        self.assertTrue(callable(method), 'Store must serialize concurrent evaluation operation IDs')
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(method, 'same-op', self.short, _spec(self.root), {'expected':b'PASS'}, compute)
            self.assertTrue(entered.wait(5))
            second = executor.submit(method, 'same-op', self.short, _spec(self.root), {'expected':b'PASS'},
                                      lambda: calls.append('unexpected rerun') or report)
            time.sleep(0.05)
            release.set()
            self.assertEqual(first.result(timeout=10), second.result(timeout=10))
        self.assertEqual(calls, ['run'])

    def test_adoption_binds_full_selection_and_recovers_same_move_receipt(self):
        short_id = self.store.put_evaluation(_record(self.short, self.root, 2))
        long_id = self.store.put_evaluation(_record(self.long, self.root, 3))
        method = getattr(self.store, 'adopt_evaluated_once', None)
        self.assertTrue(callable(method), 'Adoption receipt must bind selection input before Store.move')
        args = ('adopt-once', 'main', self.head.revision, [short_id, long_id], ['valid'])
        first = method(*args)
        second = Store(self.temp.name).adopt_evaluated_once(*args)
        self.assertEqual(first, second)
        self.assertEqual(first['evaluation_id'], long_id)
        self.assertEqual(first['branch'].tip, self.long)
        # Same winner, different evaluation set: the request digest still differs.
        with self.assertRaisesRegex(InvalidOperation, 'different input'):
            method('adopt-once', 'main', self.head.revision, [long_id], ['valid'])
        self.assertEqual(self.store.head('main').tip, self.long)

    def test_coordinator_adoption_recovers_move_committed_before_receipt_commit(self):
        short_id = self.store.put_evaluation(_record(self.short, self.root, 2))
        long_id = self.store.put_evaluation(_record(self.long, self.root, 3))
        args = ('move-before-ack', 'main', self.head.revision, [short_id, long_id], ['valid'])
        def fault(step):
            if step == 'after_evaluation_adoption_move':
                raise OSError('simulated crash after move')
        first_store = Store(self.temp.name, fault_hook=fault)
        first = object.__new__(ContextCoordinator)
        first._condition = threading.Condition()
        first._branches = set()
        first.store = first_store
        with self.assertRaisesRegex(OSError, 'after move'):
            first.adopt_evaluated_once(*args)
        moved = first_store.head('main')
        self.assertEqual(moved.revision, self.head.revision + 1)
        self.assertEqual(moved.tip, self.long)
        self.assertEqual(first_store.evaluation_operation_status(args[0])['state'], 'pending')

        reopened = Store(self.temp.name)
        retry = object.__new__(ContextCoordinator)
        retry._condition = threading.Condition()
        retry._branches = set()
        retry.store = reopened
        receipt = retry.adopt_evaluated_once(*args)
        self.assertEqual(receipt['branch'], moved)
        self.assertEqual(reopened.head('main'), moved)
        self.assertEqual(reopened.evaluation_operation_status(args[0])['state'], 'committed')

    def test_coordinator_intent_and_evaluation_receipts_cannot_share_id(self):
        report = _record(self.short, self.root, 2)
        self.store.evaluate_once('cross-endpoint', self.short, _spec(self.root), {'expected':b'PASS'},
                                 lambda: report)
        coordinator = object.__new__(ContextCoordinator)
        coordinator.store = self.store
        coordinator._intents = self.store.root / 'pool-operations'
        coordinator._intents.mkdir(mode=0o700, exist_ok=True)
        with self.assertRaisesRegex(InvalidOperation, 'evaluation operation'):
            coordinator._intent('cross-endpoint', 'signature', self.head)

    def test_evaluation_cannot_claim_existing_pool_intent_id(self):
        directory = self.store.root / 'pool-operations'
        directory.mkdir(mode=0o700, exist_ok=True)
        (directory / 'pool-owned.json').write_text('{}', encoding='ascii')
        report = _record(self.short, self.root, 2)
        calls = []
        with self.assertRaisesRegex(InvalidOperation, 'coordinator operation'):
            self.store.evaluate_once('pool-owned', self.short, _spec(self.root), {'expected':b'PASS'},
                                     lambda: calls.append('run') or report)
        self.assertEqual(calls, [])

    def test_namespace_operation_id_keeps_legacy_length_and_unicode_contract(self):
        long_id = 'n' * 128
        unicode_id = 'café/会話'
        long_namespace = self.store.create_namespace('long ID', long_id)
        unicode_namespace = self.store.create_namespace('unicode ID', unicode_id)
        self.assertEqual(long_namespace.operation_id, long_id)
        self.assertEqual(unicode_namespace.operation_id, unicode_id)
        self.assertEqual(self.store.create_namespace('long ID', long_id), long_namespace)
        self.assertEqual(self.store.create_namespace('unicode ID', unicode_id), unicode_namespace)

    def test_invalid_specification_is_rejected_before_operation_claim_or_callback(self):
        report = _record(self.short, self.root, 2)
        malformed = _spec(self.root)
        malformed['timeout_seconds'] = '5'
        calls = []
        with self.assertRaises(InvalidOperation):
            self.run_once('invalid-spec', report, malformed, {'expected':b'PASS'},
                          lambda: calls.append('run') or report)
        self.assertEqual(calls, [])
        with self.assertRaises(NotFound):
            self.store.evaluation_operation_status('invalid-spec')
        self.assertFalse((self.store.evaluation_operations/'invalid-spec.json').exists())

    def test_missing_model_contract_is_integrity_error_before_evaluation_claim(self):
        report = _record(self.short, self.root, 2)
        records = {
            self.short: {'parent': self.root, 'model_contract_id': None},
            self.root: {'parent': None, 'model_contract_id': 'model-contract'},
        }
        calls = []
        with patch.object(self.store, 'get_snapshot', side_effect=records.__getitem__):
            with self.assertRaisesRegex(IntegrityError, 'no model contract'):
                self.store.evaluate_once('missing-model-contract', self.short, _spec(self.root),
                                         {'expected': b'PASS'}, lambda: calls.append('run') or report)
        self.assertEqual(calls, [])
        self.assertFalse((self.store.evaluation_operations/'missing-model-contract.json').exists())

    def test_adapter_refuses_adoption_without_coordinator_serialization_hook(self):
        from serve.cow_adapter import CowAdapter
        class BareCoordinator:
            def __init__(self, store): self.store = store
        adapter = CowAdapter(BareCoordinator(self.store))
        with self.assertRaisesRegex(InvalidOperation, 'serialized evaluation adoption'):
            adapter.adopt_evaluations('missing-hook', 'main', self.head.revision,
                                      [], ['valid'])
        self.assertEqual(self.store.head('main'), self.head)

    def test_evaluation_holds_checkpoint_read_lease_until_operation_receipt(self):
        report = _record(self.short, self.root, 2)
        entered, release = threading.Event(), threading.Event()
        def fault(step):
            if step == 'before_evaluation_operation_commit':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test did not release evaluation receipt gate')
        store = Store(self.temp.name, fault_hook=fault)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(store.evaluate_once, 'lease-eval', self.short,
                _spec(self.root), {'expected':b'PASS'}, lambda: report)
            self.assertTrue(entered.wait(5), 'evaluation did not reach the pre-receipt boundary')
            self.assertIn(self.short, store.checkpoint_roots(),
                          'numerical candidate must stay leased through durable operation receipt')
            release.set()
            result = future.result(timeout=10)
        self.assertEqual(result['record']['snapshot_id'], self.short)
        self.assertNotIn(self.short, store.checkpoint_roots())

    def test_adoption_holds_target_lease_through_store_move_and_operation_receipt(self):
        short_id = self.store.put_evaluation(_record(self.short, self.root, 2))
        long_id = self.store.put_evaluation(_record(self.long, self.root, 3))
        entered, release = threading.Event(), threading.Event()
        def fault(step):
            if step == 'after_evaluation_adoption_move':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test did not release adoption receipt gate')
        store = Store(self.temp.name, fault_hook=fault)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(store.adopt_evaluated_once, 'lease-adopt', 'main',
                                     self.head.revision, [short_id, long_id], ['valid'])
            self.assertTrue(entered.wait(5), 'adoption did not reach the post-CAS boundary')
            self.assertIn(self.long, store.checkpoint_roots(),
                          'selected numerical target must remain leased through its receipt')
            release.set()
            result = future.result(timeout=10)
        self.assertEqual(result['branch'].tip, self.long)
        self.assertNotIn(self.long, store.checkpoint_roots())

    def test_coordinator_rejects_adoption_while_branch_has_pending_publication(self):
        coordinator = object.__new__(ContextCoordinator)
        coordinator.store = self.store
        coordinator._condition = threading.Condition(threading.RLock())
        coordinator._branches = {'main'}
        with self.assertRaisesRegex(Conflict, 'pending request/publication'):
            coordinator.adopt_evaluated_once('pending-adopt', 'main', self.head.revision,
                                              [], ['valid'])
        self.assertFalse((self.store.evaluation_operations/'pending-adopt.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
