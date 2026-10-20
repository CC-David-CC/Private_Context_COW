"""Deterministic overlap controls for asynchronous checkpoint publication.

Synthetic CPU fixture only; it makes no GPU, timing, or model-performance claim.
"""
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import Conflict

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from test_pool_coordinator import Model


class PublicationOverlapTests(unittest.TestCase):
    def start(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        trace, models = [], {}
        gate = threading.Event()
        gate.set()
        config = PoolConfig(gpus=(0,), ram_bytes=4096, disk_bytes=16384,
                            min_available_bytes=0)

        def factory(gpu):
            models[gpu] = Model(temp.name, gpu, (0,), trace, gate)
            return models[gpu]

        coordinator = ContextCoordinator(temp.name, config, factory)
        self.addCleanup(coordinator.close)
        coordinator.ready(timeout=5)
        return coordinator, models[0], trace, gate

    @staticmethod
    def wait_for(predicate, message):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError(message)

    @staticmethod
    def start_observer(model):
        starts = {}
        original = model.run_greedy_request

        def run(sid, ids, count, *, stream, on_token):
            starts.setdefault(stream, threading.Event()).set()
            return original(sid, ids, count, stream=stream, on_token=on_token)

        model.run_greedy_request = run
        return starts

    @staticmethod
    def create_branches(coordinator, model, names):
        root = coordinator.create('root', (1, 2), 'root-create').durable.result(5)['snapshot']
        for name in names:
            coordinator.fork(name, root)
        return root, tuple(model.history(root))

    @staticmethod
    def queued_events(request):
        result = []
        while True:
            try:
                result.append(request.events.get_nowait())
            except queue.Empty:
                return result

    def test_writer_overlap_computes_three_fifo_siblings_without_durable_ack_or_head_moves(self):
        coordinator, model, trace, gate = self.start()
        root, root_ids = self.create_branches(coordinator, model, ('a', 'b', 'c', 'd'))
        writer_entered, release_writer = threading.Event(), threading.Event()
        self.addCleanup(release_writer.set)
        original_save = model.save_prepared_branch

        def block_writer(name, revision, prepared, operation):
            if operation == 'op-a':
                writer_entered.set()
                if not release_writer.wait(5):
                    raise TimeoutError('test writer gate was not released')
            return original_save(name, revision, prepared, operation)

        model.save_prepared_branch = block_writer
        first = coordinator.submit('a', (10,), 2, 'op-a')
        first_result = first.snapshot.result(5)
        self.assertTrue(writer_entered.wait(2))
        self.assertFalse(first.durable.done())
        self.assertIn(first_result['snapshot'], coordinator.store.checkpoint_roots())

        siblings = [coordinator.submit(name, (token,), 2, operation)
                    for name, token, operation in (('b', 20, 'op-b'),
                                                   ('c', 30, 'op-c'),
                                                   ('d', 40, 'op-d'))]
        sibling_results = [request.snapshot.result(5) for request in siblings]
        same_branch = coordinator.submit('a', (99,), 1, 'sneak',
                                         expected_revision=coordinator.store.head('a').revision)

        self.assertEqual([row[2] for row in trace if row[0] == 'start'],
                         ['op-a', 'op-b', 'op-c', 'op-d'])
        self.assertEqual(coordinator.metrics()['max_active_per_gpu'], {0: 1})
        targets = [first_result['snapshot'], *(value['snapshot'] for value in sibling_results)]
        self.assertTrue(set(targets).issubset(coordinator.store.checkpoint_roots()))
        for name in ('a', 'b', 'c', 'd'):
            self.assertEqual(coordinator.store.head(name).tip, root)
        for request in (first, *siblings, same_branch):
            self.assertFalse(request.durable.done())
            self.assertFalse(any(event.get('event') == 'durable'
                                 for event in self.queued_events(request)))
        self.assertNotIn(('start', 0, 'sneak'), trace)

        release_writer.set()
        results = [first.durable.result(5), *(request.durable.result(5) for request in siblings)]
        with self.assertRaises(Conflict):
            same_branch.durable.result(5)
        self.assertNotIn(('start', 0, 'sneak'), trace)
        expected = [(10, 0, 1), (20, 0, 1), (30, 0, 1), (40, 0, 1)]
        for name, receipt, target, suffix in zip(('a', 'b', 'c', 'd'), results, targets, expected):
            self.assertEqual(receipt['branch']['name'], name)
            self.assertEqual(receipt['snapshot'], target)
            self.assertEqual(model.history(target), root_ids + suffix)
            self.assertEqual(coordinator.store.head(name).tip, target)
        self.wait_for(lambda: not set(targets).intersection(coordinator.store.checkpoint_roots()),
                      'publication leases remained after durable completion')

    def test_first_writer_failure_leaves_siblings_durable_and_retry_reuses_computed_target(self):
        coordinator, model, trace, gate = self.start()
        root, root_ids = self.create_branches(coordinator, model, ('a', 'b', 'c'))
        writer_entered, release_writer = threading.Event(), threading.Event()
        self.addCleanup(release_writer.set)
        original_save = model.save_prepared_branch

        def fail_first(name, revision, prepared, operation):
            if operation == 'op-a':
                writer_entered.set()
                if not release_writer.wait(5):
                    raise TimeoutError('test failure gate was not released')
                raise OSError(28, 'synthetic writer failure after computation')
            return original_save(name, revision, prepared, operation)

        model.save_prepared_branch = fail_first
        first = coordinator.submit('a', (10,), 2, 'op-a')
        first_result = first.snapshot.result(5)
        self.assertTrue(writer_entered.wait(2))
        siblings = [coordinator.submit(name, (token,), 2, operation)
                    for name, token, operation in (('b', 20, 'op-b'), ('c', 30, 'op-c'))]
        sibling_results = [request.snapshot.result(5) for request in siblings]
        before_retry_forward = model.stats['forward_tokens']

        self.assertEqual(coordinator.store.head('a').tip, root)
        release_writer.set()
        with self.assertRaises(OSError) as failure:
            first.durable.result(5)
        self.assertEqual(failure.exception.errno, 28)
        receipts = [request.durable.result(5) for request in siblings]

        first_events = self.queued_events(first)
        self.assertTrue(any(event.get('event') == 'failure' for event in first_events))
        self.assertFalse(any(event.get('event') == 'durable' for event in first_events))
        for name, receipt, result, token in zip(('b', 'c'), receipts, sibling_results, (20, 30)):
            self.assertEqual(receipt['branch']['name'], name)
            self.assertEqual(coordinator.store.head(name).tip, result['snapshot'])
            self.assertEqual(model.history(result['snapshot']), root_ids + (token, 0, 1))
        self.assertEqual(coordinator.store.head('a').tip, root)

        before_retry_starts = len([item for item in trace if item[0] == 'start'])
        model.save_prepared_branch = original_save
        retried = coordinator.retry_failed('op-a').result(5)
        self.assertEqual(retried['snapshot'], first_result['snapshot'])
        self.assertEqual(len([item for item in trace if item[0] == 'start']), before_retry_starts)
        self.assertEqual(model.stats['forward_tokens'], before_retry_forward)
        self.assertEqual(model.history(retried['snapshot']), root_ids + (10, 0, 1))
        self.assertEqual(coordinator.store.head('a').tip, retried['snapshot'])

    def test_cancelled_queued_job_does_not_consume_worker_or_reorder_remaining_jobs(self):
        coordinator, model, trace, gate = self.start()
        root, _root_ids = self.create_branches(coordinator, model, ('a', 'b', 'cancelled', 'd', 'e'))
        writer_entered, release_writer = threading.Event(), threading.Event()
        self.addCleanup(release_writer.set)
        original_save = model.save_prepared_branch

        def block_writer(name, revision, prepared, operation):
            if operation == 'op-a':
                writer_entered.set()
                if not release_writer.wait(5):
                    raise TimeoutError('test writer gate was not released')
            return original_save(name, revision, prepared, operation)

        model.save_prepared_branch = block_writer
        starts = self.start_observer(model)
        first = coordinator.submit('a', (10,), 1, 'op-a')
        first_result = first.snapshot.result(5)
        self.assertTrue(writer_entered.wait(2))

        gate.clear()
        active = coordinator.submit('b', (20,), 1, 'op-b')
        self.assertTrue(starts.setdefault('op-b', threading.Event()).wait(2))
        cancelled = coordinator.submit('cancelled', (30,), 1, 'op-cancel')
        pending = coordinator.submit('d', (40,), 1, 'op-d')
        final = coordinator.submit('e', (50,), 1, 'op-e')
        self.assertTrue(coordinator.cancel(cancelled))
        self.assertTrue(cancelled.durable.cancelled())
        self.assertNotIn(('start', 0, 'op-cancel'), trace)

        gate.set()
        active_result = active.snapshot.result(5)
        pending_result = pending.snapshot.result(5)
        final_result = final.snapshot.result(5)
        self.assertEqual([row[2] for row in trace if row[0] == 'start'],
                         ['op-a', 'op-b', 'op-d', 'op-e'])
        for name in ('a', 'b', 'cancelled', 'd', 'e'):
            self.assertEqual(coordinator.store.head(name).tip, root)
        all_targets = [first_result['snapshot'], active_result['snapshot'],
                       pending_result['snapshot'], final_result['snapshot']]
        self.assertTrue(set(all_targets).issubset(coordinator.store.checkpoint_roots()))
        self.assertFalse(any(event.get('event') == 'durable' for event in self.queued_events(first)))

        release_writer.set()
        first.durable.result(5)
        active.durable.result(5)
        pending.durable.result(5)
        final.durable.result(5)
        self.assertEqual([row[2] for row in trace if row[0] == 'start'],
                         ['op-a', 'op-b', 'op-d', 'op-e'])
        self.assertEqual(coordinator.store.head('cancelled').tip, root)
        self.assertEqual(coordinator.metrics()['max_active_per_gpu'], {0: 1})


if __name__ == '__main__':
    unittest.main()
