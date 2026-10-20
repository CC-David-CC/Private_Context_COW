"""CPU-only coordinator cache ownership regressions; no native/GPU claim."""
from dataclasses import replace
from contextlib import contextmanager
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig, PoolFull, PoolTransitionTrace
from context_time_travel.strata_backend import StrataImage
from test_pool_coordinator import Model


def observe_cache_at_unpin(coordinator, key, model, snapshot, observed):
    original = coordinator.pool.pin

    @contextmanager
    def pin(candidate):
        with original(candidate) as checkpoint:
            try:
                yield checkpoint
            finally:
                if candidate == key:
                    observed['cache_at_unpin'] = model.cache.get(snapshot)

    coordinator.pool.pin = pin


class ColdCacheOwnershipTests(unittest.TestCase):
    def _start(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        models = {}
        events = PoolTransitionTrace('prepare', process_id=501, phase_id='cold-cache')
        config = PoolConfig(gpus=(0,), ram_bytes=4096, disk_bytes=16384,
                            min_available_bytes=0)

        def factory(gpu):
            model = Model(temp.name, gpu, (0,), [], threading.Event())
            model.gate.set()
            models[gpu] = model
            return model

        coordinator = ContextCoordinator(temp.name, config, factory,
                                         transition_trace=events)
        self.addCleanup(coordinator.close)
        coordinator.ready(timeout=5)
        model = models[0]
        root = coordinator.create('main', (1, 2), 'root-create').durable.result(5)['snapshot']
        return coordinator, model, root, events

    def test_removed_pool_entry_cannot_be_bypassed_by_cached_cold_bytes(self):
        coordinator, model, root, events = self._start()
        key = coordinator._key(model, root)
        self.assertIsInstance(model.cache[root].blob, bytes)
        coordinator.pool.remove(key)
        observed = {}
        observe_cache_at_unpin(coordinator,key,model,root,observed)

        coordinator.submit('main', (3,), 1, 'cold-evicted').durable.result(5)

        succeeded = [row for row in events.snapshot()['events']
                     if row['event'] == 'disk_preload_succeeded']
        self.assertTrue(succeeded, 'evicted serialized alias bypassed shared-pool reload')
        self.assertEqual(succeeded[0]['key'], list(key))
        self.assertNotIn(root, model.cache,
                         'temporary cold serialized alias survived after source pin release')
        self.assertIsNone(observed['cache_at_unpin'],
                          'cold serialized alias must be removed before source unpin')
        self.assertEqual(coordinator.pool.metrics()['pins'], 0)
        self.assertEqual(coordinator.pool.metrics()['reserved_bytes'], 0)

    def test_cold_alias_is_pinned_and_reserved_during_execution(self):
        coordinator, model, root, _ = self._start()
        key = coordinator._key(model, root)
        entered, resume = threading.Event(), threading.Event()
        observed = {}
        observe_cache_at_unpin(coordinator,key,model,root,observed)
        original = model.run_greedy_request

        def inspect(*args, **kwargs):
            current = model.cache[root]
            observed['cold_blob'] = isinstance(current.blob, bytes)
            observed['metrics'] = coordinator.pool.metrics()
            try:
                coordinator.pool.remove(key)
            except PoolFull:
                observed['remove_blocked'] = True
            else:
                observed['remove_blocked'] = False
            entered.set()
            if not resume.wait(5):
                raise TimeoutError('test did not release execution gate')
            return original(*args, **kwargs)

        model.run_greedy_request = inspect
        request = coordinator.submit('main', (3,), 1, 'cold-protected')
        self.assertTrue(entered.wait(5))
        self.assertTrue(observed['cold_blob'])
        self.assertGreater(observed['metrics']['reserved_bytes'], 0)
        self.assertGreater(observed['metrics']['pins'], 0)
        self.assertTrue(observed['remove_blocked'])
        resume.set()
        request.durable.result(5)
        self.assertNotIn(root, model.cache)
        self.assertIsNone(observed['cache_at_unpin'],
                          'cold serialized alias must be removed before source unpin')
        self.assertEqual(coordinator.pool.metrics()['pins'], 0)
        self.assertEqual(coordinator.pool.metrics()['reserved_bytes'], 0)

    def test_execution_exception_removes_temporary_cold_alias_and_releases_ownership(self):
        coordinator, model, root, _ = self._start()
        key = coordinator._key(model, root)
        entered = threading.Event()
        observed = {}
        observe_cache_at_unpin(coordinator,key,model,root,observed)

        def fail(*args, **kwargs):
            self.assertIsInstance(model.cache[root].blob, bytes)
            self.assertGreater(coordinator.pool.metrics()['reserved_bytes'], 0)
            self.assertGreater(coordinator.pool.metrics()['pins'], 0)
            entered.set()
            raise RuntimeError('synthetic execution failure')

        model.run_greedy_request = fail
        request = coordinator.submit('main', (3,), 1, 'cold-failure')
        self.assertTrue(entered.wait(5))
        with self.assertRaisesRegex(RuntimeError, 'synthetic execution failure'):
            request.durable.result(5)
        self.assertNotIn(root, model.cache)
        self.assertIsNone(observed['cache_at_unpin'],
                          'failed execution must remove cold alias before source unpin')
        self.assertIn(key, coordinator.pool)
        self.assertEqual(coordinator.pool.metrics()['pins'], 0)
        self.assertEqual(coordinator.pool.metrics()['reserved_bytes'], 0)

    def test_native_image_uses_fast_cache_without_shared_pool_load(self):
        coordinator, model, root, events = self._start()
        key = coordinator._key(model, root)
        coordinator.pool.remove(key)
        image = StrataImage(b'image-id', b'process-id', 9, 2, 8, 4, ())
        model.cache[root] = replace(model.cache[root], blob=image)

        coordinator.submit('main', (3,), 1, 'native-hot').durable.result(5)

        acquired = [row for row in events.snapshot()['events']
                    if row['event'] == 'request_source_acquired']
        self.assertEqual(acquired[-1]['source_kind'], 'model_cache')
        self.assertFalse(any(row['event'] == 'disk_preload_succeeded'
                             for row in events.snapshot()['events']))
        self.assertEqual(coordinator.pool.metrics()['pins'], 0)
        self.assertIs(model.cache[root].blob, image,
                      'the worker-owned native image should remain eligible for the hot path')

    def test_successful_create_records_root_before_followup_request(self):
        coordinator, model, root, _ = self._start()
        worker = coordinator._workers[0]
        self.assertEqual(worker['last_root'], model.store.head('main').root)
        self.assertEqual(worker['last_root'], root)
