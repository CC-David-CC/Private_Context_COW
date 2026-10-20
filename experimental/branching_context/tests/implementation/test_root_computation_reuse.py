"""Exact-root reuse controls; synthetic CPU backend, no model qualification."""
from contextlib import contextmanager
import json
import time
from pathlib import Path
import tempfile
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel import real as real_module
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError
from context_time_travel.real import Checkpoint
from context_time_travel.strata_backend import StrataImage


class CPUBackend:
    """Small deterministic backend exercising the real BranchingModel path."""
    vocab_size = 16

    def __init__(self, profile):
        self.profile = profile
        self.stats = dict(forward_tokens=0, consumed_tokens=0, loaded_file_bytes=0)
        self.forward_calls = []

    def new_cache(self):
        return {}

    def metadata(self, key):
        return {'general.architecture': 'cpu-control'}.get(key)

    def contract_fields(self):
        return dict(attention='control', state_kind='cpu-bytes')

    @contextmanager
    def lease(self):
        state = {'ids': []}
        yield state

    def consume(self, lease, ids):
        ids = tuple(ids)
        lease['ids'].extend(ids)
        self.stats['forward_tokens'] += len(ids)
        self.stats['consumed_tokens'] += len(ids)
        self.forward_calls.append(ids)

    def export(self, lease):
        return b'cpu-state-v1:' + b','.join(str(token).encode('ascii') for token in lease['ids'])

    def logits(self, lease):
        total = sum(lease['ids'])
        return tuple(float(total + index) for index in range(self.vocab_size))

    def position(self, lease):
        return len(lease['ids']) - 1

    def work_counters(self):
        return dict(self.stats)

    def restore(self, lease, blob, *, history, frontier):
        resident = blob
        if isinstance(blob, StrataImage):
            blob = blob.identity
        expected = b'cpu-state-v1:' + b','.join(str(token).encode('ascii') for token in history)
        if blob != expected:
            raise IntegrityError('CPU control checkpoint does not bind history')
        lease['ids'] = list(history)
        self.stats['loaded_file_bytes'] += len(blob)
        return resident

    def serialize(self, lease, state):
        return state.identity if isinstance(state, StrataImage) else state

    def park(self, lease):
        # The real fallback path may park host state; this control backend has
        # no separate resident resource to release.
        return None

    def release_checkpoints(self):
        return 0

    def close(self):
        pass


class RootComputationReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile = dict(model_contract_id='cpu-model', execution_profile_id='cpu-exec',
                            model_weights_sha256='a' * 64,
                            execution={'GPU': [0]})
        self.backends = {}

        def factory(gpu):
            backend = CPUBackend(self.profile)
            self.backends[gpu] = backend
            return BranchingModel(self.tmp.name, backend, self.profile)

        self.coordinator = ContextCoordinator(self.tmp.name,
            PoolConfig(gpus=(0,), ram_bytes=1024 * 1024, disk_bytes=1024 * 1024,
                       min_available_bytes=0), factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(timeout=5)

    def create(self, branch, ids, operation):
        return self.coordinator.create(branch, tuple(ids), operation).durable.result(5)

    def wait_for_pool_release(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            metrics = self.coordinator.pool.metrics()
            if metrics['pins'] == 0 and metrics['reserved_bytes'] == 0:
                return metrics
            time.sleep(0.005)
        self.fail('Root source pin or staging reservation remained held after publication')

    def test_same_prefix_distinct_create_operations_reuse_exact_root(self):
        prefix = (2, 5, 7, 11)
        first = self.create('branch-a', prefix, 'create-a')
        backend = self.backends[0]
        first_schedule = json.loads((self.coordinator.store.root / 'checkpoints' /
            (first['snapshot'] + '.json')).read_bytes())['manifest'].get('execution_schedule')

        second = self.create('branch-b', prefix, 'create-b')

        self.assertEqual(backend.stats['forward_tokens'], len(prefix))
        self.assertEqual(first['snapshot'], second['snapshot'])
        self.assertNotEqual(first['branch']['name'], second['branch']['name'])
        status_a = self.coordinator.operation_status('create-a')
        status_b = self.coordinator.operation_status('create-b')
        self.assertEqual(status_a['state'], 'durable')
        self.assertEqual(status_b['state'], 'durable')
        self.assertEqual(status_a['receipt']['branch']['name'], 'branch-a')
        self.assertEqual(status_b['receipt']['branch']['name'], 'branch-b')
        self.assertEqual(self.coordinator.store.get_snapshot(first['snapshot'])['controller'],
                         {'status': 'idle'})
        second_schedule = json.loads((self.coordinator.store.root / 'checkpoints' /
            (first['snapshot'] + '.json')).read_bytes())['manifest'].get('execution_schedule')
        self.assertEqual(second_schedule, first_schedule)
        self.wait_for_pool_release()
        model = self.coordinator._workers[0]['model']
        pooled = self.coordinator.pool.get(self.coordinator._key(model,first['snapshot']))
        self.assertIsNot(model.cache.get(first['snapshot']),pooled,
                         'temporary shared-pool checkpoint alias escaped source release')

    def test_disk_checkpoint_reuse_after_worker_ram_cache_is_cleared(self):
        prefix = (1, 4, 8, 12)
        first = self.create('branch-a', prefix, 'create-a')
        backend = self.backends[0]
        model = self.coordinator._workers[0]['model']
        model.cache.clear()
        self.coordinator.pool.remove(self.coordinator._key(model,first['snapshot']))
        manifest = model.checkpoint_manifest(first['snapshot'],prefix)
        observed_reserved = []
        load = model._load_checkpoint

        def capture_reservation(snapshot, history):
            observed_reserved.append(self.coordinator.pool.metrics()['reserved_bytes'])
            return load(snapshot,history)

        model._load_checkpoint = capture_reservation
        before = backend.stats['forward_tokens']
        before_loaded = backend.stats['loaded_file_bytes']

        second = self.create('branch-b', prefix, 'create-b')

        self.assertEqual(second['snapshot'], first['snapshot'])
        self.assertEqual(backend.stats['forward_tokens'], before)
        self.assertEqual(len(observed_reserved),1)
        required = (2*(manifest['state_bytes']+4*manifest['logits_count'])
                    + manifest['state_bytes'] + 12*len(prefix))
        self.assertGreaterEqual(observed_reserved[0],required)
        self.assertGreater(backend.stats['loaded_file_bytes'],before_loaded)
        self.wait_for_pool_release()

    def test_standalone_disk_reuse_holds_store_read_lease(self):
        prefix = (1, 4, 8, 12)
        first = self.create('branch-a', prefix, 'create-a')
        model = self.coordinator._workers[0]['model']
        model.cache.clear()
        original_lease = model.store.read_lease
        original_load = model._load_checkpoint
        original_digest = real_module.verified_checkpoint_digest
        active = [0]
        observations = []

        @contextmanager
        def tracked_lease(snapshot):
            with original_lease(snapshot):
                active[0] += 1
                try:
                    yield snapshot
                finally:
                    active[0] -= 1

        def tracked_load(*args, **kwargs):
            observations.append(('load', active[0] > 0))
            return original_load(*args, **kwargs)

        def tracked_digest(owner):
            observations.append(('validate', active[0] > 0))
            return original_digest(owner)

        model.store.read_lease = tracked_lease
        model._load_checkpoint = tracked_load
        real_module.verified_checkpoint_digest = tracked_digest
        try:
            snapshot, reused = model.create_with_reuse(prefix)
        finally:
            real_module.verified_checkpoint_digest = original_digest
            model.store.read_lease = original_lease
            model._load_checkpoint = original_load

        self.assertEqual(snapshot, first['snapshot'])
        self.assertIs(reused, True)
        self.assertIn(('load', True), observations)
        self.assertIn(('validate', True), observations)
        self.assertEqual(active[0], 0)

    def test_hot_native_root_image_bypasses_cold_pool_staging(self):
        prefix = (1, 4, 8, 12)
        first = self.create('branch-a', prefix, 'create-a')
        model = self.coordinator._workers[0]['model']
        snapshot = first['snapshot']
        original = model.cache[snapshot]
        image = StrataImage(original.blob, b'cpu-test-process', 9, len(prefix),
                            len(original.blob), 4 * len(original.frontier), ())
        hot = Checkpoint.create(original.model_contract_id, original.execution_profile_id,
            original.consumed, image, original.frontier, original.input_identity,
            original.execution_schedule)
        model.cache[snapshot] = hot
        key = self.coordinator._key(model,snapshot)
        self.coordinator.pool.remove(key)
        before_metrics = self.coordinator.pool.metrics()
        observed = []
        root_materialization = self.coordinator._root_materialization

        def record_materialization(*args, **kwargs):
            result = root_materialization(*args, **kwargs)
            observed.append(result)
            return result

        self.coordinator._root_materialization = record_materialization
        self.coordinator._preload = lambda *args, **kwargs: self.fail(
            'hot native image must not enter cold checkpoint staging')

        second = self.create('branch-b', prefix, 'create-b')

        self.assertEqual(second['snapshot'], snapshot)
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], hot)
        self.assertIs(model.cache[snapshot], hot)
        after_metrics = self.coordinator.pool.metrics()
        self.assertEqual(after_metrics['reserved_bytes'], before_metrics['reserved_bytes'])
        self.assertEqual(after_metrics['pins'], before_metrics['pins'])
        self.assertEqual(self.backends[0].stats['forward_tokens'], len(prefix))

    def test_root_materialization_retries_eviction_between_preload_and_pin(self):
        prefix = (2, 5, 7, 11)
        first = self.create('branch-a', prefix, 'create-a')
        model = self.coordinator._workers[0]['model']
        key = self.coordinator._key(model,first['snapshot'])
        self.assertIn(key,self.coordinator.pool)
        original_pin = self.coordinator.pool.pin
        original_remove = self.coordinator.pool.remove
        attempts = []

        @contextmanager
        def evict_once(candidate):
            if candidate == key and not attempts:
                attempts.append('evicted-before-pin')
                original_remove(candidate)
                raise KeyError(candidate)
            attempts.append('pinned')
            with original_pin(candidate) as checkpoint:
                yield checkpoint

        self.coordinator.pool.pin = evict_once
        try:
            second = self.create('branch-b',prefix,'create-b')
        finally:
            self.coordinator.pool.pin = original_pin

        self.assertEqual(second['snapshot'],first['snapshot'])
        self.assertGreaterEqual(attempts.count('pinned'),1)
        self.assertEqual(attempts[0],'evicted-before-pin')
        metrics = self.wait_for_pool_release()
        self.assertEqual(metrics['pins'],0)
        self.assertEqual(metrics['reserved_bytes'],0)

    def test_missing_checkpoint_recomputes_exact_prefix(self):
        prefix = (3, 6, 9)
        first = self.create('branch-a', prefix, 'create-a')
        backend = self.backends[0]
        model = self.coordinator._workers[0]['model']
        model.cache.clear()
        self.coordinator.pool.remove(self.coordinator._key(model,first['snapshot']))
        manifest_path = model.checkpoint_directory / (first['snapshot'] + '.json')
        manifest = json.loads(manifest_path.read_bytes())['manifest']
        manifest_path.unlink()
        (model.checkpoint_directory / (manifest['payload_sha256'] + '.state')).unlink()
        before = backend.stats['forward_tokens']

        second = self.create('branch-b', prefix, 'create-b')

        self.assertEqual(second['snapshot'], first['snapshot'])
        self.assertEqual(backend.stats['forward_tokens'] - before, len(prefix))

    def test_corrupt_existing_checkpoint_is_a_hard_error_without_recompute(self):
        prefix = (1, 2, 4)
        first = self.create('branch-a', prefix, 'create-a')
        backend = self.backends[0]
        model = self.coordinator._workers[0]['model']
        model.cache.clear()
        self.coordinator.pool.remove(self.coordinator._key(model,first['snapshot']))
        manifest = json.loads((model.checkpoint_directory / (first['snapshot'] + '.json')).read_bytes())['manifest']
        payload = model.checkpoint_directory / (manifest['payload_sha256'] + '.state')
        payload.write_bytes(b'corrupt')
        before = backend.stats['forward_tokens']

        request = self.coordinator.create('branch-b', prefix, 'create-b')
        with self.assertRaises((IntegrityError, OSError)):
            request.durable.result(5)

        self.assertEqual(backend.stats['forward_tokens'], before)

    def test_dangling_manifest_is_corruption_not_a_cache_miss(self):
        prefix = (2, 4, 6)
        first = self.create('branch-a',prefix,'create-a')
        model = self.coordinator._workers[0]['model']
        backend = self.backends[0]
        model.cache.clear()
        self.coordinator.pool.remove(self.coordinator._key(model,first['snapshot']))
        manifest = model.checkpoint_directory/(first['snapshot']+'.json')
        manifest.unlink()
        manifest.symlink_to(model.checkpoint_directory/'missing-manifest-target.json')
        before = backend.stats['forward_tokens']

        request = self.coordinator.create('branch-b',prefix,'create-b')
        with self.assertRaises(IntegrityError):
            request.durable.result(5)

        self.assertEqual(backend.stats['forward_tokens'],before)

    def test_different_prefix_does_not_reuse_same_length_root(self):
        first = (2, 3, 5, 7)
        second = (2, 3, 5, 8)
        a = self.create('branch-a', first, 'create-a')
        backend = self.backends[0]
        before = backend.stats['forward_tokens']

        b = self.create('branch-b', second, 'create-b')

        self.assertNotEqual(a['snapshot'], b['snapshot'])
        self.assertEqual(backend.stats['forward_tokens'] - before, len(second))

    def test_other_execution_profile_never_reuses_saved_checkpoint(self):
        prefix = (4, 5, 6)
        first = self.create('branch-a', prefix, 'create-a')
        backend = self.backends[0]
        before = backend.stats['forward_tokens']

        other_profile = dict(self.profile, execution_profile_id='other-execution')
        other_backend = CPUBackend(other_profile)
        other_model = BranchingModel(self.tmp.name, other_backend, other_profile)
        with self.assertRaises(IntegrityError):
            other_model.create(prefix)

        self.assertEqual(other_backend.stats['forward_tokens'], 0)
        self.assertEqual(first['snapshot'], self.coordinator.store.head('branch-a').tip)


if __name__ == '__main__':
    unittest.main()
