"""CPU coordinator/filesystem checks for disk admission and validated recency."""
from pathlib import Path
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from test_pool_coordinator import Model


class DiskLRUCoordinatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.models = {}
        gate = threading.Event()
        gate.set()
        def factory(gpu):
            model = Model(temporary.name, gpu, (0,), [], gate)
            self.models[gpu] = model
            return model
        self.coordinator = ContextCoordinator(temporary.name,
            PoolConfig(gpus=(0,), ram_bytes=4096, disk_bytes=16384,
                       min_available_bytes=0), factory, admission_timeout=.2)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)

    def idle(self):
        with self.coordinator._condition:
            self.assertTrue(self.coordinator._condition.wait_for(
                lambda: not self.coordinator._workers[0]['busy'], timeout=5))

    def test_publication_protects_digest_and_uses_configured_bounded_wait(self):
        coordinator = self.coordinator
        calls = []
        original = coordinator.disk.reserve
        def reserve(size, **options):
            calls.append(dict(size=size, **options))
            return original(size, **options)
        coordinator.disk.reserve = reserve
        receipt = coordinator.create('main', (1, 2), 'create').durable.result(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].get('payload_sha256'), receipt['payload_sha256'])
        self.assertEqual(calls[0].get('timeout'), .2)
        self.assertEqual(coordinator.disk._protected_payloads, {})

    def test_concurrent_collection_preserves_payload_before_manifest_publication(self):
        coordinator = self.coordinator
        model = self.models[0]
        original = model.save_prepared_branch
        observed = []
        def save(name, revision, prepared, operation):
            path = model.checkpoint_directory/(prepared.checkpoint.checksum+'.state')
            path.write_bytes(prepared.checkpoint.blob+b'\0\0\x80?')
            self.assertFalse((model.checkpoint_directory/(prepared.snapshot+'.json')).exists())
            # Simulate maintenance at the genuine payload-to-manifest window.
            coordinator.disk.collect()
            observed.append(path.exists())
            self.assertTrue(path.exists(), 'collector deleted an in-flight publication payload')
            return original(name, revision, prepared, operation)
        model.save_prepared_branch = save
        receipt = coordinator.create('main', (1, 2), 'create').durable.result(5)
        self.assertEqual(observed, [True])
        self.assertEqual(coordinator.store.head('main').tip, receipt['snapshot'])

    def test_validated_disk_load_and_publication_update_recency(self):
        coordinator = self.coordinator
        touches = []
        original = coordinator.disk.touch
        def touch(snapshot):
            touches.append(snapshot)
            return original(snapshot)
        coordinator.disk.touch = touch
        receipt = coordinator.create('main', (1, 2), 'create').durable.result(5)
        self.idle()
        root = receipt['snapshot']
        self.assertEqual(touches, [root])
        model = self.models[0]
        key = coordinator._key(model, root)
        model.cache.clear()
        coordinator.pool.remove(key)
        before = model.stats['forward_tokens']
        coordinator._preload(model, root).result(5)
        self.assertEqual(touches, [root, root])
        self.assertEqual(model.stats['forward_tokens'], before)

    def test_failed_disk_load_does_not_update_recency(self):
        coordinator = self.coordinator
        receipt = coordinator.create('main', (1, 2), 'create').durable.result(5)
        self.idle()
        model = self.models[0]
        root = receipt['snapshot']
        model.cache.clear()
        coordinator.pool.remove(coordinator._key(model, root))
        manifest = model.checkpoint_manifest(root)
        (model.checkpoint_directory/(manifest['payload_sha256']+'.state')).unlink()
        touches = []
        coordinator.disk.touch = touches.append
        with self.assertRaises(FileNotFoundError):
            coordinator._preload(model, root).result(5)
        self.assertEqual(touches, [])
        self.assertEqual(coordinator.store.head('main').tip, root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
