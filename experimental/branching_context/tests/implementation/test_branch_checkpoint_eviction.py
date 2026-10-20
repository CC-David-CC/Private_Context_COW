"""Finite numerical caches never own canonical branch/history lifetime."""
import hashlib
import json
import tempfile
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import DiskCheckpointPool, PoolConfig, PoolFull
from context_time_travel.store import Store
from test_canonical_branch_actions import BranchActionModel


class BranchCheckpointEvictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.root = self.store.create_root('m', (1, 2), {'status': 'idle'})
        self.tip = self.store.append(self.root, (3,), {'status': 'idle'})
        self.store.fork('main', self.tip)
        self.store.retain('saved-point', self.root)
        self.pool = DiskCheckpointPool(self.store, 1024**2)

    def checkpoint(self, snapshot, payload):
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        path = self.pool.directory / (snapshot + '.json')
        path.write_text(json.dumps({'manifest': {'payload_sha256': digest}}), encoding='utf-8')
        return path, state

    def canonical_bytes(self):
        roots = [self.store.objects, self.store.references, self.store.root / 'retained']
        files = [self.store.root / 'refs.json', self.store.root / 'ref-order.json']
        for root in roots:
            files.extend(root.glob('*.json'))
        return {p.relative_to(self.store.root).as_posix(): p.read_bytes() for p in files}

    def test_unleased_branch_and_named_history_are_not_permanent_numerical_pins(self):
        self.assertEqual(self.store.checkpoint_roots(), set())
        self.assertTrue({self.root, self.tip} <= self.store.retained_snapshots())

    def test_unleased_root_and_tip_are_lru_victims_without_canonical_changes(self):
        root_files = self.checkpoint(self.root, b'prefix-state')
        tip_files = self.checkpoint(self.tip, b'continued-state')
        before = self.canonical_bytes()
        self.pool.touch(self.root)
        self.pool.touch(self.tip)
        self.pool.byte_limit = self.pool._bytes()
        with self.pool.reserve(1):
            self.assertFalse(root_files[0].exists())
            self.assertFalse(root_files[1].exists())
            self.assertTrue(tip_files[0].exists())
        with self.pool.reserve(self.pool.byte_limit):
            self.assertFalse(tip_files[0].exists())
            self.assertFalse(tip_files[1].exists())
        self.assertEqual(self.canonical_bytes(), before)
        self.assertEqual(self.store.head('main').tip, self.tip)
        self.assertEqual(self.store.lineage(self.tip)[1], [1, 2, 3])

    def test_live_root_and_tip_leases_still_bound_capacity(self):
        root_files = self.checkpoint(self.root, b'prefix-state')
        tip_files = self.checkpoint(self.tip, b'continued-state')
        self.pool.byte_limit = self.pool._bytes()
        before = self.canonical_bytes()
        with self.store.read_lease(self.root), self.store.read_lease(self.tip):
            self.assertEqual(self.store.checkpoint_roots(), {self.root, self.tip})
            with self.assertRaises(PoolFull):
                self.pool.reserve(1, timeout=.02)
            self.assertTrue(all(p.exists() for p in root_files + tip_files))
        with self.pool.reserve(self.pool.byte_limit):
            self.assertFalse(any(p.exists() for p in root_files + tip_files))
        self.assertEqual(self.canonical_bytes(), before)

    def test_shared_payload_remains_until_last_live_manifest_is_released(self):
        root_manifest, state = self.checkpoint(self.root, b'shared-state')
        tip_manifest, _ = self.checkpoint(self.tip, b'shared-state')
        self.pool.byte_limit = self.pool._bytes()
        with self.store.read_lease(self.tip):
            with self.pool.reserve(1):
                self.assertFalse(root_manifest.exists())
                self.assertTrue(tip_manifest.exists())
                self.assertTrue(state.exists())
        with self.pool.reserve(self.pool.byte_limit):
            self.assertFalse(tip_manifest.exists())
            self.assertFalse(state.exists())


class EvictedBranchContinuationTests(unittest.TestCase):
    def test_all_branch_checkpoint_eviction_allows_fork_retry_and_reported_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            models = {}
            def factory(gpu):
                model = BranchActionModel(directory, gpu, (0,))
                models[gpu] = model
                return model
            config = PoolConfig(gpus=(0,), ram_bytes=1024**2,
                disk_bytes=8*1024**2, min_available_bytes=0)
            c = ContextCoordinator(directory, config, factory)
            try:
                c.ready(5)
                root = c.create('main', (1, 2), 'create').durable.result(5)
                answer = c.submit('main', (3,), 1, 'answer').durable.result(5)
            finally:
                c.close()
            disk = DiskCheckpointPool(Store(directory), config.disk_bytes)
            old_receipt = Store(directory)._read_refs()['receipts']['answer']
            old_record = Store(directory).get_snapshot(answer['snapshot'])
            disk.byte_limit = disk._bytes()
            with disk.reserve(disk.byte_limit):
                self.assertEqual(list(disk.directory.glob('*.json')), [])
                self.assertEqual(list(disk.directory.glob('*.state')), [])
            c = ContextCoordinator(directory, config, factory)
            try:
                c.ready(5)
                fork = c.fork('copy', answer['snapshot'])
                rewound = c.rewind('copy', fork.revision, root['snapshot'], 'retry')
                self.assertEqual(models[0].work_counters()['forward_tokens'], 0)
                request = c.submit('copy', (7,), 1, 'next', expected_revision=rewound.revision)
                result = request.snapshot.result(5)
                request.durable.result(5)
                self.assertEqual(result['source_recomputation']['prefix_forward_tokens'], 2)
                self.assertIs(result['source_recomputation']['prior_computation_preserved'], False)
                self.assertEqual(models[0].history(result['snapshot']), (1, 2, 7, 0))
                self.assertEqual(c.store.get_snapshot(answer['snapshot']), old_record)
                self.assertEqual(c.store._read_refs()['receipts']['answer'], old_receipt)
                self.assertEqual(c.store.head('main').tip, answer['snapshot'])
            finally:
                c.close()


if __name__ == '__main__':
    unittest.main()
