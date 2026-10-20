"""Test-first regressions for checkpoint-disk LRU admission and reclamation."""
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from context_time_travel.pools import DiskCheckpointPool, PoolFull
from context_time_travel.store import Store


class DiskLRUTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.root = self.store.create_root('model', (1,), {'status': 'idle'})
        self.s1 = self.store.append(self.root, (2,), {'status': 'idle'})
        self.s2 = self.store.append(self.s1, (3,), {'status': 'idle'})
        self.s3 = self.store.append(self.s2, (4,), {'status': 'idle'})
        self.tip = self.store.append(self.s3, (5,), {'status': 'idle'})
        self.store.fork('main', self.tip)
        self.pool = DiskCheckpointPool(self.store, 1 << 20)

    def add_checkpoint(self, snapshot, payload):
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        if not state.exists():
            state.write_bytes(payload)
        manifest = self.pool.directory / (snapshot + '.json')
        manifest.write_text(json.dumps({'manifest': {'payload_sha256': digest}}), encoding='utf-8')
        return manifest, state

    def canonical_fingerprint(self):
        refs = (self.store.root / 'refs.json').read_bytes()
        order = (self.store.root / 'ref-order.json').read_bytes()
        branch_refs = tuple(sorted(
            (path.name, path.read_bytes())
            for path in self.store.references.glob('*.json')
        ))
        objects = self.store.root / 'objects'
        return (refs, order, branch_refs,
                tuple(sorted((p.name, p.read_bytes()) for p in objects.glob('*.json'))))

    def test_canonical_fingerprint_covers_branch_reference_files(self):
        branch = self.store.references / 'main.json'
        before = self.canonical_fingerprint()
        original = branch.read_bytes()
        try:
            branch.write_bytes(original + b' ')
            self.assertNotEqual(self.canonical_fingerprint(), before)
        finally:
            branch.write_bytes(original)

    def test_eviction_accounting_does_not_rescan_directory_per_victim(self):
        for index, sid in enumerate((self.s1, self.s2, self.s3)):
            self.add_checkpoint(sid, f'account-{index}'.encode())
        baseline = self.pool._bytes()
        self.pool.byte_limit = baseline
        byte_scans = 0
        original_bytes = self.pool._bytes
        def counted_bytes():
            nonlocal byte_scans
            byte_scans += 1
            return original_bytes()
        self.pool._bytes = counted_bytes
        with self.pool.reserve(baseline):
            self.assertLessEqual(byte_scans, 4)

    def test_recency_inventory_prunes_missing_manifests(self):
        manifest, _ = self.add_checkpoint(self.s1, b'gone')
        self.pool.touch(self.s1)
        manifest.unlink()
        self.pool.collect()
        self.assertNotIn(self.s1, self.pool._recency)

    def test_protected_orphan_survives_collection_until_reservation_release(self):
        payload = b'in-flight-state'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        self.pool.byte_limit = len(payload) + 1
        with self.pool.reserve(1, payload_sha256=digest):
            self.pool.collect()
            self.assertTrue(state.exists(), 'active publication protects its payload from orphan GC')
        self.pool.collect()
        self.assertFalse(state.exists(), 'orphan becomes reclaimable after final reservation release')

    def test_nested_payload_protection_is_reference_counted(self):
        payload = b'nested-in-flight'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        with self.pool.reserve(0, payload_sha256=digest):
            with self.pool.reserve(0, payload_sha256=digest):
                pass
            self.pool.collect()
            self.assertTrue(state.exists())
        self.pool.collect()
        self.assertFalse(state.exists())

    def test_protected_shared_payload_survives_eviction_of_last_old_manifest(self):
        payload = b'old-and-new-publication'
        digest = hashlib.sha256(payload).hexdigest()
        old_manifest, state = self.add_checkpoint(self.s1, payload)
        manifest_bytes = old_manifest.stat().st_size
        self.pool.byte_limit = state.stat().st_size + manifest_bytes
        with self.pool.reserve(manifest_bytes, payload_sha256=digest):
            self.assertFalse(old_manifest.exists())
            self.assertTrue(state.exists(), 'in-flight manifest may be the next reference')
            new_manifest = self.pool.directory / (self.s2 + '.json')
            new_manifest.write_text(json.dumps({'manifest': {'payload_sha256': digest}}), encoding='utf-8')
        self.pool.collect()
        self.assertTrue(state.exists(), 'new manifest now owns the shared payload')

    def test_failed_admission_drops_payload_protection(self):
        payload = b'failed-admission'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        self.pool.byte_limit = len(payload)
        with self.assertRaises(PoolFull):
            self.pool.reserve(1, payload_sha256=digest, timeout=0)
        self.pool.collect()
        self.assertFalse(state.exists(), 'failed admission must release its temporary protection')

    def test_byte_inventory_tolerates_payload_disappearing_after_is_file(self):
        state = self.pool.directory / ('a' * 64 + '.state')
        state.write_bytes(b'x')
        original_is_file = type(state).is_file
        original_stat = type(state).stat
        def is_file(path, *args, **kwargs):
            if path == state:
                return True
            return original_is_file(path, *args, **kwargs)
        def stat(path, *args, **kwargs):
            if path == state:
                raise FileNotFoundError(path)
            return original_stat(path, *args, **kwargs)
        with mock.patch.object(type(state), 'is_file', is_file), \
                mock.patch.object(type(state), 'stat', stat):
            self.assertEqual(self.pool._bytes(), 0)

    def test_read_lease_release_allows_waiting_reservation_without_other_release(self):
        manifest, _ = self.add_checkpoint(self.s1, b'only-eligible')
        self.pool.byte_limit = self.pool._bytes()
        acquired = threading.Event()
        errors = []
        lease = self.store.read_lease(self.s1)
        lease.__enter__()
        def waiter():
            try:
                with self.pool.reserve(1, timeout=1):
                    acquired.set()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=waiter)
        thread.start()
        deadline = time.monotonic() + 0.5
        while self.pool.metrics()['admission_waits'] == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        lease.__exit__(None, None, None)
        thread.join(0.3)
        self.assertFalse(thread.is_alive(), 'lease release should unblock bounded admission promptly')
        self.assertTrue(acquired.is_set())
        self.assertFalse(manifest.exists())
        self.assertEqual(errors, [])

    def test_mru_is_protected_and_only_enough_lru_manifests_are_removed(self):
        rows = {}
        for index, sid in enumerate((self.s1, self.s2, self.s3)):
            rows[sid] = self.add_checkpoint(sid, f'payload-{index}'.encode())
        baseline = self.pool._bytes()
        # Explicit touches establish a deterministic order: s2 oldest, s3 next,
        # then s1 most recent, despite their creation order.
        self.pool.touch(self.s2)
        self.pool.touch(self.s3)
        self.pool.touch(self.s1)
        self.pool.touch(self.s1)
        self.pool.byte_limit = baseline
        before = self.canonical_fingerprint()
        with self.pool.reserve(1):
            self.assertFalse(rows[self.s2][0].exists())
            self.assertTrue(rows[self.s1][0].exists())
            self.assertTrue(rows[self.s3][0].exists())
            metrics = self.pool.metrics()
            self.assertEqual(metrics['files_evicted'], 2)
            self.assertEqual(len(list(self.pool.directory.glob('*.json'))), 2)
            self.assertLessEqual(metrics['file_bytes'] + metrics['reserved_bytes'], baseline)
        self.assertEqual(self.canonical_fingerprint(), before)
        self.assertEqual(self.store.lineage(self.tip)[1], [1, 2, 3, 4, 5])

    def test_shared_payload_survives_until_last_manifest_reference_is_removed(self):
        m1, state = self.add_checkpoint(self.s1, b'shared-checkpoint')
        m2, state2 = self.add_checkpoint(self.s2, b'shared-checkpoint')
        self.assertEqual(state, state2)
        initial = self.pool._bytes()
        self.pool.touch(self.s2)
        self.pool.touch(self.s1)
        self.pool.byte_limit = initial
        second_manifest_bytes = m2.stat().st_size
        with self.pool.reserve(m1.stat().st_size):
            self.assertFalse(m2.exists())
            self.assertTrue(m1.exists())
            self.assertTrue(state.exists(), 'one remaining manifest still references the payload')
        self.pool.touch(self.s1)
        with self.pool.reserve(second_manifest_bytes + 1):
            self.assertFalse(m1.exists())
            self.assertFalse(state.exists(), 'payload is reclaimable only after its last manifest is gone')

    def test_active_read_lease_pins_manifest_from_unlink(self):
        m1, _ = self.add_checkpoint(self.s1, b'leased')
        m2, _ = self.add_checkpoint(self.s2, b'evictable')
        baseline = self.pool._bytes()
        self.pool.touch(self.s2)
        self.pool.touch(self.s1)
        self.pool.byte_limit = baseline
        before = self.canonical_fingerprint()
        with self.store.read_lease(self.s1):
            with self.pool.reserve(1):
                self.assertTrue(m1.exists())
                self.assertFalse(m2.exists())
        self.assertEqual(self.canonical_fingerprint(), before)

    def test_no_eligible_space_fails_boundedly_without_deleting_leased_anchors(self):
        mroot, _ = self.add_checkpoint(self.root, b'root')
        mtip, _ = self.add_checkpoint(self.tip, b'tip')
        before = (mroot.read_bytes(), mtip.read_bytes(), self.pool._bytes())
        self.pool.byte_limit = self.pool._bytes()
        started = time.monotonic()
        with self.store.read_lease(self.root), self.store.read_lease(self.tip):
            with self.assertRaises(PoolFull):
                self.pool.reserve(1, timeout=0.04)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((mroot.read_bytes(), mtip.read_bytes(), self.pool._bytes()), before)
        self.assertEqual(self.pool.metrics()['files_evicted'], 0)

    def test_releasing_reservation_wakes_waiting_admission(self):
        self.pool.byte_limit = 64
        held = self.pool.reserve(64)
        started = threading.Event()
        acquired = threading.Event()
        errors = []
        def waiter():
            started.set()
            try:
                with self.pool.reserve(1, timeout=2):
                    acquired.set()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=waiter)
        thread.start()
        self.assertTrue(started.wait(1))
        deadline = time.monotonic() + 1
        while self.pool.metrics().get('admission_waits', 0) == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreater(self.pool.metrics().get('admission_waits', 0), 0)
        held.release()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(self.pool.metrics()['reserved_bytes'], 0)

    def test_orphan_state_payload_is_reclaimed_before_manifest_lru(self):
        orphan = self.pool.directory / ('f' * 64 + '.state')
        orphan.write_bytes(b'orphan-data')
        self.pool.byte_limit = orphan.stat().st_size
        before = self.canonical_fingerprint()
        with self.pool.reserve(1):
            self.assertFalse(orphan.exists())
        self.assertEqual(self.store.lineage(self.tip)[1], [1, 2, 3, 4, 5])
        self.assertEqual(self.canonical_fingerprint(), before)

    def test_startup_recency_uses_manifest_mtime_then_snapshot_name(self):
        m1, _ = self.add_checkpoint(self.s1, b'one')
        m2, _ = self.add_checkpoint(self.s2, b'two')
        m3, _ = self.add_checkpoint(self.s3, b'three')
        stamp = 1_700_000_000_000_000_000
        for path in (m1, m2, m3):
            os.utime(path, ns=(stamp, stamp))
        restarted = DiskCheckpointPool(self.store, 1 << 20)
        baseline = restarted._bytes()
        restarted.byte_limit = baseline
        oldest = min((self.s1, self.s2, self.s3))
        with restarted.reserve(1):
            self.assertFalse((restarted.directory / (oldest + '.json')).exists())


if __name__ == '__main__':
    unittest.main()
