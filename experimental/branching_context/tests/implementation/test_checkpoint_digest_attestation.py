"""CPU controls for reusing a verified immutable checkpoint digest."""
import gc
import sys
import threading
import unittest
import weakref
from dataclasses import replace
from unittest import mock

import test_durable_restore as durable_fixtures
from context_time_travel.real import Checkpoint
from context_time_travel.pools import SharedCheckpointPool, checkpoint_digest
from context_time_travel import digest_attestation


class CheckpointDigestAttestationTests(unittest.TestCase):
    def test_prepared_publish_and_pool_handoff_reuse_verified_owner_digest(self):
        """The same prepared owner crosses writer validation and pool admission."""
        durable_fixtures.DurableRestoreControlTests.setUp(self)
        self.addCleanup(durable_fixtures.DurableRestoreControlTests.tearDown, self)
        self.model.cache[self.sid] = Checkpoint('model','execution',2,b'native-state',
            (1.,2.),Checkpoint.digest(b'native-state',(1.,2.)),
            self.model._input_identity(self.model.history(self.sid)))
        self.backend.serialize = lambda *_: b'native-state'
        prepared = self.model.prepare_checkpoint(self.sid)
        pool = SharedCheckpointPool(1024)

        # Count only in-memory checkpoint digests. File payload validation must
        # continue to hash the actual file and is intentionally outside this hook.
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            pool.admit('prepared', prepared.checkpoint, durable=True)
            self.model.publish_prepared_checkpoint(prepared)
            self.assertEqual(compute.call_count, 0)
        self.assertEqual(pool.get('prepared'), prepared.checkpoint)
        self.assertTrue((self.model.checkpoint_directory / (prepared.checkpoint.checksum + '.state')).exists())
        pool.close()

    def test_direct_constructor_is_hashed_before_it_can_be_attested(self):
        blob, frontier = b'cpu-state', (1.25, -2.5)
        cp = Checkpoint('model', 'profile', 3, blob, frontier,
                        Checkpoint.digest(blob, frontier))
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)

    def test_certified_cache_hit_does_not_rescan_large_immutable_frontier(self):
        cp = Checkpoint.create('model', 'profile', 3, b'cpu-state', tuple(float(i) for i in range(4096)))
        with mock.patch.object(digest_attestation, '_eligible',
                               wraps=digest_attestation._eligible) as eligible:
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(eligible.call_count, 0)
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(eligible.call_count, 0)

    def test_factory_registration_computes_digest_internally(self):
        expected = Checkpoint.digest(b'factory-state', (1.25,))
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            cp = Checkpoint.create('model', 'profile', 3, b'factory-state', (1.25,))
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(cp.checksum, expected)
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)

    def test_public_pool_admission_hashes_unregistered_owner_once(self):
        blob, frontier = b'cpu-state', (1.25, -2.5)
        cp = Checkpoint('model', 'profile', 3, blob, frontier,
                        Checkpoint.digest(blob, frontier))
        pool = SharedCheckpointPool(1024)
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            admitted = pool.admit('direct', cp, durable=True)
            self.assertIs(admitted, cp)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(pool.metrics()['resident_bytes'], len(blob) + 4 * len(frontier))
        pool.close()

    def test_nonweakrefable_legacy_owner_is_accepted_without_memoization(self):
        class SlotsOwner:
            __slots__ = ('blob', 'frontier', 'checksum')

        owner = SlotsOwner()
        owner.blob = b'legacy-slots-state'
        owner.frontier = (1.25, -2.5)
        owner.checksum = Checkpoint.digest(owner.blob, owner.frontier)
        pool = SharedCheckpointPool(1024)

        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertIs(pool.admit('slots-owner', owner, durable=True), owner)
            self.assertEqual(compute.call_count, 2)
        self.assertEqual(pool.metrics()['resident_bytes'], len(owner.blob) + 4 * len(owner.frontier))
        pool.close()

    def test_direct_reservation_still_rejects_corrupt_checksum_and_releases_charge(self):
        cp = Checkpoint('model', 'profile', 1, b'cpu-state', (1.0,), '0' * 64)
        pool = SharedCheckpointPool(128)
        with pool.reserve(len(cp.blob) + 4) as reservation:
            with self.assertRaises(ValueError):
                reservation.admit('bad', cp)
            self.assertEqual(pool.metrics()['resident_bytes'], 0)
            self.assertEqual(pool.metrics()['reserved_bytes'], len(cp.blob) + 4)
        self.assertEqual(pool.metrics()['resident_bytes'], 0)
        self.assertEqual(pool.metrics()['reserved_bytes'], 0)
        pool.close()

    def test_bad_checksum_and_replaced_payload_cannot_reuse_owner_attestation(self):
        cp = Checkpoint.create('model', 'profile', 3, b'cpu-state', (1.25, -2.5))
        changed_blob = replace(cp, blob=b'other-state')
        changed_frontier = replace(cp, frontier=(9.0, -2.5))
        wrong_checksum = replace(cp, checksum='0' * 64)
        changed_blob_digest = Checkpoint.digest(changed_blob.blob, changed_blob.frontier)
        changed_frontier_digest = Checkpoint.digest(changed_frontier.blob, changed_frontier.frontier)

        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(changed_blob), changed_blob_digest)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(checkpoint_digest(changed_frontier), changed_frontier_digest)
            self.assertEqual(compute.call_count, 2)
            pool = SharedCheckpointPool(1024)
            with self.assertRaises(ValueError):
                pool.admit('bad', wrong_checksum)
            self.assertEqual(compute.call_count, 3)
            self.assertEqual(pool.metrics()['resident_bytes'], 0)
            pool.close()

    def test_mutable_frontier_stays_on_full_hash_path(self):
        blob, frontier = b'cpu-state', [1.25, -2.5]
        cp = Checkpoint('model', 'profile', 3, blob, frontier,
                        Checkpoint.digest(blob, frontier))
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 2)

    def test_mutable_numeric_inside_tuple_stays_on_full_hash_path(self):
        class MutableFloat:
            def __init__(self, value): self.value = value
            def __float__(self): return self.value

        value = MutableFloat(1.25)
        blob, frontier = b'cpu-state', (value,)
        cp = Checkpoint('model', 'profile', 3, blob, frontier,
                        Checkpoint.digest(blob, frontier))
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            value.value = 9.0
            self.assertNotEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 2)

    def test_same_owner_blob_replacement_invalidates_and_releases_old_reference(self):
        blob = b'old-state'
        baseline = sys.getrefcount(blob)
        cp = Checkpoint.create('model', 'profile', 1, blob, (1.0,))
        self.assertEqual(sys.getrefcount(blob), baseline + 2)
        object.__setattr__(cp, 'blob', b'new-state')
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertNotEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(sys.getrefcount(blob), baseline)

    def test_same_owner_frontier_and_checksum_replacement_invalidate_entry(self):
        cp = Checkpoint.create('model', 'profile', 1, b'cpu-state', (1.0,))
        old_frontier = cp.frontier
        frontier_baseline = sys.getrefcount(old_frontier)
        object.__setattr__(cp, 'frontier', (2.0,))
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertNotEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(sys.getrefcount(old_frontier), frontier_baseline - 2)

        cp2 = Checkpoint.create('model', 'profile', 1, b'other-state', (1.0,))
        original_blob = cp2.blob
        blob_refs = sys.getrefcount(original_blob)
        object.__setattr__(cp2, 'checksum', '0' * 64)
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertNotEqual(checkpoint_digest(cp2), cp2.checksum)
            self.assertEqual(compute.call_count, 1)
            compute.reset_mock()
        self.assertEqual(sys.getrefcount(original_blob), blob_refs - 1)

    def test_same_owner_malformed_replacement_drops_stale_attestation(self):
        cp = Checkpoint.create('model', 'profile', 1, b'cpu-state', tuple([float('1.0')]))
        old_frontier = cp.frontier
        frontier_baseline = sys.getrefcount(old_frontier)
        object.__setattr__(cp, 'frontier', [1.0])
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(cp), cp.checksum)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(sys.getrefcount(old_frontier), frontier_baseline - 2)

    def test_concurrent_lookup_and_owner_cleanup_are_safe(self):
        cp = Checkpoint.create('model', 'profile', 1, b'cpu-state', (1.0,))
        expected = cp.checksum
        owner = weakref.ref(cp)
        gate = threading.Barrier(9)
        results = []

        def read(checkpoint):
            gate.wait()
            results.append(digest_attestation.verified_checkpoint_digest(checkpoint))

        threads = [threading.Thread(target=read, args=(cp,)) for _ in range(8)]
        for thread in threads: thread.start()
        gate.wait()
        del cp
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        gc.collect()
        self.assertEqual(results, [expected] * 8)
        self.assertIsNone(owner())
        self.assertLessEqual(digest_attestation.attested_owner_count(),
                             digest_attestation.MAX_ATTESTED_OWNERS)

    def test_concurrent_lru_eviction_lookup_and_owner_cleanup(self):
        with digest_attestation._LOCK:
            digest_attestation._OWNERS.clear()
        blob = b'concurrent-eviction-owner'
        blob_baseline = sys.getrefcount(blob)
        owner = Checkpoint.create('model', 'profile', 1, blob, (1.0,))
        expected = owner.checksum
        owner_ref = weakref.ref(owner)
        start = threading.Barrier(5)
        stop = threading.Event()
        errors = []

        def read():
            try:
                start.wait()
                while not stop.is_set():
                    checkpoint = owner_ref()
                    if checkpoint is None:
                        return
                    if digest_attestation.verified_checkpoint_digest(checkpoint) != expected:
                        errors.append('digest changed')
                        return
                    del checkpoint
            except BaseException as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=read) for _ in range(4)]
        for thread in threads:
            thread.start()
        start.wait()
        churn = [Checkpoint.create('model', 'profile', index, bytes([index % 251]) * 8,
                                   (float(index),))
                 for index in range(digest_attestation.MAX_ATTESTED_OWNERS * 2)]
        del owner
        gc.collect()
        stop.set()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        del churn
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertEqual(sys.getrefcount(blob), blob_baseline)

    def test_lru_eviction_falls_back_to_full_hash(self):
        # Isolate global memo state left by earlier tests so these are exactly
        # MAX+1 live owners and the first one must be the eviction victim.
        with digest_attestation._LOCK:
            digest_attestation._OWNERS.clear()
        owners = [Checkpoint.create('model', 'profile', index, bytes([index % 251]) * 8,
                                    tuple([float(index)]))
                  for index in range(digest_attestation.MAX_ATTESTED_OWNERS)]
        oldest_blob = owners[0].blob
        cached_refs = sys.getrefcount(oldest_blob)
        owners.append(Checkpoint.create('model', 'profile', digest_attestation.MAX_ATTESTED_OWNERS,
                                        b'last-ownr', tuple([42.0])))
        self.assertEqual(sys.getrefcount(oldest_blob), cached_refs - 1)
        with mock.patch.object(digest_attestation, '_compute_digest',
                               wraps=digest_attestation._compute_digest) as compute:
            self.assertEqual(checkpoint_digest(owners[0]), owners[0].checksum)
            self.assertEqual(compute.call_count, 1)
            # Mock call_args retains the hash input until reset; exclude that
            # test-harness reference from the cache lifetime assertion.
            compute.reset_mock()
        self.assertEqual(sys.getrefcount(oldest_blob), cached_refs)

    def test_owner_collection_releases_cache_payload_references(self):
        blob = b'x' * 4096
        baseline = sys.getrefcount(blob)
        cp = Checkpoint.create('model', 'profile', 1, blob, (0.5,))
        self.assertEqual(sys.getrefcount(blob), baseline + 2)  # owner + bounded cache entry
        owner = weakref.ref(cp)
        del cp
        gc.collect()
        self.assertIsNone(owner())
        self.assertEqual(sys.getrefcount(blob), baseline)

    def test_cache_is_bounded_when_owners_remain_alive(self):
        owners = [Checkpoint.create('model', 'profile', index, b'x' * 8, (float(index),))
                  for index in range(digest_attestation.MAX_ATTESTED_OWNERS * 2)]
        self.assertLessEqual(digest_attestation.attested_owner_count(),
                             digest_attestation.MAX_ATTESTED_OWNERS)
        self.assertEqual(len(owners), digest_attestation.MAX_ATTESTED_OWNERS * 2)


if __name__ == '__main__':
    unittest.main()
