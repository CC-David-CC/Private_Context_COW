"""CPU accounting controls for root reuse; no model or GPU qualification."""
import unittest

from context_time_travel.pools import checkpoint_bytes, PoolFull
import test_root_computation_reuse as fixtures


class RootRestoreReservationTests(unittest.TestCase):
    setUp = fixtures.RootComputationReuseTests.setUp
    create = fixtures.RootComputationReuseTests.create
    wait_for_pool_release = fixtures.RootComputationReuseTests.wait_for_pool_release

    def check_restore(self, *, disk):
        prefix = (1, 4, 8, 12)
        first = self.create('a', prefix, 'create-a')
        self.wait_for_pool_release()
        model = self.coordinator._workers[0]['model']
        backend = self.backends[0]
        key = self.coordinator._key(model, first['snapshot'])
        cold = self.coordinator.pool.get(key)
        required = 3 * checkpoint_bytes(cold) + 12 * cold.consumed
        model.cache.clear()
        if disk:
            self.coordinator.pool.remove(key)
        observed = []
        original = backend.restore

        def restore(*args, **kwargs):
            observed.append(self.coordinator.pool.metrics())
            return original(*args, **kwargs)

        backend.restore = restore
        second = self.create('b', prefix, 'create-b')
        self.assertEqual(second['snapshot'], first['snapshot'])
        self.assertTrue(observed, 'control did not enter native restoration')
        for metrics in observed:
            self.assertGreaterEqual(metrics['reserved_bytes'], required,
                'source pin alone does not account for native restore scratch')
            self.assertGreater(metrics['pins'], 0)
        self.assertEqual(backend.stats['forward_tokens'], len(prefix))
        self.wait_for_pool_release()

    def test_pool_hit_reserves_transfer_through_native_restore(self):
        self.check_restore(disk=False)

    def test_disk_hit_reserves_transfer_after_preload_admission(self):
        self.check_restore(disk=True)

    def test_insufficient_restore_capacity_fails_before_backend_activation(self):
        prefix = (1, 4, 8, 12)
        first = self.create('a', prefix, 'create-a')
        self.wait_for_pool_release()
        coordinator = self.coordinator
        model = coordinator._workers[0]['model']
        key = coordinator._key(model, first['snapshot'])
        cold = coordinator.pool.get(key)
        model.cache.clear()
        # Enough for the existing capture reservation, but insufficient for
        # the larger restore scratch extent. A rejection during capture alone
        # would not demonstrate the missing before-activation admission.
        coordinator.pool.byte_limit = (checkpoint_bytes(cold)
            + 3 * len(cold.blob) + 8 * len(cold.frontier))
        coordinator._admission_timeout = .05
        entered = []
        original = self.backends[0].restore

        def restore(*args, **kwargs):
            entered.append(True)
            return original(*args, **kwargs)

        self.backends[0].restore = restore
        request = coordinator.create('b', prefix, 'create-b')
        with self.assertRaises(PoolFull):
            request.durable.result(5)
        self.assertEqual(entered, [], 'capacity rejection occurred after native activation')
        self.wait_for_pool_release()


if __name__ == '__main__':
    unittest.main()
