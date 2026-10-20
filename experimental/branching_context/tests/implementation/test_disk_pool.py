"""Real filesystem pool/retention tests; synthetic checkpoint payloads."""
from pathlib import Path
import tempfile
import unittest
from context_time_travel.store import Store
from context_time_travel.pools import DiskCheckpointPool, PoolFull


class DiskPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.root = self.store.create_root('m',(1,),{'status':'idle'})
        self.tip = self.store.append(self.root,(2,),{'status':'idle'})
        self.store.fork('main',self.tip)
        self.pool = DiskCheckpointPool(self.store,64)

    def test_history_roots_tips_and_saved_points_do_not_permanently_pin_numerical_copies(self):
        later = self.store.append(self.tip,(3,),{'status':'idle'})
        self.store.move('main',0,later,'later')
        self.assertEqual(self.store.checkpoint_roots(),set())
        self.assertTrue({self.root,later} <= self.store.retained_snapshots())
        self.store.retain('fork-point',self.tip)
        self.assertEqual(self.store.checkpoint_roots(),set())
        self.assertTrue({self.root,self.tip,later} <= self.store.retained_snapshots())

    def test_read_lease_is_a_disk_eviction_pin(self):
        later = self.store.append(self.tip,(3,),{'status':'idle'})
        self.store.move('main',0,later,'later')
        with self.store.read_lease(self.tip): self.assertIn(self.tip,self.store.checkpoint_roots())
        self.assertNotIn(self.tip,self.store.checkpoint_roots())

    def test_orphan_payload_collection_never_deletes_canonical_history(self):
        file = self.pool.directory/('0'*64+'.state');file.write_bytes(b'x'*60)
        with self.pool.reserve(32):
            self.assertFalse(file.exists())
            self.assertEqual(self.pool.metrics()['reserved_bytes'],32)
        self.assertEqual(self.store.get_snapshot(self.tip)['inputs'],[2])

    def test_failed_reservation_releases_and_does_not_exceed_limit(self):
        with self.assertRaises(RuntimeError):
            with self.pool.reserve(64):
                with self.assertRaises(PoolFull): self.pool.reserve(1)
                raise RuntimeError('write failed')
        self.assertEqual(self.pool.metrics()['reserved_bytes'],0)


if __name__ == '__main__': unittest.main()
