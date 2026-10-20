"""Pool control-plane tests. These do not qualify GPU computation."""
from dataclasses import replace
import tempfile
import threading
import unittest

from context_time_travel.real import Checkpoint
from context_time_travel.pools import PoolConfig, SharedCheckpointPool, PoolFull


def image(blob=b'abcd', snapshot='a'):
    return Checkpoint('m', 'e', 1, blob, (1.,), Checkpoint.digest(blob, (1.,)), snapshot)


class SharedPoolTests(unittest.TestCase):
    def test_gpu_list_is_the_only_worker_count(self):
        self.assertEqual(PoolConfig(gpus=(1,5,6)).worker_count, 3)
        self.assertEqual(PoolConfig(gpus=(0,)).gpus, (0,))
        for ids in ((), (1,1), (-1,), (True,), ('1',)):
            with self.assertRaises(ValueError): PoolConfig(gpus=ids)

    def test_groups_are_one_worker_and_physical_devices_cannot_overlap(self):
        grouped = PoolConfig(gpus=((0,1,2),5,6))
        self.assertEqual(grouped.worker_count,3)
        self.assertEqual(grouped.physical_gpus,(0,1,2,5,6))
        self.assertEqual(PoolConfig(gpus=[[0,1,2]]).worker_count,1)
        for ids in (((0,1),1),((0,0),),([],),((True,1),),((0,(1,2)),)):
            with self.assertRaises(ValueError): PoolConfig(gpus=ids)

    def test_empty_and_nonfinite_frontiers_cannot_enter_shared_pool(self):
        pool = SharedCheckpointPool(64)
        for cp in (image(b''),replace(image(),frontier=()),replace(image(),frontier=(float('nan'),))):
            with self.assertRaises(ValueError): pool.admit('a',cp)
        self.assertEqual(pool.metrics()['resident_bytes'],0)

    def test_clean_lru_eviction_and_pin(self):
        pool = SharedCheckpointPool(16)
        pool.admit('a', image(), durable=True)
        pool.admit('b', image(b'efgh'), durable=True)
        with pool.pin('a'):
            pool.admit('c', image(b'ijkl'), durable=True)
            self.assertNotIn('b', pool)
            self.assertIn('a', pool)
        self.assertEqual(pool.metrics()['resident_bytes'],16)
        self.assertEqual(pool.metrics()['clean_evictions'],1)

    def test_dirty_images_are_never_automatic_victims(self):
        pool = SharedCheckpointPool(8)
        pool.admit('a', image(), durable=False)
        with self.assertRaises(PoolFull): pool.reserve(8)
        self.assertIn('a',pool)
        pool.mark_durable('a')
        with pool.reserve(8): pass
        self.assertNotIn('a',pool)

    def test_payload_aliases_count_once_and_remain_after_one_alias_evicted(self):
        pool = SharedCheckpointPool(16)
        pool.admit('a', image(), durable=True)
        pool.admit('alias', replace(image(),input_identity='other'), durable=True)
        self.assertEqual(pool.metrics()['resident_bytes'],8)
        self.assertIs(pool.get('a').blob,pool.get('alias').blob)
        pool.remove('a')
        self.assertEqual(pool.metrics()['resident_bytes'],8)
        self.assertEqual(pool.get('alias').frontier,(1.,))

    def test_reservation_is_charged_before_allocation_and_released_on_error(self):
        pool = SharedCheckpointPool(16)
        with self.assertRaisesRegex(RuntimeError,'allocation'):
            with pool.reserve(12):
                self.assertEqual(pool.metrics()['reserved_bytes'],12)
                with self.assertRaises(PoolFull): pool.reserve(8)
                raise RuntimeError('allocation failed')
        self.assertEqual(pool.metrics()['reserved_bytes'],0)
        with pool.reserve(16) as reservation:
            reservation.admit('a',image(),durable=True)
        self.assertEqual(pool.metrics()['resident_bytes'],8)

    def test_failed_admission_leaves_existing_value(self):
        pool = SharedCheckpointPool(8)
        old = image()
        pool.admit('a',old,durable=False)
        with self.assertRaises(PoolFull): pool.admit('a',image(b'x'*9))
        self.assertIs(pool.get('a'),old)

    def test_invalid_payload_rejected_without_changing_pool(self):
        pool = SharedCheckpointPool(16)
        with self.assertRaises(ValueError): pool.admit('bad',replace(image(),checksum='bad'))
        self.assertEqual(pool.metrics()['resident_bytes'],0)

    def test_coalesced_load_and_cancelled_waiter_do_not_duplicate_payload(self):
        pool = SharedCheckpointPool(16)
        started,release = threading.Event(),threading.Event()
        calls = []
        def load():
            calls.append(1);started.set();release.wait(5);return image()
        one = pool.preload('a',8,load)
        self.assertTrue(started.wait(2))
        two = pool.preload('a',8,load)
        self.assertIs(one,two)
        release.set()
        self.assertEqual(one.result(5).blob,b'abcd')
        self.assertEqual(len(calls),1)
        self.assertEqual(pool.metrics()['reserved_bytes'],0)
        pool.close()

    def test_failed_load_releases_reservation_and_allows_retry(self):
        pool = SharedCheckpointPool(8)
        def fail(): raise OSError('disk read failed')
        with self.assertRaises(OSError): pool.preload('a',8,fail).result(5)
        self.assertEqual(pool.metrics()['reserved_bytes'],0)
        self.assertEqual(pool.preload('a',8,lambda:image()).result(5).blob,b'abcd')
        pool.close()

    def test_foreground_load_waits_for_capacity_and_coalesces_waiters(self):
        pool=SharedCheckpointPool(16)
        held=pool.reserve(16)
        calls=[]
        one=pool.preload('a',8,lambda:(calls.append(1) or image()),timeout=2)
        two=pool.preload('a',8,lambda:(calls.append(2) or image()),timeout=2)
        self.assertIs(one,two)
        held.release()
        self.assertEqual(one.result(3).blob,b'abcd')
        self.assertEqual(calls,[1]);self.assertEqual(pool.metrics()['reserved_bytes'],0)
        pool.close()


if __name__ == '__main__': unittest.main()
