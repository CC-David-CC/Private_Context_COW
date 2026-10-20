"""VC02 image embeddings share the existing RAM budget and lease machinery."""
from dataclasses import replace
import hashlib
import struct
import threading
import unittest

from context_time_travel import IntegrityError
from context_time_travel.pools import PoolFull, SharedCheckpointPool
from context_time_travel.real import Checkpoint
from context_time_travel.vision_embeddings import parse_sve


def embedding(value=0.25):
    raw=struct.pack('<5i4f',0x31455653,2,2,1,2,*([value]*4))
    return parse_sve(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),expected_width=2)


def checkpoint():
    raw=b'abcd'
    return Checkpoint('m','e',1,raw,(1.,),Checkpoint.digest(raw,(1.,)),'s')


class VisionRamPoolTests(unittest.TestCase):
    def pool(self, size=44, **options):
        pool=SharedCheckpointPool(size,**options)
        self.addCleanup(pool.close)
        return pool

    def test_vc006_embedding_and_checkpoint_use_one_budget_and_lru(self):
        pool=self.pool()
        pool.admit('image',embedding())  # 36 bytes, no disk copy claimed.
        pool.admit('state',checkpoint(),durable=True)  # 8 bytes.
        self.assertEqual(pool.metrics()['resident_bytes'],44)
        self.assertFalse(pool.is_durable('image'))
        pool.admit('next-image',embedding(0.5))
        self.assertNotIn('image',pool)
        self.assertIn('state',pool)
        self.assertEqual(pool.metrics()['resident_bytes'],44)

    def test_vc006_last_ram_embedding_can_expire_without_claiming_disk_durability(self):
        clock=[0.]
        pool=self.pool(checkpoint_idle_seconds=1800,clock=lambda:clock[0])
        pool.admit('image',embedding())
        clock[0]=1800.
        self.assertEqual(pool.collect_expired(),0)
        clock[0]=1800.001
        self.assertEqual(pool.collect_expired(),1)
        self.assertEqual(pool.metrics()['resident_bytes'],0)

    def test_vc006_active_embedding_lease_prevents_expiry_and_pressure_eviction(self):
        clock=[0.]
        pool=self.pool(36,checkpoint_idle_seconds=10,clock=lambda:clock[0])
        pool.admit('image',embedding())
        with pool.pin('image') as held:
            clock[0]=11.
            self.assertEqual(pool.collect_expired(),0)
            with self.assertRaises(PoolFull):
                pool.admit('other',embedding(0.5))
            self.assertEqual(held.sha256,embedding().sha256)
        self.assertEqual(pool.collect_expired(),0)  # Releasing counts as a use.
        clock[0]=22.
        self.assertEqual(pool.collect_expired(),1)

    def test_vc006_embedding_admission_cannot_evict_unpublished_checkpoint(self):
        pool=self.pool(36)
        pool.admit('dirty',checkpoint())
        with self.assertRaises(PoolFull):
            pool.admit('image',embedding())
        self.assertIn('dirty',pool)
        pool.mark_durable('dirty')
        pool.admit('image',embedding())
        self.assertNotIn('dirty',pool)

    def test_vc006_identical_embedding_aliases_share_one_immutable_payload(self):
        pool=self.pool(36)
        pool.admit('one',embedding())
        pool.admit('two',embedding())
        self.assertEqual(pool.metrics()['unique_payloads'],1)
        self.assertEqual(pool.metrics()['resident_bytes'],36)
        self.assertIs(pool.get('one').payload,pool.get('two').payload)
        pool.remove('one')
        self.assertEqual(pool.get('two').data.tobytes(),embedding().data.tobytes())

    def test_vc006_forged_embedding_metadata_or_bytes_rejected_before_admission(self):
        pool=self.pool()
        value=embedding()
        for changed in (replace(value,rows=1), replace(value,sha256='0'*64),
                        replace(value,payload=value.payload[:-1])):
            with self.subTest(changed=changed):
                with self.assertRaises(IntegrityError):
                    pool.admit('bad',changed)
                self.assertEqual(pool.metrics()['resident_bytes'],0)
                self.assertEqual(pool.metrics()['reserved_bytes'],0)

    def test_vc006_coalesced_embedding_load_reserves_real_bytes_once(self):
        pool=self.pool(36)
        entered,release=threading.Event(),threading.Event()
        calls=[]
        def loader():
            calls.append(1)
            entered.set()
            if not release.wait(5):raise TimeoutError('test loader not released')
            return embedding()
        one=pool.preload('image',36,loader)
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(pool.metrics()['reserved_bytes'],36)
            two=pool.preload('image',36,loader)
            self.assertIs(one,two)
        finally:
            release.set()
        self.assertEqual(one.result(timeout=5).nbytes,36)
        self.assertEqual(calls,[1])
        self.assertEqual(pool.metrics()['resident_bytes'],36)
        self.assertEqual(pool.metrics()['reserved_bytes'],0)

    def test_vc006_pool_close_accounts_active_image_until_lease_release(self):
        pool=self.pool(36)
        pool.admit('image',embedding())
        with pool.pin('image'):
            pool.close()
            self.assertEqual(pool.metrics()['resident_bytes'],36)
            self.assertEqual(pool.metrics()['pins'],1)
        self.assertEqual(pool.metrics()['resident_bytes'],0)
        self.assertEqual(pool.metrics()['pins'],0)


if __name__=='__main__':
    unittest.main()
