"""VC02 real filesystem/lock tests; payloads are small synthetic bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from context_time_travel import Store
from context_time_travel.pools import DiskCheckpointPool, PoolFull


class VisionDiskPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name)
        self.clock=[10_000_000_000]
        self.pool=DiskCheckpointPool(self.store,64,checkpoint_idle_seconds=10,
                                     clock_ns=lambda:self.clock[0])

    def put(self,raw=b'x'*48):
        digest=hashlib.sha256(raw).hexdigest()
        path=self.pool.directory/(digest+'.sve')
        path.write_bytes(raw)
        os.utime(path,ns=(self.clock[0],self.clock[0]))
        return digest,path

    def test_vc006_embedding_counts_toward_existing_budget_and_can_be_evicted(self):
        digest,path=self.put()
        self.assertEqual(self.pool.metrics()['file_bytes'],48)
        with self.pool.reserve(32):
            self.assertFalse(path.exists())
            self.assertEqual(self.pool.metrics()['reserved_bytes'],32)

    def test_vc006_zero_byte_digest_lease_protects_an_embedding_from_another_pool(self):
        digest,path=self.put()
        other=DiskCheckpointPool(Store(self.tmp.name),64,checkpoint_idle_seconds=10,
                                 clock_ns=lambda:self.clock[0])
        with self.pool.reserve(0,payload_sha256=digest):
            self.clock[0]+=11_000_000_000
            self.assertEqual(other.collect_expired(),0)
            with self.assertRaises(PoolFull):
                with other.reserve(32):pass
            self.assertTrue(path.exists())
        self.assertEqual(other.collect_expired(),1)
        self.assertFalse(path.exists())

    def test_vc006_embedding_touch_persists_idle_age_across_pool_restart(self):
        digest,path=self.put()
        self.clock[0]+=9_000_000_000
        self.pool.touch_embedding(digest)
        reopened=DiskCheckpointPool(Store(self.tmp.name),64,checkpoint_idle_seconds=10,
                                    clock_ns=lambda:self.clock[0])
        self.clock[0]+=9_000_000_000
        self.assertEqual(reopened.collect_expired(),0)
        self.clock[0]+=2_000_000_000
        self.assertEqual(reopened.collect_expired(),1)
        self.assertFalse(path.exists())

    def test_vc006_checkpoint_and_embedding_share_one_recency_order(self):
        root=self.store.create_root('text',[1],{})
        digest=hashlib.sha256(b'state').hexdigest()
        manifest=self.pool.directory/(root+'.json')
        manifest.write_text(json.dumps(dict(manifest=dict(payload_sha256=digest))))
        (self.pool.directory/(digest+'.state')).write_bytes(b'state')
        os.utime(manifest,ns=(20_000_000_000,20_000_000_000))
        image,path=self.put(b'x'*32)
        used=self.pool.metrics()['file_bytes']
        pool=DiskCheckpointPool(self.store,used)
        with pool.reserve(32):
            self.assertFalse(path.exists())
            self.assertTrue(manifest.exists())

    def test_vc006_process_death_releases_embedding_eviction_pin(self):
        digest,path=self.put()
        script='''
import sys
from context_time_travel import Store
from context_time_travel.pools import DiskCheckpointPool
pool=DiskCheckpointPool(Store(sys.argv[1]),64)
with pool.reserve(0,payload_sha256=sys.argv[2]):
    print('READY',flush=True)
    sys.stdin.readline()
'''
        proc=subprocess.Popen([sys.executable,'-B','-u','-c',script,self.tmp.name,digest],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(),'READY')
            self.clock[0]+=11_000_000_000
            self.assertEqual(self.pool.collect_expired(),0)
            proc.kill()
            proc.wait(timeout=5)
            self.assertEqual(self.pool.collect_expired(),1)
            self.assertFalse(path.exists())
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=5)
            for stream in (proc.stdin,proc.stdout,proc.stderr):stream.close()


if __name__=='__main__':unittest.main()
