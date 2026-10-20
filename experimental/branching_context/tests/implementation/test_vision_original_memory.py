"""Original read allocation/admission; real files/pools, synthetic embeddings."""
from contextlib import contextmanager
from concurrent.futures import CancelledError
import os
import struct
import tempfile
import unittest
from unittest import mock

from context_time_travel.pools import DiskCheckpointPool,PoolFull,SharedCheckpointPool
from context_time_travel.store import IntegrityError,Store
import context_time_travel.vision_artifacts as module


class VisionOriginalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.store=Store(self.temp.name)
        self.disk=DiskCheckpointPool(self.store,4*1024**2);self.pool=SharedCheckpointPool(2*1024**2)
        self.addCleanup(self.pool.close)
        self.original=bytes(range(256))*2048;self.normalized=b'normalized'
        self.payload=struct.pack('<5i4f',0x31455653,2,2,1,2,*([.25]*4))
        self.owner=self.make_owner()
        with self.owner.prepare(original=self.original,normalized=self.normalized,encode=lambda _:self.payload,pad_token_id=7) as held:
            self.span=held.span
        self.root=self.store.create_input_root('memory-fixture',[self.span],{})
        self.path=self.owner.originals/(self.span['original_sha256']+'.input');self.reads=[]

    def make_owner(self):
        return module.ImageArtifacts(self.store,self.pool,self.disk,encoder_profile_id='a'*64,
            expected_width=2,original_bytes_limit=2*1024**2)

    def small_pool(self,size=1024,*,embedding=True):
        self.pool.close();self.pool=SharedCheckpointPool(size);self.addCleanup(self.pool.close);self.owner=self.make_owner()
        if not embedding:
            for path in self.disk.directory.glob('*.sve'):path.unlink()

    @contextmanager
    def watched_reads(self,hook=None):
        fdopen=module.os.fdopen;case=self
        class Stream:
            def __init__(self,stream):self.stream=stream
            def __getattr__(self,key):return getattr(self.stream,key)
            def __enter__(self):self.stream.__enter__();return self
            def __exit__(self,*args):return self.stream.__exit__(*args)
            def read(self,size=-1):
                if hook is not None:hook()
                metrics=case.pool.metrics();case.reads.append(dict(requested=size,reserved=metrics['reserved_bytes']))
                case.assertGreaterEqual(size,0,'Original reads must be bounded before allocation')
                case.assertGreaterEqual(metrics['reserved_bytes'],size,'Original read allocation preceded RAM admission')
                return self.stream.read(size)
        def tracked(fd,*args,**options):
            original=os.readlink('/proc/self/fd/'+str(fd)).removesuffix(' (deleted)')==str(self.path)
            stream=fdopen(fd,*args,**options)
            return Stream(stream) if original else stream
        with mock.patch.object(module.os,'fdopen',tracked):yield

    def assert_released(self):
        self.assertEqual(self.pool.metrics()['reserved_bytes'],0)
        self.assertEqual(self.pool.metrics()['pins'],0)
        self.assertLessEqual(self.pool.metrics()['peak_accounted_bytes'],self.pool.byte_limit)

    def test_large_cached_original_is_hashed_in_admitted_chunks_without_retaining_it(self):
        self.small_pool()
        with self.watched_reads(),self.owner.lease(self.span) as held:
            self.assertEqual(held.embedding.payload,self.payload)
            self.assertEqual(self.pool.metrics()['reserved_bytes'],0)
        self.assertTrue(self.reads);self.assertLessEqual(max(row['requested'] for row in self.reads),self.pool.byte_limit)
        self.assert_released()

    def test_regeneration_requires_whole_original_admission_before_callback(self):
        self.small_pool(embedding=False);calls=[]
        with self.watched_reads(),self.assertRaises(PoolFull):
            with self.owner.lease(self.span,reencode=lambda raw:(calls.append(raw),self.payload)):pass
        self.assertEqual(calls,[]);self.assert_released()

    def test_regeneration_holds_original_reservation_through_callback_then_releases(self):
        self.small_pool(1024**2,embedding=False);seen=[]
        def encode(raw):
            seen.append(self.pool.metrics()['reserved_bytes'])
            self.assertEqual(raw,self.original);self.assertGreaterEqual(seen[-1],len(raw))
            return self.normalized,self.payload
        with self.watched_reads(),self.owner.lease(self.span,reencode=encode) as held:
            self.assertEqual(held.source,'reencoded');self.assertEqual(self.pool.metrics()['reserved_bytes'],0)
        self.assertEqual(len(seen),1);self.assert_released()

    def test_two_active_cached_leases_share_budget_with_hash_scratch(self):
        self.small_pool(128)
        with self.watched_reads(),self.owner.lease(self.span) as first,self.owner.lease(self.span) as second:
            self.assertEqual(first.embedding.sha256,second.embedding.sha256)
            self.assertEqual(self.pool.metrics()['reserved_bytes'],0)
        self.assert_released()

    def test_original_corruption_fails_even_when_embedding_is_cached(self):
        self.small_pool();raw=self.original[:-1]+bytes([self.original[-1]^1]);self.path.write_bytes(raw)
        with self.watched_reads(),self.assertRaises(IntegrityError):
            with self.owner.lease(self.span):pass
        self.assert_released()

    def test_growth_after_admission_cannot_trigger_a_larger_read(self):
        self.small_pool();changed=[]
        def grow():
            if not changed:
                changed.append(True)
                with self.path.open('ab') as stream:stream.write(b'growth')
        with self.watched_reads(grow),self.assertRaises(IntegrityError):
            with self.owner.lease(self.span):pass
        self.assertLessEqual(max(row['requested'] for row in self.reads),self.pool.byte_limit);self.assert_released()

    def test_original_truncation_releases_hash_reservation(self):
        self.small_pool();changed=[]
        def truncate():
            if not changed:changed.append(True);self.path.write_bytes(b'truncated')
        with self.watched_reads(truncate),self.assertRaises(IntegrityError):
            with self.owner.lease(self.span):pass
        self.assert_released()

    def test_symlink_original_is_rejected_without_reading_or_reserving(self):
        self.small_pool();other=self.path.with_suffix('.other');other.write_bytes(self.original)
        self.path.unlink();self.path.symlink_to(other)
        with self.watched_reads(),self.assertRaises(IntegrityError):
            with self.owner.lease(self.span):pass
        self.assertEqual(self.reads,[]);self.assert_released()

    def test_caller_and_encoder_errors_release_read_reservations_and_pins(self):
        self.small_pool()
        with self.watched_reads(),self.assertRaisesRegex(RuntimeError,'caller'):
            with self.owner.lease(self.span):raise RuntimeError('caller')
        self.assert_released();self.small_pool(1024**2,embedding=False)
        def fail(_):raise RuntimeError('encoder')
        with self.watched_reads(),self.assertRaisesRegex(RuntimeError,'encoder'):
            with self.owner.lease(self.span,reencode=fail):pass
        self.assert_released()

    def test_cancelled_encoder_releases_original_reservation(self):
        self.small_pool(1024**2,embedding=False)
        def cancel(_):raise CancelledError('encoder cancelled')
        with self.watched_reads(),self.assertRaises(CancelledError):
            with self.owner.lease(self.span,reencode=cancel):pass
        self.assert_released()

    def test_fully_reserved_pool_refuses_before_any_original_read(self):
        self.small_pool()
        with self.pool.reserve(self.pool.byte_limit),self.watched_reads(),self.assertRaises(PoolFull):
            with self.owner.lease(self.span):pass
        self.assertEqual(self.reads,[]);self.assert_released()

    def test_fdopen_failure_does_not_leak_owned_descriptor(self):
        self.small_pool();opened=[];fdopen=module.os.fdopen
        def fail(fd,*args,**options):
            if os.readlink('/proc/self/fd/'+str(fd))==str(self.path):
                opened.append(fd);raise MemoryError('synthetic file-wrapper allocation failure')
            return fdopen(fd,*args,**options)
        with mock.patch.object(module.os,'fdopen',fail),self.assertRaises(MemoryError):
            with self.owner.lease(self.span):pass
        leaked=[]
        for fd in opened:
            try:os.fstat(fd)
            except OSError:continue
            leaked.append(fd);os.close(fd)
        self.assertEqual(len(opened),1);self.assertEqual(leaked,[]);self.assert_released()

    def test_encoder_file_error_is_not_reported_as_a_missing_original(self):
        self.small_pool(1024**2,embedding=False)
        def fail(_):raise FileNotFoundError('encoder executable disappeared')
        with self.assertRaisesRegex(FileNotFoundError,'encoder executable disappeared'):
            with self.owner.lease(self.span,reencode=fail):pass
        self.assert_released()


if __name__=='__main__':unittest.main()
