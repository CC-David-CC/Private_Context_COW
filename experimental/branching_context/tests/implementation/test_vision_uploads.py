"""Namespace/expiry ownership of original bytes; no image inference is claimed."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from context_time_travel.pools import SharedCheckpointPool, DiskCheckpointPool, PoolFull
from context_time_travel.store import Store, IntegrityError, InvalidOperation, NotFound
from context_time_travel.vision_artifacts import ImageArtifacts


class ImageUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name);self.ram=SharedCheckpointPool(1024**2);self.addCleanup(self.ram.close)
        self.disk=DiskCheckpointPool(self.store,1024**2);self.clock=[time.time_ns()]
        self.a=self.store.create_namespace('A','namespace-A').namespace_id
        self.b=self.store.create_namespace('B','namespace-B').namespace_id
        self.images=self.artifacts()

    def artifacts(self,**options):
        return ImageArtifacts(self.store,self.ram,self.disk,encoder_profile_id='a'*64,expected_width=4,
            original_bytes_limit=options.pop('original_bytes_limit',1024**2),
            upload_idle_seconds=30,clock_ns=lambda:self.clock[0],**options)

    def test_identical_uploads_share_bytes_but_require_namespace_grant(self):
        first=self.images.upload(self.a,b'original bytes');second=self.images.upload(self.a,b'original bytes')
        self.assertEqual(first,second);self.assertEqual(first['image_id'],hashlib.sha256(b'original bytes').hexdigest())
        with self.assertRaises(NotFound):
            with self.images.lease_uploaded(self.b,first['image_id']):self.fail('Wrong namespace was admitted')
        self.images.upload(self.b,b'original bytes')
        self.assertEqual(len(list(self.images.originals.glob('*.input'))),1)
        with self.images.lease_uploaded(self.b,first['image_id']) as grant:self.assertEqual(grant['bytes'],14)

    def test_new_store_process_view_resolves_the_same_upload(self):
        first=self.images.upload(self.a,b'original bytes')
        owner=ImageArtifacts(Store(self.temp.name),self.ram,self.disk,encoder_profile_id='a'*64,
            expected_width=4,original_bytes_limit=1024**2,upload_idle_seconds=30,clock_ns=lambda:self.clock[0])
        with owner.lease_uploaded(self.a,first['image_id']):
            self.assertEqual(owner._original(first['image_id']),b'original bytes')

    def test_unexpired_upload_survives_collection_then_expires(self):
        upload=self.images.upload(self.a,b'original bytes');identity=upload['image_id']
        self.images.collect();self.assertTrue((self.images.originals/(identity+'.input')).exists())
        self.clock[0]+=31*10**9;self.images.collect()
        self.assertFalse((self.images.originals/(identity+'.input')).exists())
        with self.assertRaises(NotFound):
            with self.images.lease_uploaded(self.a,identity):self.fail('Expired upload was admitted')

    def test_active_read_survives_expiry_until_release(self):
        upload=self.images.upload(self.a,b'original bytes');identity=upload['image_id']
        with self.images.lease_uploaded(self.a,identity):
            self.clock[0]+=31*10**9;self.images.collect()
            self.assertEqual(self.images._original(identity),b'original bytes')
        self.images.collect();self.assertFalse((self.images.originals/(identity+'.input')).exists())

    def test_canonical_history_retains_original_after_upload_expiry(self):
        upload=self.images.upload(self.a,b'original bytes')
        from test_vision_execution import spans
        image=spans(original_sha256=upload['image_id'])[1]
        self.store.create_input_root('model',[image],{})
        self.clock[0]+=31*10**9;self.images.collect()
        self.assertEqual(self.images._original(upload['image_id']),b'original bytes')

    def test_refreshed_upload_moves_expiry_without_changing_identity(self):
        first=self.images.upload(self.a,b'original bytes');self.clock[0]+=20*10**9
        again=self.images.upload(self.a,b'original bytes')
        self.assertEqual(first['image_id'],again['image_id']);self.assertGreater(again['expires_ns'],first['expires_ns'])
        self.clock[0]+=20*10**9;self.images.collect()
        with self.images.lease_uploaded(self.a,first['image_id']):pass

    def test_corrupt_grant_fails_closed_without_collecting_originals(self):
        upload=self.images.upload(self.a,b'original bytes')
        path=next(self.images.upload_grants.glob('*.json'));path.write_text('{}')
        with self.assertRaises(IntegrityError):
            with self.images.lease_uploaded(self.a,upload['image_id']):self.fail('Corruption was admitted')
        with self.assertRaises(IntegrityError):self.images.collect()
        self.assertTrue((self.images.originals/(upload['image_id']+'.input')).exists())

    def test_invalid_namespace_and_capacity_refuse_without_grant(self):
        with self.assertRaises(NotFound):self.images.upload('f'*32,b'bytes')
        small=self.artifacts(original_bytes_limit=1)
        with self.assertRaises(PoolFull):small.upload(self.a,b'too large')
        self.assertEqual(list(self.images.upload_grants.glob('*.json')),[])

    def test_dangling_grant_symlink_is_rejected_instead_of_repaired(self):
        self.images.upload(self.a,b'original bytes');path=next(self.images.upload_grants.glob('*.json'))
        path.unlink();path.symlink_to(self.store.root/'missing-grant')
        with self.assertRaises(IntegrityError):self.images.upload(self.a,b'original bytes')
        self.assertTrue(path.is_symlink())


if __name__=='__main__':unittest.main()
