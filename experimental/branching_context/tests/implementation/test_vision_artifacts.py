"""VC02 durable byte ownership; synthetic encoder callbacks are not GPU evidence."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest

from context_time_travel import IntegrityError, Store
from context_time_travel.pools import DiskCheckpointPool, PoolFull, SharedCheckpointPool


def encoded(value=0.25):
    return struct.pack('<5i4f',0x31455653,2,2,1,2,*([value]*4))


class VisionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name)
        self.ram=SharedCheckpointPool(1024**2,checkpoint_idle_seconds=10)
        self.addCleanup(self.ram.close)
        self.clock=[time.time_ns()]
        self.disk=DiskCheckpointPool(self.store,1024**2,checkpoint_idle_seconds=10,
                                     clock_ns=lambda:self.clock[0])
        self.calls=[]

    def artifacts(self,**options):
        from context_time_travel.vision_artifacts import ImageArtifacts
        settings=dict(encoder_profile_id='a'*64,expected_width=2,original_bytes_limit=1024**2)
        settings.update(options)
        return ImageArtifacts(self.store,self.ram,self.disk,**settings)

    def encoder(self,normalized):
        self.calls.append(normalized)
        return encoded()

    def prepare(self,images,**changes):
        args=dict(original=b'original-image',normalized=b'normalized-image',
                  encode=self.encoder,pad_token_id=123)
        args.update(changes)
        return images.prepare(**args)

    def test_vc006_identical_upload_reuses_artifact_without_an_encoder_call(self):
        images=self.artifacts()
        with self.prepare(images) as first:
            span=first.span
            root=self.store.create_input_root('mixed',[span],{})
        with self.prepare(images,encode=None) as second:
            self.assertEqual(second.span,span)
            self.assertEqual(second.source,'ram')
            self.assertEqual(second.embedding.payload,encoded())
        self.assertEqual(self.calls,[b'normalized-image'])
        self.assertEqual(self.store.input_lineage(root)[1],[span])
        self.assertEqual(len(list(images.originals.glob('*.input'))),1)

    def test_vc006_clean_process_restart_loads_disk_without_encoder(self):
        images=self.artifacts()
        with self.prepare(images) as first:span=first.span
        script='''
import json,sys
from context_time_travel import Store
from context_time_travel.pools import SharedCheckpointPool,DiskCheckpointPool
from context_time_travel.vision_artifacts import ImageArtifacts
store=Store(sys.argv[1]);ram=SharedCheckpointPool(1024**2)
images=ImageArtifacts(store,ram,DiskCheckpointPool(store,1024**2),
    encoder_profile_id='a'*64,expected_width=2,original_bytes_limit=1024**2)
with images.lease(json.loads(sys.argv[2])) as value:
    print(json.dumps(dict(source=value.source,sha256=value.embedding.sha256)))
ram.close()
'''
        result=subprocess.run([sys.executable,'-B','-c',script,self.tmp.name,json.dumps(span)],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(source='disk',sha256=span['embedding_sha256']))

    def test_vc006_last_embedding_expires_but_original_and_expected_identity_survive(self):
        images=self.artifacts()
        with self.prepare(images) as first:
            span=first.span
            root=self.store.create_input_root('mixed',[span],{})
        for key in list(self.ram._entries):self.ram.remove(key)
        self.clock[0]+=11_000_000_000
        self.assertEqual(self.disk.collect_expired(),1)
        self.assertEqual(list(self.disk.directory.glob('*.sve')),[])
        self.assertTrue((images.originals/(span['original_sha256']+'.input')).exists())
        self.assertEqual(images.collect()['originals_deleted'],[])
        before=self.store.get_snapshot(root)
        seen=[]
        def reencode(original):
            seen.append(original)
            return b'normalized-image',encoded()
        with images.lease(span,reencode=reencode) as restored:
            self.assertEqual(restored.source,'reencoded')
            self.assertEqual(restored.embedding.sha256,span['embedding_sha256'])
        self.assertEqual(seen,[b'original-image'])
        self.assertEqual(self.store.get_snapshot(root),before)

    def test_vc006_reencoding_mismatch_never_rewrites_expected_input_or_branch(self):
        images=self.artifacts()
        with self.prepare(images) as first:
            span=first.span
            root=self.store.create_input_root('mixed',[span],{})
            head=self.store.fork('main',root)
        for key in list(self.ram._entries):self.ram.remove(key)
        self.clock[0]+=11_000_000_000
        self.disk.collect_expired()
        for normalized,payload in ((b'changed',encoded()),(b'normalized-image',encoded(0.5))):
            with self.subTest(normalized=normalized):
                with self.assertRaises(IntegrityError):
                    with images.lease(span,reencode=lambda _:(normalized,payload)):pass
                self.assertEqual(self.store.head('main'),head)
                self.assertEqual(self.store.input_lineage(root)[1],[span])
                self.assertEqual(list(self.disk.directory.glob('*.sve')),[])

    def test_vc006_missing_embedding_without_encoder_is_explicit_failure(self):
        from context_time_travel import NotFound
        images=self.artifacts()
        with self.prepare(images) as first:span=first.span
        for key in list(self.ram._entries):self.ram.remove(key)
        self.clock[0]+=11_000_000_000
        self.disk.collect_expired()
        with self.assertRaisesRegex(NotFound,'encod'):
            with images.lease(span):pass

    def test_vc006_wrong_profile_and_original_corruption_fail_before_encoder(self):
        images=self.artifacts()
        with self.prepare(images) as first:span=first.span
        wrong=self.artifacts(encoder_profile_id='b'*64)
        with self.assertRaises(IntegrityError):
            with wrong.lease(span,reencode=lambda _:self.fail('must not encode')):pass
        path=images.originals/(span['original_sha256']+'.input')
        path.write_bytes(b'corrupt')
        with self.assertRaises(IntegrityError):
            with images.lease(span,reencode=lambda _:self.fail('must not encode')):pass

    def test_vc006_original_capacity_failure_does_not_encode_or_publish(self):
        images=self.artifacts(original_bytes_limit=4)
        with self.assertRaises(PoolFull):
            with self.prepare(images):pass
        self.assertEqual(self.calls,[])
        self.assertEqual(list(images.originals.glob('*.input')),[])

    def test_vc006_failed_derived_save_returns_error_without_a_usable_ack(self):
        self.disk=DiskCheckpointPool(self.store,32)
        images=self.artifacts()
        with self.assertRaises(PoolFull):
            with self.prepare(images):self.fail('failed save must not yield success')
        self.assertEqual(self.disk.metrics()['reserved_bytes'],0)
        self.assertEqual(list(self.disk.directory.glob('*.sve')),[])

    def test_vc006_active_prepare_prevents_original_gc_until_history_is_published(self):
        images=self.artifacts()
        with self.prepare(images) as first:
            self.assertEqual(images.collect()['originals_deleted'],[])
            root=self.store.create_input_root('mixed',[first.span],{})
        self.assertEqual(images.collect()['originals_deleted'],[])
        self.assertEqual(self.store.get_snapshot(root)['schema'],2)

    def test_vc006_unreferenced_inputs_can_be_collected_after_lease_release(self):
        images=self.artifacts()
        with self.prepare(images) as first:span=first.span
        self.assertEqual(images.collect()['originals_deleted'],[span['original_sha256']])
        self.assertEqual(list(images.originals.glob('*.input')),[])

    def test_vc006_record_publication_failure_does_not_yield_success_and_can_retry(self):
        images=self.artifacts()
        def fail(step):
            if step=='image_before_encoding_publish':raise OSError('injected write refusal')
        self.store.fault_hook=fail
        with self.assertRaisesRegex(OSError,'write refusal'):
            with self.prepare(images):self.fail('must not yield')
        self.store.fault_hook=None
        with self.prepare(images) as value:
            self.assertEqual(value.embedding.payload,encoded())
        self.assertEqual(self.disk.metrics()['reserved_bytes'],0)

    def test_vc006_restricted_store_cannot_bypass_artifact_authorization(self):
        from context_time_travel.vision_artifacts import ImageArtifacts
        with self.assertRaises(PermissionError):
            ImageArtifacts(self.store.for_principal('guest'),self.ram,self.disk,
                encoder_profile_id='a'*64,expected_width=2,original_bytes_limit=1024**2)

    def test_vc006_invalid_pad_token_fails_before_encoding_or_original_publication(self):
        from context_time_travel import InvalidOperation
        images=self.artifacts()
        with self.assertRaises(InvalidOperation):
            with self.prepare(images,pad_token_id=-1):pass
        self.assertEqual(self.calls,[])
        self.assertEqual(list(images.originals.glob('*.input')),[])


if __name__=='__main__':unittest.main()
