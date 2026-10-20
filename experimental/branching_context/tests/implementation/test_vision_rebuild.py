"""VC-015 control-plane recovery tests; synthetic CPU arithmetic only."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import SharedCheckpointPool, DiskCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation, NotFound, Store, _json_bytes
from context_time_travel.vision_artifacts import ImageArtifacts
from context_time_travel.vision_execution import RESUME_CONTRACT, POSITION_CONVENTION
from test_vision_api_binding import MixedCPUBackend


class RebuildMixedBackend(MixedCPUBackend):
    def __init__(self, profile):
        super().__init__(profile); self.closed=False; self.bad_work=False

    def consume_inputs(self, lease, inputs, *, embeddings):
        before=self.stats['forward_tokens']
        super().consume_inputs(lease,inputs,embeddings=embeddings)
        if self.bad_work:self.stats['forward_tokens']-=1
        self.stats['consumed_tokens']+=self.stats['forward_tokens']-before

    def close(self): self.closed=True


class VisionRebuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-profile',
            model_weights_sha256='a'*64,execution=dict(resume_contract=RESUME_CONTRACT,
                prefill_chunk=128,embedding_width=4,capacity=128,eos_tokens=[],
                generated_operator='normal-verifier-decode',native_request_protocol='consumed-greedy-ack-v1'))
        self.backend=RebuildMixedBackend(self.profile)
        self.model=BranchingModel(self.tmp.name,self.backend,self.profile)
        self.ram=SharedCheckpointPool(1024**2);self.addCleanup(self.ram.close)
        self.disk=DiskCheckpointPool(self.model.store,1024**2)
        self.artifacts=ImageArtifacts(self.model.store,self.ram,self.disk,encoder_profile_id='c'*64,
            expected_width=4,original_bytes_limit=1024**2)
        self.original=b'retained original image';self.normalized=b'normalized encoder input'
        self.sve=struct.pack('<5i24f',0x31455653,6,3,2,4,*[n/32 for n in range(24)])
        with self.artifacts.prepare(original=self.original,normalized=self.normalized,
            encode=lambda _:self.sve,pad_token_id=7) as held:
            self.span=held.span
            self.spans=[dict(kind='text',tokens=[1,2]),self.span,dict(kind='text',tokens=[3])]
            self.root=self.model.create_inputs(self.spans,artifacts=self.artifacts)
        self.model.store.fork('main',self.root)
        self.calls=[]

    def evict(self,*,keep_embedding=False):
        self.model.cache.clear();self.backend.release_checkpoints()
        for path in self.model.checkpoint_directory.iterdir():
            if path.is_file() and (not keep_embedding or path.suffix!='.sve'):path.unlink()
        for key in list(self.ram._entries):self.ram.remove(key)

    def canonical(self):
        root=self.model.store.root
        return {str(p.relative_to(root)):p.read_bytes() for folder in
            ('objects','references','image-originals','image-encodings')
            for p in (root/folder).glob('*') if p.is_file()}

    def reencode(self,original):
        self.calls.append(original);return self.normalized,self.sve

    def rebuild(self,snapshot=None,**options):
        return self.model.rebuild_materialization(snapshot or self.root,
            artifacts=self.artifacts,reencode=self.reencode,**options)

    def test_vc015_all_derived_copies_evicted_reencodes_and_reports_actual_prefix_work(self):
        self.model.checkpoint(self.root);self.evict();before=self.canonical()
        work=self.backend.work_counters()['forward_tokens']
        receipt=self.rebuild()
        self.assertEqual(self.calls,[self.original])
        self.assertEqual(receipt['reencode_callback_calls'],1)
        self.assertEqual(receipt['image_artifacts'][0]['source'],'reencoded')
        self.assertEqual(receipt['forward_tokens'],9)
        self.assertEqual(self.backend.work_counters()['forward_tokens']-work,9)
        self.assertEqual(receipt['positions'],POSITION_CONVENTION)
        self.assertEqual(receipt['source_kind'],'recomputed-history')
        self.assertIs(receipt['durable'],False)
        self.assertIs(receipt['prior_computation_preserved'],False)
        self.assertEqual(self.model.history(self.root),InputHistory(self.spans))
        self.assertEqual(self.canonical(),before)
        self.assertFalse((self.model.checkpoint_directory/(self.root+'.json')).exists())
        self.assertEqual(self.model.cache[self.root].input_identity,InputHistory(self.spans).digest)

    def test_cached_embedding_rebuilds_model_without_reencoding(self):
        self.evict(keep_embedding=True)
        def forbidden(_):self.fail('A cached embedding must not re-encode')
        receipt=self.model.rebuild_materialization(self.root,artifacts=self.artifacts,reencode=forbidden)
        self.assertEqual(receipt['reencode_callback_calls'],0)
        self.assertEqual(receipt['image_artifacts'][0]['source'],'disk')
        self.assertEqual(receipt['forward_tokens'],9)

    def test_missing_encoder_is_explicit_before_native_work(self):
        self.evict();before=self.backend.work_counters();canonical=self.canonical()
        with self.assertRaisesRegex(NotFound,'encoder'):
            self.model.rebuild_materialization(self.root,artifacts=self.artifacts)
        self.assertEqual(self.backend.work_counters(),before)
        self.assertEqual(self.canonical(),canonical)
        self.assertNotIn(self.root,self.model.cache)
        self.assertFalse(self.backend.closed)

    def test_strict_restore_does_not_invoke_artifacts_or_replay(self):
        self.evict();before=self.backend.work_counters()
        with self.assertRaises(IntegrityError):self.model.restore_checkpoint(self.root)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.backend.work_counters(),before)

    def test_reencoding_identity_mismatch_never_advances_native_or_history(self):
        self.evict();before=self.backend.work_counters();canonical=self.canonical()
        changed=self.sve[:-4]+struct.pack('<f',99.)
        for normalized,payload in ((b'wrong',self.sve),(self.normalized,changed)):
            with self.subTest(normalization_changed=normalized!=self.normalized):
                with self.assertRaises(IntegrityError):
                    self.model.rebuild_materialization(self.root,artifacts=self.artifacts,
                        reencode=lambda _:(normalized,payload))
                self.assertEqual(self.backend.work_counters(),before)
                self.assertEqual(self.canonical(),canonical)
                self.assertNotIn(self.root,self.model.cache)
                self.assertFalse(list(self.disk.directory.glob('*.sve')))
                self.assertFalse(self.backend.closed)

    def test_existing_bad_checkpoint_cannot_be_bypassed_by_reencoding(self):
        self.evict()
        (self.model.checkpoint_directory/(self.root+'.json')).write_bytes(b'corrupt')
        before=self.backend.work_counters()
        with self.assertRaisesRegex(InvalidOperation,'existing materialization'):
            self.rebuild()
        self.assertEqual(self.calls,[]);self.assertEqual(self.backend.work_counters(),before)

    def test_foreign_artifact_owner_cannot_reconstruct_this_store(self):
        self.evict();other=tempfile.TemporaryDirectory();self.addCleanup(other.cleanup)
        wrong_store=Store(other.name)
        wrong=ImageArtifacts(wrong_store,self.ram,DiskCheckpointPool(wrong_store,1024**2),
            encoder_profile_id='c'*64,expected_width=4,original_bytes_limit=1024**2)
        before=self.backend.work_counters()
        with self.assertRaisesRegex(InvalidOperation,'canonical store'):
            self.model.rebuild_materialization(self.root,artifacts=wrong,reencode=self.reencode)
        self.assertEqual(self.calls,[]);self.assertEqual(self.backend.work_counters(),before)

    def test_generated_snapshot_preserves_controller_and_canonical_branch(self):
        active=self.model.start_generation(self.root,dict(temperature=0,eos_tokens=[]),'saved-stream')
        generated=self.model.generate(active,4);snapshot=generated.snapshot_id
        self.model.store.move('main',0,snapshot,'publish-generated')
        state=self.model.controller(snapshot);self.evict();canonical=self.canonical()
        receipt=self.rebuild(snapshot)
        self.assertEqual(receipt['consumed'],13)
        self.assertEqual(receipt['forward_tokens'],13)
        self.assertEqual(receipt['controller_sha256'],hashlib.sha256(_json_bytes(state)).hexdigest())
        self.assertEqual(self.model.controller(snapshot),state)
        self.assertEqual(self.canonical(),canonical)

    def test_unsupported_sampler_fails_before_encoder_or_model(self):
        active=self.model.start_generation(self.root,dict(temperature=0.7),'sampled')
        self.evict();before=self.backend.work_counters()
        with self.assertRaisesRegex(InvalidOperation,'tested greedy controller'):
            self.rebuild(active)
        self.assertEqual(self.calls,[]);self.assertEqual(self.backend.work_counters(),before)

    def test_incomplete_native_work_is_not_cached_and_backend_is_closed(self):
        self.evict();canonical=self.canonical();self.backend.bad_work=True
        with self.assertRaises(IntegrityError):self.rebuild()
        self.assertNotIn(self.root,self.model.cache)
        self.assertEqual(self.canonical(),canonical)
        self.assertTrue(self.backend.closed)


if __name__=='__main__':unittest.main()
