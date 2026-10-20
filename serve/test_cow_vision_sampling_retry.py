"""Changed image retries retain inputs and regenerate only on the GPU worker.

These are CPU control/encoder fixtures, not real GPU qualification.
"""
from concurrent.futures import CancelledError
import hashlib
import threading
import unittest

from context_time_travel.store import IntegrityError, InvalidOperation, NotFound
import serve.test_cow_vision_sampling_flow as fixtures


class CowVisionSamplingRetryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CowVisionSamplingFlowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.c=self.f.f
        self.c.text.create('main',[dict(role='user',content=self.c.parts)],'context',
            namespace_id=self.c.namespace).durable.result(5)
        self.c.text.ask('main',self.c.parts,1,'question',4,
            settings=fixtures.POLICY,seed=42).durable.result(5)
        self.original=self.c.manager.committed_generation('question')
        self.head=self.c.store.head('main')

    def evict_embeddings(self,*,producer=True):
        self.c.close()
        for path in (self.c.store.root/'checkpoints').glob('*.sve'):
            path.unlink()
        self.c.without_encoder=not producer;self.c.start()

    def retry(self,operation='changed'):
        return self.c.text.retry('question','main',self.head.revision,operation,
            settings=dict(temperature=.9,top_k=2,top_p=.95))

    def test_missing_delta_embedding_reencodes_on_worker_without_prefix_replay(self):
        self.evict_embeddings();calls=len(self.c.encoder.calls);thread=threading.get_ident()
        before=self.c.models[-1].backend.stats['forward_tokens']
        request=self.retry();result=request.durable.result(5)
        descriptor=self.c.manager.committed_generation('changed')['generation']
        self.assertEqual(descriptor['ids'],self.original['generation']['ids'])
        self.assertEqual(len(self.c.encoder.calls)-calls,1)
        self.assertNotEqual(self.c.encoder.calls[-1],thread)
        count=sum(len(s['tokens']) if s['kind']=='text' else s['rows'] for s in descriptor['ids'])
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens']-before,count+4)
        self.assertEqual(self.c.store.get_snapshot(result['snapshot'])['controller']['next_draw'],4)
        self.assertEqual(request.request.durable.result(5)['snapshot'],result['snapshot'])
        # The normal request answer must expose actual encoder work.
        self.assertEqual(request.request.snapshot.result(5)['input_preparation']['encoder_calls'],1)

    def test_cached_retry_needs_no_encoder_or_new_upload_grant(self):
        calls=len(self.c.encoder.calls)
        self.c.close();self.c.without_encoder=True;self.c.start()
        request=self.retry();request.durable.result(5)
        self.assertEqual(len(self.c.encoder.calls),calls)
        self.assertEqual(request.request.snapshot.result(5)['input_preparation']['encoder_calls'],0)

    def test_absent_producer_preserves_branch_when_delta_embedding_is_missing(self):
        self.evict_embeddings(producer=False);before=self.c.models[-1].backend.stats['forward_tokens']
        with self.assertRaises(NotFound):
            self.retry().durable.result(5)
        self.assertEqual(self.c.store.head('main'),self.head)
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens'],before)

    def test_encoder_failure_releases_original_and_pool_leases(self):
        self.evict_embeddings();self.c.encoder.fail=True
        with self.assertRaisesRegex(InvalidOperation,'Synthetic encoder failure'):
            self.retry().durable.result(5)
        self.assertEqual(self.c.store.head('main'),self.head)
        self.c.encoder.fail=False
        request=self.retry('recovered');request.durable.result(5)
        self.assertEqual(request.request.snapshot.result(5)['input_preparation']['encoder_calls'],1)

    def test_cancellation_during_reencoding_does_not_publish_a_child(self):
        self.evict_embeddings();entered=threading.Event();release=threading.Event();cancel=threading.Event()
        original=self.c.encoder.encode
        def encode(*args,**options):
            entered.set()
            if not release.wait(5):raise TimeoutError('Test did not release encoder')
            return original(*args,**options)
        self.c.encoder.encode=encode
        request=self.c.text.retry('question','main',self.head.revision,'cancelled',
            settings=dict(temperature=.9),cancelled=cancel.is_set)
        try:
            self.assertTrue(entered.wait(5));cancel.set()
        finally:release.set()
        with self.assertRaises(CancelledError):request.durable.result(5)
        self.assertEqual(self.c.store.head('main'),self.head)
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens'],0)

    def test_reencoding_identity_mismatch_preserves_branch_and_original_receipt(self):
        from types import SimpleNamespace
        from context_time_travel.vision_embeddings import parse_sve
        self.evict_embeddings();original=self.c.encoder.encode
        def encode(*args,**options):
            result=original(*args,**options);raw=result.embedding.payload
            raw=raw[:-1]+b'\x00'
            return SimpleNamespace(embedding=parse_sve(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_width=4),receipt=result.receipt)
        self.c.encoder.encode=encode
        with self.assertRaises(IntegrityError):self.retry().durable.result(5)
        self.assertEqual(self.c.store.head('main'),self.head)
        self.assertEqual(self.c.manager.committed_generation('question'),self.original)
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens'],0)

    def test_changed_retry_exact_replay_after_reopen_needs_no_derived_bytes(self):
        self.evict_embeddings();first=self.retry();receipt=first.durable.result(5);text=first.text_result()
        self.evict_embeddings(producer=False)
        replay=self.c.text.retry('changed','main',self.c.store.head('main').revision,'replay-changed')
        self.assertTrue(replay['replayed']);self.assertEqual(replay['answer']['text'],text)
        self.assertEqual(replay['receipt']['snapshot'],receipt['snapshot'])
        self.assertEqual(self.c.models[-1].backend.stats['forward_tokens'],0)


if __name__=='__main__':unittest.main()
