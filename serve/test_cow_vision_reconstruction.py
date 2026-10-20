"""Serving reconstruction with real Store/pools, synthetic CPU model/encoder."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import errno
import struct
import threading
import unittest

from context_time_travel.store import IntegrityError,InvalidOperation,NotFound
from context_time_travel.vision_artifacts import ImageArtifacts
from context_time_travel.vision_artifacts import RetainedImageArtifacts
from context_time_travel.vision_embeddings import parse_sve
import serve.test_cow_vision_text_flow as fixture


class CowVisionReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests();self.addCleanup(self.flow.doCleanups);self.flow.setUp()
        self.root=self.flow.text.create('main',self.flow.messages,'context',namespace_id=self.flow.namespace).durable.result(5)['snapshot']
        self.history=self.flow.tokens.history('main');self.root_record=self.flow.store.get_snapshot(self.root)

    def reopen(self,*,keep_embeddings=False,without_producer=False):
        self.flow.close()
        for path in (self.flow.store.root/'checkpoints').iterdir():
            if path.is_file() and path.suffix in ('.json','.state','.sve'):
                if keep_embeddings and path.suffix=='.sve':continue
                path.unlink()
        self.flow.without_encoder=without_producer;self.flow.start()
        # This synthetic backend previously counted forward work only. Count
        # consumed cells at its actual fixture consumption site as well.
        backend=self.flow.models[-1].backend;consume=backend.consume_inputs
        def counted(lease,inputs,**options):
            before=backend.stats['forward_tokens'];consume(lease,inputs,**options)
            backend.stats['consumed_tokens']+=backend.stats['forward_tokens']-before
        backend.consume_inputs=counted

    def ask(self,operation='question',**options):
        return self.flow.text.ask('main','What color?',1,operation,2,**options)

    def assert_source_intact(self):
        self.assertEqual(self.flow.store.get_snapshot(self.root),self.root_record)
        self.assertEqual(self.flow.models[-1].history(self.root),self.history)

    def test_cached_embedding_rebuilds_with_zero_encoding_and_reports_paid_work(self):
        self.reopen(keep_embeddings=True);calls=len(self.flow.encoder.calls)
        handle=self.ask();receipt=handle.durable.result(5);answer=handle.snapshot.result(5)
        work=answer['source_recomputation']
        self.assertEqual(work['source_kind'],'recomputed-history');self.assertFalse(work['prior_computation_preserved'])
        self.assertFalse(work['durable']);self.assertEqual(work['prefix_forward_tokens'],len(self.history))
        self.assertRegex(work['frontier_sha256'],'^[0-9a-f]{64}$')
        self.assertEqual(work['input_preparation']['encoder_calls'],0)
        self.assertEqual(work['image_artifacts'][0]['source'],'disk')
        self.assertEqual(len(self.flow.encoder.calls),calls);self.assert_source_intact()
        self.assertEqual(receipt['branch']['revision'],2)
        events=list(handle.iter_events());names=[event['event'] for event in events]
        self.assertIn('source_recomputation_started',names);self.assertIn('source_recomputed',names)

    def test_cached_embedding_needs_no_configured_producer(self):
        self.reopen(keep_embeddings=True,without_producer=True);calls=len(self.flow.encoder.calls)
        handle=self.ask();handle.durable.result(5);work=handle.snapshot.result(5)['source_recomputation']
        self.assertEqual(work['reencode_callback_calls'],0);self.assertEqual(work['image_artifacts'][0]['source'],'disk')
        self.assertEqual(len(self.flow.encoder.calls),calls);self.assert_source_intact()

    def test_all_derived_copies_gone_reencodes_on_worker_and_continues(self):
        self.reopen();calls=len(self.flow.encoder.calls);handle=self.ask();handle.durable.result(5)
        work=handle.snapshot.result(5)['source_recomputation']
        self.assertEqual(work['reencode_callback_calls'],1);self.assertEqual(work['input_preparation']['encoder_calls'],1)
        self.assertEqual(work['image_artifacts'][0]['source'],'reencoded')
        self.assertEqual(work['image_rows'],6);self.assertEqual(work['text_tokens'],len(self.history)-6)
        self.assertEqual(len(self.flow.encoder.calls),calls+1)
        self.assertEqual(self.flow.encoder.calls[-1],self.flow.manager._threads[0].ident)
        self.assertEqual(handle.text_result(),'xx');self.assert_source_intact()

    def test_missing_producer_without_embeddings_fails_before_model_work(self):
        self.reopen(without_producer=True)
        with self.assertRaisesRegex(NotFound,'encoder'):self.ask().durable.result(5)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.flow.store.head('main').tip,self.root);self.assert_source_intact()

    def test_regenerated_payload_mismatch_preserves_branch_and_native_state(self):
        self.reopen();original=self.flow.encoder.encode
        def changed(*args,**options):
            result=original(*args,**options);raw=result.embedding.payload[:-4]+struct.pack('<f',99.)
            return replace(result,embedding=parse_sve(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),expected_width=4))
        self.flow.encoder.encode=changed
        with self.assertRaises(IntegrityError):self.ask().durable.result(5)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.flow.store.head('main').tip,self.root);self.assert_source_intact()
        self.assertFalse(list((self.flow.store.root/'checkpoints').glob('*.sve')))

    def test_changed_normalization_is_rejected_before_encoder(self):
        self.reopen();normal=self.flow.preparer.normalizer.normalize;calls=len(self.flow.encoder.calls)
        def changed(raw):return replace(normal(raw),payload=b'different normalized bytes')
        self.flow.preparer.normalizer.normalize=changed
        with self.assertRaises(IntegrityError):self.ask().durable.result(5)
        self.assertEqual(len(self.flow.encoder.calls),calls)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0);self.assert_source_intact()

    def test_present_corrupt_embedding_is_not_silently_reencoded(self):
        self.reopen(keep_embeddings=True);calls=len(self.flow.encoder.calls)
        path=next((self.flow.store.root/'checkpoints').glob('*.sve'));raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(IntegrityError):self.ask().durable.result(5)
        self.assertEqual(len(self.flow.encoder.calls),calls)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0);self.assert_source_intact()

    def test_present_corrupt_checkpoint_is_not_bypassed(self):
        self.flow.close();(self.flow.store.root/'checkpoints'/(self.root+'.json')).write_bytes(b'corrupt')
        self.flow.start();calls=len(self.flow.encoder.calls)
        with self.assertRaises(IntegrityError):self.ask().durable.result(5)
        self.assertEqual(len(self.flow.encoder.calls),calls)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0);self.assert_source_intact()

    def test_cancel_after_encoding_prevents_prefix_forward_and_publication(self):
        self.reopen();stopped=threading.Event();original=self.flow.encoder.encode
        def cancel(*args,**options):
            result=original(*args,**options);stopped.set();return result
        self.flow.encoder.encode=cancel
        with self.assertRaises(CancelledError):self.ask(cancelled=stopped.is_set).durable.result(5)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.flow.store.head('main').tip,self.root);self.assert_source_intact()

    def test_failure_can_be_followed_by_a_new_successful_operation(self):
        self.reopen();self.flow.encoder.fail=True
        with self.assertRaisesRegex(InvalidOperation,'encoder failure'):self.ask('failed').durable.result(5)
        self.flow.encoder.fail=False;handle=self.ask('successful');handle.durable.result(5)
        self.assertEqual(handle.text_result(),'xx');self.assert_source_intact()

    def test_stopped_generated_history_reconstructs_without_changing_saved_controller(self):
        self.flow.models[-1].backend.next_token=257;first=self.ask('first');first.durable.result(5)
        source=self.flow.store.head('main').tip;record=self.flow.store.get_snapshot(source);history=self.flow.tokens.history('main')
        self.reopen();handle=self.flow.text.append_user('main','Next question',2,'next',2);handle.durable.result(5)
        work=handle.snapshot.result(5)['source_recomputation'];self.assertEqual(work['prefix_forward_tokens'],len(history))
        self.assertEqual(self.flow.store.get_snapshot(source),record);self.assert_source_intact()

    def test_unconfigured_retention_budget_allows_reading_but_not_original_admission(self):
        owner=ImageArtifacts(self.flow.store,self.flow.manager.pool,self.flow.manager.disk,
            encoder_profile_id=self.flow.preparer.encoder_profile_id,expected_width=4,original_bytes_limit=None)
        span=next(span for span in self.history.spans if span['kind']=='image')
        with owner.lease(span) as item:self.assertEqual(item.embedding.rows,6)
        with self.assertRaisesRegex(InvalidOperation,'retention|admission|budget'):owner.upload(self.flow.namespace,b'new')
        with self.assertRaisesRegex(InvalidOperation,'retention|admission|budget'),owner.prepare(
                original=b'new',normalized=b'new',encode=lambda _:b'',pad_token_id=span['pad_token_id']):pass

    def test_cancel_after_cached_inputs_are_leased_prevents_native_computation(self):
        self.reopen(keep_embeddings=True,without_producer=True);stopped=threading.Event()
        original=RetainedImageArtifacts.lease
        @contextmanager
        def cancel(owner,span,**options):
            with original(owner,span,**options) as item:
                stopped.set();yield item
        RetainedImageArtifacts.lease=cancel
        try:
            with self.assertRaises(CancelledError):self.ask(cancelled=stopped.is_set).durable.result(5)
        finally:RetainedImageArtifacts.lease=original
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.flow.store.head('main').tip,self.root);self.assert_source_intact()

    def test_rejected_publication_retries_without_reconstruction_or_reencoding(self):
        self.reopen();model=self.flow.models[-1];save=model.save_prepared_branch
        def failed(*args,**options):raise OSError(errno.ENOSPC,'synthetic publication rejection')
        model.save_prepared_branch=failed
        try:
            handle=self.ask()
            with self.assertRaises(OSError):handle.durable.result(5)
        finally:model.save_prepared_branch=save
        self.assertEqual(self.flow.store.head('main').tip,self.root)
        calls=len(self.flow.encoder.calls);work=model.backend.stats['forward_tokens']
        receipt=self.flow.manager.retry_publication('question').result(5)
        self.assertEqual(receipt['branch']['revision'],2)
        self.assertEqual(len(self.flow.encoder.calls),calls);self.assertEqual(model.backend.stats['forward_tokens'],work)

    def test_corrupted_retained_original_fails_before_encoder_or_model(self):
        self.reopen();calls=len(self.flow.encoder.calls)
        span=next(span for span in self.history.spans if span['kind']=='image')
        (self.flow.store.root/'image-originals'/(span['original_sha256']+'.input')).write_bytes(b'corrupt original')
        with self.assertRaises(IntegrityError):self.ask().durable.result(5)
        self.assertEqual(len(self.flow.encoder.calls),calls);self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],0)
        self.assertEqual(self.flow.store.head('main').tip,self.root);self.assert_source_intact()


if __name__=='__main__':unittest.main()
