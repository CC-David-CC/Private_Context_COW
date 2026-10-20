"""Resolved image deltas over real Store/pools; synthetic CPU inference only."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
import copy
import errno
import hashlib
import io
import json
import threading
import time
import unittest

from PIL import Image
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolFull, SharedCheckpointPool
from context_time_travel.store import Conflict, IntegrityError, InvalidOperation, _json_bytes
from context_time_travel.vision_embeddings import encoding_key
import serve.test_cow_vision_text_flow as fixture


class VisionDeltaCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests();self.addCleanup(self.flow.doCleanups);self.flow.setUp()
        self.clock=[time.time_ns()];self.flow.preparer.clock_ns=lambda:self.clock[0]
        handle=self.flow.text.create('main',self.flow.messages,'context',namespace_id=self.flow.namespace)
        self.root=handle.durable.result(5)['snapshot'];self.prior=self.flow.tokens.history('main')
        raw=io.BytesIO();Image.new('RGB',(4,3),'blue').save(raw,format='PNG');self.raw=raw.getvalue()
        self.upload=self.flow.preparer.upload(self.flow.namespace,self.raw,pool=self.flow.manager.pool,disk=self.flow.manager.disk)
        normalized=self.flow.preparer.normalizer.normalize(self.raw)
        self.artifacts=self.flow.preparer.artifacts(self.flow.manager.pool,self.flow.manager.disk)
        pad=next(span['pad_token_id'] for span in self.prior.spans if span['kind']=='image')
        with self.artifacts.prepare(original=self.raw,normalized=normalized.payload,
            encode=lambda value:self.flow.encoder.encode(value).embedding.payload,pad_token_id=pad) as held:
            self.span=held.span
        self.delta=InputHistory([dict(kind='text',tokens=[65,66]),self.span,dict(kind='text',tokens=[67,68])])
        self.ram_key=('image',encoding_key(self.span['normalized_sha256'],self.span['encoder_profile_id']))

    def presentation(self,operation='delta',namespace=None):
        return self.flow.text._presentation(namespace or self.flow.namespace,'initial_turn',
            [dict(role='user',content='new image')],operation,self.flow.text._message_id(operation,'assistant'),canonical=self.prior)

    def submit(self,operation='delta',branch='main',revision=1,*,artifacts=True,presentation=None,delta=None):
        options=dict(expected_revision=revision,presentation=self.presentation(operation) if presentation is None else presentation)
        if artifacts:options['artifacts']=self.artifacts
        return self.flow.manager.submit(branch,self.delta if delta is None else delta,2,operation,**options)

    def test_queue_consumes_only_delta_and_publishes_typed_descriptor(self):
        before=self.flow.models[-1].backend.stats['forward_tokens'];calls=len(self.flow.encoder.calls)
        handle=self.submit();receipt=handle.durable.result(5);answer=handle.snapshot.result(5)
        self.assertEqual(answer['counter_deltas']['forward_tokens'],len(self.delta)+2)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens']-before,len(self.delta)+2)
        self.assertEqual(len(self.flow.encoder.calls),calls)
        source=self.flow.manager.committed_generation('delta');generation=source['generation']
        self.assertEqual(generation['schema'],9);self.assertEqual(generation['ids'],self.delta.spans)
        self.assertEqual(source['receipt']['presentation']['input_span']['count'],len(self.delta))
        self.assertEqual(self.flow.tokens.history('main'),self.prior+self.delta+tuple(answer['tokens']))
        self.assertEqual(receipt['branch']['revision'],2)

    def test_reopen_replays_without_artifacts_or_producer_and_text_retry_uses_cell_counts(self):
        first=self.submit().durable.result(5);self.flow.close();self.flow.without_encoder=True;self.flow.start()
        before=self.flow.models[-1].backend.stats['forward_tokens']
        second=self.submit(artifacts=False);again=second.durable.result(5)
        self.assertEqual(first['branch'],again['branch']);self.assertTrue(second.snapshot.result(5)['recovered'])
        replay=self.flow.text.retry('delta','main',2,'replay-delta')
        self.assertEqual(replay['answer']['text'],'xx');self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_changed_image_with_same_pad_projection_cannot_reuse_operation(self):
        self.submit().durable.result(5);before=self.flow.models[-1].backend.stats['forward_tokens']
        changed=copy.deepcopy(self.delta.spans);changed[1]['original_sha256']='f'*64
        self.assertEqual(InputHistory(changed).native_tokens,self.delta.native_tokens)
        with self.assertRaisesRegex(ValueError,'different input'):
            self.submit(delta=InputHistory(changed))
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_stale_revision_and_wrong_namespace_fail_before_intent(self):
        before=self.flow.models[-1].backend.stats['forward_tokens']
        with self.assertRaises(Conflict):self.submit('stale',revision=0)
        other=self.flow.store.create_namespace('Other','other').namespace_id
        with self.assertRaises(InvalidOperation):self.submit('wrong-namespace',presentation=self.presentation('wrong-namespace',other))
        for name in ('stale','wrong-namespace'):self.assertFalse((self.flow.store.root/'pool-operations'/(name+'.json')).exists())
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_missing_artifact_owner_rejects_before_intent_or_forward(self):
        before=self.flow.models[-1].backend.stats['forward_tokens']
        with self.assertRaisesRegex(InvalidOperation,'artifact|image'):self.submit('missing',artifacts=False)
        self.assertFalse((self.flow.store.root/'pool-operations/missing.json').exists())
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_signed_descriptor_with_altered_image_or_flattened_ids_is_rejected(self):
        self.submit().durable.result(5);path=self.flow.store.root/'pool-operations/delta.json';raw=path.read_bytes()
        for flatten in (False,True):
            value=json.loads(raw);generation=value['intent']['generation']
            if flatten:generation['ids']=list(self.delta.native_tokens)
            else:generation['ids'][1]['embedding_sha256']='0'*64
            value['checksum']=hashlib.sha256(_json_bytes(value['intent'])).hexdigest();path.write_bytes(_json_bytes(value))
            try:
                with self.assertRaises(IntegrityError):self.flow.manager.committed_generation('delta')
            finally:path.write_bytes(raw)

    def block_worker(self):
        entered=threading.Event();gate=threading.Event();self.addCleanup(gate.set)
        backend=self.flow.models[-1].backend;original=backend.generate_request
        @contextmanager
        def blocked(lease,count):
            entered.set()
            if not gate.wait(5):raise TimeoutError('CPU fixture gate timed out')
            with original(lease,count) as tokens:yield tokens
        backend.generate_request=blocked
        self.flow.tokens.fork('other',self.root,source_branch='main',operation_id='fork-other')
        first=self.flow.text.ask('main','ordinary question',1,'block',2)
        self.assertTrue(entered.wait(5));return first,gate

    def test_queued_image_is_pinned_across_expiry_then_released_on_cancel(self):
        first,gate=self.block_worker()
        try:
            request=self.submit('queued','other',0)
            self.clock[0]+=1801*10**9;self.artifacts.collect()
            with self.assertRaises(PoolFull):self.flow.manager.pool.remove(self.ram_key)
            self.assertTrue((self.artifacts.originals/(self.span['original_sha256']+'.input')).exists())
            self.assertTrue(self.flow.manager.cancel(request))
            with self.assertRaises(CancelledError):request.durable.result(5)
            self.flow.manager.pool.remove(self.ram_key);self.assertNotIn(self.ram_key,self.flow.manager.pool);self.artifacts.collect()
            self.assertFalse((self.artifacts.originals/(self.span['original_sha256']+'.input')).exists())
        finally:gate.set()
        first.durable.result(5)

    def test_intent_failure_releases_image_and_source_admission(self):
        original=self.flow.manager._intent
        def fail(*args,**options):raise OSError(errno.ENOSPC,'synthetic intent write failure')
        self.flow.manager._intent=fail
        try:
            with self.assertRaises(OSError):self.submit('intent-failure')
        finally:self.flow.manager._intent=original
        self.flow.manager.pool.remove(self.ram_key);self.assertNotIn(self.ram_key,self.flow.manager.pool)
        self.clock[0]+=1801*10**9;self.artifacts.collect()
        self.assertFalse((self.artifacts.originals/(self.span['original_sha256']+'.input')).exists())

    def test_rejected_publication_retries_without_reencoding_or_forward_work(self):
        model=self.flow.models[-1];original=model.save_prepared_branch
        def fail(*args,**options):raise OSError(errno.ENOSPC,'synthetic durable write failure')
        model.save_prepared_branch=fail
        try:
            request=self.submit('save-failure')
            with self.assertRaises(OSError):request.durable.result(5)
        finally:model.save_prepared_branch=original
        self.assertEqual(self.flow.store.head('main').tip,self.root)
        before=model.backend.stats['forward_tokens'];calls=len(self.flow.encoder.calls)
        receipt=self.flow.manager.retry_publication('save-failure').result(5)
        self.assertEqual(receipt['branch']['revision'],2)
        self.assertEqual(model.backend.stats['forward_tokens'],before);self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_changed_sampler_retry_and_new_sampler_controls_remain_explicitly_unqualified(self):
        with self.assertRaisesRegex(InvalidOperation,'greedy'):
            self.flow.manager.submit('main',self.delta,2,'sampled',expected_revision=1,
                presentation=self.presentation('sampled'),artifacts=self.artifacts,seed=7)
        self.assertFalse((self.flow.store.root/'pool-operations/sampled.json').exists())
        self.submit().durable.result(5)
        with self.assertRaisesRegex(InvalidOperation,'not yet qualified'):
            self.flow.manager.submit_retry_generation('delta','main',2,'changed',None,self.presentation('changed'),seed=7)

    def test_foreign_pool_owner_is_rejected_before_intent_or_forward(self):
        pool=SharedCheckpointPool(1024**3);self.addCleanup(pool.close)
        foreign=self.flow.preparer.artifacts(pool,self.flow.manager.disk)
        before=self.flow.models[-1].backend.stats['forward_tokens']
        with self.assertRaisesRegex(InvalidOperation,'owned by this coordinator'):
            self.flow.manager.submit('main',self.delta,2,'foreign-pool',expected_revision=1,
                presentation=self.presentation('foreign-pool'),artifacts=foreign)
        self.assertFalse((self.flow.store.root/'pool-operations/foreign-pool.json').exists())
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_invalid_geometry_and_image_count_reject_before_intent(self):
        bad=copy.deepcopy(self.delta.spans);bad[1].update(width=5,embedding_bytes=20+self.span['rows']*5*4)
        many=[span for _ in range(9) for span in self.delta.spans]
        before=self.flow.models[-1].backend.stats['forward_tokens']
        for name,delta in (('wrong-width',InputHistory(bad)),('too-many',InputHistory(many))):
            with self.subTest(name=name),self.assertRaises(InvalidOperation):self.submit(name,delta=delta)
            self.assertFalse((self.flow.store.root/'pool-operations'/(name+'.json')).exists())
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)


if __name__=='__main__':unittest.main()
