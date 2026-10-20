"""Production preparer/control path with synthetic CPU model/encoder only."""
from concurrent.futures import CancelledError
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest

from PIL import Image
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store, IntegrityError, InvalidOperation, NotFound
from context_time_travel.vision_embeddings import parse_sve
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.cow_vision_encoder import EncodedImage
from serve.cow_vision_preparer import CowVisionPreparer
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from test_vision_coordinator import ServingMixedCPUBackend

ROOT=Path(__file__).resolve().parents[1]


class EncoderFixture:
    identity='d'*64;width=4;max_rows=1024
    def __init__(self):
        self.calls=[];self.entered=threading.Event();self.gate=threading.Event();self.gate.set();self.fail=False
    def encode(self,normalized,**options):
        self.calls.append(threading.get_ident());self.entered.set()
        if not self.gate.wait(5):raise TimeoutError('Fixture gate was not released')
        if self.fail:raise InvalidOperation('Synthetic encoder failure')
        raw=struct.pack('<5i24f',0x31455653,6,3,2,4,*[i/32 for i in range(24)])
        value=parse_sve(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),expected_width=4)
        return EncodedImage(value,dict(scope='synthetic-cpu-only',output_sha256=value.sha256))


class WideCPUBackend(ServingMixedCPUBackend):vocab_size=512


class CowVisionPreparerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name);self.namespace=self.store.create_namespace('A','namespace-A').namespace_id
        self.other=self.store.create_namespace('B','namespace-B').namespace_id
        self.clock=[time.time_ns()];self.encoder=EncoderFixture();self.models=[];self.threads=[];self.manager=None
        self.normalizer=CowImageNormalizer()
        base=CowPromptEncoder(ByteTokenizer(),ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':False})
        self.renderer=CowVisionPromptEncoder(base,renderer_id='c'*64)
        self.preparer=CowVisionPreparer(self.store,self.normalizer,self.renderer,self.encoder,
            original_bytes_limit=16*1024**2,upload_idle_seconds=30,clock_ns=lambda:self.clock[0])
        self.start();self.addCleanup(self.close)
        image=io.BytesIO();Image.new('RGB',(3,2),'red').save(image,format='PNG');self.raw=image.getvalue()
        self.upload=self.preparer.upload(self.namespace,self.raw,pool=self.manager.pool,disk=self.manager.disk)

    def start(self):
        profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-preparer',model_weights_sha256='a'*64,
            execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',
                generated_operator='normal-verifier-decode',native_request_protocol='consumed-greedy-ack-v1',
                prefill_chunk=128,embedding_width=4,capacity=2048,eos_tokens=[]))
        def factory(gpu):
            self.threads.append(threading.get_ident());model=BranchingModel(self.temp.name,WideCPUBackend(profile),profile)
            self.models.append(model);return model
        self.manager=ContextCoordinator(self.temp.name,PoolConfig(gpus=(0,),ram_bytes=1024**3,
            disk_bytes=16*1024**2,min_available_bytes=0,checkpoint_idle_seconds=getattr(self,'idle_period',None)),
            factory,input_preparer=self.preparer)
        self.manager.ready(5)

    def close(self):
        self.encoder.gate.set()
        if self.manager is not None:self.manager.close();self.manager=None

    def body(self,namespace=None):
        return dict(schema=1,namespace_id=namespace or self.namespace,add_generation_prompt=True,
            messages=[dict(role='user',content=[dict(type='image',image_id=self.upload['image_id']),
                dict(type='text',text=' Describe the diagram.')])])

    def root(self,branch='main',operation='root-op',body=None,presentation=None):
        body=self.body() if body is None else body
        presentation=self.preparer.presentation(body,operation) if presentation is None else presentation
        return self.manager.prepare_root(branch,body,operation,presentation=presentation)

    def test_production_preparer_uses_worker_and_publishes_typed_history(self):
        request=self.root();durable=request.durable.result(5);answer=request.snapshot.result(5)
        history=self.models[-1].history(durable['snapshot']);self.assertIsInstance(history,InputHistory)
        self.assertEqual(self.encoder.calls,self.threads[-1:])
        self.assertEqual(answer['counter_deltas']['forward_tokens'],len(history))
        self.assertEqual(len([s for s in history.spans if s['kind']=='image']),1)
        self.assertEqual(answer['input_preparation']['images'][0]['original_sha256'],self.upload['image_id'])

    def test_identical_upload_roots_reuse_encoder_and_prefix(self):
        first=self.root('first','first-op').durable.result(5)
        request=self.root('second','second-op');second=request.durable.result(5)
        self.assertEqual(first['snapshot'],second['snapshot']);self.assertEqual(len(self.encoder.calls),1)
        self.assertEqual(request.snapshot.result(5)['counter_deltas']['forward_tokens'],0)

    def test_wrong_namespace_refuses_before_intent_or_encoder(self):
        with self.assertRaises(NotFound):self.root(body=self.body(self.other))
        self.assertEqual(self.encoder.calls,[])
        self.assertFalse((self.store.root/'pool-operations/root-op.json').exists())

    def test_forged_presentation_refuses_before_intent_or_encoder(self):
        body=self.body();presentation=self.preparer.presentation(body,'root-op');presentation['namespace_id']=self.other
        with self.assertRaises(InvalidOperation):self.root(body=body,presentation=presentation)
        self.assertEqual(self.encoder.calls,[])

    def test_queued_admissions_survive_upload_expiry_and_gc(self):
        self.encoder.gate.clear();first=self.root('first','first-op')
        try:
            self.assertTrue(self.encoder.entered.wait(5));second=self.root('second','second-op')
            self.clock[0]+=31*10**9;artifacts=self.preparer.artifacts(self.manager.pool,self.manager.disk)
            artifacts.collect();self.assertEqual(artifacts._original(self.upload['image_id']),self.raw)
        finally:self.encoder.gate.set()
        a=first.durable.result(5);b=second.durable.result(5)
        self.assertEqual(a['snapshot'],b['snapshot']);self.assertEqual(len(self.encoder.calls),1)

    def test_cancellation_after_root_prefill_does_not_publish_a_branch(self):
        prior=self.root('retained','retained-op').durable.result(5)
        retained=self.store.head('retained');cancelled=threading.Event()
        model=self.models[-1];original=model.create_with_reuse
        def consumed_then_cancelled(*args,**kwargs):
            result=original(*args,**kwargs)
            cancelled.set()
            return result
        model.create_with_reuse=consumed_then_cancelled
        body=self.body();body['messages'][0]['content'][-1]['text']+=' A different prefilling fixture.'
        presentation=self.preparer.presentation(body,'cancel-after-prefill')
        before=model.backend.stats['forward_tokens']
        handle=self.manager.prepare_root('unpublished',body,'cancel-after-prefill',
            presentation=presentation,cancelled=cancelled.is_set)
        with self.assertRaises(CancelledError):handle.durable.result(5)
        self.assertTrue(cancelled.is_set())
        self.assertGreater(model.backend.stats['forward_tokens'],before)
        self.assertNotIn('unpublished',[head.name for head in self.store.branches()])
        self.assertEqual(self.store.head('retained'),retained)

    def test_cancelled_queue_releases_original_after_other_failure(self):
        self.encoder.gate.clear();self.encoder.fail=True;first=self.root('first','first-op')
        try:
            self.assertTrue(self.encoder.entered.wait(5));second=self.root('second','second-op')
            self.assertTrue(self.manager.cancel(second));self.clock[0]+=31*10**9
        finally:self.encoder.gate.set()
        with self.assertRaises(InvalidOperation):first.durable.result(5)
        with self.assertRaises(CancelledError):second.durable.result(5)
        artifacts=self.preparer.artifacts(self.manager.pool,self.manager.disk);artifacts.collect()
        self.assertFalse((artifacts.originals/(self.upload['image_id']+'.input')).exists())

    def test_completed_replay_after_reopen_and_upload_expiry_never_encodes(self):
        first=self.root().durable.result(5);self.close();self.clock[0]+=31*10**9;self.start()
        self.preparer.artifacts(self.manager.pool,self.manager.disk).collect();calls=len(self.encoder.calls)
        again=self.root().durable.result(5)
        self.assertEqual(first['branch'],again['branch']);self.assertEqual(len(self.encoder.calls),calls)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)

    def test_wrong_width_and_corrupt_original_fail_before_encoding(self):
        self.models[-1].profile['execution']['embedding_width']=5
        with self.assertRaises(InvalidOperation):self.root(operation='width').durable.result(5)
        self.models[-1].profile['execution']['embedding_width']=4
        artifacts=self.preparer.artifacts(self.manager.pool,self.manager.disk)
        (artifacts.originals/(self.upload['image_id']+'.input')).write_bytes(b'corrupt')
        with self.assertRaises(IntegrityError):self.root(operation='corrupt').durable.result(5)
        self.assertEqual(self.encoder.calls,[])
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],0)

    def test_background_expiry_collects_unadmitted_uploads(self):
        self.close();self.idle_period=.04;self.start();self.clock[0]+=31*10**9
        artifacts=self.preparer.artifacts(self.manager.pool,self.manager.disk)
        original=artifacts.originals/(self.upload['image_id']+'.input');deadline=time.monotonic()+2
        while original.exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertFalse(original.exists());self.assertEqual(self.encoder.calls,[])
        self.assertIsNone(self.manager.metrics()['idle_expiry']['last_error'])


if __name__=='__main__':unittest.main()
