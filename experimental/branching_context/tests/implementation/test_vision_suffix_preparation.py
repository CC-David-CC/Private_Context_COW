"""Worker-owned suffix preparation; real Store/pools, synthetic CPU encoder."""
from concurrent.futures import CancelledError
import copy
import io
import threading
import time
import unittest

from PIL import Image
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolFull
from context_time_travel.store import Conflict, InvalidOperation, NotFound
from context_time_travel.vision_embeddings import encoding_key
import serve.test_cow_vision_text_flow as fixture


class VisionSuffixPreparationTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests();self.addCleanup(self.flow.doCleanups);self.flow.setUp()
        self.root=self.flow.text.create('main',self.flow.messages,'context',namespace_id=self.flow.namespace).durable.result(5)['snapshot']
        self.prior=self.flow.tokens.history('main');self.before=self.flow.models[-1].backend.stats['forward_tokens']
        self.clock=[time.time_ns()];self.flow.preparer.clock_ns=lambda:self.clock[0]
        raw=io.BytesIO();Image.new('RGB',(4,3),'blue').save(raw,format='PNG')
        self.upload=self.flow.preparer.upload(self.flow.namespace,raw.getvalue(),pool=self.flow.manager.pool,disk=self.flow.manager.disk)
        self.body=dict(schema=2,namespace_id=self.flow.namespace,kind='initial_turn',add_generation_prompt=True,
            messages=[dict(role='user',content=[dict(type='text',text=' New diagram: '),
                dict(type='image',image_id=self.upload['image_id']),dict(type='text',text=' Which color?')])])

    def prepare(self,*,body=None,revision=1,cancelled=None):
        body=self.body if body is None else body
        presentation=self.flow.preparer.presentation(body,'suffix')
        return self.flow.manager.prepare_suffix('main',revision,body,'suffix',2,
            presentation=presentation,cancelled=cancelled)

    def test_preparation_is_worker_owned_and_does_not_create_history_or_evaluate(self):
        entries=set(self.flow.store.objects.iterdir())
        with self.prepare() as prepared:
            self.assertIsInstance(prepared.history,InputHistory)
            self.assertEqual(prepared.history.spans[1]['original_sha256'],self.upload['image_id'])
            self.assertEqual(self.flow.encoder.calls[-1],self.flow.encoder.calls[0])
            self.assertNotEqual(self.flow.encoder.calls[-1],threading.get_ident())
            self.assertEqual(prepared.receipt['forward_tokens'],0)
            self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)
            self.assertEqual(self.flow.store.head('main').tip,self.root)
            self.assertEqual(set(self.flow.store.objects.iterdir()),entries)
            self.assertFalse((self.flow.store.root/'pool-operations/suffix.json').exists())

    def test_prepared_suffix_submits_without_replaying_source_and_remains_pinned(self):
        with self.prepare() as prepared:
            key=('image',encoding_key(prepared.history.spans[1]['normalized_sha256'],prepared.history.spans[1]['encoder_profile_id']))
            with self.assertRaises(PoolFull):self.flow.manager.pool.remove(key)
            request=self.flow.manager.submit('main',prepared.history,2,'suffix',expected_revision=1,
                presentation=self.flow.preparer.presentation(self.body,'suffix'),artifacts=prepared.artifacts)
        request.durable.result(5)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens']-self.before,len(prepared.history)+2)
        self.assertEqual(self.flow.tokens.history('main'),self.prior+prepared.history+tuple(request.snapshot.result(5)['tokens']))

    def test_stale_source_and_wrong_namespace_reject_before_encoding(self):
        calls=len(self.flow.encoder.calls)
        with self.assertRaises(Conflict),self.prepare(revision=0):pass
        other=self.flow.store.create_namespace('Other','other').namespace_id
        body=copy.deepcopy(self.body);body['namespace_id']=other
        with self.assertRaises((InvalidOperation,NotFound)),self.prepare(body=body):pass
        self.assertEqual(len(self.flow.encoder.calls),calls)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)

    def test_rejects_multiple_messages_non_user_and_non_generation_suffixes(self):
        for field,value in (('messages',self.body['messages']*2),('messages',[dict(role='assistant',content='no')]),
                            ('add_generation_prompt',False),('kind','root')):
            body=copy.deepcopy(self.body);body[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(InvalidOperation),self.prepare(body=body):pass

    def test_cancelled_preparation_releases_uploaded_originals(self):
        with self.assertRaises(CancelledError),self.prepare(cancelled=lambda:True):pass
        self.clock[0]+=1801*10**9;artifacts=self.flow.preparer.artifacts(self.flow.manager.pool,self.flow.manager.disk)
        artifacts.collect();self.assertFalse((artifacts.originals/(self.upload['image_id']+'.input')).exists())
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)

    def test_encoder_failure_preserves_source_and_releases_leases(self):
        self.flow.encoder.fail=True
        with self.assertRaisesRegex(InvalidOperation,'encoder failure'),self.prepare():pass
        self.assertEqual(self.flow.store.head('main').tip,self.root)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)
        self.clock[0]+=1801*10**9;artifacts=self.flow.preparer.artifacts(self.flow.manager.pool,self.flow.manager.disk)
        artifacts.collect();self.assertFalse((artifacts.originals/(self.upload['image_id']+'.input')).exists())

    def test_next_user_prefix_uses_existing_consumed_eos_separator(self):
        self.flow.models[-1].backend.next_token=257
        self.flow.text.ask('main','finish',1,'finish',2).durable.result(5)
        prior=self.flow.tokens.history('main');body=copy.deepcopy(self.body);body['kind']='user_turn'
        with self.flow.manager.prepare_suffix('main',2,body,'next',2,
                presentation=self.flow.preparer.presentation(body,'next')) as prepared:
            spans=[span for span in prepared.history.spans if span['kind']=='image']
            frame=self.flow.mixed.encode(body['messages'],spans,add_generation_prompt=True)
            self.assertEqual(prepared.history,InputHistory([dict(kind='text',tokens=self.flow.tokenizer.encode('\n',parse_special=False))])+frame)
            self.assertEqual(self.flow.tokens.history('main'),prior)

    def run_blocked(self,action):
        self.flow.encoder.entered.clear();self.flow.encoder.gate.clear();outcome=[]
        def run():
            try:
                with action() as prepared:outcome.append(prepared.history)
            except BaseException as error:outcome.append(error)
        thread=threading.Thread(target=run);thread.start()
        self.addCleanup(thread.join,5);self.addCleanup(self.flow.encoder.gate.set)
        self.assertTrue(self.flow.encoder.entered.wait(5));return thread,outcome

    def test_request_is_detached_and_no_branch_is_published_during_encoding(self):
        original=copy.deepcopy(self.body)
        thread,outcome=self.run_blocked(lambda:self.prepare())
        self.body['messages'][0]['content'][0]['text']='tampered after admission'
        self.assertEqual(self.flow.store.head('main').tip,self.root)
        self.flow.encoder.gate.set();thread.join(5);self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0],InputHistory)
        images=[span for span in outcome[0].spans if span['kind']=='image']
        self.assertEqual(outcome[0],self.flow.mixed.encode(original['messages'],images,add_generation_prompt=True))

    def test_source_changed_during_encoding_is_rejected_without_forward_work(self):
        thread,outcome=self.run_blocked(lambda:self.prepare())
        self.flow.store.move('main',1,self.root,'external-source-move')
        self.flow.encoder.gate.set();thread.join(5);self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0],Conflict)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)

    def test_cancellation_during_encoding_waits_for_cleanup_and_preserves_source(self):
        stopped=threading.Event();thread,outcome=self.run_blocked(lambda:self.prepare(cancelled=stopped.is_set))
        stopped.set();self.flow.encoder.gate.set();thread.join(5);self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0],CancelledError)
        self.assertEqual(self.flow.store.head('main').tip,self.root)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],self.before)

    def test_forged_presentation_rejects_before_encoder_or_history_changes(self):
        presentation=self.flow.preparer.presentation(self.body,'suffix');presentation['messages'][0]['content']='forged'
        calls=len(self.flow.encoder.calls)
        with self.assertRaisesRegex(InvalidOperation,'presentation'),self.flow.manager.prepare_suffix(
                'main',1,self.body,'suffix',2,presentation=presentation):pass
        self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_encoder_timeout_is_reported_instead_of_being_treated_as_a_wait(self):
        stopped=threading.Event();outcome=[]
        def timeout(*args,**options):raise TimeoutError('actual encoder timeout')
        self.flow.encoder.encode=timeout
        def cancellation():
            if stopped.is_set():raise CancelledError('Clean up the deliberately failing timeout probe')
            return False
        def run():
            try:
                with self.prepare(cancelled=cancellation):pass
            except BaseException as error:outcome.append(error)
        thread=threading.Thread(target=run);thread.start()
        try:
            thread.join(1)
            self.assertFalse(thread.is_alive(),'A terminal encoder timeout must not become an endless Future wait')
            self.assertIsInstance(outcome[0],TimeoutError)
            self.assertIn('actual encoder timeout',str(outcome[0]))
        finally:stopped.set();thread.join(5)

    def test_close_releases_a_prepared_borrow_without_publishing_or_leaking(self):
        with self.prepare() as prepared:
            self.assertTrue(prepared.history)
            self.flow.close()
        self.assertEqual(self.flow.store.head('main').tip,self.root)


if __name__=='__main__':unittest.main()
