"""Image-bearing saved-chat questions; real Store and synthetic CPU inference."""
from contextlib import contextmanager
import copy
import hashlib
import http.client
import io
import json
import threading
import time
import unittest

from PIL import Image
from context_time_travel.input_spans import InputHistory
from context_time_travel.store import Conflict, InvalidOperation
from serve.cow_http import make_server
from serve.cow_text_adapter import UnsupportedTextRequest
import serve.test_cow_vision_text_flow as fixture


class CowVisionSuffixFlowTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests();self.addCleanup(self.flow.doCleanups);self.flow.setUp()
        self.root=self.flow.text.create('main',self.flow.messages,'context',namespace_id=self.flow.namespace).durable.result(5)['snapshot']
        raw=io.BytesIO();Image.new('RGB',(4,3),'blue').save(raw,format='PNG')
        self.upload=self.flow.preparer.upload(self.flow.namespace,raw.getvalue(),pool=self.flow.manager.pool,disk=self.flow.manager.disk)
        self.content=[dict(type='text',text='Before '),dict(type='image',image_id=self.upload['image_id']),
                      dict(type='text',text=' after literal <|image_pad|>')]

    def ask(self,content=None,revision=1,operation='question',max_tokens=2,**options):
        return self.flow.text.ask('main',self.content if content is None else content,revision,operation,max_tokens,**options)

    def test_new_image_question_publishes_only_delta_and_generated_text(self):
        prior=self.flow.tokens.history('main');before=self.flow.models[-1].backend.stats['forward_tokens']
        handle=self.ask();receipt=handle.durable.result(5)
        self.assertEqual(handle.text_result(),'xx')
        generation=self.flow.manager.committed_generation('question')['generation']
        self.assertEqual(handle.input_preparation['forward_tokens'],0)
        self.assertEqual(handle.input_preparation['images'][0]['original_sha256'],self.upload['image_id'])
        self.assertEqual(generation['schema'],9);delta=InputHistory(generation['ids'])
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens']-before,len(delta)+2)
        self.assertEqual(self.flow.tokens.history('main'),prior+delta+tuple(handle.snapshot.result(5)['tokens']))
        view=self.flow.workspace.history('main')
        self.assertEqual(len(view['image_inputs']),2)
        self.assertEqual(view['messages'][-2]['content'],
            'Before [image sha256:'+self.upload['image_id']+'] after literal <|image_pad|>')
        self.assertEqual(receipt['branch']['revision'],2)

    def test_completed_replay_after_reopen_needs_no_encoder_upload_or_embedding(self):
        first=self.ask();receipt=first.durable.result(5);self.flow.close()
        for path in self.flow.store.root.glob('checkpoints/*.sve'):path.unlink()
        for path in self.flow.store.root.glob('image-upload-grants/*.json'):path.unlink()
        self.flow.without_encoder=True;self.flow.start()
        calls=len(self.flow.encoder.calls);before=self.flow.models[-1].backend.stats['forward_tokens']
        replay=self.ask();self.assertEqual(replay.durable.result(5)['branch'],receipt['branch'])
        self.assertTrue(replay.snapshot.result(5)['recovered']);self.assertEqual(replay.text_result(),'xx')
        self.assertEqual(replay.input_preparation['source'],'committed-replay')
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)
        self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_changed_text_image_bound_or_role_cannot_rebind_completed_operation(self):
        self.ask().durable.result(5);head=self.flow.store.head('main');calls=len(self.flow.encoder.calls)
        changed=copy.deepcopy(self.content);changed[0]['text']='different'
        other=copy.deepcopy(self.content);other[1]['image_id']='f'*64
        for content,bound in ((changed,2),(other,2),(self.content,3)):
            with self.subTest(content=content,bound=bound),self.assertRaises((ValueError,InvalidOperation)):
                self.ask(content=content,max_tokens=bound)
        self.assertEqual(self.flow.store.head('main'),head);self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_next_user_image_preserves_consumed_eos_and_required_separator(self):
        self.flow.models[-1].backend.next_token=257;self.flow.text.ask('main','finish',1,'first',2).durable.result(5)
        prior=self.flow.tokens.history('main');self.flow.models[-1].backend.next_token=120
        handle=self.flow.text.append_user('main',self.content,2,'next',2);handle.durable.result(5)
        generation=self.flow.manager.committed_generation('next')['generation'];delta=InputHistory(generation['ids'])
        images=[span for span in delta.spans if span['kind']=='image']
        frame=self.flow.mixed.encode([dict(role='user',content=self.content)],images,add_generation_prompt=True)
        expected=InputHistory([dict(kind='text',tokens=self.flow.tokenizer.encode('\n',parse_special=False))])+frame
        self.assertEqual(delta,expected);self.assertEqual(self.flow.tokens.history('main').segment(0,len(prior)),prior)
        replay=self.flow.text.append_user('main',self.content,2,'next',2)
        self.assertEqual(replay.durable.result(5)['branch'],handle.durable.result(5)['branch'])

    def test_unsupported_controls_and_stale_source_do_not_encode(self):
        calls=len(self.flow.encoder.calls)
        for options in (dict(settings={'temperature':.7}),dict(seed=3),dict(grammar_state='READ'),dict(text_stops=['END'])):
            with self.subTest(options=options),self.assertRaises(UnsupportedTextRequest):self.ask(**options)
        with self.assertRaises(Conflict):self.ask(revision=0)
        self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_missing_producer_for_new_image_has_a_clear_error(self):
        self.flow.text.vision_preparer=None
        with self.assertRaisesRegex(UnsupportedTextRequest,'preparation.*configured'):self.ask()

    def test_pending_duplicate_reports_existing_work_without_another_encoding(self):
        backend=self.flow.models[-1].backend;original=backend.generate_request;entered=threading.Event();gate=threading.Event()
        @contextmanager
        def blocked(lease,count):
            entered.set()
            if not gate.wait(5):raise TimeoutError('CPU generation gate timed out')
            with original(lease,count) as tokens:yield tokens
        backend.generate_request=blocked
        handle=self.ask()
        try:
            self.assertTrue(entered.wait(5));calls=len(self.flow.encoder.calls)
            with self.assertRaisesRegex(Conflict,'operation|pending|recover'):self.ask()
            self.assertEqual(len(self.flow.encoder.calls),calls)
        finally:gate.set()
        handle.durable.result(5)


class CowVisionSuffixHTTPTests(unittest.TestCase):
    def setUp(self):
        self.case=CowVisionSuffixFlowTests();self.addCleanup(self.case.doCleanups);self.case.setUp();self.flow=self.case.flow
        self.server=make_server(self.flow.tokens,api_key='secret',text_adapter=self.flow.text,workspace=self.flow.workspace)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.addCleanup(self.close)

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5)

    def post(self,document,key='secret'):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        self.addCleanup(connection.close)
        connection.request('POST','/cow/text/ask',json.dumps(document),
            {'Content-Type':'application/json','Authorization':'Bearer '+key})
        response=connection.getresponse();return response.status,json.loads(response.read())

    def body(self):
        return dict(branch='main',revision=1,content=self.case.content,operation_id='http-question',max_tokens=2,stream=False)

    def test_http_mixed_question_and_exact_replay_preserve_ids(self):
        code,first=self.post(self.body());self.assertEqual(code,200,first)
        before=self.flow.models[-1].backend.stats['forward_tokens'];code,again=self.post(self.body())
        self.assertEqual(code,200,again);self.assertEqual(first['receipt']['branch'],again['receipt']['branch'])
        self.assertEqual(first['answer']['input_preparation']['forward_tokens'],0)
        self.assertEqual(again['answer']['input_preparation']['source'],'committed-replay')
        self.assertEqual(again['answer']['text'],'xx');self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_auth_and_invalid_url_parts_cannot_trigger_encoder(self):
        calls=len(self.flow.encoder.calls);self.assertEqual(self.post(self.body(),key='wrong')[0],401)
        body=self.body();body['content']=[dict(type='image',url='file:///etc/passwd')]
        self.assertEqual(self.post(body)[0],400);self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_head_includes_namespace_without_loading_conversation(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5);self.addCleanup(connection.close)
        connection.request('GET','/cow/head?branch=main',headers={'Authorization':'Bearer secret'})
        response=connection.getresponse();self.assertEqual(response.status,200)
        value=json.loads(response.read());self.assertEqual(value['namespace_id'],self.flow.namespace)
        self.assertEqual(value['tip'],self.case.root)

    def test_disconnect_during_encoding_cancels_before_model_work_or_publication(self):
        self.flow.encoder.entered.clear();self.flow.encoder.gate.clear();self.addCleanup(self.flow.encoder.gate.set)
        before=self.flow.models[-1].backend.stats['forward_tokens']
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        connection.request('POST','/cow/text/ask',json.dumps(self.body()),
            {'Content-Type':'application/json','Authorization':'Bearer secret'})
        try:self.assertTrue(self.flow.encoder.entered.wait(5))
        finally:connection.close();self.flow.encoder.gate.set()
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with self.flow.manager._condition:
                if not self.flow.manager._input_preparations:break
            time.sleep(.02)
        else:self.fail('Disconnected image preparation did not retire')
        self.assertEqual(self.flow.store.head('main').tip,self.case.root)
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)
        self.assertFalse((self.flow.store.root/'pool-operations/http-question.json').exists())


if __name__=='__main__':unittest.main()
