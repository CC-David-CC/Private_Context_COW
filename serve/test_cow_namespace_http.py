"""Real HTTP sockets around synthetic metadata/text adapters; not GPU evidence."""
import json
from concurrent.futures import Future
import queue
import threading
import unittest

from serve import test_cow_http as fixtures
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextRequest


class NamespaceAdapter(fixtures.Adapter):
    def namespaces(self):
        self.calls.append(('list',));return [dict(namespace_id='a'*32,title='Code')]
    def create_namespace(self,title,operation):
        self.calls.append(('namespace',title,operation))
        return dict(namespace_id='a'*32,title=title)
    def presentation_catalog(self,namespace):
        self.calls.append(('catalog',namespace))
        return dict(namespace=dict(namespace_id=namespace),heads=[],events=[])


class Text:
    def __init__(self,outer):self.outer=outer
    def prepared(self):
        request=self.outer.generate()
        request.events=queue.SimpleQueue()
        request.snapshot=Future();request.snapshot.set_result(dict(tokens=[]))
        return request
    def prepare(self,*args,**kwargs):
        self.outer.calls.append(('prepare',args,kwargs));return self.prepared()
    def create(self,*args,**kwargs):
        self.outer.calls.append(('context',args,kwargs));return self.prepared()
    def result(self,kind,*args,**kwargs):
        self.outer.calls.append((kind,args,kwargs))
        class Tokenizer:
            def token_bytes(self,token):return b'hello'
        return CowTextRequest(self.outer.generate(),Tokenizer(),
            controller_reader=lambda snapshot:dict(status='active',policy=dict(eos_tokens=[])))
    def answer(self,*args,**kwargs):return self.result('answer',*args,**kwargs)
    def ask(self,*args,**kwargs):return self.result('ask',*args,**kwargs)


class NamespaceHTTPTests(unittest.TestCase):
    call=fixtures.CowHTTPTests.call
    tearDown=fixtures.CowHTTPTests.tearDown
    def setUp(self):
        self.adapter=NamespaceAdapter()
        self.server=make_server(self.adapter,api_key='secret',text_adapter=Text(self.adapter))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def test_namespace_create_list_and_catalog_are_authenticated(self):
        self.assertEqual(self.call('/cow/namespaces',key='wrong')[0],401)
        self.assertEqual(self.adapter.calls,[])
        code,body=self.call('/cow/namespace',dict(title='Code',operation_id='namespace-op'))
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['namespace_id'],'a'*32)
        code,body=self.call('/cow/namespaces')
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['namespaces'][0]['title'],'Code')
        code,body=self.call('/cow/presentation?namespace='+('a'*32))
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['heads'],[])

    def test_catalog_requires_one_namespace_and_rejects_unknown_query(self):
        for query in ('','?namespace=x&namespace=y','?root=x','?namespace=x&other=y'):
            with self.subTest(query=query):
                self.assertEqual(self.call('/cow/presentation'+query)[0],400)
        self.assertEqual(self.adapter.calls,[])

    def test_prepare_and_context_bind_namespace_without_inventing_an_answer(self):
        body=dict(branch='A',messages=[dict(role='system',content='code context')],
                  namespace_id='a'*32,operation_id='prepare',stream=False)
        for route,kind in (('prepare','prepare'),('context','context')):
            code,raw=self.call('/cow/text/'+route,body)
            self.assertEqual(code,200)
            self.assertNotIn('text',json.loads(raw)['answer'])
            self.assertEqual(json.loads(raw)['answer']['tokens'],[])
            call=[row for row in self.adapter.calls if row[0]==kind][-1]
            self.assertEqual(set(call[2]), {'namespace_id', 'cancelled'})
            self.assertEqual(call[2]['namespace_id'], 'a'*32)
            self.assertTrue(callable(call[2]['cancelled']))

    def test_answer_and_first_question_return_text_only_after_durable_success(self):
        body=dict(branch='A',revision=1,operation_id='answer',max_tokens=8,stream=False)
        code,raw=self.call('/cow/text/answer',body)
        self.assertEqual(code,200);self.assertEqual(json.loads(raw)['answer']['text'],'hello')
        code,raw=self.call('/cow/text/ask',dict(body,content='first question',stream=True))
        self.assertEqual(code,200)
        events=[json.loads(line) for line in raw.splitlines()]
        self.assertEqual(events[0]['event'],'text')
        self.assertEqual(events[-1]['event'],'complete')
        self.assertEqual(events[-1]['answer']['text'],'hello')
        call=[row for row in self.adapter.calls if row[0]=='ask'][0]
        self.assertEqual(call[1],('A','first question',1,'answer',8))
        self.assertTrue(callable(call[2]['cancelled']))

    def test_hidden_presentation_descriptor_and_bad_stream_fail_before_dispatch(self):
        body=dict(branch='A',revision=1,operation_id='answer',max_tokens=8,stream=False)
        self.assertEqual(self.call('/cow/text/answer',dict(body,presentation={}))[0],400)
        self.assertEqual(self.call('/cow/text/answer',dict(body,stream='false'))[0],400)
        self.assertEqual(self.adapter.calls,[])


if __name__=='__main__':unittest.main()
