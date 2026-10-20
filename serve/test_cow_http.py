"""HTTP transport fixtures only; no GPU inference qualification."""
from concurrent.futures import CancelledError, Future
import json
import queue
import socket
import threading
import unittest
import urllib.error
import urllib.request
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextRequest


class Adapter:
    def __init__(self): self.calls=[]
    def generate(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        request=type('Request',(),{})()
        request.events=queue.SimpleQueue()
        request.events.put(dict(event='token',token=5))
        request.snapshot=Future();request.snapshot.set_result(dict(tokens=[5]))
        request.durable=Future();request.durable.set_result(dict(snapshot='saved',branch=dict(revision=1)))
        return request
    def head(self,branch): return dict(name=branch,revision=0,tip='root')
    def fork(self,name,snapshot): return dict(name=name,tip=snapshot,revision=0)
    def rewind(self,*args): return dict(name=args[0],tip=args[2],revision=args[1]+1)


class CowHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter=Adapter()
        self.server=make_server(self.adapter,api_key='secret')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5)
    def call(self,path,body=None,key='secret',origin=None):
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
        if origin: headers['Origin']=origin
        req=urllib.request.Request(self.base+path,headers=headers,
                                  data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req,timeout=5) as response:
                return response.status,response.read().decode()
        except urllib.error.HTTPError as error:
            with error: return error.code,error.read().decode()
    def request(self,**more):
        return dict(branch='A',revision=0,operation_id='request-a',rendered_token_ids=[1,2],
                    max_tokens=8,settings={'temperature':0},**more)
    def test_auth_and_foreign_origin_reject_before_admission(self):
        self.assertEqual(self.call('/cow/generate',self.request(),key='wrong')[0],401)
        self.assertEqual(self.call('/cow/generate',self.request(),origin='http://evil.example')[0],403)
        self.assertEqual(self.adapter.calls,[])
    def test_stream_separates_provisional_token_and_durable_success(self):
        code,body=self.call('/cow/generate',self.request())
        self.assertEqual(code,200)
        events=[json.loads(row) for row in body.splitlines()]
        self.assertEqual([e['event'] for e in events],['token','complete'])
        self.assertEqual(events[-1]['receipt']['snapshot'],'saved')
        args,kwargs=self.adapter.calls[0]
        self.assertEqual(args[:3],('A',[1,2],0))
        self.assertTrue(callable(kwargs['cancelled']))
    def test_nonstream_response_contains_acknowledged_receipt(self):
        code,body=self.call('/cow/generate',self.request(stream=False))
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['receipt']['snapshot'],'saved')
    def test_publication_failure_never_emits_complete(self):
        original=self.adapter.generate
        def failed(*args,**kwargs):
            req=original(*args,**kwargs);req.durable=Future()
            req.durable.set_exception(RuntimeError('disk full'));return req
        self.adapter.generate=failed
        code,body=self.call('/cow/generate',self.request())
        self.assertEqual(code,200)
        self.assertEqual([json.loads(row)['event'] for row in body.splitlines()],['token','error'])
    def test_missing_request_fields_reject_before_admission(self):
        self.assertEqual(self.call('/cow/generate',{'branch':'A'})[0],400)
        self.assertEqual(self.adapter.calls,[])
    def test_unknown_options_do_not_silently_change_the_contract(self):
        self.assertEqual(self.call('/cow/generate',self.request(tools=[{}]))[0],400)
        self.assertEqual(self.adapter.calls,[])
    def test_head_fork_and_rewind_use_adapter_authority(self):
        self.assertEqual(json.loads(self.call('/cow/head?branch=A')[1])['tip'],'root')
        code,body=self.call('/cow/fork',dict(name='B',snapshot='root'))
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['name'],'B')
        code,body=self.call('/cow/rewind',dict(branch='A',revision=2,snapshot='root',operation_id='rewind'))
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['revision'],3)

    def test_catalog_route_is_authenticated_read_only_and_root_scoped(self):
        roots=[]
        def branches(root=None):
            roots.append(root)
            return (dict(name='A',root=root,tip='saved',revision=2),)
        self.adapter.branches=branches
        code,body=self.call('/cow/branches?root=chosen')
        self.assertEqual(code,200)
        self.assertEqual(json.loads(body)['branches'],[dict(name='A',root='chosen',tip='saved',revision=2)])
        self.assertEqual(self.call('/cow/branches',key='wrong')[0],401)
        self.assertEqual(self.call('/cow/branches?root=x&root=y')[0],400)
        self.assertEqual(self.call('/cow/branches?unexpected=x')[0],400)
        self.assertEqual(roots,['chosen'])
        self.assertEqual(self.adapter.calls,[])
    def test_public_bind_requires_key(self):
        with self.assertRaises(ValueError): make_server(self.adapter,host='0.0.0.0')

    def test_text_routes_are_explicitly_disabled_without_text_adapter(self):
        code,_=self.call('/cow/text/generate',dict(branch='A',messages=[{'role':'user','content':'x'}],
            revision=0,operation_id='text-a',max_tokens=8))
        self.assertEqual(code,404);self.assertEqual(self.adapter.calls,[])

    def test_text_stream_and_nonstream_decode_same_canonical_tokens(self):
        outer=self.adapter
        class Tokenizer:
            def token_bytes(self,token): return b'hello'
        class Text:
            def create(self,*a): return outer.generate(*a)
            def generate(self,*a,**kw):
                return CowTextRequest(outer.generate(*a,**kw),Tokenizer(),
                    controller_reader=lambda snapshot:dict(status='active',policy=dict(eos_tokens=[])))
            def resume(self,*a,**kw): return self.generate(*a,**kw)
        server=make_server(outer,api_key='secret',text_adapter=Text())
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        old=self.base;self.base='http://127.0.0.1:'+str(server.server_port)
        try:
            body=dict(branch='A',messages=[{'role':'user','content':'question'}],revision=0,
                      operation_id='text-a',max_tokens=8)
            code,raw=self.call('/cow/text/generate',body)
            self.assertEqual(code,200)
            events=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual(events[0],dict(event='text',token_id=5,text='hello'))
            self.assertEqual(events[-1]['answer']['text'],'hello')
            code,raw=self.call('/cow/text/generate',dict(body,stream=False))
            self.assertEqual(code,200);self.assertEqual(json.loads(raw)['answer']['text'],'hello')
            resume=dict(branch='A',revision=1,operation_id='text-resume',max_tokens=8,stream=False)
            code,raw=self.call('/cow/text/resume',resume)
            self.assertEqual(code,200);self.assertEqual(json.loads(raw)['answer']['text'],'hello')
            before=len(outer.calls)
            self.assertEqual(self.call('/cow/text/resume',dict(resume,messages=[]))[0],400)
            self.assertEqual(len(outer.calls),before)
        finally:
            self.base=old;server.shutdown();server.server_close();thread.join(5)

    def test_explicit_cancel_stops_an_admitted_request_at_its_cancel_boundary(self):
        admitted=threading.Event();stopped=threading.Event();response={}
        def pending(*args,**kwargs):
            request=type('Request',(),{})()
            request.events=queue.SimpleQueue();request.snapshot=Future();request.durable=Future()
            def observe():
                admitted.set()
                deadline=threading.Event()
                for _ in range(100):
                    if kwargs['cancelled']():
                        stopped.set()
                        request.snapshot.set_exception(CancelledError())
                        request.durable.set_exception(CancelledError())
                        return
                    deadline.wait(.02)
            threading.Thread(target=observe,daemon=True).start()
            return request
        self.adapter.generate=pending
        def send():
            response['result']=self.call('/cow/generate',self.request())
        thread=threading.Thread(target=send);thread.start()
        self.assertTrue(admitted.wait(5))
        code,body=self.call('/cow/cancel',{'operation_id':'request-a'})
        self.assertEqual(code,200,body)
        self.assertEqual(json.loads(body),{'operation_id':'request-a','active':True,'cancel_requested':True})
        self.assertTrue(stopped.wait(5))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        status,raw=response['result']
        self.assertEqual(status,200)
        events=[json.loads(line) for line in raw.splitlines()]
        self.assertEqual(events[-1]['event'],'error')
        self.assertTrue(events[-1]['cancelled'])
        self.assertNotIn('complete',[event['event'] for event in events])

    def test_disconnect_sets_transient_cancellation_before_publication(self):
        admitted=threading.Event();stopped=threading.Event()
        def pending(*args,**kwargs):
            request=type('Request',(),{})()
            request.events=queue.SimpleQueue();request.snapshot=Future();request.durable=Future()
            def observe():
                admitted.set()
                for _ in range(100):
                    if kwargs['cancelled'](): stopped.set();return
                    stopped.wait(.05)
            threading.Thread(target=observe,daemon=True).start()
            return request
        self.adapter.generate=pending
        body=json.dumps(self.request()).encode()
        client=socket.create_connection(self.server.server_address,timeout=5)
        client.sendall(b'POST /cow/generate HTTP/1.0\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
        self.assertTrue(admitted.wait(5))
        client.shutdown(socket.SHUT_RDWR);client.close()
        self.assertTrue(stopped.wait(5),'Disconnect did not cancel pending computation')

    def test_append_user_route_passes_only_new_content_and_streams_text(self):
        outer=self.adapter
        admitted=[]
        class Tokenizer:
            def token_bytes(self, token): return b'hello'
        class Text:
            def append_user(self, *args, **kwargs):
                admitted.append((args, kwargs))
                return CowTextRequest(outer.generate(*args, **kwargs), Tokenizer(),
                    controller_reader=lambda snapshot: dict(status='active',policy=dict(eos_tokens=[])))
        server=make_server(outer,api_key='secret',text_adapter=Text())
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        old=self.base;self.base='http://127.0.0.1:'+str(server.server_port)
        try:
            body=dict(branch='A',content='a new question',revision=3,
                      operation_id='append-next',max_tokens=8)
            code,raw=self.call('/cow/text/append-user',body)
            self.assertEqual(code,200)
            events=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual(events[0],dict(event='text',token_id=5,text='hello'))
            self.assertEqual(events[-1]['answer']['text'],'hello')
            self.assertEqual(admitted[0][0],('A','a new question',3,'append-next',8))
            self.assertTrue(callable(admitted[0][1]['cancelled']))
            before=len(admitted)
            self.assertEqual(self.call('/cow/text/append-user',dict(body,messages=[]))[0],400)
            self.assertEqual(len(admitted),before)
        finally:
            self.base=old;server.shutdown();server.server_close();thread.join(5)


if __name__=='__main__': unittest.main()
