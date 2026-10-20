"""Real HTTP/store/pool path; synthetic CPU inference and encoder only."""
import hashlib
import http.client
import io
import json
import socket
import threading
import unittest

from PIL import Image
from serve.cow_http import make_server
import serve.test_cow_vision_text_flow as fixture


class CowVisionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests()
        self.addCleanup(self.flow.doCleanups)
        self.flow.setUp()
        self.server=make_server(self.flow.tokens,api_key='secret',text_adapter=self.flow.text,
                                workspace=self.flow.workspace)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close_server)
        output=io.BytesIO();Image.new('RGB',(5,3),'blue').save(output,format='PNG');self.raw=output.getvalue()
        self.path='/cow/images?namespace='+self.flow.namespace

    def close_server(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5)

    def call(self,path,raw,*,headers=None,key='secret'):
        values={'Authorization':'Bearer '+key,'Content-Type':'application/octet-stream',
                'X-COW-Content-SHA256':hashlib.sha256(raw).hexdigest()}
        values.update(headers or {})
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        self.addCleanup(connection.close)
        connection.request('POST',path,body=raw,headers=values)
        response=connection.getresponse();return response.status,json.loads(response.read())

    def json_call(self,path,document):
        return self.call(path,json.dumps(document).encode(),headers={'Content-Type':'application/json'})

    def inventory(self):
        return {str(p.relative_to(self.flow.store.root)):p.read_bytes()
                for pattern in ('image-originals/*.input','image-upload-grants/*.json','objects/*.json')
                for p in self.flow.store.root.glob(pattern)}

    def header_only(self,*,headers=None,path=None,key='secret'):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
        self.addCleanup(connection.close)
        connection.putrequest('POST',self.path if path is None else path)
        values={'Authorization':'Bearer '+key,'Content-Type':'application/octet-stream',
                'X-COW-Content-SHA256':hashlib.sha256(self.raw).hexdigest(),'Content-Length':str(len(self.raw))}
        values.update(headers or {})
        for name,value in values.items():connection.putheader(name,value)
        connection.endheaders()
        response=connection.getresponse();return response.status,json.loads(response.read())

    def test_upload_is_content_addressed_and_does_not_encode_or_forward(self):
        before=self.flow.models[-1].backend.stats['forward_tokens']
        code,receipt=self.call(self.path,self.raw)
        self.assertEqual(code,200);self.assertEqual(receipt['image_id'],hashlib.sha256(self.raw).hexdigest())
        self.assertEqual((receipt['width'],receipt['height']),(5,3))
        self.assertEqual(receipt['media_type'],'image/png')
        self.assertEqual(self.call(self.path,self.raw)[1]['image_id'],receipt['image_id'])
        self.assertEqual(self.flow.encoder.calls,[])
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_upload_then_context_fork_and_question_uses_existing_routes(self):
        code,image=self.call(self.path,self.raw);self.assertEqual(code,200)
        messages=[dict(role='user',content=[dict(type='image',image_id=image['image_id']),
                                          dict(type='text',text='Keep this diagram.')])]
        code,root=self.json_call('/cow/text/context',dict(branch='main',namespace_id=self.flow.namespace,
            messages=messages,operation_id='context',stream=False))
        self.assertEqual(code,200);self.assertEqual(root['event'],'complete')
        code,fork=self.json_call('/cow/fork',dict(name='other',snapshot=root['receipt']['snapshot'],
            source_branch='main',operation_id='fork'))
        self.assertEqual(code,200)
        code,answer=self.json_call('/cow/text/ask',dict(branch='other',revision=fork['revision'],
            content='What color?',operation_id='question',max_tokens=2,stream=False))
        self.assertEqual(code,200);self.assertEqual(answer['answer']['text'],'xx')
        self.assertEqual(len(self.flow.encoder.calls),1)

    def test_disconnect_during_initial_encoding_never_publishes_a_root(self):
        import time
        # Real HTTP/coordinator/store; only the encoder/model are CPU fixtures.
        code,image=self.call(self.path,self.raw);self.assertEqual(code,200)
        baseline=self.flow.text.create('retained',self.flow.messages,'retained-root',
                                      namespace_id=self.flow.namespace).durable.result(5)
        retained=self.flow.store.head('retained')
        for index,route in enumerate(('/cow/text/context','/cow/text/prepare')):
            with self.subTest(route=route):
                variant=io.BytesIO();Image.new('RGB',(5+index,3),(0,64+32*index,255)).save(variant,format='PNG')
                code,image=self.call(self.path,variant.getvalue());self.assertEqual(code,200)
                self.flow.encoder.entered.clear();self.flow.encoder.gate.clear()
                self.addCleanup(self.flow.encoder.gate.set)
                branch='disconnected-'+str(index)
                body=dict(branch=branch,namespace_id=self.flow.namespace,operation_id='disconnect-root-'+str(index),
                    messages=[dict(role='user',content=[dict(type='image',image_id=image['image_id']),
                        dict(type='text',text='Retain this image context.')])],stream=False)
                work=self.flow.models[-1].backend.stats['forward_tokens']
                connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
                connection.request('POST',route,json.dumps(body),
                    {'Content-Type':'application/json','Authorization':'Bearer secret'})
                try:self.assertTrue(self.flow.encoder.entered.wait(5))
                finally:connection.close();self.flow.encoder.gate.set()
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    metrics=self.flow.manager.metrics()
                    if not metrics['queued'] and not any(metrics['active_per_gpu'].values()) and not metrics['pending_branches']:break
                    time.sleep(.01)
                else:self.fail('Disconnected root preparation did not retire')
                self.assertNotIn(branch,[head.name for head in self.flow.store.branches()])
                self.assertEqual(self.flow.store.head('retained'),retained)
                self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],work)
                self.assertEqual(self.flow.manager.pool.metrics()['reserved_bytes'],0)
                self.assertEqual(self.flow.manager.disk.metrics()['reserved_bytes'],0)
                # Ensure the next arm needs actual encoding rather than a derived hit.

    def test_authentication_and_origin_reject_before_reading_body(self):
        before=self.inventory()
        self.assertEqual(self.header_only(key='wrong')[0],401)
        self.assertEqual(self.header_only(headers={'Origin':'http://evil.example'})[0],403)
        self.assertEqual(self.inventory(),before)

    def test_digest_mismatch_and_invalid_image_publish_nothing(self):
        before=self.inventory()
        code,value=self.call(self.path,self.raw,headers={'X-COW-Content-SHA256':'0'*64})
        self.assertEqual(code,400);self.assertIn('digest',value['error'].lower())
        self.assertEqual(self.inventory(),before)
        code,_=self.call(self.path,b'not an image');self.assertEqual(code,400)
        self.assertEqual(self.inventory(),before);self.assertEqual(self.flow.encoder.calls,[])

    def test_unknown_namespace_rejects_before_reading_body(self):
        code,value=self.header_only(path='/cow/images?namespace='+'f'*32)
        self.assertEqual(code,404);self.assertIn('error',value)

    def test_invalid_headers_and_query_reject_before_reading_body(self):
        cases=[({'Content-Length':'0'},None),({'Content-Length':str(16*1024**2+1)},None),
            ({'Content-Length':'-1'},None),({'Content-Length':'x'},None),
            ({'X-COW-Content-SHA256':'bad'},None),({'Content-Type':'application/json'},None),
            ({'Transfer-Encoding':'chunked'},None),({},self.path+'&namespace='+self.flow.namespace),
            ({},self.path+'&path=/etc/passwd'),({},'/cow/images?namespace=bad')]
        for headers,path in cases:
            with self.subTest(headers=headers,path=path):
                self.assertEqual(self.header_only(headers=headers,path=path)[0],400)
        self.assertEqual(self.flow.encoder.calls,[])

    def test_conflicting_duplicate_headers_reject_before_reading(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
        self.addCleanup(connection.close);connection.putrequest('POST',self.path)
        for name,value in [('Authorization','Bearer secret'),('Content-Type','application/octet-stream'),
            ('X-COW-Content-SHA256',hashlib.sha256(self.raw).hexdigest()),
            ('Content-Length',str(len(self.raw))),('Content-Length','1')]:connection.putheader(name,value)
        connection.endheaders();response=connection.getresponse()
        self.assertEqual(response.status,400);response.read()

    def test_truncated_body_releases_reservation_and_keeps_store_unchanged(self):
        before=self.inventory();connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        self.addCleanup(connection.close);connection.putrequest('POST',self.path)
        for name,value in [('Authorization','Bearer secret'),('Content-Type','application/octet-stream'),
            ('X-COW-Content-SHA256',hashlib.sha256(self.raw).hexdigest()),
            ('Content-Length',str(len(self.raw)))]:connection.putheader(name,value)
        connection.endheaders();connection.send(self.raw[:10]);connection.sock.shutdown(socket.SHUT_WR)
        response=connection.getresponse();self.assertEqual(response.status,400);response.read()
        self.assertEqual(self.inventory(),before)
        with self.flow.manager.pool.reserve(self.flow.manager.pool.byte_limit):pass

    def test_capacity_refusal_happens_before_body_read(self):
        with self.flow.manager.pool.reserve(self.flow.manager.pool.byte_limit-1):
            code,value=self.header_only()
        self.assertEqual(code,409);self.assertIn('error',value)
        self.assertEqual(self.flow.encoder.calls,[])

    def test_restore_only_server_has_no_upload_endpoint(self):
        self.flow.text.vision_preparer=None
        self.assertEqual(self.header_only()[0],404)

    def test_upload_requires_explicit_server_key_even_on_loopback(self):
        self.close_server();self.server=make_server(self.flow.tokens,text_adapter=self.flow.text)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.assertEqual(self.header_only()[0],403)


if __name__=='__main__':unittest.main()
