"""CPU-only real-socket tests for the stdlib terminal COW client."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from tools import cow_cli


class FakeCowHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    calls = []
    stream_mode = 'good'
    retry_mode = 'good'

    def log_message(self, *_args):
        pass

    def _send(self, code, value):
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.calls.append(('GET', self.path, self.headers.get('Authorization')))
        if self.path.startswith('/cow/workspace/history?'):
            return self._send(200, {'branch':'b-test','messages':[{'role':'assistant','content':'Tokyo \u6771\u4eac \u2603'}]})
        if self.path == '/cow/head?branch=b-test':
            return self._send(200, {'name':'b-test','revision':7,'tip':'snap-7','root':'snap-0'})
        if self.path.startswith('/cow/status'):
            return self._send(200, {'status':'ok'})
        return self._send(404, {'error':'missing'})

    def do_POST(self):
        size = int(self.headers.get('Content-Length', '0'))
        data = json.loads(self.rfile.read(size))
        self.calls.append(('POST', self.path, data, self.headers.get('Authorization')))
        if self.path == '/cow/publication/retry':
            receipt={'branch':{'name':'b-test','revision':8,'tip':'snap-8','root':'snap-0'},
                     'snapshot':'snap-8','operation_id':data['operation_id']}
            if type(self).retry_mode == 'wrong-tip':
                receipt['branch']['tip']='snap-old'
            elif type(self).retry_mode == 'bad-revision':
                receipt['branch']['revision']=True
            return self._send(200, {'event':'complete','receipt':{
                **receipt}})
        if self.path == '/cow/text/ask':
            mode = 'unicode-good' if data.get('content') == 'unicode-case' else type(self).stream_mode
            self.send_response(200)
            self.send_header('Content-Type','application/x-ndjson')
            self.send_header('Transfer-Encoding','chunked')
            self.end_headers()
            events = {
                'good': [
                    {'event':'text','text':'héllo ☃'},
                    {'event':'complete','answer':{'text':'héllo ☃','tokens':[1],'snapshot':'snap-8'},
                     'receipt':{'branch':{'name':'b-test','revision':8,'tip':'snap-8','root':'snap-0'},
                                'snapshot':'snap-8'}}],
                'unicode-good': [
                    {'event':'text','text':'Tokyo \u6771\u4eac \u2603'},
                    {'event':'complete','answer':{'text':'Tokyo \u6771\u4eac \u2603','tokens':[1],'snapshot':'snap-8'},
                     'receipt':{'branch':{'name':'b-test','revision':8,'tip':'snap-8','root':'snap-0',
                                          'label':'Tokyo \u6771\u4eac \u2603'},'snapshot':'snap-8'}}],
                'error': [{'event':'text','text':'partial'}, {'event':'error','error':'stale revision'}],
                'eof': [{'event':'text','text':'partial'}],
                'malformed': ['not-json\n'],
                'no-receipt': [{'event':'text','text':'partial'}, {'event':'complete','answer':{},'receipt':None}],
                'empty-receipt': [{'event':'complete','answer':{},'receipt':{}}],
                'wrong-branch': [{'event':'complete','answer':{'snapshot':'snap-8'},
                    'receipt':{'branch':{'name':'b-other','revision':8,'tip':'snap-8','root':'snap-0'},'snapshot':'snap-8'}}],
                'wrong-tip': [{'event':'complete','answer':{'snapshot':'snap-8'},
                    'receipt':{'branch':{'name':'b-test','revision':8,'tip':'snap-old','root':'snap-0'},'snapshot':'snap-8'}}],
                'bad-revision': [{'event':'complete','answer':{'snapshot':'snap-8'},
                    'receipt':{'branch':{'name':'b-test','revision':True,'tip':'snap-8','root':'snap-0'},'snapshot':'snap-8'}}],
                'wrong-answer-snapshot': [{'event':'complete','answer':{'snapshot':'snap-old'},
                    'receipt':{'branch':{'name':'b-test','revision':8,'tip':'snap-8','root':'snap-0'},'snapshot':'snap-8'}}],
            }.get(mode, [])
            for event in events:
                raw = (event if isinstance(event,str) else json.dumps(event,ensure_ascii=False)) .encode() + b'\n'
                self.wfile.write(f'{len(raw):X}\r\n'.encode()+raw+b'\r\n')
                self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            return
        self._send(409, {'error':'unexpected'})


class InterruptedResponse:
    closed = False
    def __enter__(self): return self
    def __exit__(self,*_args): self.close()
    def close(self): type(self).closed = True
    def __iter__(self): raise KeyboardInterrupt


class HeadResponse:
    def __enter__(self): return self
    def __exit__(self,*_args): pass
    def read(self): return json.dumps({'name':'b-test','revision':7,'tip':'snap-7','root':'snap-0'}).encode()


class CowCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeCowHandler.calls = []
        cls.server = ThreadingHTTPServer(('127.0.0.1',0), FakeCowHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        FakeCowHandler.calls = []
        FakeCowHandler.stream_mode = 'good'
        FakeCowHandler.retry_mode = 'good'

    def test_cli_ask_uses_current_revision_and_bearer_auth(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
             mock.patch.dict('os.environ', {'COW_API_KEY':'test-secret'}):
            result = cow_cli.main(['--url',self.url,'ask','--branch','b-test','--text','question',
                                   '--operation-id','op-test'])
        self.assertEqual(result,0)
        self.assertNotIn('operation_id=',out.getvalue())
        self.assertIn('operation_id=op-test',err.getvalue())
        self.assertIn('héllo ☃',out.getvalue())
        self.assertNotIn('snap-8',out.getvalue())
        self.assertIn('snap-8',err.getvalue())
        self.assertEqual([row[0:2] for row in FakeCowHandler.calls],
                         [('GET','/cow/head?branch=b-test'),('POST','/cow/text/ask')])
        body = FakeCowHandler.calls[-1][2]
        self.assertEqual(body['revision'],7)
        self.assertEqual(body['operation_id'],'op-test')
        self.assertEqual(FakeCowHandler.calls[-1][3],'Bearer test-secret')

    def test_stream_errors_and_eof_never_print_a_durable_receipt(self):
        for mode in ('error','eof','malformed','no-receipt','empty-receipt',
                     'wrong-branch','wrong-tip','bad-revision','wrong-answer-snapshot'):
            with self.subTest(mode=mode):
                FakeCowHandler.stream_mode = mode
                provisional=[]
                with self.assertRaises(cow_cli.ClientError):
                    cow_cli.CowClient(self.url).stream('/cow/text/ask',{'branch':'b-test'},provisional.append)
                if mode in ('error','eof','malformed','no-receipt'):
                    self.assertTrue(provisional or mode=='malformed')
                else:
                    self.assertEqual(provisional,[])

    def test_operation_ids_and_terminal_controls_are_safe(self):
        with self.assertRaises(cow_cli.ClientError):
            cow_cli.operation_id('bad id')
        with self.assertRaises(cow_cli.ClientError):
            cow_cli.operation_id('')
        self.assertEqual(cow_cli.safe_text('snow ☃\x1b[31m'), 'snow ☃\\u001b[31m')
        with self.assertRaises(cow_cli.ClientError):
            cow_cli.CowClient('http://remote.example:8099')
        with self.assertRaises(cow_cli.ClientError):
            cow_cli.CowClient('http://user:password@127.0.0.1:8099')
        for timeout in (0,-1,float('nan'),float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(cow_cli.ClientError):
                cow_cli.CowClient('http://127.0.0.1:8099',timeout=timeout)
        with tempfile.TemporaryDirectory() as directory:
            invalid=Path(directory)/'invalid.txt';invalid.write_bytes(b'\xff')
            with self.assertRaisesRegex(cow_cli.ClientError,'valid UTF-8'):
                cow_cli._messages_file(invalid)

    def test_publication_retry_reuses_the_given_operation_id(self):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            result=cow_cli.main(['--url',self.url,'retry-publication','--operation-id','op-recover'])
        self.assertEqual(result,0)
        self.assertIn('operation_id=op-recover',err.getvalue())
        self.assertIn('snap-8',err.getvalue())
        posts=[row for row in FakeCowHandler.calls if row[0]=='POST']
        self.assertEqual(len(posts),1)
        self.assertEqual(posts[0][1],'/cow/publication/retry')
        self.assertEqual(posts[0][2],{'operation_id':'op-recover'})

    def test_publication_retry_rejects_unbound_receipts(self):
        for mode in ('wrong-tip','bad-revision'):
            with self.subTest(mode=mode):
                FakeCowHandler.retry_mode=mode
                out,err=io.StringIO(),io.StringIO()
                with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                    result=cow_cli.main(['--url',self.url,'retry-publication','--operation-id','op-recover'])
                self.assertEqual(result,2)
                self.assertIn('operation_id=op-recover',err.getvalue())
                self.assertNotIn('durable receipt:',err.getvalue())

    def test_mutation_failure_prints_operation_before_attempt_and_does_not_retry(self):
        out,err=io.StringIO(),io.StringIO()
        FakeCowHandler.stream_mode='error'
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            result=cow_cli.main(['--url',self.url,'ask','--branch','b-test','--text','same',
                                 '--operation-id','op-stable'])
        self.assertEqual(result,2)
        self.assertIn('operation_id=op-stable',err.getvalue())
        self.assertIn('did not complete durably',err.getvalue())
        self.assertEqual(sum(call[0]=='POST' for call in FakeCowHandler.calls),1)

    def test_ctrl_c_is_clean_and_closes_stream_response(self):
        out,err=io.StringIO(),io.StringIO()
        InterruptedResponse.closed=False
        def open_response(request,**_kwargs):
            return HeadResponse() if request.get_method()=='GET' else InterruptedResponse()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err), \
             mock.patch('tools.cow_cli.urlopen',side_effect=open_response):
            result=cow_cli.main(['--url',self.url,'ask','--branch','b-test','--text','question',
                                 '--operation-id','op-interrupted'])
        self.assertEqual(result,130)
        self.assertTrue(InterruptedResponse.closed)
        self.assertEqual(out.getvalue(),'')
        self.assertIn('operation_id=op-interrupted',err.getvalue())
        self.assertIn('no durable completion is claimed',err.getvalue())
        self.assertNotIn('Traceback',err.getvalue())

    def test_subprocess_stdio_is_utf8_when_parent_requests_cp1252(self):
        cli_path = str(Path(cow_cli.__file__).resolve())
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'cp1252'
        history = subprocess.run(
            [sys.executable, cli_path, '--url', self.url, 'history', '--branch', 'b-test'],
            capture_output=True, env=env, timeout=10)
        self.assertEqual(history.returncode, 0, history.stderr.decode('utf-8', 'replace'))
        self.assertIn('Tokyo \u6771\u4eac \u2603', history.stdout.decode('utf-8'))

        answer = subprocess.run(
            [sys.executable, cli_path, '--url', self.url, 'ask', '--branch', 'b-test',
             '--text', 'unicode-case', '--operation-id', 'op-unicode'],
            capture_output=True, env=env, timeout=10)
        self.assertEqual(answer.returncode, 0, answer.stderr.decode('utf-8', 'replace'))
        self.assertIn('Tokyo \u6771\u4eac \u2603', answer.stdout.decode('utf-8'))
        self.assertIn('Tokyo \u6771\u4eac \u2603', answer.stderr.decode('utf-8'))

    def test_real_cow_http_synthetic_fixture_context_answer_resume_history(self):
        try:
            from serve.cow_browser_fixture import CowBrowserFixtureServer
        except ImportError:
            self.skipTest('assembled COW fixture dependencies are not on PYTHONPATH')

        def invoke(*argv):
            stdout,stderr=io.StringIO(),io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                code=cow_cli.main(['--url',server.base_url,*argv])
            self.assertEqual(code,0,stderr.getvalue())
            return stdout.getvalue(),stderr.getvalue()

        with CowBrowserFixtureServer(api_key='cli-fixture-key') as server, \
             mock.patch.dict('os.environ', {'COW_API_KEY':'cli-fixture-key'}), \
             tempfile.TemporaryDirectory() as directory:
            denied_out,denied_err=io.StringIO(),io.StringIO()
            with mock.patch.dict('os.environ', {}, clear=True), \
                 contextlib.redirect_stdout(denied_out), contextlib.redirect_stderr(denied_err):
                denied=cow_cli.main(['--url',server.base_url,'status'])
            self.assertEqual(denied,2)
            self.assertEqual(denied_out.getvalue(),'')
            self.assertIn('HTTP 401',denied_err.getvalue())
            stdout,stderr=invoke('status')
            self.assertIn('worker_count',stdout)
            stdout,stderr=invoke('namespace','--title','CLI test','--operation-id','op-cli-ns')
            namespace=json.loads(stderr.split('durable receipt:\n',1)[1])['namespace_id']
            context=Path(directory)/'context.txt';context.write_text('Only cite this tiny fixture.\n',encoding='utf-8')
            stdout,stderr=invoke('load','--namespace-id',namespace,'--branch','b-cli-test',
                '--file',str(context),'--operation-id','op-cli-load')
            self.assertIn('op-cli-load',stderr)
            invoke('ask','--branch','b-cli-test','--text','What is in scope?',
                   '--max-tokens','4','--operation-id','op-cli-ask')
            invoke('continue','--branch','b-cli-test','--max-tokens','4','--operation-id','op-cli-resume')
            history,_=invoke('history','--branch','b-cli-test')
            projected=json.loads(history)
            self.assertEqual(projected['controller_status'],'stopped')
            self.assertEqual([m['role'] for m in projected['messages']],['system','user','assistant'])
            invoke('append-user','--branch','b-cli-test','--text','one more',
                   '--max-tokens','4','--operation-id','op-cli-append')
            context_list,_=invoke('contexts')
            self.assertEqual(json.loads(context_list)['namespaces'][0]['namespace_id'],namespace)
            branch_list,_=invoke('branches')
            self.assertIn('b-cli-test',[row['name'] for row in json.loads(branch_list)['branches']])
            invoke('fork','--source-branch','b-cli-test','--name','b-cli-fork',
                   '--operation-id','op-cli-fork')
            branch_list,_=invoke('branches')
            self.assertIn('b-cli-fork',[row['name'] for row in json.loads(branch_list)['branches']])
            saved_head=cow_cli._head(cow_cli.CowClient(server.base_url,'cli-fixture-key'),'b-cli-test')
            invoke('rewind','--branch','b-cli-test','--snapshot',saved_head['root'],
                   '--operation-id','op-cli-rewind')
            rewound,_=invoke('history','--branch','b-cli-test')
            self.assertEqual(json.loads(rewound)['controller_status'],'idle')
            invoke('prepare','--namespace-id',namespace,'--branch','b-prepared',
                   '--text','prepared prompt','--operation-id','op-cli-prepare')
            invoke('answer','--branch','b-prepared','--max-tokens','4','--operation-id','op-cli-answer')
            prepared_history,_=invoke('history','--branch','b-prepared')
            self.assertEqual([m['role'] for m in json.loads(prepared_history)['messages']],
                             ['user','assistant'])
            operation,_=invoke('operation','--operation-id','op-cli-ask')
            self.assertIn('operation_id',json.loads(operation))

            # A real concurrent rewind after head-read makes ask's CAS stale.
            invoke('load','--namespace-id',namespace,'--branch','b-stale',
                '--file',str(context),'--operation-id','op-stale-load')
            invoke('ask','--branch','b-stale','--text','first turn','--max-tokens','4',
                   '--operation-id','op-stale-ask')
            original_request=cow_cli.CowClient.request
            original_stream=cow_cli.CowClient.stream
            state={'raced':False,'stream_calls':0}
            def race_after_head(client,method,path,body=None):
                result=original_request(client,method,path,body)
                if (method=='GET' and path.startswith('/cow/head?branch=b-stale')
                        and not state['raced']):
                    original_request(client,'POST','/cow/rewind',{
                        'branch':'b-stale','revision':result['revision'],'snapshot':result['root'],
                        'operation_id':'op-external-race'})
                    state['raced']=True
                return result
            def count_stream(client,*args,**kwargs):
                state['stream_calls']+=1
                return original_stream(client,*args,**kwargs)
            stale_out,stale_err=io.StringIO(),io.StringIO()
            with mock.patch.object(cow_cli.CowClient,'request',race_after_head), \
                 mock.patch.object(cow_cli.CowClient,'stream',count_stream), \
                 contextlib.redirect_stdout(stale_out),contextlib.redirect_stderr(stale_err):
                stale_result=cow_cli.main(['--url',server.base_url,'ask','--branch','b-stale',
                    '--text','race with a head move','--operation-id','op-stale-cas'])
            self.assertEqual(stale_result,2)
            self.assertTrue(state['raced'])
            self.assertEqual(state['stream_calls'],1)
            self.assertIn('HTTP 409',stale_err.getvalue())
            self.assertIn('operation_id=op-stale-cas',stale_err.getvalue())
            self.assertTrue(server.synthetic)


if __name__ == '__main__':
    unittest.main()
