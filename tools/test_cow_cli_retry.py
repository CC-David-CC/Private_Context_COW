"""CPU socket regressions for durable client-side operation intent."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import stat

from tools import cow_cli


class LostAckHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    calls = []
    get_calls = 0
    committed = {}
    close_first = True
    stream_mode = 'complete'
    ack_mode = 'correct'

    def log_message(self, *_args):
        pass

    def _send_json(self, value):
        raw = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        type(self).get_calls += 1
        if self.path == '/cow/head?branch=b-journal':
            self._send_json({'name':'b-journal','revision':4,'tip':'snap-4','root':'snap-0'})
        else:
            self._send_json({'error':'unexpected GET'})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        body = json.loads(raw)
        type(self).calls.append((self.path, raw, self.headers.get('Authorization')))
        op = body['operation_id']
        if self.path == '/cow/fork':
            head = {'name':body['name'],'revision':0,'tip':body['snapshot'],'root':'snap-0'}
            if type(self).ack_mode == 'fork-wrong-tip':
                head['tip'] = 'snap-wrong'
            elif type(self).ack_mode == 'fork-wrong-revision':
                head['revision'] = 3
            self._send_json(head)
            return
        if type(self).stream_mode == 'failure':
            encoded = json.dumps({'event':'failure','error':'[Errno 28] No space left on device'}).encode() + b'\n'
            self.send_response(200)
            self.send_header('Content-Type','application/x-ndjson')
            self.send_header('Content-Length',str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        receipt = type(self).committed.get(op)
        if receipt is None:
            receipt = {'branch':{'name':body['branch'],'revision':5,'tip':'snap-5','root':'snap-0'},
                       'snapshot':'snap-5','operation_id':op}
            if type(self).ack_mode == 'wrong-revision':
                receipt['branch']['revision'] = 6
            type(self).committed[op] = receipt
            if type(self).close_first:
                type(self).close_first = False
                self.close_connection = True
                return
        event = {'event':'complete','answer':{'snapshot':'snap-5','text':'done'},'receipt':receipt}
        encoded = json.dumps(event, ensure_ascii=False).encode('utf-8') + b'\n'
        self.send_response(200)
        self.send_header('Content-Type','application/x-ndjson')
        self.send_header('Content-Length',str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class CowCliRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LostAckHandler.calls=[]
        LostAckHandler.get_calls=0
        LostAckHandler.committed={}
        LostAckHandler.close_first=True
        LostAckHandler.stream_mode='complete'
        LostAckHandler.ack_mode='correct'
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),LostAckHandler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        cls.cli=str(Path(cow_cli.__file__).resolve())

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        LostAckHandler.calls=[]
        LostAckHandler.get_calls=0
        LostAckHandler.committed={}
        LostAckHandler.close_first=True
        LostAckHandler.stream_mode='complete'
        LostAckHandler.ack_mode='correct'

    def _run_cli(self, state, *args, secret='journal-secret'):
        env=os.environ.copy()
        env['COW_API_KEY']=secret
        return subprocess.run([sys.executable,self.cli,'--url',self.url,'--state-dir',str(state),*args],
                              capture_output=True,env=env,timeout=20)

    def test_lost_ack_retries_exact_intent_in_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            first=self._run_cli(state,'ask','--branch','b-journal','--text','same question',
                                '--max-tokens','7','--operation-id','op-lost-ack')
            self.assertNotEqual(first.returncode,0,first.stderr.decode('utf-8','replace'))
            retry=self._run_cli(state,'retry-request','--operation-id','op-lost-ack')
            self.assertEqual(retry.returncode,0,retry.stderr.decode('utf-8','replace'))
            self.assertIn('snap-5',retry.stderr.decode('utf-8'))
            self.assertEqual(len(LostAckHandler.committed),1)
            self.assertEqual(len(LostAckHandler.calls),2)
            self.assertEqual(LostAckHandler.calls[0][:2],LostAckHandler.calls[1][:2])
            self.assertEqual(LostAckHandler.calls[0][0],'/cow/text/ask')
            self.assertEqual(LostAckHandler.get_calls,1,
                             'retry must not fetch a newer head/revision')
            self.assertEqual(LostAckHandler.calls[0][2],'Bearer journal-secret')
            journal=json.loads(next((state/'operations').glob('*.json')).read_text(encoding='utf-8'))
            self.assertNotIn('journal-secret',json.dumps(journal))
            self.assertIn('"revision": 4',journal['body_json'])
            self.assertEqual(journal['state'],'complete')
            if os.name != 'nt':
                self.assertEqual(stat.S_IMODE(state.stat().st_mode),0o700)
                self.assertEqual(stat.S_IMODE((state/'operations').stat().st_mode),0o700)
                self.assertEqual(stat.S_IMODE(next((state/'operations').glob('*.json')).stat().st_mode),0o600)

    def test_completed_retry_is_local_and_new_id_is_new_work(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            self._run_cli(state,'ask','--branch','b-journal','--text','first',
                          '--operation-id','op-first')
            resolved=self._run_cli(state,'retry-request','--operation-id','op-first')
            self.assertEqual(resolved.returncode,0,resolved.stderr.decode('utf-8','replace'))
            before=len(LostAckHandler.calls)
            completed=self._run_cli(state,'retry-request','--operation-id','op-first')
            self.assertEqual(completed.returncode,0,completed.stderr.decode('utf-8','replace'))
            self.assertEqual(len(LostAckHandler.calls),before)
            new=self._run_cli(state,'ask','--branch','b-journal','--text','second',
                              '--operation-id','op-second')
            self.assertEqual(new.returncode,0,new.stderr.decode('utf-8','replace'))
            self.assertEqual(len(LostAckHandler.committed),2)

    def test_tampered_completed_receipt_is_rejected_before_local_success(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            self._run_cli(state,'ask','--branch','b-journal','--text','first',
                          '--operation-id','op-sealed')
            self._run_cli(state,'retry-request','--operation-id','op-sealed')
            record_path=next((state/'operations').glob('*.json'))
            record=json.loads(record_path.read_text(encoding='utf-8'))
            record['receipt']['snapshot']='snap-forged'
            record['receipt']['branch']['tip']='snap-forged'
            record_path.write_text(json.dumps(record),encoding='utf-8')
            before=len(LostAckHandler.calls)
            result=self._run_cli(state,'retry-request','--operation-id','op-sealed')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('record checksum',result.stderr.decode('utf-8').lower())
            self.assertNotIn('durable receipt:',result.stderr.decode('utf-8'))
            self.assertEqual(len(LostAckHandler.calls),before)

    def test_wrong_success_receipts_remain_pending(self):
        for mode,command,args in (
            ('wrong-revision','ask',('--branch','b-journal','--text','wrong revision')),
            ('fork-wrong-tip','fork',('--source-branch','b-journal','--snapshot','snap-4','--name','b-fork')),
            ('fork-wrong-revision','fork',('--source-branch','b-journal','--snapshot','snap-4','--name','b-fork'))):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                state=Path(directory)/'state'
                LostAckHandler.ack_mode=mode
                result=self._run_cli(state,command,*args,'--operation-id','op-wrong-ack')
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stderr.decode('utf-8').count('durable receipt:'),0)
                record=json.loads(next((state/'operations').glob('*.json')).read_text(encoding='utf-8'))
                self.assertEqual(record['state'],'pending')
                self.assertEqual(record['body_json'] and record['operation_id'],'op-wrong-ack')

    def test_same_operation_id_cannot_be_rebound_to_changed_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            first=self._run_cli(state,'ask','--branch','b-journal','--text','original',
                                '--operation-id','op-fixed')
            self.assertNotEqual(first.returncode,0)
            before=len(LostAckHandler.calls)
            changed=self._run_cli(state,'ask','--branch','b-journal','--text','different',
                                  '--operation-id','op-fixed')
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('already',changed.stderr.decode('utf-8').lower())
            self.assertEqual(len(LostAckHandler.calls),before)
            changed_path=self._run_cli(state,'fork','--source-branch','b-journal',
                '--snapshot','snap-4','--name','b-other','--operation-id','op-fixed')
            self.assertNotEqual(changed_path.returncode,0)
            self.assertIn('already',changed_path.stderr.decode('utf-8').lower())
            self.assertEqual(len(LostAckHandler.calls),before)

    def test_corrupt_intent_is_rejected_without_network_post(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'/'operations'
            state.mkdir(parents=True)
            (state/'op-corrupt.json').write_text('{broken',encoding='utf-8')
            before=len(LostAckHandler.calls)
            result=self._run_cli(state.parent.parent,'retry-request','--operation-id','op-corrupt')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('journal',result.stderr.decode('utf-8').lower())
            self.assertEqual(len(LostAckHandler.calls),before)

    def test_modified_valid_json_intent_is_rejected_by_binding_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            self._run_cli(state,'ask','--branch','b-journal','--text','original',
                          '--operation-id','op-tamper')
            record_path=next((state/'operations').glob('*.json'))
            record=json.loads(record_path.read_text(encoding='utf-8'))
            record['body_json']=record['body_json'].replace('original','changed')
            record_path.write_text(json.dumps(record),encoding='utf-8')
            before=len(LostAckHandler.calls)
            result=self._run_cli(state,'retry-request','--operation-id','op-tamper')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('checksum',result.stderr.decode('utf-8').lower())
            self.assertEqual(len(LostAckHandler.calls),before)

    def test_local_journal_failure_prevents_mutating_post(self):
        with tempfile.TemporaryDirectory() as directory:
            not_directory=Path(directory)/'file'
            not_directory.write_text('occupied',encoding='utf-8')
            before=len(LostAckHandler.calls)
            result=self._run_cli(not_directory,'ask','--branch','b-journal','--text','must not send',
                                 '--operation-id','op-no-journal')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('journal',result.stderr.decode('utf-8').lower())
            self.assertEqual(len(LostAckHandler.calls),before)

    def test_failure_event_is_reported_and_intent_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            LostAckHandler.stream_mode='failure'
            result=self._run_cli(state,'ask','--branch','b-journal','--text','disk failure',
                                 '--operation-id','op-failure')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('No space left on device',result.stderr.decode('utf-8'))
            self.assertNotIn('durable receipt:',result.stderr.decode('utf-8'))
            self.assertEqual(len(LostAckHandler.calls),1)
            record=next((state/'operations').glob('*.json'))
            self.assertEqual(json.loads(record.read_text(encoding='utf-8'))['state'],'pending')

    def test_retry_publication_does_not_replace_original_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            first=self._run_cli(state,'ask','--branch','b-journal','--text','original',
                                '--operation-id','op-pub')
            self.assertNotEqual(first.returncode,0)
            record=next((state/'operations').glob('*.json'))
            before=json.loads(record.read_text(encoding='utf-8'))
            published=self._run_cli(state,'retry-publication','--operation-id','op-pub')
            self.assertEqual(published.returncode,0,published.stderr.decode('utf-8','replace'))
            after=json.loads(record.read_text(encoding='utf-8'))
            self.assertEqual(after['path'],before['path'])
            self.assertEqual(after['body_json'],before['body_json'])
            self.assertEqual(after['operation_id'],before['operation_id'])
            self.assertEqual(after['intent_sha256'],before['intent_sha256'])
            self.assertEqual(after['state'],'complete')
            self.assertEqual(len(list((state/'operations').glob('*.json'))),1)


if __name__ == '__main__':
    unittest.main()
