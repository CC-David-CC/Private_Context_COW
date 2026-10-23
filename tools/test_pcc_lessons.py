"""Real disk/HTTP admission and lost-response cases; fake runner is labelled."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from tools.pcc_lessons import LessonService

def lesson():
    return dict(schema=1,title='Owned correction',model_contract_id='a'*64,prompt='2 + 2 = ',completion='4 .',
        rationale='A checked arithmetic correction',scope='owned-training',evidence=dict(method='host-verified-correction-v1',
        failed_result_sha256='b'*64,corrected_result_sha256='c'*64,verification_sha256='d'*64))

class Lessons(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.calls=[]
        def runner(request,examples,path,parent,fd):
            self.calls.append((request,examples,parent));return dict(mode=request['mode'],candidate_sha256='e'*64,
                candidate_path=str(path/'candidate.gguf'),receipt=dict(steps=[],example_cursor=len(examples)))
        self.service=LessonService(Path(self.temp.name)/'private','a'*64,lambda text:list(text.encode()),runner,max_tokens=32)
    def tearDown(self):self.service.close();self.temp.cleanup()
    def add(self,op='lesson-one'):
        return self.service.submit('/cow/training/lessons',dict(operation_id=op,lesson=lesson()))['lesson_id']
    def request(self,ids,op='train-one',parent=None):
        return dict(operation_id=op,lesson_ids=ids,mode='lora',batch_size=2,steps=1,learning_rate=.001,rank=4,parent_job_id=parent)
    def wait(self,op):
        for _ in range(200):
            value=self.service.job(op)
            if value['state'] not in ('queued','running'):return value
            time.sleep(.01)
        self.fail('Training did not settle')
    def test_single_and_batch_are_explicit_and_no_duplicate_update(self):
        identity=self.add();request=self.request([identity]);accept=self.service.submit('/cow/training/jobs',request)
        self.assertEqual(self.wait('train-one')['state'],'complete')
        self.assertEqual(self.service.submit('/cow/training/jobs',request),accept)
        self.assertEqual(self.service.operation('train-one'),accept);self.assertEqual(len(self.calls),1)
        self.assertFalse(self.service.job('train-one')['promoted'])
        self.assertEqual(self.calls[0][1][0]['targets'][:-1],self.calls[0][1][0]['tokens'][1:])
        self.assertFalse(self.calls[0][1][0]['loss_mask'][0])
    def test_operation_reuse_different_bytes_rejected(self):
        identity=self.add();request=self.request([identity]);self.service.submit('/cow/training/jobs',request)
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',dict(request,steps=2))
    def test_durable_operation_detects_changed_request_bytes(self):
        identity=self.add();self.service.submit('/cow/training/jobs',self.request([identity]));self.wait('train-one')
        path=self.service.root/'operations/train-one.json';record=json.loads(path.read_bytes())
        record['request']['learning_rate']=.002;path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):self.service.operation('train-one')
    def test_pending_selection_and_batch_snapshot(self):
        first=self.add();v=lesson();v['title']='Second checked correction'
        second=self.service.submit('/cow/training/lessons',dict(operation_id='lesson-two',lesson=v))['lesson_id']
        self.assertTrue(all(row['pending'] for row in self.service.lessons()['lessons']))
        self.service.submit('/cow/training/jobs',self.request([first,second]));self.wait('train-one')
        self.assertEqual(len(self.calls[0][1]),2)
        self.assertTrue(all(not row['pending'] for row in self.service.lessons()['lessons']))
    def test_step_budget_cannot_silently_leave_selected_lessons_untrained(self):
        first=self.add();v=lesson();v['title']='Another correction'
        second=self.service.submit('/cow/training/lessons',dict(operation_id='lesson-two',lesson=v))['lesson_id']
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',dict(self.request([first,second]),batch_size=1,steps=1))
        self.assertTrue(all(row['pending'] for row in self.service.lessons()['lessons']))
    def test_mutated_training_tokens_are_not_accepted_as_same_lesson(self):
        identity=self.add();p=self.service.root/'lessons'/(identity+'.json')
        value=json.loads(p.read_bytes());value['example']['tokens'][0]+=1;self.service._write(p,value)
        with self.assertRaises(ValueError):self.service.lesson(identity)
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',self.request([identity]))
    def test_reject_foreign_model_and_client_paths(self):
        with self.assertRaises(ValueError):self.service.submit('/cow/training/lessons',dict(operation_id='foreign',lesson=dict(lesson(),model_contract_id='f'*64)))
        identity=self.add()
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',dict(self.request([identity]),model='/tmp/model'))
    def test_no_silent_truncation_or_unverified_failure(self):
        with self.assertRaises(ValueError):self.service.submit('/cow/training/lessons',dict(operation_id='long',lesson=dict(lesson(),prompt='x'*100)))
        v=lesson();v['evidence']['corrected_result_sha256']=v['evidence']['failed_result_sha256']
        with self.assertRaises(ValueError):self.service.submit('/cow/training/lessons',dict(operation_id='same',lesson=v))
    def test_resume_binds_completed_candidate_and_new_selected_lessons(self):
        identity=self.add();self.service.submit('/cow/training/jobs',self.request([identity]));self.wait('train-one')
        self.service.submit('/cow/training/jobs',self.request([identity],op='next-update',parent='train-one'));self.wait('next-update')
        self.assertEqual(self.calls[-1][2]['candidate_sha256'],'e'*64)
    def test_parent_mode_and_rank_cannot_change_silently(self):
        identity=self.add();self.service.submit('/cow/training/jobs',self.request([identity]));self.wait('train-one')
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',dict(self.request([identity],op='wrong-rank',parent='train-one'),rank=8))
        with self.assertRaises(ValueError):self.service.submit('/cow/training/jobs',dict(self.request([identity],op='wrong-mode',parent='train-one'),mode='full'))
    def test_restart_never_replays_interrupted_update(self):
        identity=self.add();request=self.request([identity]);self.service.submit('/cow/training/jobs',request);self.wait('train-one')
        p=self.service.root/'operations/train-one.json';v=json.loads(p.read_bytes());v.pop('candidate');v['state']='running';self.service._write(p,v)
        root=self.service.root;self.service.close()
        self.service=LessonService(root,'a'*64,lambda text:list(text.encode()),lambda *a:self.fail('Blind replay'))
        self.assertEqual(self.service.job('train-one')['state'],'interrupted')
        self.assertEqual(self.service.submit('/cow/training/jobs',request)['job_id'],'train-one')
    def test_http_disabled_auth_and_lost_ack_reconciliation(self):
        from serve.cow_http import make_server
        with self.assertRaises(ValueError):make_server(object(),training_service=self.service)
        server=make_server(object(),api_key='private-test-key',training_service=self.service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            def call(path,data=None,key='private-test-key'):
                body=None if data is None else json.dumps(data).encode()
                with urlopen(Request(url+path,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})) as response:return json.load(response)
            with self.assertRaises(HTTPError) as rejected:call('/cow/training/lessons',key='wrong')
            rejected.exception.close()
            request=dict(operation_id='http-lesson',lesson=lesson())
            accept=call('/cow/training/lessons',request)
            self.assertEqual(call('/cow/training/operations/http-lesson'),accept)
            self.assertEqual(call('/cow/training/lessons',request),accept)
        finally:server.shutdown();server.server_close();thread.join()
    def test_cli_missing_reconciliation_receipt_is_indeterminate_not_rejection(self):
        from tools import cow_cli
        directory=Path(self.temp.name)/'cli-journal';journal=cow_cli.RequestJournal(directory)
        body=dict(operation_id='uncertain-job',lesson_ids=['a'*64],mode='lora',batch_size=1,steps=1,learning_rate=.001,rank=4,parent_job_id=None)
        journal.begin('uncertain-job','http://127.0.0.1:1','/cow/training/jobs',json.dumps(body,ensure_ascii=False).encode())
        class Client:
            base_url='http://127.0.0.1:1'
            def request(self,method,path):
                if method!='GET':raise AssertionError('An uncertain update was posted again')
                raise cow_cli.ExplicitRejection('Receipt not found yet')
        with patch.object(cow_cli,'_post_may_have_been_sent',False):
            with self.assertRaises(cow_cli.ClientError) as caught:cow_cli._training_post(Client(),journal,'/cow/training/jobs',body)
            self.assertNotIsInstance(caught.exception,cow_cli.ExplicitRejection)
            self.assertTrue(cow_cli._post_may_have_been_sent)

if __name__=='__main__':unittest.main()
