"""CLI/real-HTTP/store integration; synthetic CPU inference, no GPU claim."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from tools import cow_cli
import serve.test_cow_vision_http as http_fixture


class CowCliImagesTests(unittest.TestCase):
    def setUp(self):
        self.fixture=http_fixture.CowVisionHTTPTests();self.addCleanup(self.fixture.doCleanups);self.fixture.setUp()
        self.flow=self.fixture.flow;self.folder=Path(self.flow.temp.name)/'client';self.folder.mkdir()
        self.image=self.folder/'diagram.png';self.image.write_bytes(self.fixture.raw)
        self.url='http://127.0.0.1:'+str(self.fixture.server.server_port)
        self.journal=self.folder/'journal'

    def cli(self,*args):
        output=io.StringIO();error=io.StringIO()
        with mock.patch.dict(os.environ,{'COW_API_KEY':'secret'}),contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
            code=cow_cli.main(['--url',self.url,'--state-dir',str(self.journal),'--machine-result',*args])
        return code,output.getvalue(),error.getvalue()

    def test_upload_command_sends_bytes_and_returns_bound_identity(self):
        code,output,error=self.cli('upload-image','--namespace-id',self.flow.namespace,'--file',str(self.image))
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['image_id'],hashlib.sha256(self.fixture.raw).hexdigest())
        self.assertEqual(self.flow.encoder.calls,[])

    def test_prepare_and_answer_use_uploaded_identity_not_client_path(self):
        code,output,error=self.cli('prepare','--namespace-id',self.flow.namespace,'--branch','main',
            '--text','Describe this.','--image',str(self.image),'--operation-id','image-prepare')
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['schema'],1)
        record=cow_cli.RequestJournal(self.journal).get('image-prepare');body=json.loads(record['body_json'])
        self.assertNotIn(str(self.image),record['body_json'])
        self.assertEqual(body['messages'][0]['content'],[
            dict(type='image',image_id=hashlib.sha256(self.fixture.raw).hexdigest()),dict(type='text',text='Describe this.')])
        code,output,error=self.cli('answer','--branch','main','--max-tokens','2','--operation-id','image-answer')
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['answer']['text'],'xx')

    def test_context_command_keeps_question_boundary_and_repeated_images(self):
        code,_,error=self.cli('context','--namespace-id',self.flow.namespace,'--branch','main','--text','Shared diagrams.',
            '--image',str(self.image),'--image',str(self.image),'--operation-id','image-context')
        self.assertEqual(code,0,error);view=self.flow.workspace.history('main')
        self.assertEqual(view['initial_root_mode'],'context_only');self.assertEqual(len(view['image_inputs']),2)
        self.assertEqual(len(self.flow.encoder.calls),1)
        code,output,error=self.cli('ask','--branch','main','--text','What color?','--max-tokens','2','--operation-id','image-ask')
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['answer']['text'],'xx')

    def test_ordered_parts_resolve_local_files_and_preserve_order(self):
        parts=[dict(type='text',text='Before '),dict(type='image',file='diagram.png'),dict(type='text',text=' after')]
        path=self.folder/'parts.json';path.write_text(json.dumps(parts))
        code,_,error=self.cli('prepare','--namespace-id',self.flow.namespace,'--branch','main',
            '--parts-file',str(path),'--operation-id','ordered-prepare')
        self.assertEqual(code,0,error)
        body=json.loads(cow_cli.RequestJournal(self.journal).get('ordered-prepare')['body_json'])
        self.assertEqual(body['messages'][0]['content'],[
            dict(type='text',text='Before '),dict(type='image',image_id=hashlib.sha256(self.fixture.raw).hexdigest()),
            dict(type='text',text=' after')])

    def test_completed_retry_needs_no_local_image_or_new_model_work(self):
        args=['prepare','--namespace-id',self.flow.namespace,'--branch','main','--text','Diagram.',
              '--image',str(self.image),'--operation-id','saved-image']
        self.assertEqual(self.cli(*args)[0],0);self.image.unlink()
        before=self.flow.models[-1].backend.stats['forward_tokens']
        code,_,error=self.cli('retry-request','--operation-id','saved-image')
        self.assertEqual(code,0,error);self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before)

    def test_changed_file_cannot_rebind_the_same_operation(self):
        args=['prepare','--namespace-id',self.flow.namespace,'--branch','main','--text','Diagram.',
              '--image',str(self.image),'--operation-id','same-operation']
        self.assertEqual(self.cli(*args)[0],0);head=self.flow.store.head('main')
        self.image.write_bytes((self.flow.store.root/'image-originals'/(self.flow.upload['image_id']+'.input')).read_bytes())
        code,_,error=self.cli(*args)
        self.assertEqual(code,2);self.assertIn('journal entry',error.lower());self.assertEqual(self.flow.store.head('main'),head)

    def test_invalid_parts_or_oversized_file_fail_before_upload(self):
        before=self.fixture.inventory();parts=self.folder/'bad.json'
        for value in ([dict(type='image',url='https://example.com/x')],
                      [dict(type='image',file=str(self.image))]*9):
            parts.write_text(json.dumps(value))
            code,_,_=self.cli('prepare','--namespace-id',self.flow.namespace,'--branch','main','--parts-file',str(parts))
            self.assertEqual(code,2);self.assertEqual(self.fixture.inventory(),before)
        huge=self.folder/'huge.png'
        with huge.open('wb') as stream:stream.truncate(16*1024**2+1)
        code,_,_=self.cli('upload-image','--namespace-id',self.flow.namespace,'--file',str(huge))
        self.assertEqual(code,2);self.assertEqual(self.fixture.inventory(),before)

    def test_actual_cli_subprocess_upload(self):
        completed=subprocess.run([sys.executable,'-B','-m','tools.cow_cli','--url',self.url,
            'upload-image','--namespace-id',self.flow.namespace,'--file',str(self.image)],
            env={**os.environ,'COW_API_KEY':'secret'},capture_output=True,text=True,timeout=10)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['image_id'],hashlib.sha256(self.fixture.raw).hexdigest())


if __name__=='__main__':unittest.main()
