"""CLI to authenticated HTTP to real Store; synthetic CPU model/encoder."""
import io
import json
import unittest

from PIL import Image
from tools import cow_cli
import tools.test_cow_cli_images as fixture


class CowCLIImageSuffixTests(unittest.TestCase):
    def setUp(self):
        self.case=fixture.CowCliImagesTests();self.addCleanup(self.case.doCleanups);self.case.setUp()
        self.flow=self.case.flow;self.cli=self.case.cli
        code,_,error=self.cli('context','--namespace-id',self.flow.namespace,'--branch','main','--text','Shared diagrams.',
            '--image',str(self.case.image),'--operation-id','context')
        self.assertEqual(code,0,error)
        raw=io.BytesIO();Image.new('RGB',(4,3),'red').save(raw,format='PNG')
        self.image=self.case.folder/'new.png';self.image.write_bytes(raw.getvalue())

    def ask(self,operation='question'):
        return self.cli('ask','--branch','main','--text','What changed?','--image',str(self.image),
            '--max-tokens','2','--operation-id',operation)

    def journal_body(self,operation):
        return json.loads(cow_cli.RequestJournal(self.case.journal).get(operation)['body_json'])

    def test_ask_uses_selected_branch_namespace_and_journals_only_image_ids(self):
        code,output,error=self.ask();self.assertEqual(code,0,error)
        self.assertEqual(json.loads(output)['answer']['text'],'xx')
        body=self.journal_body('question');self.assertIsInstance(body['content'],list)
        self.assertEqual(body['content'][-1],dict(type='text',text='What changed?'))
        self.assertNotIn(str(self.image),json.dumps(body));self.assertEqual(body['revision'],1)
        self.assertEqual(len(self.flow.workspace.history('main')['image_inputs']),2)

    def test_image_request_retry_needs_no_local_file_or_new_work(self):
        self.assertEqual(self.ask()[0],0);self.image.unlink()
        before=self.flow.models[-1].backend.stats['forward_tokens'];calls=len(self.flow.encoder.calls)
        code,output,error=self.cli('retry-request','--operation-id','question')
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['answer']['text'],'xx')
        self.assertEqual(self.flow.models[-1].backend.stats['forward_tokens'],before);self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_same_operation_is_refused_before_uploading_a_changed_file(self):
        self.assertEqual(self.ask()[0],0);before=self.case.fixture.inventory();calls=len(self.flow.encoder.calls)
        self.image.write_bytes(self.case.fixture.raw)
        code,_,error=self.ask();self.assertEqual(code,2);self.assertIn('journal',error.lower())
        self.assertEqual(self.case.fixture.inventory(),before);self.assertEqual(len(self.flow.encoder.calls),calls)

    def test_ordered_question_parts_preserve_text_image_order(self):
        path=self.case.folder/'question-parts.json';path.write_text(json.dumps([
            dict(type='text',text='Before '),dict(type='image',file='new.png'),dict(type='text',text=' after')]))
        code,_,error=self.cli('ask','--branch','main','--parts-file',str(path),'--max-tokens','2','--operation-id','parts')
        self.assertEqual(code,0,error);content=self.journal_body('parts')['content']
        self.assertEqual(content[0],dict(type='text',text='Before '));self.assertEqual(content[-1],dict(type='text',text=' after'))
        self.assertEqual(content[1]['type'],'image')

    def test_next_user_can_add_another_image_without_rewriting_history(self):
        self.flow.models[-1].backend.next_token=257
        code,_,error=self.cli('ask','--branch','main','--text','finish','--max-tokens','2','--operation-id','first')
        self.assertEqual(code,0,error);prior=self.flow.tokens.history('main');self.flow.models[-1].backend.next_token=120
        code,output,error=self.cli('append-user','--branch','main','--text','Next diagram','--image',str(self.image),
            '--max-tokens','2','--operation-id','next')
        self.assertEqual(code,0,error);self.assertEqual(json.loads(output)['answer']['text'],'xx')
        self.assertEqual(self.flow.tokens.history('main').segment(0,len(prior)),prior)

    def test_invalid_question_parts_fail_before_any_upload(self):
        path=self.case.folder/'invalid-parts.json';path.write_text(json.dumps([dict(type='image',url='https://example.com/image.png')]))
        before=self.case.fixture.inventory();calls=len(self.flow.encoder.calls)
        code,_,_=self.cli('ask','--branch','main','--parts-file',str(path),'--operation-id','bad')
        self.assertEqual(code,2);self.assertEqual(self.case.fixture.inventory(),before)
        self.assertEqual(len(self.flow.encoder.calls),calls)


if __name__=='__main__':unittest.main()
