import json
from pathlib import Path
import tempfile
import unittest
import urllib.request
import urllib.error

from serve.chat_workspace import ChatWorkspace
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer,MockEngine,Service,serve


class Recording(MockEngine):
    def generate(self,ids,*args,**kwargs):
        self.prompts.append(self.tok.decode(ids))
        self.last_sampling=args[1]
        yield from super().generate(ids,*args,**kwargs)


class WorkspaceHTTP(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();tok=ByteTokenizer()
        self.engine=Recording(tok,'</think>\n\nAnswer \\(\\frac{1}{2}\\).');self.engine.prompts=[]
        self.service=Service(self.engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        self.service.workspace=ChatWorkspace(Path(self.tmp.name)/'history.sqlite3')
        self.server=serve(self.service,port=0);self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.service.workspace.close();self.tmp.cleanup()

    def call(self,path,body=None,headers=None):
        req=urllib.request.Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,
                                   headers=headers or {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=30) as r:return r.status,json.loads(r.read())
        except urllib.error.HTTPError as e:return e.code,json.loads(e.read())

    def create(self):
        code,data=self.call('/workspace/chats/create',{'title':'Test chat','operation_id':'create'})
        self.assertEqual(code,200);return data['branch']

    def generate(self,branch,text,op):
        return self.call('/workspace/generate',{'branch_id':branch['id'],'revision':branch['revision'],
                  'message':{'role':'user','text':text},'operation_id':op,'max_tokens':128,'reasoning_effort':'none'})

    def test_reply_is_saved_and_next_branch_uses_only_its_history(self):
        branch=self.create();code,result=self.generate(branch,'P','p')
        self.assertEqual(code,200)
        history=self.call('/workspace/history?branch='+branch['id'])[1]
        self.assertEqual(len(history['messages']),2)
        fork=self.call('/workspace/branches/fork',{'branch_id':branch['id'],'node_id':history['tip'],'name':'B','operation_id':'fork'})[1]
        self.generate(history,'Suffix A','a');self.generate(fork,'Suffix B','b')
        self.assertIn('Suffix B',self.engine.prompts[-1]);self.assertNotIn('Suffix A',self.engine.prompts[-1])
        self.assertIn('P',self.engine.prompts[-1])

    def test_stream_publishes_durable_receipt_before_done(self):
        branch=self.create()
        body={'branch_id':branch['id'],'revision':0,'operation_id':'stream',
              'message':{'role':'user','text':'P'},'stream':True,'reasoning_effort':'none','max_tokens':128}
        req=urllib.request.Request(self.base+'/workspace/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as r:
            chunks=[json.loads(line[6:]) for line in r if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]']
        receipt=next(c['context_workspace'] for c in chunks if c.get('context_workspace'))
        self.assertTrue(receipt['saved']);self.assertEqual(self.service.workspace.history(branch['id'])['tip'],receipt['tip'])
        self.assertEqual(self.service.workspace.request('stream')['status'],'completed')

    def test_archived_request_disables_native_checkpoint_saving(self):
        branch=self.create()
        self.service.workspace.archive_chat(branch['chat_id'],True)
        code,result=self.generate(branch,'Preserve this text','archived')
        self.assertEqual(code,200)
        self.assertIs(self.engine.last_sampling['strata_checkpoint'],False)
        self.assertEqual(self.service.workspace.history(branch['id'])['messages'][0]['text'],'Preserve this text')

    def test_history_has_auth_and_mutation_has_origin_checks(self):
        self.service.api_key='test-key'
        self.assertEqual(self.call('/workspace/chats')[0],401)
        headers={'Content-Type':'application/json','Authorization':'Bearer test-key','Origin':'http://evil.example'}
        self.assertEqual(self.call('/workspace/chats/create',{'title':'Wrong','operation_id':'evil'},headers)[0],403)

    def test_stale_and_repeated_requests_do_not_run_model_again(self):
        branch=self.create();self.generate(branch,'P','p')
        count=len(self.engine.prompts)
        self.assertEqual(self.generate(branch,'Again','stale')[0],409)
        self.assertEqual(self.generate(branch,'P','p')[0],409)
        self.assertEqual(len(self.engine.prompts),count)

    def test_import_retry_does_not_duplicate_messages(self):
        body={'operation_id':'import','title':'Imported','messages':[{'role':'user','text':'literal </aaa>','time':1},{'role':'assistant','text':'ok','time':2}]}
        first=self.call('/workspace/import',body)[1];second=self.call('/workspace/import',body)[1]
        self.assertEqual(first,second)
        history=self.call('/workspace/history?branch='+first['branch']['id'])[1]
        self.assertEqual(len(history['messages']),2)

    def test_graph_is_authorized_and_missing_chat_is_404(self):
        branch=self.create();self.service.api_key='test-key'
        path='/workspace/graph?chat='+branch['chat_id']
        self.assertEqual(self.call(path)[0],401)
        headers={'Authorization':'Bearer test-key'}
        code,graph=self.call(path,headers=headers)
        self.assertEqual(code,200);self.assertEqual(graph['branches'][0]['saved_messages'],0)
        self.assertEqual(self.call('/workspace/graph?chat=missing',headers=headers)[0],404)

    def test_graph_and_chat_summary_follow_fork_and_rewind(self):
        branch=self.create();self.generate(branch,'P','p')
        history=self.call('/workspace/history?branch='+branch['id'])[1]
        fork=self.call('/workspace/branches/fork',{'branch_id':branch['id'],'node_id':history['tip'],'name':'B','operation_id':'fork'})[1]
        self.generate(fork,'B suffix','b')
        graph=self.call('/workspace/graph?chat='+branch['chat_id'])[1]
        self.assertEqual(graph['shared_prefix_messages'],2)
        summary=self.call('/workspace/chats')[1]['chats'][0]
        self.assertEqual((summary['branch_count'],summary['max_saved_messages'],summary['shared_prefix_messages']),(2,4,2))
        head=self.call('/workspace/history?branch='+fork['id'])[1]
        self.call('/workspace/branches/rewind',{'branch_id':fork['id'],'revision':head['revision'],'node_id':history['tip'],'operation_id':'rewind'})
        active=self.call('/workspace/graph?chat='+branch['chat_id'])[1]
        old=self.call('/workspace/graph?chat='+branch['chat_id']+'&previous=true')[1]
        self.assertNotIn(head['tip'],{n['id'] for n in active['nodes']})
        self.assertIn(head['tip'],{p['node_id'] for p in old['previous_tips']})
