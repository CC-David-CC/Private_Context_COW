"""Authenticated HTTP grammar admission with real files and coordinator fixture."""
import json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
import serve.test_cow_grammar_canonical_flow as fixture
from serve.cow_grammar_disk import DurableGrammarCatalog
from serve.cow_http import make_server
class GrammarRegistrationHttpTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixture.CowCanonicalGrammarFlowTests('runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.catalog=DurableGrammarCatalog(Path(self.temp.name)/'grammars',self.fixture.tokenizer,[self.fixture.tokenizer.EOS]);self.addCleanup(self.catalog.close)
 def start(self,enabled=True):
  f=self.fixture
  server=make_server(f.token_adapter,api_key='owned-key',text_adapter=f.text,**(dict(grammar_catalog=self.catalog) if enabled else {}))
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  def close():server.shutdown();server.server_close();thread.join(5)
  self.addCleanup(close);return 'http://127.0.0.1:'+str(server.server_port)
 def post(self,url,path,payload,key='owned-key'):
  request=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
  try:
   with urllib.request.urlopen(request,timeout=10) as response:return response.status,json.load(response)
  except urllib.error.HTTPError as error:return error.code,json.load(error)
 def config(self):return dict(schema=1,states=dict(ANSWER=dict(format='gbnf',source='root ::= "A" "B"')))
 def test_unauthorized_registration_has_no_definition_effect(self):
  status,body=self.post(self.start(),'/cow/grammars',self.config(),key='wrong')
  self.assertEqual(status,401);self.assertEqual(self.catalog.states,())
  self.assertEqual(list(Path(self.temp.name).rglob('*.json')),[])
 def test_opt_in_registration_is_idempotent_and_routes_real_http_answer(self):
  url=self.start();f=self.fixture
  status,receipt=self.post(url,'/cow/grammars',self.config());self.assertEqual(status,200,receipt)
  self.assertEqual(self.post(url,'/cow/grammars',self.config()),(200,receipt))
  root,a,b=f._canonical_forked_root()
  status,response=self.post(url,'/cow/text/answer',dict(branch=a.name,revision=a.revision,operation_id='http-registered-answer',max_tokens=1,grammar_state=receipt['states']['ANSWER'],stream=False))
  self.assertEqual(status,200,response)
  head=f.coordinator.store.head(a.name);controller=f.coordinator.store.get_snapshot(head.tip)['controller']
  self.assertEqual(controller['constraint']['engine_id'],receipt['engine_id'])
  self.assertEqual(f.backends[0].selected_tokens[-1:], [ord('A')])
 def test_default_server_does_not_expose_registration(self):
  self.assertEqual(self.post(self.start(False),'/cow/grammars',self.config())[0],404)

 def test_duplicate_json_keys_reject_without_registration(self):
  url=self.start()
  raw=('{"schema":1,"schema":1,"states":'+json.dumps(self.config()["states"])+'}').encode()
  request=urllib.request.Request(url+'/cow/grammars',data=raw,headers={'Authorization':'Bearer owned-key','Content-Type':'application/json'})
  with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request,timeout=10)
  self.assertEqual(caught.exception.code,400)
  self.assertEqual(self.catalog.states,())
