"""Durable grammar metadata, actual files/process restart; no GPU claims."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from test_grammar_first import TinyByteTokenizer
from context_time_travel.store import InvalidOperation
from serve.cow_grammar_disk import DurableGrammarCatalog
class DurableGrammarTests(unittest.TestCase):
 def config(self):return dict(schema=1,states=dict(ANSWER=dict(format='gbnf',source='root ::= "A" "B"')))
 def test_new_process_reloads_identical_group_and_controller_identity(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'grammars';tokenizer=TinyByteTokenizer()
   with DurableGrammarCatalog(root,tokenizer,[tokenizer.EOS]) as catalog:
    receipt=catalog.register(self.config())
    registry=catalog.resolve(receipt['states']['ANSWER'])
    controller=registry.start(receipt['states']['ANSWER'],'stream',[],len(tokenizer.tokens))
   payload=dict(root=str(root),receipt=receipt,controller=controller)
   script="import json,sys;from test_grammar_first import TinyByteTokenizer;from serve.cow_grammar_disk import DurableGrammarCatalog;p=json.load(sys.stdin);t=TinyByteTokenizer();c=DurableGrammarCatalog(p['root'],t,[t.EOS]);r=c.resolve(controller=p['controller']);r.restore(p['controller'],[]);print(r.engine_id);c.close()"
   child=subprocess.run([sys.executable,'-B','-c',script],input=json.dumps(payload),capture_output=True,text=True,timeout=15)
   self.assertEqual(child.returncode,0,child.stderr)
   self.assertEqual(child.stdout.strip(),receipt['engine_id'])
 def test_altered_definition_rejects_on_reload(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'grammars';t=TinyByteTokenizer()
   with DurableGrammarCatalog(root,t,[t.EOS]) as c:r=c.register(self.config())
   p=root/(r['group_id']+'.json');data=json.loads(p.read_text())
   data['config']['states']['ANSWER']['source']='root ::= "X"';p.write_text(json.dumps(data))
   before={q.name:q.read_bytes() for q in root.glob('*.json')}
   with self.assertRaises(InvalidOperation):DurableGrammarCatalog(root,t,[t.EOS])
   self.assertEqual(before,{q.name:q.read_bytes() for q in root.glob('*.json')})
 def test_symlinked_root_is_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);actual=base/'real';actual.mkdir(mode=0o700);link=base/'link';link.symlink_to(actual,target_is_directory=True);t=TinyByteTokenizer()
   with self.assertRaises((InvalidOperation,OSError)):DurableGrammarCatalog(link,t,[t.EOS])
 def test_failed_file_publication_does_not_admit_group(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'grammars';t=TinyByteTokenizer()
   with DurableGrammarCatalog(root,t,[t.EOS]) as c:
    with patch('serve.cow_grammar_disk.os.fsync',side_effect=OSError(28,'injected no space')):
     with self.assertRaises(OSError):c.register(self.config())
    self.assertEqual(c._groups,{})
    self.assertEqual(list(root.glob('*.json')),[])
    receipt=c.register(self.config());self.assertIsNotNone(c.resolve(receipt['states']['ANSWER']))

 def test_operation_binding_survives_reopen_and_rejects_changed_definition(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'grammars';t=TinyByteTokenizer()
   with DurableGrammarCatalog(root,t,[t.EOS]) as c:
    first=c.register_operation(self.config(),'register-one')
   with DurableGrammarCatalog(root,t,[t.EOS]) as c:
    self.assertEqual(c.register_operation(self.config(),'register-one'),first)
    changed=self.config();changed['states']['ANSWER']['source']='root ::= "X"'
    with self.assertRaises(InvalidOperation):c.register_operation(changed,'register-one')
    self.assertEqual(c.register_operation(self.config(),'register-two')['group_id'],first['group_id'])
