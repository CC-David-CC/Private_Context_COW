"""Canonical coordinator routes immutable registered engines; CPU backend."""
import tempfile,unittest
from pathlib import Path
import serve.test_cow_grammar_canonical_flow as fixture
from serve.cow_grammar_catalog import GrammarCatalog
from serve.cow_grammar_router import GrammarRouter
class GrammarRouterTests(unittest.TestCase):
 def setup_fixture(self):
  f=fixture.CowCanonicalGrammarFlowTests('runTest');f.setUp();self.addCleanup(f.doCleanups)
  catalog=GrammarCatalog(f.tokenizer,[f.tokenizer.EOS])
  router=GrammarRouter(catalog,startup=f.registry)
  f.text.grammar_registry=router;f.token_adapter.grammar_registry=router
  return f,catalog,router
 def config(self,text):return dict(schema=1,states=dict(ANSWER=dict(format='gbnf',source='root ::= '+text)))
 def test_registered_a_continues_after_b_registration_and_execution(self):
  f,catalog,router=self.setup_fixture()
  root,a,b=f._canonical_forked_root()
  first=catalog.register(self.config('"A" "B"'))
  request=f.text.answer(a.name,a.revision,'registered-a-start',1,grammar_state=first['states']['ANSWER'])
  saved=request.durable.result(5)
  original=f.coordinator.store.get_snapshot(saved['snapshot'])['controller']
  self.assertEqual(request.snapshot.result()['tokens'],[ord('A')])
  second=catalog.register(self.config('"X"'))
  other=f.text.answer(b.name,b.revision,'registered-b-start',1,grammar_state=second['states']['ANSWER'])
  other.durable.result(5);self.assertEqual(other.snapshot.result()['tokens'],[ord('X')])
  self.assertEqual(f.coordinator.store.get_snapshot(saved['snapshot'])['controller'],original)
  head=f.coordinator.store.head(a.name)
  resumed=f.text.resume(a.name,head.revision,'registered-a-resume',2)
  resumed.durable.result(5)
  self.assertEqual(resumed.snapshot.result()['tokens'],[ord('B'),f.tokenizer.EOS])
 def test_startup_registry_remains_available_after_dynamic_registration(self):
  f,catalog,router=self.setup_fixture();root,a,b=f._canonical_forked_root()
  catalog.register(self.config('"X"'))
  request=f.text.answer(a.name,a.revision,'legacy-read-start',1,grammar_state='READ')
  saved=request.durable.result(5)
  self.assertEqual(request.snapshot.result()['tokens'],[ord('A')])
  self.assertEqual(f.coordinator.store.get_snapshot(saved['snapshot'])['controller']['constraint']['engine_id'],f.registry.engine_id)
