"""Adding a grammar group cannot redefine an existing controller engine."""
import unittest
from test_grammar_first import TinyByteTokenizer
from context_time_travel.store import InvalidOperation
from serve.cow_grammar_catalog import GrammarCatalog
class GrammarCatalogTests(unittest.TestCase):
 def setUp(self):
  self.tokenizer=TinyByteTokenizer()
  self.catalog=GrammarCatalog(self.tokenizer,[self.tokenizer.EOS])
 def config(self,text):
  return dict(schema=1,states=dict(ANSWER=dict(format='gbnf',source='root ::= '+text)))
 def test_second_group_preserves_first_identity_and_parser_restore(self):
  a=self.catalog.register(self.config('"A" "B"'))
  name=a['states']['ANSWER'];first=self.catalog.resolve(name)
  state=first.start(name,'stream',[],len(self.tokenizer.tokens))
  before=first.descriptor(name)
  b=self.catalog.register(self.config('"X"'))
  self.assertNotEqual(name,b['states']['ANSWER'])
  self.assertIs(self.catalog.resolve(name),first)
  self.assertEqual(first.descriptor(name),before)
  session=self.catalog.resolve(controller=state).restore(state,[])
  self.assertEqual(session.engine.engine_id,first.engine_id)
 def test_duplicate_registration_is_idempotent_and_detached(self):
  spec=self.config('"A"');one=self.catalog.register(spec);two=self.catalog.register(spec)
  self.assertEqual(one,two)
  one['states']['ANSWER']='corrupt'
  self.assertNotEqual(one,self.catalog.register(spec))
 def test_unknown_controller_and_failed_registration_do_not_mutate_groups(self):
  first=self.catalog.register(self.config('"A"'))
  with self.assertRaises(InvalidOperation):self.catalog.resolve('unknown')
  with self.assertRaises(InvalidOperation):self.catalog.register(dict(schema=1,states={}))
  self.assertEqual(first,self.catalog.register(self.config('"A"')))
 def test_group_capacity_error_keeps_existing_group_usable(self):
  catalog=GrammarCatalog(self.tokenizer,[self.tokenizer.EOS],max_groups=1)
  first=catalog.register(self.config('"A"'))
  with self.assertRaises(InvalidOperation):catalog.register(self.config('"B"'))
  self.assertEqual(catalog.resolve(first['states']['ANSWER']).engine_id,first['engine_id'])

 def test_caller_mutation_during_compilation_cannot_change_group_binding(self):
  from unittest.mock import patch
  from serve.cow_grammar import GrammarRegistry
  config=self.config('"A"')
  original=GrammarRegistry.from_config
  expected=original(config,self.tokenizer,[self.tokenizer.EOS]).engine_id
  def compile_actual(value,tokenizer,eos):
   registry=original(value,tokenizer,eos)
   config['states']['ANSWER']['source']='root ::= "B"'
   return registry
  with patch.object(GrammarRegistry,'from_config',side_effect=compile_actual):
   receipt=self.catalog.register(config)
  self.assertEqual(receipt['group_id'],expected)
  expected_alias=dict(schema=1,states={receipt['states']['ANSWER']:dict(format='gbnf',source='root ::= "A"')})
  self.assertEqual(receipt['engine_id'],original(expected_alias,self.tokenizer,[self.tokenizer.EOS]).engine_id)
  repeat=self.catalog.register(self.config('"A"'))
  self.assertEqual(repeat,receipt)
