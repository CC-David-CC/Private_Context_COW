"""Inline admission uses the same immutable engine as startup files."""
import json,tempfile,unittest
from pathlib import Path
from serve.cow_grammar import GrammarRegistry
from test_grammar_first import TinyByteTokenizer
from context_time_travel.store import InvalidOperation
class InlineGrammarTests(unittest.TestCase):
 def test_inline_and_file_definitions_have_identical_engine_identity(self):
  tokenizer=TinyByteTokenizer();spec=dict(schema=1,states=dict(READ=dict(format='gbnf',source='root ::= "A"')))
  inline=GrammarRegistry.from_config(spec,tokenizer,[tokenizer.EOS])
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'registry.json';p.write_text(json.dumps(spec))
   file=GrammarRegistry.load(p,tokenizer,[tokenizer.EOS])
  self.assertEqual(inline.engine_id,file.engine_id)
  self.assertEqual(inline.spec_sha256,file.spec_sha256)
  original=inline.descriptor('READ')
  spec['states']['READ']['source']='root ::= "B"'
  self.assertEqual(inline.descriptor('READ'),original)
 def test_inline_source_paths_and_unknown_fields_are_rejected(self):
  tokenizer=TinyByteTokenizer()
  for spec in [
   dict(schema=1,states=dict(READ=dict(format='gbnf',source='root ::= "A"',path='/etc/passwd'))),
   dict(schema=True,states=dict(READ=dict(format='gbnf',source='root ::= "A"'))),
   dict(schema=1,states={},extra=True)]:
   with self.subTest(spec=spec),self.assertRaises(InvalidOperation):
    GrammarRegistry.from_config(spec,tokenizer,[tokenizer.EOS])
