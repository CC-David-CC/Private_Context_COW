"""Explicit Lark uses the same canonical controller and bounded matcher."""
import copy
import json
import tempfile
from pathlib import Path
import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError, InvalidOperation
from serve.cow_grammar import GrammarRegistry
from test_grammar_first import TinyByteTokenizer
from test_thinking_grammar import ThinkingTokenizer

SOURCE = '''%llguidance {}
start: "[" (entry ("," entry){0,127})? "]"
entry: STRING
STRING: /"[a-z]{0,16}"/
'''

class LarkGrammarTests(unittest.TestCase):
    def engine(self, source=SOURCE, fmt='lark'):
        return GrammarEngine(TinyByteTokenizer(), {'ACTION':dict(format=fmt,source=source)}, [TinyByteTokenizer.EOS])

    def consume(self, engine, text, saved=None, prefix=''):
        state = engine.start('ACTION','lark-test',[],engine.vocab_size) if saved is None else copy.deepcopy(saved)
        session = engine.restore(state,list(prefix.encode()))
        for token in text.encode():
            state = session.consume(state,token,engine.tokenizer.token_bytes(token),False)
        return state,session

    def test_long_bounded_array_and_rejection_against_json_reference(self):
        engine=self.engine()
        for length in (0,1,127,128):
            text=json.dumps(['ab']*length,separators=(',',':'))
            _,session=self.consume(engine,text)
            self.assertEqual(len(json.loads(text)),length)
            self.assertTrue(session.is_complete())
        for text in [json.dumps(['a']*129,separators=(',',':')), '["'+('a'*17)+'"]', '["A"]', '[1]', '["a",]']:
            with self.subTest(text=text[:40]), self.assertRaises(IntegrityError):
                self.consume(engine,text)

    def test_partial_controller_restore_and_fork_isolation(self):
        engine=self.engine();prefix='["a",'
        saved,original=self.consume(engine,prefix);snapshot=copy.deepcopy(saved)
        _,left=self.consume(engine,'"b"]',saved,prefix)
        _,right=self.consume(engine,'"c"]',saved,prefix)
        self.assertTrue(left.is_complete());self.assertTrue(right.is_complete())
        self.assertFalse(original.is_complete());self.assertEqual(saved,snapshot)
        incompatible=self.engine(SOURCE.replace('{0,127}','{0,126}'))
        with self.assertRaises(IntegrityError):incompatible.restore(saved,list(prefix.encode()))

    def test_format_identity_and_legacy_gbnf_bytes_unchanged(self):
        lark=self.engine();legacy=self.engine(SOURCE,fmt='gbnf')
        self.assertNotEqual(lark.engine_id,legacy.engine_id)
        self.assertNotEqual(lark.spec_sha256,legacy.spec_sha256)
        gbnf=self.engine('root ::= "A" | "B"',fmt='gbnf')
        self.assertEqual(gbnf.engine_id,LEGACY_ENGINE_ID)

    def test_registry_persists_exact_source_and_thinking_final(self):
        cfg=dict(schema=1,states={
            'ACTION':dict(format='lark',source=SOURCE),
            'THINK':dict(format='thinking',source=dict(schema=1,max_tokens=16,final=dict(format='lark',source=SOURCE)))})
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'grammars.json';path.write_text(json.dumps(cfg))
            registry=GrammarRegistry.load(path,ThinkingTokenizer(),[ThinkingTokenizer.EOS])
            again=GrammarRegistry.from_config(cfg,ThinkingTokenizer(),[ThinkingTokenizer.EOS])
            self.assertEqual(registry.engine_id,again.engine_id)
            self.assertEqual(registry.states,('ACTION','THINK'))

    def test_invalid_sources_fail_closed(self):
        for source in ('',{},'start: missing_rule', 'start: "unterminated'):
            with self.subTest(source=source),self.assertRaises(InvalidOperation):self.engine(source)

LEGACY_ENGINE_ID = '9336c9ad3853598084545ee568edf3784c223654e3fac52f45912d5969db170f'

if __name__=='__main__':unittest.main()
