"""Adapter wiring and literal-boundary controls with a CPU reference double.

Native fastokens/model token-ID qualification is a separate explicit probe.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.strata_tokenizer import BYTE_TO_UNICODE,Tokenizer
from tools.fastokens_tokenizer import FastokensTokenizer,load_tokenizer


class NativeDouble:
    calls=[]
    document=None

    @classmethod
    def from_json_str(cls,raw,**kw):
        cls.document=json.loads(raw);cls.calls=[]
        instance=cls()
        vocabulary=cls.document['model']['vocab']
        tokens=[None]*len(vocabulary)
        for token,identity in vocabulary.items():tokens[identity]=token
        merges=[' '.join(pair) for pair in cls.document['model']['merges']]
        instance.reference=Tokenizer(tokens,merges)
        return instance

    @property
    def vocab_size(self):return len(self.reference.tokens)

    def encode_batch(self,pieces,add_special_tokens=False):
        self.calls.append(tuple(pieces))
        if add_special_tokens:raise AssertionError('Native BPE cannot inject template controls')
        return [SimpleNamespace(ids=self.reference._encode_plain(piece)) for piece in pieces]


class FastokensBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tokens=list(BYTE_TO_UNICODE.values())+['he','hel','hell','hello','<|control|>','<tool_call>']
        self.merges=['h e','he l','hel l','hell o']
        self.types=[1]*260+[3,4]
        self.reference=Tokenizer(self.tokens,self.merges,self.types)
        self.addCleanup(patch.stopall)
        patch('importlib.metadata.version',return_value='0.3.4').start()
        patch.dict('sys.modules',{'fastokens._native':SimpleNamespace(Tokenizer=NativeDouble)}).start()
        patch.dict(os.environ,{'FASTOKENS_INPUT_CACHE':'0','FASTOKENS_SEGMENT_CACHE':'0'}).start()
        self.fast=FastokensTokenizer(self.tokens,self.merges,self.types)

    def test_original_special_literal_and_plain_span_policy_is_preserved(self):
        text='hello <|control|> <tool_call> hello'
        for parse in (False,True):
            for spans in ((),[(0,len(text))],[(6,20)]):
                with self.subTest(parse=parse,spans=spans):
                    self.assertEqual(self.fast.encode(text,parse_special=parse,plain=spans),
                                     self.reference.encode(text,parse_special=parse,plain=spans))
        document=NativeDouble.document
        self.assertEqual(document['added_tokens'],[])
        self.assertIsNone(document['normalizer'])
        self.assertFalse(document['pre_tokenizer']['use_regex'])
        self.assertEqual(document['model']['vocab'],self.reference.ids)

    def test_cow_keeps_unicode_splitting_and_cached_long_piece_results(self):
        text="a\u0301"*300+' X_é \r\n '+'hello'*100
        self.fast.encode(text)
        self.assertTrue(NativeDouble.calls)
        self.assertTrue(all(len(batch)<=256 for batch in NativeDouble.calls))
        expected={piece for piece in self.reference._re.findall(text) if len(piece.encode())>self.reference.HEAP_MIN}
        self.assertEqual({piece for call in NativeDouble.calls for piece in call},expected)
        before=len(NativeDouble.calls)
        self.fast.encode(text)
        self.assertEqual(len(NativeDouble.calls),before)

    def test_native_token_ids_must_belong_to_the_pinned_vocabulary(self):
        with patch.object(self.fast._native,'encode_batch',return_value=[SimpleNamespace(ids=[999999])]):
            with self.assertRaises(ValueError):self.fast.encode('hello'*100)

    def test_unknown_version_and_hidden_cache_overrides_are_rejected(self):
        with patch('importlib.metadata.version',return_value='99.0'):
            with self.assertRaisesRegex(RuntimeError,'0.3.4'):FastokensTokenizer(self.tokens,self.merges,self.types)
        with patch.dict(os.environ,{'FASTOKENS_INPUT_CACHE':'999'}):
            with self.assertRaisesRegex(ValueError,'cache'):FastokensTokenizer(self.tokens,self.merges,self.types)

    def test_python_default_never_imports_optional_backend(self):
        with patch.object(Tokenizer,'from_gguf',return_value=self.reference) as load:
            with patch.dict('sys.modules',{'fastokens._native':None}):
                self.assertIs(load_tokenizer('fixture.gguf'),self.reference)
            load.assert_called_once_with('fixture.gguf')
        with self.assertRaises(ValueError):load_tokenizer('fixture.gguf',backend='unknown')

    def test_short_code_pieces_keep_the_reference_fast_path(self):
        with patch.object(self.fast._native,'encode_batch',side_effect=AssertionError('short-piece native overhead')):
            self.assertEqual(self.fast.encode('hello + value_123\n'),self.reference.encode('hello + value_123\n'))

    def test_text_components_select_the_same_tokenizer_for_all_text_operations(self):
        from serve.cow_http import model_text_components
        template=(Path(__file__).resolve().parents[4]/'serve/chat_template.jinja').read_text()
        with patch('tools.fastokens_tokenizer.load_tokenizer',return_value=self.fast) as load, \
             patch('tools.gguf_reader.GGUFFile',return_value=SimpleNamespace(metadata={'tokenizer.chat_template':template})):
            components=model_text_components('fixture.gguf',tokenizer_backend='fastokens')
        load.assert_called_once_with('fixture.gguf',backend='fastokens')
        self.assertIs(components['tokenizer'],self.fast)
        self.assertIs(components['encode_rendered'].__self__.tokenizer,self.fast)

    def test_launcher_defaults_and_optional_backend_are_explicit(self):
        from tools.cow_server import configured_argv
        from tools.cow_server_setup import tokenizer_config
        config=dict(profile='profile',store='store',host='127.0.0.1',port=8099,
            ram_gib=100,disk_gib=100,checkpoint_idle_seconds=1800)
        self.assertEqual(tokenizer_config(SimpleNamespace()),{})
        self.assertNotIn('--tokenizer-backend',configured_argv(config,Path('/tmp/server'),'key'))
        config.update(tokenizer_config(SimpleNamespace(tokenizer_backend='fastokens')))
        argv=configured_argv(config,Path('/tmp/server'),'key')
        self.assertEqual(argv[argv.index('--tokenizer-backend')+1],'fastokens')
        for invalid in ('auto',True,None):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                tokenizer_config(SimpleNamespace(tokenizer_backend=invalid))
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                configured_argv(dict(config,tokenizer_backend=invalid),Path('/tmp/server'),'key')

    def test_explicit_backend_failure_does_not_fall_back(self):
        from serve.cow_http import model_text_components
        with patch('tools.fastokens_tokenizer.load_tokenizer',side_effect=RuntimeError('Optional dependency missing')):
            with self.assertRaisesRegex(RuntimeError,'Optional dependency missing'):
                model_text_components('fixture.gguf',tokenizer_backend='fastokens')


if __name__=='__main__':unittest.main()
