"""Metadata/configuration-only startup checks; no model or GPU is started."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from context_time_travel.store import Store, IntegrityError
from serve import cow_http
from serve.server import ByteTokenizer


class CowVisionStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.store=Store(self.folder/'store');self.model=self.folder/'model.gguf';self.model.write_bytes(b'metadata fixture')
        self.exe=self.folder/'encoder';self.exe.write_bytes(b'not executed')
        self.projector=self.folder/'projector';self.projector.write_bytes(b'projector fixture')
        self.profile=dict(model_path=str(self.model),model_contract_id='model-contract',
            execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',embedding_width=4))
        self.template=(Path(__file__).resolve().parent/'chat_template.jinja').read_text()
        self.config=self.folder/'vision.json'
        self.document=dict(schema=1,executable=str(self.exe),projector=str(self.projector),
            pinned_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.exe,self.projector,self.model)},
            embedding_width=4,original_bytes_limit=16*1024**2,threads=2)
        self.config.write_text(json.dumps(self.document))

    def components(self,*,mixed):
        tokenizer=ByteTokenizer()
        tokenizer.tokens=[chr(value) for value in range(256)]+list(tokenizer.SPECIALS)
        tokenizer.pre='qwen35';tokenizer.token_types=[1]*256+[3]*len(tokenizer.SPECIALS)
        tokenizer.special_ids={'tokenizer.ggml.eos_token_id':257};tokenizer.ranks={('a','b'):0}
        with mock.patch('tools.strata_tokenizer.Tokenizer.from_gguf',return_value=tokenizer), \
             mock.patch('tools.gguf_reader.GGUFFile',return_value=SimpleNamespace(metadata={'tokenizer.chat_template':self.template})):
            return cow_http.model_text_components(self.model,{'enable_thinking':False},
                model_contract_id=self.profile['model_contract_id'],mixed_inputs=mixed)

    def test_restore_only_mixed_metadata_has_renderer_without_producer(self):
        value=self.components(mixed=True)
        self.assertIsNotNone(value['mixed_renderer'])
        self.assertNotEqual(value['renderer_id'],value['mixed_renderer'].renderer_id)
        self.assertNotIn('vision_preparer',value)

    def test_legacy_components_keep_the_base_renderer(self):
        text=self.components(mixed=False);mixed=self.components(mixed=True)
        self.assertIsNone(text.get('mixed_renderer'))
        self.assertEqual(text['renderer_id'],mixed['renderer_id'])
        messages=[dict(role='user',content='literal <|image_pad|>')]
        self.assertEqual(text['encode_rendered'](messages,text['render'](messages,add_generation_prompt=True),True),
                         mixed['encode_rendered'](messages,mixed['render'](messages,add_generation_prompt=True),True))

    def test_pinned_producer_configuration_is_bound_to_profile_and_renderer(self):
        parts=self.components(mixed=True)
        producer=cow_http.vision_preparer_from_config(self.store,parts['mixed_renderer'],self.profile,self.config)
        self.assertEqual(producer.encoder.width,4);self.assertEqual(producer.original_bytes_limit,16*1024**2)
        self.assertIs(producer.renderer,parts['mixed_renderer'])
        self.assertIn(str(self.model),producer.encoder.files)

    def test_config_rejects_unknown_fields_wrong_profile_and_bad_hash(self):
        parts=self.components(mixed=True)
        for change in (dict(extra=True),dict(schema=2),dict(original_bytes_limit=0),dict(model='/other')):
            self.config.write_text(json.dumps(dict(self.document,**change)))
            with self.subTest(change=change),self.assertRaises(ValueError):
                cow_http.vision_preparer_from_config(self.store,parts['mixed_renderer'],self.profile,self.config)
        self.config.write_text(json.dumps(self.document));self.profile['execution']['embedding_width']=5
        with self.assertRaises(ValueError):
            cow_http.vision_preparer_from_config(self.store,parts['mixed_renderer'],self.profile,self.config)
        self.profile['execution']['embedding_width']=4;self.exe.write_bytes(b'changed')
        with self.assertRaises(IntegrityError):
            cow_http.vision_preparer_from_config(self.store,parts['mixed_renderer'],self.profile,self.config)


if __name__=='__main__':unittest.main()
