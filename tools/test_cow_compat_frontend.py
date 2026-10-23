import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental/branching_context/src'))
from serve.cow_compat import Frontend

class Encoder:
    template=SimpleNamespace(source='owned-test-template')

class CompatibilityFrontend(unittest.TestCase):
    def frontend(self,thinking=False):
        return Frontend(object(),Encoder(),dict(model_contract_id='model',execution_profile_id='execution',execution=dict(capacity=128)), 'owned',16,thinking=thinking)
    def test_default_and_thinking_renderer_are_explicit(self):
        calls=[]
        def encode(encoder,messages,tools,kwargs):
            calls.append(kwargs);return [1,2,3] if kwargs['enable_thinking'] else [1,2]
        with patch('serve.cow_compat.Service.encode_prompt',side_effect=encode):
            request=dict(model='owned',messages=[dict(role='user',content='hello')])
            self.assertEqual(len(self.frontend().prepare(request,'chat')['full']),2)
            self.assertEqual(len(self.frontend(True).prepare(request,'chat')['full']),3)
            with self.assertRaises(ValueError):self.frontend().prepare(dict(request,enable_thinking=True),'chat')
            self.assertFalse(calls[0]['enable_thinking']);self.assertTrue(calls[2]['enable_thinking'])
    def test_rendered_prompt_includes_output_reserve(self):
        with patch('serve.cow_compat.Service.encode_prompt',return_value=list(range(120))):
            with self.assertRaises(ValueError):self.frontend().prepare(dict(messages=[dict(role='user',content='hello')]),'chat')
