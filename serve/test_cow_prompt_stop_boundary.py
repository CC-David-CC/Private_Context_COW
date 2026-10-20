"""Prompt encoding after a canonical text-stop completion."""
from pathlib import Path
import unittest

from serve.cow_prompt_encoder import CowPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]


class CowPromptStopBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()
        self.template = ChatTemplate(ROOT / 'serve' / 'chat_template.jinja')
        self.encoder = CowPromptEncoder(self.tokenizer, self.template, {'enable_thinking': False})
        self.end = self.tokenizer.encode('<|im_end|>', parse_special=True)[0]

    def test_text_stop_appends_eot_in_new_suffix_without_rewriting_history(self):
        canonical = tuple(map(ord, 'answer-before-stop'))
        original = canonical
        controller = {
            'status': 'stopped',
            'policy': {'eos_tokens': [self.end]},
            'text_stop': {
                'schema': 1, 'stop_sequences': ['<STOP>'], 'pending_hex': '',
                'stopped': True, 'matched_stop': '<STOP>', 'finished': False,
            },
        }
        suffix = self.encoder.append_user_suffix(canonical, controller, 'next question')
        newline = tuple(self.tokenizer.encode('\n', parse_special=False))
        self.assertTrue(suffix[:1] == (self.end,))
        self.assertEqual(suffix[1:1 + len(newline)], newline)
        full = [{'role': 'user', 'content': 'next question'}]
        rendered = self.encoder.render(full, add_generation_prompt=True)
        frame = tuple(self.encoder.encode_rendered(full, rendered, True))
        self.assertEqual(suffix, (self.end,) + newline + frame)
        self.assertEqual(canonical, original)
        self.assertNotIn(self.end, canonical)

    def test_arbitrary_stopped_state_without_matched_text_stop_still_rejects(self):
        canonical = tuple(map(ord, 'not-eos'))
        controller = {'status': 'stopped', 'policy': {'eos_tokens': [self.end]}}
        with self.assertRaises(ValueError):
            self.encoder.append_user_suffix(canonical, controller, 'next question')


if __name__ == '__main__':
    unittest.main()
