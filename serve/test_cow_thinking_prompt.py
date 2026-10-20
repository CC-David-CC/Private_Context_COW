"""Opt-in thinking roots use their own exact rendering identity and suffix."""
from pathlib import Path
import unittest

from serve.cow_prompt_encoder import CowPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer


class ThinkingPromptTests(unittest.TestCase):
    def test_prepared_thinking_root_is_recognized_without_retagging_closed_roots(self):
        tok = ByteTokenizer()
        template = ChatTemplate(Path(__file__).resolve().parent / 'chat_template.jinja')
        messages = [dict(role='user', content='A question.')]
        thinking = CowPromptEncoder(tok, template, {'enable_thinking': True})
        closed = CowPromptEncoder(tok, template, {'enable_thinking': False})
        def encode(encoder, generation):
            return encoder.encode_rendered(messages,
                encoder.render(messages, add_generation_prompt=generation), generation)
        active = encode(thinking, True)
        self.assertEqual(thinking.initial_root_mode(active, {'status': 'idle'}), 'prepared_answer')
        self.assertEqual(thinking.initial_root_mode(encode(thinking, False), {'status': 'idle'}), 'context_only')
        self.assertEqual(thinking.initial_root_mode(encode(closed, True), {'status': 'idle'}), 'unsupported')
        self.assertEqual(closed.initial_root_mode(active, {'status': 'idle'}), 'unsupported')
        self.assertEqual(thinking.initial_root_mode(active, {'status': 'active'}), 'unsupported')
        with self.assertRaises(ValueError):
            thinking.append_user_suffix(active, {'status': 'stopped'}, 'Unqualified new turn')


if __name__ == '__main__':
    unittest.main()
