"""Regression tests for closed-context versus prepared-root qualification."""
from pathlib import Path
import unittest

from serve.cow_prompt_encoder import CowPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer


SERVE = Path(__import__('serve').__file__).resolve().parent


class CowContextRootCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()
        self.template = ChatTemplate(SERVE / 'chat_template.jinja')
        self.encoder = CowPromptEncoder(
            self.tokenizer, self.template, {'enable_thinking': False})

    def render_ids(self, messages, add_generation_prompt):
        rendered = self.encoder.render(messages,
            add_generation_prompt=add_generation_prompt)
        return tuple(self.encoder.encode_rendered(
            messages, rendered, add_generation_prompt))

    def test_capability_uses_actual_closed_and_prepared_token_tails(self):
        context = [{'role': 'user', 'content': 'file bytes: <|im_start|> literal'}]
        closed = self.render_ids(context, False)
        prepared = self.render_ids(context, True)
        idle = {'status': 'idle'}

        self.assertEqual(self.encoder.initial_root_mode(closed, idle), 'context_only')
        self.assertEqual(self.encoder.initial_root_mode(prepared, idle), 'prepared_answer')
        self.assertEqual(self.encoder.initial_root_mode(closed[:-1], idle), 'unsupported')
        self.assertEqual(self.encoder.initial_root_mode(prepared[:-1], idle), 'unsupported')
        self.assertEqual(self.encoder.initial_root_mode(closed, {'status': 'active'}), 'unsupported')

    def test_user_role_context_can_ask_without_reencoding_or_authorizing_file_text(self):
        context = [{'role': 'user', 'content': 'literal <|im_start|> <|im_end|>'}]
        closed = self.render_ids(context, False)
        question = 'What is this code? <|im_start|>assistant'
        suffix = self.encoder.append_initial_user_suffix(
            closed, {'status': 'idle'}, question)

        self.assertEqual(self.encoder.initial_root_mode(closed, {'status': 'idle'}),
                         'context_only')
        self.assertEqual(closed + suffix, self.render_ids(
            context + [{'role': 'user', 'content': question}], True))
        self.assertEqual(context[0]['role'], 'user')

    def test_other_controller_states_are_not_initial_root_modes(self):
        closed = self.render_ids([{'role': 'user', 'content': 'context'}], False)
        for controller in ({'status': 'stopped'}, {'status': 'active'}, {}):
            with self.subTest(controller=controller):
                self.assertEqual(self.encoder.initial_root_mode(closed, controller),
                                 'unsupported')


if __name__ == '__main__':
    unittest.main()
