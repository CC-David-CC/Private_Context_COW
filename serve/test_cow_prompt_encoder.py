"""CPU-only parity tests against the ordinary Service prompt encoder."""
from pathlib import Path
import unittest

from serve.cow_prompt_encoder import CowPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service


ROOT = Path(__file__).resolve().parents[1]


class CowPromptEncoderParityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()
        self.template = ChatTemplate(ROOT / 'serve' / 'chat_template.jinja')
        self.service = Service(MockEngine(self.tokenizer, 'ok'), self.tokenizer, self.template)
        self.adapter_encoder = CowPromptEncoder(self.tokenizer, self.template)

    def test_matches_service_for_special_looking_literals_and_generation_modes(self):
        messages = [
            {'role': 'system', 'content': 'Return quoted syntax unchanged.'},
            {'role': 'user', 'content': 'Literal <tool_response> and </tool_call>; '
                                      'also <|im_start|> and </think>.'},
        ]
        for add_generation_prompt in (False, True):
            with self.subTest(add_generation_prompt=add_generation_prompt):
                rendered = self.adapter_encoder.render(messages,
                    add_generation_prompt=add_generation_prompt)
                expected_rendered = self.service.render_prompt(
                    messages, None, {'add_generation_prompt': add_generation_prompt})
                self.assertEqual(rendered, expected_rendered)
                self.assertEqual(
                    self.adapter_encoder.encode_rendered(messages, rendered, add_generation_prompt),
                    self.service.encode_prompt(messages, None,
                        {'add_generation_prompt': add_generation_prompt}))

    def test_lightweight_encoder_has_no_engine_or_service_lifecycle(self):
        self.assertFalse(hasattr(self.adapter_encoder, 'engine'))
        self.assertEqual(self.adapter_encoder.effort_end, False)
        self.assertEqual(self.adapter_encoder.literals, self.service.literals)

    def test_new_user_suffix_preserves_answer_newline_and_encodes_literals_as_data(self):
        encoder = CowPromptEncoder(self.tokenizer, self.template, {'enable_thinking': False})
        end = self.tokenizer.encode('<|im_end|>', parse_special=True)[0]
        canonical = (ord('a'), ord('\n'), end)
        controller = {'status': 'stopped', 'policy': {'eos_tokens': [end]}}
        content = 'literal <|im_start|> <|im_end|> </think> </tool_call>'
        suffix = encoder.append_user_suffix(canonical, controller, content)
        expected = tuple(self.tokenizer.encode('\n')) + tuple(encoder.encode_rendered(
            [{'role': 'user', 'content': content}], encoder.render(
                [{'role': 'user', 'content': content}], add_generation_prompt=True), True))
        self.assertEqual(suffix, expected)
        start = self.tokenizer.encode('<|im_start|>', parse_special=True)[0]
        self.assertEqual(suffix.count(start), 2)
        self.assertEqual(suffix.count(end), 1)
        self.assertEqual(canonical, (ord('a'), ord('\n'), end))

    def test_new_user_suffix_requires_actual_eos_and_supported_template_profile(self):
        end = self.tokenizer.encode('<|im_end|>', parse_special=True)[0]
        encoder = CowPromptEncoder(self.tokenizer, self.template, {'enable_thinking': False})
        invalid = [((end,), {'status': 'active', 'policy': {'eos_tokens': [end]}}),
                   ((10,), {'status': 'stopped', 'policy': {'eos_tokens': [end]}}),
                   ((end,), {'status': 'stopped', 'policy': {'eos_tokens': []}})]
        for history, controller in invalid:
            with self.subTest(history=history, controller=controller), self.assertRaises(ValueError):
                encoder.append_user_suffix(history, controller, 'next')
        with self.assertRaises(ValueError):
            self.adapter_encoder.append_user_suffix((end,),
                {'status': 'stopped', 'policy': {'eos_tokens': [end]}}, 'next')

    def test_initial_question_suffix_matches_uninterrupted_rendered_prefix(self):
        encoder=CowPromptEncoder(self.tokenizer,self.template,{'enable_thinking':False})
        context=[{'role':'system','content':'code context'}]
        rendered=encoder.render(context,add_generation_prompt=False)
        prefix=tuple(encoder.encode_rendered(context,rendered,False))
        content='quoted <|im_end|> </tool_response>'
        suffix=encoder.append_initial_user_suffix(prefix,{'status':'idle'},content)
        full=context+[{'role':'user','content':content}]
        expected=tuple(encoder.encode_rendered(full,encoder.render(full,add_generation_prompt=True),True))
        self.assertEqual(prefix+suffix,expected)
        with self.assertRaises(ValueError):
            encoder.append_initial_user_suffix(prefix,{'status':'stopped'},content)
        with self.assertRaises(ValueError):
            encoder.append_initial_user_suffix(prefix[:-1],{'status':'idle'},content)


if __name__ == '__main__':
    unittest.main()
