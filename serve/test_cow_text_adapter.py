"""CPU-only contract tests for the opt-in COW text rendering boundary."""
from concurrent.futures import Future
import queue
from types import SimpleNamespace
import unittest

from serve.cow_text_adapter import CowTextAdapter, UnsupportedTextRequest
from serve.cow_prefix import RenderedPrefixMismatch


class Tokenizer:
    def __init__(self):
        self.parse_special_flags = []

    def encode(self, text, parse_special=False):
        self.parse_special_flags.append(parse_special)
        return list(text.encode('utf-8'))

    def token_bytes(self, token):
        return {1: b'\xe2', 2: b'\x82', 3: b'\xac', 4: b''}[token]

    def decode(self, ids, errors='replace'):
        return bytes(ids).decode('utf-8', errors=errors)


class TokenAdapter:
    def __init__(self, canonical=()):
        self.canonical = tuple(canonical)
        self.calls = []
        self.controller = {'status': 'idle'}
        self.request = SimpleNamespace(events=queue.SimpleQueue(), snapshot=Future(), durable=Future())

    def source_state(self, branch, revision, operation_id):
        return None, self.canonical, self.controller

    def create(self, branch, rendered_ids, operation_id):
        self.calls.append(('create', branch, tuple(rendered_ids), operation_id))
        self.canonical = tuple(rendered_ids)
        return self.request

    def generate(self, branch, rendered_ids, revision, operation_id, max_tokens, **kwargs):
        if tuple(rendered_ids[:len(self.canonical)]) != self.canonical:
            raise AssertionError('text adapter bypassed token-prefix validation')
        self.calls.append(('generate', branch, tuple(rendered_ids), revision, operation_id, max_tokens, kwargs))
        return self.request


class CowTextAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.tokens = TokenAdapter()
        self.render_calls = []
        self.encode_calls = []

        def render(messages, *, add_generation_prompt):
            self.render_calls.append((messages, add_generation_prompt))
            rendered = ''.join(f'<{m["role"]}>{m["content"]}<|im_end|>' for m in messages)
            return rendered + ('<assistant>' if add_generation_prompt else '')

        def encode_rendered(messages, rendered, add_generation_prompt):
            self.encode_calls.append((messages, rendered, add_generation_prompt))
            return self.tokenizer.encode(rendered, parse_special=True)

        self.adapter = CowTextAdapter(self.tokens, self.tokenizer, render, encode_rendered)

    def test_create_and_generate_render_complete_history_and_forward_exact_ids(self):
        first = [{'role': 'user', 'content': 'hello'}]
        self.adapter.create('main', first, 'create-1')
        prefix = self.tokens.canonical
        # Simulate committed generated IDs; the next complete template render
        # must preserve the entire earlier token stream exactly.
        self.tokens.canonical += tuple(b'<assistant>answer<|im_end|>')
        history = first + [{'role': 'assistant', 'content': 'answer'},
                           {'role': 'user', 'content': 'next'}]
        request = self.adapter.generate('main', history, 1, 'gen-1', 16)
        rendered = ''.join(f'<{m["role"]}>{m["content"]}<|im_end|>' for m in history) + '<assistant>'
        full = tuple(self.tokenizer.encode(rendered))
        self.assertIs(request.request, self.tokens.request)
        self.assertEqual(self.tokens.calls[0][2], prefix)
        self.assertEqual(self.tokens.calls[1][2], full)
        self.assertEqual(self.tokens.calls[1][6]['settings'], None)
        self.assertEqual(self.render_calls, [(first, False), (history, True)])
        self.assertEqual(self.tokenizer.parse_special_flags[:2], [True, True])

    def test_tokenizer_boundary_mismatch_is_rejected_by_token_adapter_before_submit(self):
        self.tokens.canonical = (ord('a'),)
        with self.assertRaises(RenderedPrefixMismatch):
            self.adapter.generate('main', [{'role': 'user', 'content': 'ab'}], 0, 'merge', 8)
        self.assertEqual([call[0] for call in self.tokens.calls], [])

    def test_literal_special_looking_text_is_sent_through_frontend_encoder(self):
        content = 'quoted <|im_end|> remains text'
        messages = [{'role': 'user', 'content': content}]
        self.adapter.create('literal', messages, 'literal-create')
        self.assertEqual(self.encode_calls[0][0][0]['content'], content)
        self.assertIn(content, self.encode_calls[0][1])

    def test_rejects_roles_images_tools_and_options_before_dispatch(self):
        unsupported_messages = [
            [{'role': 'tool', 'content': 'result'}],
            [{'role': 'user', 'content': [{'type': 'image', 'url': 'x'}]}],
            [{'role': 'assistant', 'content': 'x', 'tool_calls': []}],
        ]
        for messages in unsupported_messages:
            with self.subTest(messages=messages), self.assertRaises(UnsupportedTextRequest):
                self.adapter.generate('main', messages, 0, 'unsupported', 8)
        for options in ({'stop': ['x']}, {'force': 'tool'}, {'tools': []}, {'top_p': 1}):
            with self.subTest(options=options), self.assertRaises(UnsupportedTextRequest):
                self.adapter.generate('main', [{'role': 'user', 'content': 'x'}], 0,
                                      'unsupported', 8, **options)
        self.assertEqual(self.tokens.calls, [])

    def test_requires_linear_token_byte_decoder_before_any_request(self):
        class NoTokenBytes:
            def encode(self, text, parse_special=False):
                return list(text.encode())

        with self.assertRaisesRegex(TypeError, 'tokenizer.token_bytes'):
            CowTextAdapter(self.tokens, NoTokenBytes(), self.adapter.render, self.adapter.encode_rendered)
        self.assertEqual(self.tokens.calls, [])

    def test_resume_submits_canonical_ids_without_rendering_and_carries_pending_utf8(self):
        canonical = (11, 12, 13)
        self.tokens.canonical = canonical
        self.tokens.controller = {
            'status': 'active', 'policy': {'temperature': 0},
            'output_pending_hex': 'e282',
        }
        render_before = list(self.render_calls)
        encode_before = list(self.encode_calls)

        resumed = self.adapter.resume('main', 7, 'resume-1', 5)

        self.assertIs(resumed.request, self.tokens.request)
        self.assertEqual(self.tokens.calls, [(
            'generate', 'main', canonical, 7, 'resume-1', 5,
            {'settings': {'temperature': 0}, 'cancelled': None})])
        self.assertEqual(self.render_calls, render_before)
        self.assertEqual(self.encode_calls, encode_before)
        self.tokens.request.events.put({'event': 'token', 'token': 3})
        self.tokens.request.events.put({'event': 'token', 'token': 4, 'turn_complete': True})
        self.tokens.request.events.put({'event': 'durable', 'receipt': {'ok': True}})
        self.tokens.request.snapshot.set_result({'tokens': [3]})
        self.tokens.request.durable.set_result({'ok': True})
        events = list(resumed.iter_events(timeout=0))
        text = [event['text'] for event in events if event['event'] == 'text']
        self.assertEqual(text, ['€'])
        self.assertIn({'event': 'turn_complete', 'token_id': 4}, events)

    def test_resume_rejects_idle_stopped_and_non_greedy_before_submit(self):
        states = [
            {'status': 'idle'},
            {'status': 'stopped', 'policy': {'temperature': 0}},
            {'status': 'active', 'policy': {'temperature': 0.7}},
            {'status': 'active', 'policy': {}},
        ]
        for state in states:
            with self.subTest(state=state):
                self.tokens.calls.clear()
                self.tokens.controller = state
                with self.assertRaises(UnsupportedTextRequest):
                    self.adapter.resume('main', 7, 'resume-reject', 5)
                self.assertEqual(self.tokens.calls, [])
        self.assertEqual(self.render_calls, [])
        self.assertEqual(self.encode_calls, [])

    def test_append_user_preserves_original_ids_and_uses_request_bound_controller(self):
        self.tokens.canonical = (11, 12, 198, 257)
        self.tokens.controller = {'status': 'stopped', 'policy': {'eos_tokens': [257]}}
        seen = []
        def suffix(canonical, controller, content):
            seen.append((canonical, controller, content))
            return (10, 40, 41)
        adapter = CowTextAdapter(self.tokens, self.tokenizer, self.adapter.render,
                                 self.adapter.encode_rendered, append_user_suffix=suffix)
        request = adapter.append_user('main', 'next question', 7, 'next-1', 8)
        self.assertIs(request.request, self.tokens.request)
        self.assertEqual(self.tokens.calls[0][2], (11, 12, 198, 257, 10, 40, 41))
        self.assertEqual(seen, [(self.tokens.canonical, self.tokens.controller, 'next question')])
        self.assertEqual(self.render_calls, [])
        self.assertEqual(self.encode_calls, [])

    def test_append_user_rejects_active_or_missing_suffix_encoder_before_submit(self):
        with self.assertRaises(UnsupportedTextRequest):
            self.adapter.append_user('main', 'next', 0, 'next', 8)
        adapter = CowTextAdapter(self.tokens, self.tokenizer, self.adapter.render,
                                 self.adapter.encode_rendered, append_user_suffix=lambda *a: (10,))
        for status in ('active', 'idle'):
            self.tokens.controller = {'status': status}
            with self.subTest(status=status), self.assertRaises(UnsupportedTextRequest):
                adapter.append_user('main', 'next', 0, 'next', 8)
        self.assertEqual(self.tokens.calls, [])

    def test_incremental_utf8_text_events_preserve_original_token_ids(self):
        messages = [{'role': 'user', 'content': 'x'}]
        rendered = '<user>x<|im_end|><assistant>'
        self.tokens.canonical = tuple(rendered.encode())
        self.tokens.controller = {'status': 'active', 'output_pending_hex': 'e2'}

        request = self.adapter.generate('main', messages, 0, 'utf8-first', 4)
        self.tokens.request.events.put({'event': 'token', 'token': 2})  # 0x82 completes only later
        self.tokens.request.events.put({'event': 'snapshot', 'tokens': [2]})
        self.tokens.request.events.put({'event': 'durable', 'receipt': {'ok': True}})
        self.tokens.request.snapshot.set_result({'tokens': [2]})
        self.tokens.request.durable.set_result({'ok': True})
        first_events = list(request.iter_events(timeout=0))
        self.assertEqual([event['text'] for event in first_events if event['event'] == 'text'], [''])
        self.assertFalse(any(event.get('text') == '�' for event in first_events))
        self.assertTrue(any(event.get('event') == 'durable' for event in first_events))
        self.assertEqual(self.tokens.request.snapshot.result(), {'tokens': [2]})
        self.assertEqual(self.tokens.request.durable.result(), {'ok': True})

        # The next request resumes from the canonical controller's pending
        # bytes; an explicit turn-complete acknowledgement may now flush.
        self.tokens.controller = {'status': 'active', 'output_pending_hex': 'e282'}
        self.tokens.request = SimpleNamespace(events=queue.SimpleQueue(), snapshot=Future(), durable=Future())
        continued = self.adapter.generate('main', messages, 0, 'utf8-second', 4)
        self.tokens.request.events.put({'event': 'token', 'token': 3})
        self.tokens.request.events.put({'event': 'token', 'token': 4, 'turn_complete': True})
        self.tokens.request.events.put({'event': 'durable', 'receipt': {'ok': True}})
        self.tokens.request.snapshot.set_result({'tokens': [3]})
        self.tokens.request.durable.set_result({'ok': True})
        events = list(continued.iter_events(timeout=0))
        text = [event for event in events if event['event'] == 'text']
        self.assertEqual([item['text'] for item in text], ['€'])
        self.assertIn({'event': 'turn_complete', 'token_id': 4}, events)
        self.assertEqual(self.tokens.request.snapshot.result(), {'tokens': [3]})


if __name__ == '__main__':
    unittest.main()
