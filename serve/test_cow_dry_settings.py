"""CPU contract tests for opt-in DRY generation settings."""
from concurrent.futures import Future
from contextlib import contextmanager
import json
import queue
import threading
import unittest
import urllib.request

from serve.cow_adapter import CowAdapter, UnsupportedGreedyRequest
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextAdapter
from tools import cow_cli


class _Request:
    def __init__(self):
        self.events = queue.Queue()
        self.snapshot = Future()
        self.snapshot.set_result({'tokens': [5]})
        self.durable = Future()
        self.durable.set_result({'snapshot': 'saved', 'branch': {'revision': 1}})


class _ActiveTokenAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.generate_call = None
        self.head_value = type('Head', (), {'name': 'branch-a', 'tip': 'snap-1',
                                            'root': 'snap-0'})()

    def source_state(self, branch, revision, operation_id):
        self.source_call = (branch, revision, operation_id)
        return self.head_value, (10, 11), {
            'schema': 3, 'status': 'active', 'constraint': None,
            'policy': dict(self.policy), 'output_pending_hex': ''}

    def namespace(self, _branch):
        return None

    def generate(self, *args, **kwargs):
        self.generate_call = (args, kwargs)
        return _Request()


class _CoordinatorStore:
    def __init__(self, controller):
        self.controller = controller

    @contextmanager
    def read_lease(self, _snapshot):
        yield

    def lineage(self, _snapshot):
        return 'root', [10, 11]

    def get_snapshot(self, _snapshot):
        return {'schema': 1, 'controller': dict(self.controller)}


class _Coordinator:
    def __init__(self, controller):
        self.store = _CoordinatorStore(controller)
        self.submitted = None

    def request_source_head(self, branch, revision, operation_id):
        self.source = (branch, revision, operation_id)
        return type('Head', (), {'tip': 'intent-bound-source'})()

    def submit(self, *args, **kwargs):
        self.submitted = (args, kwargs)
        return 'accepted'


class _Tokenizer:
    def token_bytes(self, _token):
        return b'x'


class DryPolicyTests(unittest.TestCase):
    def test_legacy_policy_remains_five_fields_when_dry_is_omitted(self):
        value = CowAdapter._validate_settings({'temperature': .7})
        self.assertEqual(set(value), {
            'temperature', 'min_p', 'top_k', 'top_p', 'repetition_penalty'})

    def test_any_dry_override_gets_full_defaulted_policy(self):
        value = CowAdapter._validate_settings({'temperature': .7, 'dry_multiplier': 1.2})
        self.assertEqual({key: value[key] for key in (
            'dry_multiplier', 'dry_base', 'dry_allowed_length', 'dry_penalty_last_n')},
            {'dry_multiplier': 1.2, 'dry_base': 1.75,
             'dry_allowed_length': 2, 'dry_penalty_last_n': 256})

    def test_dry_policy_rejects_invalid_numeric_types_and_ranges(self):
        bad = (
            {'dry_multiplier': True}, {'dry_multiplier': float('nan')},
            {'dry_multiplier': -0.01}, {'dry_base': .99},
            {'dry_allowed_length': True}, {'dry_allowed_length': -1},
            {'dry_allowed_length': 0x100000000},
            {'dry_penalty_last_n': True}, {'dry_penalty_last_n': 0x100000000},
        )
        for item in bad:
            with self.subTest(item=item), self.assertRaises(UnsupportedGreedyRequest):
                CowAdapter._validate_settings({'temperature': .7, **item})

    def test_partial_dry_continue_overrides_saved_policy_without_resetting_other_fields(self):
        saved = {'temperature': .63, 'min_p': .08, 'top_k': 41, 'top_p': .91,
                 'repetition_penalty': 1.0, 'dry_multiplier': .7, 'dry_base': 1.6,
                 'dry_allowed_length': 3, 'dry_penalty_last_n': 128}
        token_adapter = _ActiveTokenAdapter(saved)
        text = CowTextAdapter(token_adapter, _Tokenizer(), lambda *_a, **_k: '',
                              lambda *_a, **_k: ())
        request = text.resume('branch-a', 6, 'op-resume', 12,
                              settings={'dry_multiplier': 1.25})
        call_args, kwargs = token_adapter.generate_call
        self.assertEqual(call_args[:5], ('branch-a', (10, 11), 6, 'op-resume', 12))
        self.assertEqual(kwargs['settings'], {**saved, 'dry_multiplier': 1.25})

    def test_token_adapter_resumes_from_intent_source_and_preserves_saved_dry_fields(self):
        saved = {'temperature': .7, 'min_p': .05, 'top_k': 0, 'top_p': 1.0,
                 'repetition_penalty': 1.0, 'dry_multiplier': .6, 'dry_base': 1.4,
                 'dry_allowed_length': 4, 'dry_penalty_last_n': 96}
        coordinator = _Coordinator({'schema': 3, 'status': 'active', 'constraint': None,
                                    'policy': saved})
        adapter = CowAdapter(coordinator)
        result = adapter.generate('branch-a', [10, 11], 9, 'op-retry', 8,
                                  settings={'dry_multiplier': 1.3})
        self.assertEqual(result, 'accepted')
        self.assertEqual(coordinator.source, ('branch-a', 9, 'op-retry'))
        args, kwargs = coordinator.submitted
        self.assertEqual(args[:3], ('branch-a', (), 8))
        self.assertEqual(kwargs['expected_revision'], 9)
        self.assertTrue(kwargs['resume'])
        self.assertEqual(kwargs['sampling'], {**saved, 'dry_multiplier': 1.3})

    def test_cli_preserves_legacy_defaults_and_only_sends_explicit_continue_overrides(self):
        parse = cow_cli.parser()
        ask = parse.parse_args(['ask', '--branch', 'a', '--text', 'question'])
        self.assertIsNone(cow_cli.sampling_settings(ask))
        sampled = parse.parse_args(['ask', '--branch', 'a', '--text', 'question',
                                    '--temperature', '.8'])
        self.assertEqual(cow_cli.sampling_settings(sampled), {
            'temperature': .8, 'min_p': .05, 'top_k': 0, 'top_p': 1.0})

        ask_dry = parse.parse_args(['ask', '--branch', 'a', '--text', 'question',
                                    '--dry-multiplier', '1.3'])
        self.assertEqual(cow_cli.sampling_settings(ask_dry), {
            'temperature': .7, 'min_p': .05, 'top_k': 0, 'top_p': 1.0,
            'dry_multiplier': 1.3, 'dry_base': 1.75,
            'dry_allowed_length': 2, 'dry_penalty_last_n': 256})

        continued = parse.parse_args(['continue', '--branch', 'a', '--dry-base', '1.8'])
        self.assertEqual(cow_cli.sampling_settings(continued), {'dry_base': 1.8})

    def test_cli_registers_dry_flags_for_each_generation_action(self):
        parse = cow_cli.parser()
        for command, suffix in (
            ('ask', ['--text', 'question']),
            ('answer', []), ('append-user', ['--text', 'question']), ('continue', [])):
            with self.subTest(command=command):
                args = parse.parse_args([command, '--branch', 'a', *suffix,
                                         '--dry-multiplier', '2', '--dry-base', '1.9',
                                         '--dry-allowed-length', '4', '--dry-penalty-last-n', '512'])
                self.assertEqual(cow_cli.sampling_settings(args)['dry_penalty_last_n'], 512)

    def test_cli_rejects_dry_uint32_overflow_before_request(self):
        parse = cow_cli.parser()
        for flag in ('--dry-allowed-length', '--dry-penalty-last-n'):
            with self.subTest(flag=flag):
                args = parse.parse_args(['ask', '--branch', 'a', '--text', 'question',
                                         flag, str(0x100000000)])
                with self.assertRaises(cow_cli.ClientError):
                    cow_cli.sampling_settings(args)


class _HttpAdapter:
    def __init__(self):
        self.settings = None

    def generate(self, branch, ids, revision, operation_id, max_tokens, **kwargs):
        self.settings = kwargs['settings']
        return _Request()

    def head(self, branch):
        return {'name': branch, 'revision': 0, 'tip': 'root'}


class _TextHttpRequest:
    def __init__(self):
        self.request = _Request()
        self.snapshot = self.request.snapshot
        self.durable = self.request.durable

    def decode_event(self, event):
        return [event]

    def text_result(self):
        return 'text'


class _TextHttpAdapter:
    def __init__(self):
        self.calls = []

    def _record(self, action, *args, **kwargs):
        self.calls.append((action, args, kwargs))
        return _TextHttpRequest()

    def answer(self, *args, **kwargs): return self._record('answer', *args, **kwargs)
    def ask(self, *args, **kwargs): return self._record('ask', *args, **kwargs)
    def resume(self, *args, **kwargs): return self._record('resume', *args, **kwargs)
    def append_user(self, *args, **kwargs): return self._record('append_user', *args, **kwargs)


class DryHttpTests(unittest.TestCase):
    def test_http_forwards_extended_settings_unchanged(self):
        adapter = _HttpAdapter()
        server = make_server(adapter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = {'temperature': .7, 'min_p': .05, 'top_k': 0, 'top_p': 1.0,
                        'repetition_penalty': 1.0, 'dry_multiplier': 1.1,
                        'dry_base': 1.75, 'dry_allowed_length': 2,
                        'dry_penalty_last_n': 256}
            body = {'branch': 'a', 'rendered_token_ids': [1], 'revision': 0,
                    'operation_id': 'op-http', 'max_tokens': 2, 'settings': settings,
                    'stream': False}
            request = urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}/cow/generate',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(adapter.settings, settings)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_http_text_generation_routes_forward_dry_settings_unchanged(self):
        adapter = _HttpAdapter()
        text_adapter = _TextHttpAdapter()
        server = make_server(adapter, text_adapter=text_adapter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = {'dry_multiplier': .8}
            variants = (
                ('/cow/text/answer', 'answer', {}),
                ('/cow/text/ask', 'ask', {'content': 'question'}),
                ('/cow/text/resume', 'resume', {}),
                ('/cow/text/append-user', 'append_user', {'content': 'next question'}),
            )
            for index, (path, action, extra) in enumerate(variants):
                with self.subTest(path=path):
                    body = {'branch': 'a', 'revision': 0,
                            'operation_id': f'op-{index}', 'max_tokens': 2,
                            'settings': settings, 'stream': False, **extra}
                    request = urllib.request.Request(
                        f'http://127.0.0.1:{server.server_port}{path}',
                        data=json.dumps(body).encode(),
                        headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                    self.assertEqual(text_adapter.calls[-1][0], action)
                    self.assertEqual(text_adapter.calls[-1][2]['settings'], settings)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == '__main__':
    unittest.main()
