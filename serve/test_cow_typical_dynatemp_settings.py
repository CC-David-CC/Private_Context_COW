"""Focused serving and CLI routing tests for Typical and DynaTemp settings."""
from concurrent.futures import Future
from contextlib import contextmanager
import json
import math
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
        self.snapshot.set_result({'tokens': [7]})
        self.durable = Future()
        self.durable.set_result({'snapshot': 'saved', 'branch': {'revision': 1}})


class _TokenAdapter:
    def __init__(self, controller):
        self.controller = dict(controller)
        self.generate_call = None

    def source_state(self, branch, revision, operation_id):
        self.source_call = (branch, revision, operation_id)
        return type('Head', (), {'name': branch, 'tip': 'snap-1', 'root': 'snap-0'})(), (10, 11), dict(self.controller)

    def namespace(self, _branch):
        return None

    def generate(self, *args, **kwargs):
        self.generate_call = (args, kwargs)
        return _Request()


class _Store:
    def __init__(self, controller):
        self.controller = dict(controller)

    @contextmanager
    def read_lease(self, _snapshot):
        yield

    def lineage(self, _snapshot):
        return 'root', [10, 11]

    def get_snapshot(self, _snapshot):
        return {'schema': 1, 'controller': dict(self.controller)}


class _Coordinator:
    def __init__(self, controller):
        self.store = _Store(controller)
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


def _saved_policy():
    return {
        'temperature': .61, 'min_p': .08, 'top_k': 37, 'top_p': .92,
        'repetition_penalty': 1.0, 'typical_p': .84,
        'dynatemp_range': .22, 'dynatemp_exponent': 1.3,
    }


class TypicalDynaTempPolicyTests(unittest.TestCase):
    def test_adapter_normalizes_typical_and_dyna_defaults(self):
        policy = CowAdapter._validate_settings({'temperature': .7, 'typical_p': .8})
        self.assertEqual(policy['typical_p'], .8)
        self.assertEqual(policy.get('dynatemp_range', 0.0), 0.0)
        self.assertEqual(policy.get('dynatemp_exponent', 1.0), 1.0)

        policy = CowAdapter._validate_settings({'temperature': .7, 'dynatemp_range': .2})
        self.assertEqual(policy.get('typical_p', 1.0), 1.0)
        self.assertEqual(policy['dynatemp_range'], .2)
        self.assertEqual(policy['dynatemp_exponent'], 1.0)

    def test_token_adapter_resume_merges_partial_settings_from_intent_source(self):
        saved = _saved_policy()
        coordinator = _Coordinator({
            'schema': 3, 'status': 'active', 'constraint': None, 'policy': saved,
        })
        adapter = CowAdapter(coordinator)
        self.assertEqual(adapter.generate(
            'branch-a', [10, 11], 8, 'op-continue', 12,
            settings={'dynatemp_range': .35},
        ), 'accepted')
        self.assertEqual(coordinator.source, ('branch-a', 8, 'op-continue'))
        args, kwargs = coordinator.submitted
        self.assertEqual(args[:3], ('branch-a', (), 12))
        self.assertTrue(kwargs['resume'])
        self.assertEqual(kwargs['sampling'], {**saved, 'dynatemp_range': .35})

    def test_text_resume_inherits_omitted_typical_and_dynatemp_fields(self):
        saved = _saved_policy()
        token_adapter = _TokenAdapter({
            'schema': 3, 'status': 'active', 'constraint': None,
            'policy': saved, 'output_pending_hex': '',
        })
        adapter = CowTextAdapter(token_adapter, _Tokenizer(), lambda *_a, **_k: '',
                                 lambda *_a, **_k: ())
        adapter.resume('branch-a', 4, 'op-text-continue', 16,
                       settings={'typical_p': .73})
        args, kwargs = token_adapter.generate_call
        self.assertEqual(args[:5], ('branch-a', (10, 11), 4, 'op-text-continue', 16))
        self.assertEqual(kwargs['settings'], {**saved, 'typical_p': .73})

    def test_explicit_dyna_settings_upgrade_active_legacy_greedy_without_resetting_progress(self):
        token_adapter = _TokenAdapter({
            'schema': 1, 'status': 'active', 'constraint': None,
            'policy': {'temperature': 0}, 'output_pending_hex': '',
        })
        adapter = CowTextAdapter(token_adapter, _Tokenizer(), lambda *_a, **_k: '',
                                 lambda *_a, **_k: ())
        adapter.resume('branch-a', 2, 'op-upgrade', 8,
                       settings={'dynatemp_range': .15})
        kwargs = token_adapter.generate_call[1]
        self.assertEqual(kwargs['settings']['temperature'], .7)
        self.assertEqual(kwargs['settings']['dynatemp_range'], .15)
        self.assertEqual(kwargs['settings']['dynatemp_exponent'], 1.0)

    def test_policy_rejects_out_of_domain_or_nonfinite_values(self):
        invalid = (
            {'typical_p': 0.0}, {'typical_p': 1.01}, {'typical_p': float('nan')},
            {'dynatemp_range': -0.1}, {'dynatemp_range': float('inf')},
            {'dynatemp_exponent': 0.0}, {'dynatemp_exponent': float('nan')},
            {'temperature': 1e308, 'dynatemp_range': 1e308},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(UnsupportedGreedyRequest):
                CowAdapter._validate_settings({'temperature': .7, **settings})


class TypicalDynaTempCliTests(unittest.TestCase):
    def test_cli_adds_flags_defaults_new_sampling_and_keeps_continue_partial(self):
        parse = cow_cli.parser()
        ask = parse.parse_args([
            'ask', '--branch', 'a', '--text', 'q', '--typical-p', '.8',
            '--dynatemp-range', '.2',
        ])
        settings = cow_cli.sampling_settings(ask)
        self.assertEqual(settings, {
            'temperature': .7, 'min_p': .05, 'top_k': 0, 'top_p': 1.0,
            'typical_p': .8, 'dynatemp_range': .2, 'dynatemp_exponent': 1.0,
        })
        continued = parse.parse_args([
            'continue', '--branch', 'a', '--dynatemp-exponent', '1.4',
        ])
        self.assertEqual(cow_cli.sampling_settings(continued), {'dynatemp_exponent': 1.4})

    def test_cli_registers_settings_for_all_generation_actions_and_rejects_bad_values(self):
        parse = cow_cli.parser()
        for command, extra in (
            ('ask', ['--text', 'q']), ('answer', []), ('continue', []),
            ('append-user', ['--text', 'q']),
        ):
            with self.subTest(command=command):
                args = parse.parse_args([
                    command, '--branch', 'a', *extra,
                    '--typical-p', '.7', '--dynatemp-range', '.1',
                    '--dynatemp-exponent', '1.2',
                ])
                self.assertEqual(cow_cli.sampling_settings(args)['typical_p'], .7)

        for flag, value in (
            ('--typical-p', '0'), ('--typical-p', 'nan'),
            ('--dynatemp-range', '-1'), ('--dynatemp-range', 'inf'),
            ('--dynatemp-exponent', '0'), ('--dynatemp-exponent', 'nan'),
        ):
            with self.subTest(flag=flag, value=value):
                args = parse.parse_args(['ask', '--branch', 'a', '--text', 'q', flag, value])
                with self.assertRaises(cow_cli.ClientError):
                    cow_cli.sampling_settings(args)

        overflowing = parse.parse_args([
            'ask', '--branch', 'a', '--text', 'q',
            '--temperature', '1e308', '--dynatemp-range', '1e308',
        ])
        with self.assertRaises(cow_cli.ClientError):
            cow_cli.sampling_settings(overflowing)


class _HttpRequest:
    def __init__(self):
        self.events = queue.Queue()
        self.snapshot = Future()
        self.snapshot.set_result({'tokens': [7]})
        self.durable = Future()
        self.durable.set_result({'snapshot': 'saved', 'branch': {'revision': 1}})


class _HttpTextRequest:
    def __init__(self):
        self.request = _HttpRequest()
        self.snapshot = self.request.snapshot
        self.durable = self.request.durable

    def decode_event(self, event):
        return [event]

    def text_result(self):
        return 'text'


class _HttpAdapter:
    def head(self, branch):
        return {'name': branch, 'revision': 0, 'tip': 'root'}


class _HttpTextAdapter:
    def __init__(self):
        self.calls = []

    def _record(self, action, *args, **kwargs):
        self.calls.append((action, args, kwargs))
        return _HttpTextRequest()

    def answer(self, *args, **kwargs): return self._record('answer', *args, **kwargs)
    def ask(self, *args, **kwargs): return self._record('ask', *args, **kwargs)
    def resume(self, *args, **kwargs): return self._record('resume', *args, **kwargs)
    def append_user(self, *args, **kwargs): return self._record('append_user', *args, **kwargs)


class TypicalDynaTempHttpTests(unittest.TestCase):
    def test_text_generation_routes_forward_policy_settings_unchanged(self):
        text_adapter = _HttpTextAdapter()
        server = make_server(_HttpAdapter(), text_adapter=text_adapter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        settings = {
            'temperature': .7, 'min_p': .05, 'top_k': 0, 'top_p': 1.0,
            'repetition_penalty': 1.0, 'typical_p': .8,
            'dynatemp_range': .2, 'dynatemp_exponent': 1.2,
        }
        try:
            routes = (
                ('/cow/text/answer', 'answer', {}),
                ('/cow/text/ask', 'ask', {'content': 'q'}),
                ('/cow/text/resume', 'resume', {}),
                ('/cow/text/append-user', 'append_user', {'content': 'next'}),
            )
            for index, (path, method, extra) in enumerate(routes):
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
                    self.assertEqual(text_adapter.calls[-1][0], method)
                    self.assertEqual(text_adapter.calls[-1][2]['settings'], settings)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == '__main__':
    unittest.main()
