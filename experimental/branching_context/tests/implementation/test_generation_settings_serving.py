"""Generation-boundary sampler settings on COW adapters, HTTP, and CLI.

These tests exercise request forwarding and saved-controller continuation with
CPU fixtures. They do not qualify native sampling or GPU inference.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from context_time_travel.sampled_controller import start
from serve.cow_adapter import CowAdapter, UnsupportedGreedyRequest
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextAdapter, CowTextRequest, UnsupportedTextRequest
from tools import cow_cli
import serve.test_cow_http as http_fixtures
import serve.test_cow_text_adapter as text_fixtures


SAVED = dict(temperature=0.7, min_p=0.05, top_k=0, top_p=1.0,
             repetition_penalty=1.0)


class _Store:
    def __init__(self, controller):
        self.controller = controller

    @contextmanager
    def read_lease(self, _snapshot):
        yield

    def lineage(self, _snapshot):
        return None, (1, 2, 3)

    def get_snapshot(self, _snapshot):
        return {'schema': 1, 'controller': self.controller}


class _Coordinator:
    def __init__(self, controller):
        self.store = _Store(controller)
        self.calls = []

    def request_source_head(self, _branch, revision, _operation):
        return SimpleNamespace(tip='snapshot', revision=revision)

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return object()


class GenerationSettingsAdapterTests(unittest.TestCase):
    def setUp(self):
        self.state = start(SAVED, 'saved-stream', (1, 2, 3), 8, [7])
        self.coordinator = _Coordinator(self.state)
        self.adapter = CowAdapter(self.coordinator)

    def test_resume_without_settings_reuses_saved_policy_and_cursor(self):
        self.state['next_draw'] = 11
        self.adapter.generate('main', (1, 2, 3), 4, 'resume-saved', 8)
        _, kwargs = self.coordinator.calls[-1]
        self.assertTrue(kwargs.get('resume'), 'resume must continue the saved controller')
        self.assertEqual(kwargs.get('sampling'), SAVED)
        self.assertEqual(self.state['next_draw'], 11)

    def test_partial_settings_override_saved_policy_only_for_this_generation(self):
        overrides = {'temperature': 0.9, 'top_k': 3, 'top_p': 0.8}
        expected = dict(SAVED, **overrides)
        self.adapter.generate('main', (1, 2, 3), 4, 'resume-override', 8,
                              settings=overrides)
        _, kwargs = self.coordinator.calls[-1]
        self.assertTrue(kwargs.get('resume'))
        self.assertEqual(kwargs.get('sampling'), expected)
        self.assertEqual(self.state['next_draw'], 0)

    def test_zero_temperature_override_keeps_the_sampled_controller_at_its_boundary(self):
        self.state['next_draw'] = 11
        self.adapter.generate('main', (1, 2, 3), 4, 'resume-greedy-boundary', 8,
                              settings={'temperature': 0})
        _, kwargs = self.coordinator.calls[-1]
        self.assertTrue(kwargs.get('resume'))
        self.assertEqual(kwargs.get('sampling'), dict(SAVED, temperature=0))
        self.assertEqual(self.state['next_draw'], 11)

    def test_new_sampling_request_fills_omitted_fields_from_documented_defaults(self):
        self.state = {'status': 'idle'}
        self.adapter = CowAdapter(_Coordinator(self.state))
        self.adapter.generate('main', (1, 2, 3, 4), 4, 'new-sampling', 8,
                              settings={'top_k': 5, 'top_p': 0.75})
        _, kwargs = self.adapter.coordinator.calls[-1]
        self.assertEqual(kwargs.get('sampling'), dict(SAVED, top_k=5, top_p=0.75))

    def test_active_legacy_greedy_turn_can_opt_into_sampling_without_losing_resume_state(self):
        legacy = {'schema': 1, 'status': 'active', 'policy': {'temperature': 0},
                  'history': [1, 2, 3], 'generated_tokens': [7], 'position': 4}
        coordinator = _Coordinator(legacy)
        adapter = CowAdapter(coordinator)
        adapter.generate('main', (1, 2, 3), 4, 'legacy-sampled-resume', 8,
                         settings={'temperature': 0.9, 'top_k': 2})
        _, kwargs = coordinator.calls[-1]
        self.assertTrue(kwargs.get('resume'))
        self.assertEqual(kwargs.get('sampling'),
                         dict(SAVED, temperature=0.9, top_k=2))
        self.assertEqual(legacy['generated_tokens'], [7])
        self.assertEqual(legacy['position'], 4)

    def test_sampling_knobs_cannot_bypass_missing_saved_grammar_registry(self):
        constrained = dict(self.state, constraint={'grammar_id': 'pinned'})
        coordinator = _Coordinator(constrained)
        adapter = CowAdapter(coordinator)
        with self.assertRaisesRegex(UnsupportedGreedyRequest, 'Saved grammar registry identity'):
            adapter.generate('main', (1, 2, 3), 4, 'grammar-sampling', 8,
                             settings={'temperature': 0.9})
        self.assertEqual(coordinator.calls, [])

    def test_sampling_override_preserves_an_available_active_grammar(self):
        constrained=dict(self.state,constraint={'grammar_id':'pinned','agent_state':'answer'})
        coordinator=_Coordinator(constrained)
        registry=SimpleNamespace(matches_controller=lambda value:value==constrained,
                                 descriptor=lambda name:{'grammar_id':'pinned'})
        CowAdapter(coordinator,registry).generate('main',(1,2,3),4,'grammar-sampling',8,
                                                 settings={'temperature':0.9})
        _,options=coordinator.calls[-1]
        self.assertTrue(options['resume'])
        self.assertIs(options['grammar_registry'],registry)
        self.assertEqual(options['grammar_state'],'answer')
        self.assertEqual(options['sampling']['temperature'],0.9)
        self.assertEqual(constrained['constraint'],{'grammar_id':'pinned','agent_state':'answer'})


class GenerationSettingsTextAdapterTests(unittest.TestCase):
    def test_partial_resume_override_preserves_saved_custom_sampler_fields(self):
        saved = dict(temperature=0.7, min_p=0.1, top_k=40, top_p=0.9,
                     repetition_penalty=1.0)
        tokens = text_fixtures.TokenAdapter((1, 2, 3))
        tokens.controller = start(saved, 'saved-stream', (1, 2, 3), 8, [7])
        adapter = CowTextAdapter(tokens, text_fixtures.Tokenizer(), lambda *a, **kw: None,
                                 lambda *a, **kw: None)
        adapter.resume('main', 2, 'text-partial-resume', 8,
                       settings={'temperature': 1.2})
        self.assertEqual(tokens.calls[-1][6]['settings'], dict(saved, temperature=1.2))

    def test_legacy_greedy_resume_can_opt_into_sampling_and_zero_temperature(self):
        for override, expected in (
                ({'temperature': 0.9, 'top_k': 3}, dict(SAVED, temperature=0.9, top_k=3)),
                ({'temperature': 0, 'top_k': 3}, dict(SAVED, temperature=0, top_k=3)),
                (None, {'temperature': 0}),
                ({'temperature': 0}, {'temperature': 0})):
            with self.subTest(override=override):
                tokens = text_fixtures.TokenAdapter((1, 2, 3))
                tokens.controller = {'schema': 1, 'status': 'active',
                                     'policy': {'temperature': 0},
                                     'generated_tokens': [7], 'position': 4}
                adapter = CowTextAdapter(tokens, text_fixtures.Tokenizer(), lambda *a, **kw: None,
                                         lambda *a, **kw: None)
                adapter.resume('main', 2, 'legacy-text-resume', 8, settings=override)
                self.assertEqual(tokens.calls[-1][6]['settings'], expected)
                self.assertEqual(tokens.controller['generated_tokens'], [7])
                self.assertEqual(tokens.controller['position'], 4)

    def test_boolean_temperature_is_not_accepted_as_legacy_zero(self):
        tokens = text_fixtures.TokenAdapter((1, 2, 3))
        tokens.controller = {'schema': 1, 'status': 'active',
                             'policy': {'temperature': 0}, 'generated_tokens': [7]}
        adapter = CowTextAdapter(tokens, text_fixtures.Tokenizer(), lambda *a, **kw: None,
                                 lambda *a, **kw: None)
        with self.assertRaisesRegex(UnsupportedTextRequest, 'Unsupported sampled continuation policy'):
            adapter.resume('main', 2, 'legacy-bool-temp', 8,
                           settings={'temperature': False})
        self.assertEqual(tokens.calls, [])

    def test_resume_accepts_generation_local_overrides_without_mutating_saved_controller(self):
        tokens = text_fixtures.TokenAdapter((1, 2, 3))
        tokens.controller = start(SAVED, 'saved-stream', (1, 2, 3), 8, [7])
        tokens.controller['next_draw'] = 6
        adapter = CowTextAdapter(tokens, text_fixtures.Tokenizer(), lambda *a, **kw: None,
                                 lambda *a, **kw: None)
        overrides = {'temperature': 0.85, 'top_k': 4}
        adapter.resume('main', 2, 'text-resume', 8, settings=overrides)
        forwarded = tokens.calls[-1][6]['settings']
        self.assertEqual(forwarded, dict(SAVED, temperature=0.85, top_k=4))
        self.assertEqual(tokens.controller['next_draw'], 6)

    def test_zero_temperature_continuation_keeps_other_saved_sampler_fields(self):
        tokens = text_fixtures.TokenAdapter((1, 2, 3))
        tokens.controller = start(SAVED, 'saved-stream', (1, 2, 3), 8, [7])
        tokens.controller['next_draw'] = 6
        adapter = CowTextAdapter(tokens, text_fixtures.Tokenizer(), lambda *a, **kw: None,
                                 lambda *a, **kw: None)
        adapter.resume('main', 2, 'text-resume-zero', 8, settings={'temperature': 0})
        self.assertEqual(tokens.calls[-1][6]['settings'], dict(SAVED, temperature=0))
        self.assertEqual(tokens.controller['next_draw'], 6)


class GenerationSettingsHTTPTests(unittest.TestCase):
    def test_resume_route_whitelists_and_forwards_settings(self):
        outer = http_fixtures.Adapter()
        captured = []

        class Tokenizer:
            def token_bytes(self, _token):
                return b'x'

        class Text:
            def resume(self, branch, revision, operation, max_tokens, *, settings=None, cancelled=None):
                captured.append(settings)
                request = outer.generate(branch, [], revision, operation, max_tokens,
                                         settings=settings, cancelled=cancelled)
                return CowTextRequest(request, Tokenizer(),
                    controller_reader=lambda _snapshot: {'status': 'active',
                        'policy': {'eos_tokens': []}})

        server = make_server(outer, api_key='secret', text_adapter=Text())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = dict(branch='A', revision=2, operation_id='resume-settings', max_tokens=8,
                    settings={'temperature': 0.9, 'min_p': 0.05, 'top_k': 3, 'top_p': 0.8})
        request = urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}/cow/text/resume',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer secret'})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                events = [json.loads(line) for line in response.read().decode().splitlines()]
            self.assertEqual(events[-1]['event'], 'complete')
            self.assertEqual(captured, [body['settings']])
            self.assertEqual(outer.calls[-1][1]['settings'], body['settings'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


class GenerationSettingsCLITests(unittest.TestCase):
    def test_all_generation_commands_accept_the_four_sampler_knobs(self):
        parser = cow_cli.parser()
        commands = (
            ['ask', '--branch', 'main', '--text', 'q'],
            ['answer', '--branch', 'main'],
            ['append-user', '--branch', 'main', '--text', 'next'],
            ['continue', '--branch', 'main'],
        )
        for command in commands:
            args = parser.parse_args(command + ['--temperature', '0.9', '--min-p', '0.04',
                                                '--top-k', '4', '--top-p', '0.8'])
            with self.subTest(command=command):
                self.assertEqual(cow_cli.sampling_settings(args),
                    dict(temperature=0.9, min_p=0.04, top_k=4, top_p=0.8))

    def test_any_sampling_knob_uses_documented_defaults_and_no_knobs_stays_greedy(self):
        parser = cow_cli.parser()
        args = parser.parse_args(['ask', '--branch', 'main', '--text', 'q', '--top-k', '3'])
        self.assertEqual(cow_cli.sampling_settings(args),
            dict(temperature=0.7, min_p=0.05, top_k=3, top_p=1.0))
        greedy = parser.parse_args(['ask', '--branch', 'main', '--text', 'q'])
        self.assertIsNone(cow_cli.sampling_settings(greedy))
        explicit_greedy = parser.parse_args(['continue', '--branch', 'main', '--temperature', '0'])
        self.assertEqual(cow_cli.sampling_settings(explicit_greedy), {'temperature': 0})

    def test_filter_ranges_and_zero_temperature_policy(self):
        parser = cow_cli.parser()
        invalid = (
            ['--top-k', '-1'], ['--top-k', '3.0'], ['--top-k', '18446744073709551616'], ['--top-p', '0'],
            ['--top-p', '1.01'],
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parser.parse_args(['ask', '--branch', 'main', '--text', 'q'] + options)
        for value in ('-0.1', 'nan', 'inf'):
            args = parser.parse_args(['ask', '--branch', 'main', '--text', 'q',
                                      '--temperature', value])
            with self.subTest(temperature=value), self.assertRaises(cow_cli.ClientError):
                cow_cli.sampling_settings(args)
        args = parser.parse_args(['continue', '--branch', 'main', '--temperature', '0',
                                  '--top-k', '2'])
        self.assertEqual(cow_cli.sampling_settings(args), {'temperature': 0, 'top_k': 2})

    def test_zero_temperature_cli_continuation_preserves_other_saved_settings(self):
        args = cow_cli.parser().parse_args(['continue', '--branch', 'main',
                                            '--temperature', '0'])
        self.assertEqual(cow_cli.sampling_settings(args), {'temperature': 0})

    def test_continue_sends_only_explicit_fields_for_saved_policy_merge(self):
        parser = cow_cli.parser()
        args = parser.parse_args(['continue', '--branch', 'main', '--temperature', '1.2'])
        self.assertEqual(cow_cli.sampling_settings(args), {'temperature': 1.2})

    def test_each_mutating_text_command_sends_the_resolved_generation_settings(self):
        commands = (
            ['ask', '--branch', 'main', '--text', 'q'],
            ['answer', '--branch', 'main'],
            ['append-user', '--branch', 'main', '--text', 'next'],
            ['continue', '--branch', 'main'],
        )
        expected = dict(temperature=0.9, min_p=0.04, top_k=3, top_p=0.8)
        for command in commands:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                args = cow_cli.parser().parse_args(['--state-dir', directory] + command +
                    ['--temperature', '0.9', '--min-p', '0.04', '--top-k', '3', '--top-p', '0.8'])
                with (patch.object(cow_cli, 'CowClient') as client_type,
                      patch.object(cow_cli, '_head', return_value={'revision': 9}),
                      patch.object(cow_cli, '_post_mutation', return_value={'receipt': {}}) as post,
                      patch.object(cow_cli, '_print_receipt'),
                      patch.object(cow_cli, 'operation_id', return_value='sample-op')):
                    cow_cli.run(args)
                self.assertEqual(post.call_args.args[4]['settings'], expected)
                if command[0] == 'continue':
                    self.assertEqual(post.call_args.args[3], '/cow/text/resume')


if __name__ == '__main__':
    unittest.main()
