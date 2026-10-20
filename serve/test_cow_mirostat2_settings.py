"""Generation-scoped Mirostat routing; CPU fixtures are not GPU evidence."""
import unittest

from serve.cow_adapter import CowAdapter, UnsupportedGreedyRequest
from serve.cow_text_adapter import CowTextAdapter
from serve.test_cow_typical_dynatemp_settings import _Coordinator, _TokenAdapter, _Tokenizer
from tools import cow_cli


def saved_policy():
    return dict(temperature=.8, min_p=0., top_k=0, top_p=1.,
                repetition_penalty=1., mirostat=2, mirostat_tau=4., mirostat_eta=.05)


class MirostatGenerationRoutingTests(unittest.TestCase):
    def test_adapter_accepts_standalone_mode_and_explicit_filter_conflict_rejects(self):
        self.assertEqual(CowAdapter._validate_settings(saved_policy()), saved_policy())
        partial = CowAdapter._validate_settings({'mirostat': 2})
        self.assertEqual(partial['min_p'], 0.)
        self.assertEqual(partial['mirostat_tau'], 5.)
        for override in ({'min_p': .05}, {'top_k': 60}, {'top_p': .95},
                         {'typical_p': .95}, {'dynatemp_range': .2},
                         {'temperature': 0}, {'xtc_probability': 1.}):
            with self.subTest(override=override):
                with self.assertRaises(UnsupportedGreedyRequest):
                    CowAdapter._validate_settings(dict(saved_policy(), **override))

    def test_token_resume_preserves_feedback_and_only_changes_requested_policy(self):
        policy = saved_policy()
        state = {'protocol': 'mirostat2-state-v1', 'version': 1, 'mu': -1.25}
        coordinator = _Coordinator(dict(schema=4, status='active', constraint=None,
            policy=policy, sampler_state=state, next_draw=9))
        self.assertEqual(CowAdapter(coordinator).generate(
            'a', [10, 11], 3, 'mirostat-resume', 16,
            settings={'temperature': .65, 'mirostat_tau': 3.}), 'accepted')
        _, options = coordinator.submitted
        self.assertTrue(options['resume'])
        self.assertEqual(options['sampling'], dict(policy, temperature=.65, mirostat_tau=3.))
        self.assertEqual(coordinator.store.controller['sampler_state'], state)
        self.assertEqual(coordinator.store.controller['next_draw'], 9)

    def test_text_resume_preserves_feedback_policy_and_pending_decoder_bytes(self):
        policy = saved_policy()
        state = {'protocol': 'mirostat2-state-v1', 'version': 1, 'mu': 6.125}
        token_adapter = _TokenAdapter(dict(schema=4, status='active', constraint=None,
            policy=policy, sampler_state=state, next_draw=9, output_pending_hex='e2'))
        adapter = CowTextAdapter(token_adapter, _Tokenizer(), lambda *_a, **_k: '',
                                 lambda *_a, **_k: ())
        adapter.resume('a', 3, 'mirostat-text-resume', 16, settings={'mirostat_eta': .03})
        self.assertEqual(token_adapter.generate_call[1]['settings'], dict(policy, mirostat_eta=.03))
        self.assertEqual(token_adapter.controller['sampler_state'], state)
        self.assertEqual(token_adapter.controller['output_pending_hex'], 'e2')

    def test_cli_registers_mode_and_parameters_for_every_generation_action(self):
        parser = cow_cli.parser()
        for action, extra in [('ask', ['--text', 'q']), ('answer', []),
                              ('append-user', ['--text', 'q']), ('continue', [])]:
            with self.subTest(action=action):
                args = parser.parse_args([action, '--branch', 'a', *extra,
                    '--temperature', '.8', '--mirostat', '2',
                    '--mirostat-tau', '4', '--mirostat-eta', '.05'])
                settings = cow_cli.sampling_settings(args)
                self.assertEqual(settings['mirostat'], 2)
                self.assertEqual(settings['mirostat_tau'], 4.)
                self.assertEqual(settings['mirostat_eta'], .05)
                if action != 'continue':
                    self.assertEqual(settings['min_p'], 0.)

    def test_cli_partial_continuation_does_not_reset_feedback_or_other_knobs(self):
        parser = cow_cli.parser()
        for flags, expected in [(['--mirostat-tau', '3'], {'mirostat_tau': 3.}),
                                (['--mirostat', '0'], {'mirostat': 0}),
                                (['--temperature', '.6'], {'temperature': .6})]:
            args = parser.parse_args(['continue', '--branch', 'a', *flags])
            self.assertEqual(cow_cli.sampling_settings(args), expected)

    def test_invalid_mode_and_feedback_parameters_fail_before_generation(self):
        for key, values in [('mirostat', (True, 1, 2., '2')),
                            ('mirostat_tau', (0., -1., True, float('inf'), 1e308)),
                            ('mirostat_eta', (-.01, True, float('nan')))]:
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(UnsupportedGreedyRequest):
                        CowAdapter._validate_settings(dict(saved_policy(), **{key: value}))
        parser = cow_cli.parser()
        for flag, value in [('--mirostat-tau', '0'), ('--mirostat-tau', '1e308'),
                            ('--mirostat-tau', 'nan'), ('--mirostat-eta', '-.01')]:
            args = parser.parse_args(['continue', '--branch', 'a', flag, value])
            with self.assertRaises(cow_cli.ClientError):
                cow_cli.sampling_settings(args)

    def test_legacy_cli_defaults_remain_unchanged(self):
        parser = cow_cli.parser()
        self.assertIsNone(cow_cli.sampling_settings(parser.parse_args(
            ['ask', '--branch', 'a', '--text', 'q'])))
        settings = cow_cli.sampling_settings(parser.parse_args(
            ['ask', '--branch', 'a', '--text', 'q', '--temperature', '.7']))
        self.assertEqual(settings, dict(temperature=.7, min_p=.05, top_k=0, top_p=1.))


if __name__ == '__main__':
    unittest.main()
