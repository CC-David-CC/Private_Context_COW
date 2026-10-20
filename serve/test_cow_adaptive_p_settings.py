"""CPU routing for generation-scoped Adaptive-P; no GPU claims."""
import unittest

from serve.cow_adapter import CowAdapter, UnsupportedGreedyRequest
from serve.cow_text_adapter import CowTextAdapter
from serve.test_cow_typical_dynatemp_settings import _Coordinator, _TokenAdapter, _Tokenizer
from tools import cow_cli


def saved_policy():
    return dict(temperature=.7, min_p=.05, top_k=0, top_p=1.,
                repetition_penalty=1., adaptive_target=.55, adaptive_decay=.9)


class AdaptiveGenerationRoutingTests(unittest.TestCase):
    def test_adapter_accepts_terminal_policy_and_rejects_other_terminal_samplers(self):
        policy = saved_policy()
        self.assertEqual(CowAdapter._validate_settings(policy), policy)
        partial = CowAdapter._validate_settings({'adaptive_target': .55})
        self.assertEqual(partial['adaptive_decay'], .9)
        combined = dict(policy, top_k=60, top_p=.95, typical_p=.95,
                        dynatemp_range=.2, dry_multiplier=.8)
        self.assertEqual(CowAdapter._validate_settings(combined)['adaptive_target'], .55)
        for override in ({'temperature': 0}, {'mirostat': 2, 'min_p': 0.},
                         {'xtc_probability': .12, 'xtc_threshold': .15}):
            with self.subTest(override=override), self.assertRaises(UnsupportedGreedyRequest):
                CowAdapter._validate_settings(dict(policy, **override))

    def test_token_partial_resume_preserves_ema_and_inherits_other_settings(self):
        policy = saved_policy()
        state = dict(version='adaptive-p-state-v1', weighted_sum=5.25, total_weight=10.)
        coordinator = _Coordinator(dict(schema=4, status='active', constraint=None,
            policy=policy, sampler_state=state, next_draw=9))
        self.assertEqual(CowAdapter(coordinator).generate('a', [10, 11], 3,
            'adaptive-resume', 16, settings={'adaptive_target': .6}), 'accepted')
        _, options = coordinator.submitted
        self.assertTrue(options['resume'])
        self.assertEqual(options['sampling'], dict(policy, adaptive_target=.6))
        self.assertEqual(coordinator.store.controller['sampler_state'], state)
        self.assertEqual(coordinator.store.controller['next_draw'], 9)

    def test_text_resume_preserves_ema_and_output_pending_bytes(self):
        policy = saved_policy()
        state = dict(version='adaptive-p-state-v1', weighted_sum=5.25, total_weight=10.)
        token = _TokenAdapter(dict(schema=4, status='active', constraint=None,
            policy=policy, sampler_state=state, next_draw=9, output_pending_hex='e2'))
        adapter = CowTextAdapter(token, _Tokenizer(), lambda *_a, **_k: '', lambda *_a, **_k: ())
        adapter.resume('a', 3, 'adaptive-text', 16, settings={'adaptive_decay': .8})
        self.assertEqual(token.generate_call[1]['settings'], dict(policy, adaptive_decay=.8))
        self.assertEqual(token.controller['sampler_state'], state)
        self.assertEqual(token.controller['output_pending_hex'], 'e2')

    def test_cli_knobs_apply_to_all_generation_actions_and_continue_sends_only_changes(self):
        parser = cow_cli.parser()
        for action, extra in [('ask', ['--text', 'q']), ('answer', []),
                              ('append-user', ['--text', 'q']), ('continue', [])]:
            with self.subTest(action=action):
                args = parser.parse_args([action, '--branch', 'a', *extra,
                    '--adaptive-target', '.55', '--adaptive-decay', '.9'])
                settings = cow_cli.sampling_settings(args)
                self.assertEqual(settings['adaptive_target'], .55)
                self.assertEqual(settings['adaptive_decay'], .9)
                if action != 'continue':
                    self.assertEqual(settings['temperature'], .7)
                    self.assertEqual(settings['min_p'], .05)
        args = parser.parse_args(['continue', '--branch', 'a', '--adaptive-target', '-1'])
        self.assertEqual(cow_cli.sampling_settings(args), {'adaptive_target': -1.})

    def test_invalid_target_decay_fail_before_generation(self):
        for key, values in [('adaptive_target', (1.01, True, float('inf'))),
                            ('adaptive_decay', (-.01, 1., True, float('nan')))]:
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(UnsupportedGreedyRequest):
                    CowAdapter._validate_settings(dict(saved_policy(), **{key:value}))
        parser = cow_cli.parser()
        for flag, value in [('--adaptive-target', '1.01'), ('--adaptive-target', 'nan'),
                            ('--adaptive-decay', '-.01'), ('--adaptive-decay', '1')]:
            args = parser.parse_args(['continue', '--branch', 'a', flag, value])
            with self.assertRaises(cow_cli.ClientError):
                cow_cli.sampling_settings(args)


if __name__ == '__main__':
    unittest.main()
