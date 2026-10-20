"""CPU routing checks for generation-scoped XTC; no GPU claim."""
import unittest

from serve.cow_adapter import CowAdapter, UnsupportedGreedyRequest
from serve.cow_text_adapter import CowTextAdapter
from serve.test_cow_typical_dynatemp_settings import _Coordinator, _TokenAdapter, _Tokenizer
from tools import cow_cli


def saved_policy():
    return dict(temperature=.9, min_p=.02, top_k=0, top_p=1.,
                repetition_penalty=1., xtc_probability=.12, xtc_threshold=.15)


class XtcGenerationRoutingTests(unittest.TestCase):
    def test_adapter_accepts_pair_and_normalizes_partial_new_policy(self):
        self.assertEqual(CowAdapter._validate_settings(saved_policy()), saved_policy())
        partial = CowAdapter._validate_settings({'xtc_probability': .12})
        self.assertEqual(partial['xtc_probability'], .12)
        self.assertEqual(partial['xtc_threshold'], .1)

    def test_schema4_token_resume_preserves_omitted_settings(self):
        policy = saved_policy()
        coordinator = _Coordinator(dict(schema=4, status='active', constraint=None,
                                        policy=policy, sampler_state=None, next_draw=9))
        result = CowAdapter(coordinator).generate(
            'a', [10, 11], 3, 'xtc-resume', 16, settings={'temperature': .8})
        self.assertEqual(result, 'accepted')
        args, options = coordinator.submitted
        self.assertEqual(args[:3], ('a', (), 16))
        self.assertTrue(options['resume'])
        self.assertEqual(options['sampling'], dict(policy, temperature=.8))
        self.assertEqual(coordinator.store.controller['next_draw'], 9)

    def test_schema4_text_resume_preserves_omitted_settings(self):
        policy = saved_policy()
        token_adapter = _TokenAdapter(dict(schema=4, status='active', constraint=None,
            policy=policy, sampler_state=None, next_draw=9, output_pending_hex=''))
        adapter = CowTextAdapter(token_adapter, _Tokenizer(), lambda *_a, **_k: '',
                                 lambda *_a, **_k: ())
        adapter.resume('a', 3, 'xtc-text-resume', 16, settings={'xtc_probability': 0.})
        self.assertEqual(token_adapter.generate_call[1]['settings'],
                         dict(policy, xtc_probability=0.))
        self.assertEqual(token_adapter.controller['next_draw'], 9)

    def test_cli_registers_pair_for_each_generation_action(self):
        parser = cow_cli.parser()
        for action, extra in [('ask', ['--text', 'q']), ('answer', []),
                              ('append-user', ['--text', 'q']), ('continue', [])]:
            with self.subTest(action=action):
                args = parser.parse_args([action, '--branch', 'a', *extra,
                    '--temperature', '.9', '--min-p', '.02',
                    '--xtc-probability', '.12', '--xtc-threshold', '.15'])
                settings = cow_cli.sampling_settings(args)
                self.assertEqual(settings['xtc_probability'], .12)
                self.assertEqual(settings['xtc_threshold'], .15)
                self.assertEqual(settings['temperature'], .9)
        args = parser.parse_args(['continue', '--branch', 'a', '--xtc-probability', '0'])
        self.assertEqual(cow_cli.sampling_settings(args), {'xtc_probability': 0.})

    def test_invalid_xtc_settings_reject_before_generation(self):
        for field in ('xtc_probability', 'xtc_threshold'):
            for value in (-.01, 1.01, True, float('nan'), float('inf')):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(UnsupportedGreedyRequest):
                        CowAdapter._validate_settings({field: value})
        parser = cow_cli.parser()
        for flag in ('--xtc-probability', '--xtc-threshold'):
            for value in ('-.01', '1.01', 'nan', 'inf'):
                with self.subTest(flag=flag, value=value):
                    args = parser.parse_args(['continue', '--branch', 'a', flag, value])
                    with self.assertRaises(cow_cli.ClientError):
                        cow_cli.sampling_settings(args)


if __name__ == '__main__':
    unittest.main()
