"""Diagnostic sweeps keep selective escalation explicit and bounded."""
import unittest

from run_thinking_cli_probe import natural_end_outcome, parse_probe_args


class ThinkingProbeArgumentsTests(unittest.TestCase):
    required = [part for name in ('runtime', 'branchwrite', 'profile', 'dependency-root',
        'vision-config', 'image', 'output', 'expected-image-sha256') for part in ('--' + name, 'fixture')]

    def test_existing_campaign_defaults_are_preserved(self):
        args = parse_probe_args(self.required)
        self.assertEqual(args.budgets, [0, 128])
        self.assertEqual(args.questions, ['expanded', 'short'])
        self.assertFalse(args.greedy_only)
        self.assertFalse(args.until_natural_end)

    def test_only_failed_question_can_escalate_greedy_to_natural_closure(self):
        args = parse_probe_args(self.required + ['--questions', 'short', '--greedy-only',
            '--budgets', '256', '512', '1024', '--until-natural-end'])
        self.assertEqual(args.questions, ['short'])
        self.assertEqual(args.budgets, [256, 512, 1024])
        self.assertTrue(args.greedy_only and args.until_natural_end)

    def test_duplicate_or_nonincreasing_budgets_are_not_retried_with_same_ids(self):
        for values in (['256', '256'], ['512', '256'], ['32769'], ['-1']):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                parse_probe_args(self.required + ['--budgets', *values])

    def test_natural_end_mode_requires_single_sampler_and_unique_questions(self):
        for extra in (['--until-natural-end'], ['--questions', 'short', 'short']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_probe_args(self.required + extra)

    def test_natural_wrong_answer_stops_but_forced_correct_answer_does_not(self):
        def arm(reason, quality, final):
            return dict(budget=256, first=dict(controller=dict(status='active')),
                greedy=dict(controller=dict(status='stopped'), image_quality=quality, final=final,
                    progress=dict(close_reason=reason, reasoning_tokens=200)))
        self.assertIsNone(natural_end_outcome(arm('budget', 'PASS', 'LEFT')))
        outcome = natural_end_outcome(arm('natural', 'FAIL', 'NOT_VISIBLE'))
        self.assertEqual(outcome['quality'], 'FAIL')
        self.assertEqual(outcome['close_reason'], 'natural')


if __name__ == '__main__':
    unittest.main()
