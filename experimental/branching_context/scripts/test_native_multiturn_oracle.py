"""CPU contract tests for the independent fresh native multi-turn oracle."""
import unittest
import json
from pathlib import Path

import build_fast_prefill_probe as base
from build_native_multiturn_oracle import multiturn_source, validate_stopped_turn
from run_native_multiturn_oracle import compose, profile_command, runtime_environment


class StoppedTurnGuardTests(unittest.TestCase):
    def test_full_fixture_keeps_root_and_first_user_as_distinct_prefills(self):
        fixture = {
            'prefix': [1, 2, 3], 'first_suffix': [4, 5],
            'first_limit': 2048, 'first_outputs': [8, 9],
            'new_user_suffix': [198, 7], 'next_limit': 128,
        }
        first, lengths, suffix, expected, limit = compose(fixture)
        self.assertEqual(first, [1, 2, 3, 4, 5])
        self.assertEqual(lengths, [3, 2])
        self.assertEqual(suffix, [198, 7])
        self.assertEqual(expected, [8, 9])
        self.assertEqual(limit, 2048)

    def test_rejection_fixture_is_small_and_has_no_expected_generated_output(self):
        fixture = {
            'prefix': list(range(200)), 'new_user_suffix': [198],
            'next_limit': 128,
        }
        first, lengths, suffix, expected, limit = compose(fixture, rejection=True)
        self.assertEqual((len(first), lengths, suffix, expected, limit),
                         (128, [128], [198], None, 1))

    def test_dump_stem_and_pinned_environment_are_exact(self):
        profile = {'execution': {'eos_tokens': [248044, 248046],
                                 'environment': {'CUDA_VISIBLE_DEVICES': '0', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '0'}}}
        env = runtime_environment(profile,
            {'PATH': '/bin', 'STRATA_DUMP_FIRST_LOGITS': 'stale', 'CUDA_VISIBLE_DEVICES': '9',
             'NVIDIA_VISIBLE_DEVICES': '9', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '1'},
            '/run/oracle', '/run/suffix.ids', [65579, 8258], '/run/expected.ids')
        self.assertEqual(env['STRATA_DUMP_FIRST_LOGITS'], '/run/oracle')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(env['GGML_CUDA_ENABLE_UNIFIED_MEMORY'], '0')
        self.assertEqual(env['STRATA_PREFILL_REQUESTS'], '65579,8258')
        self.assertNotIn('NVIDIA_VISIBLE_DEVICES', env)
        self.assertEqual(env['STRATA_EXPECT_FIRST_FILE'], '/run/expected.ids')
        self.assertEqual('/run/oracle.multiturn.json', env['STRATA_DUMP_FIRST_LOGITS'] + '.multiturn.json')

    def test_first_frontier_uses_bare_output_stem(self):
        self.assertIn('std::ofstream file(output, std::ios::binary);', base.FRESH)
        runner = Path(__file__).with_name('run_native_multiturn_oracle.py').read_text()
        self.assertIn("output_stem.stat().st_size != VOCAB*4", runner)

    def test_profile_without_token_argument_gets_file_prompt(self):
        profile = {'execution': {'oracle_command': ['/old/bin', '--max-new', '0', '--gpu', '0']}}
        command = profile_command(profile, '/new/bin', '/run/prompt.ids', 1)
        self.assertEqual(command, ['/new/bin', '--max-new', '1', '--gpu', '0',
                                   '--tokens-file', '/run/prompt.ids'])

    def test_loaded_pinned_profile_command_shape(self):
        pinned = Path('/home/dflanag3/branching-g3/request-stop-20261014/text-next-user-q8-v1/profile.json')
        if not pinned.exists():
            self.skipTest('llm-60 pinned profile is not available on this CPU host')
        profile = json.loads(pinned.read_text())
        command = profile_command(profile, '/new/bin', '/run/prompt.ids', 1)
        self.assertEqual(command[0], '/new/bin')
        self.assertEqual(command[command.index('--max-new')+1], '1')
        self.assertEqual(command.count('--tokens-file'), 1)
        self.assertNotIn('--tokens', command)
        self.assertEqual(command[command.index('--tokens-file')+1], '/run/prompt.ids')

    def test_profile_rejects_duplicate_or_missing_option_values(self):
        base = ['/old/bin', '--max-new', '0', '--gpu', '0']
        for malformed in (base+['--tokens-file', '--gpu', '0'],
                          ['/old/bin', '--max-new'],
                          base+['--tokens', '1', '--tokens-file', 'other']):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    profile_command({'execution': {'oracle_command': malformed}},
                                    '/new/bin', '/run/prompt.ids', 1)

    def test_accepts_only_recorded_actual_eos(self):
        self.assertTrue(validate_stopped_turn([10, 248046], {248044, 248046}, 8))

    def test_rejects_max_bound_before_suffix_prefill(self):
        with self.assertRaisesRegex(ValueError, 'did not stop on EOS'):
            validate_stopped_turn([10, 11, 12], {248044, 248046}, 3)

    def test_rejects_non_eos_terminal_token(self):
        with self.assertRaisesRegex(ValueError, 'did not stop on EOS'):
            validate_stopped_turn([10, 11], {248044, 248046}, 8)

    def test_rejects_empty_first_turn(self):
        with self.assertRaisesRegex(ValueError, 'no first-turn tokens'):
            validate_stopped_turn([], {248044, 248046}, 8)

    def test_rejects_invalid_eos_contract(self):
        with self.assertRaisesRegex(ValueError, 'EOS set is empty'):
            validate_stopped_turn([10], set(), 8)

    def test_generated_oracle_uses_fresh_prefill_decode_operators(self):
        source = multiturn_source(base.FRESH)
        self.assertIn('STRATA_PREFILL_REQUESTS', source)
        self.assertIn('request_counts', source)
        self.assertIn('request_frontiers', source)
        self.assertIn('batched.run(suffix_wide.data()', source)
        self.assertIn('decode.run(1, &token, position, pool', source)
        self.assertIn('first turn did not stop on EOS; suffix prefill refused', source)
        self.assertIn('EOS_GUARD_REJECT first_generated_tokens=', source)
        self.assertIn('prefill_work_since_first_prompt=', source)
        self.assertIn('batched.stats().tokens-first_prefill_work_tokens', source)
        self.assertIn('STRATA_EXPECT_FIRST_FILE', source)
        self.assertIn('first-turn tokens differ from fixture; suffix prefill refused', source)
        self.assertIn('STRATA_PREFILL_REQUESTS', source)
        self.assertIn('output + ".second.frontier.f32"', source)
        self.assertIn('output + ".second.final.f32"', source)
        self.assertIn('first_chunk_count', source)
        self.assertNotIn('BranchSession session', source)
        self.assertNotIn('ConversationCache ram_cache', source)
        self.assertNotIn('session.restore', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
