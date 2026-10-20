"""CPU contract tests for the resident-code-root question campaign harness."""
import importlib.util
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'run_resident_questions.py'
_SPEC = importlib.util.spec_from_file_location('run_resident_questions_under_test', SCRIPT) if SCRIPT.is_file() else None
RUNNER = importlib.util.module_from_spec(_SPEC) if _SPEC is not None else None
if _SPEC is not None and _SPEC.loader is not None:
    _SPEC.loader.exec_module(RUNNER)
else:
    RUNNER = None


def runner():
    if RUNNER is None:
        raise AssertionError('resident-question harness is not implemented yet (baseline RED)')
    return RUNNER


class ResidentQuestionValidationTests(unittest.TestCase):
    def test_expected_controller_preserves_canonical_json_eos_list(self):
        r = runner()

        class Tokenizer:
            tokens = ('a', 'b', '<eos>')

            def token_bytes(self, token):
                return (b'a', b'b', b'')[token]

        profile = {'execution': {'eos_tokens': [2]}}
        state = r._expected_controller('resident-question-test', (1,), (0, 2),
                                       Tokenizer(), profile)
        self.assertIsInstance(state['policy']['eos_tokens'], list)
        self.assertEqual(state['policy']['eos_tokens'], [2])

    def test_exact_question_work_accepts_zero_old_prefix_work(self):
        r = runner()
        before = {'forward_tokens': 7231, 'prefill_tokens': 7231,
                  'loaded_file_bytes': 0, 'integrity_mismatches': 0}
        after = {'forward_tokens': 7231 + 47 + 19,
                 'prefill_tokens': 7231 + 47,
                 'loaded_file_bytes': 0, 'integrity_mismatches': 0}
        result = r.validate_question_work(before, after, question_token_ids=tuple(range(47)),
                                          output_token_ids=tuple(range(19)))
        self.assertEqual(result['question_tokens'], 47)
        self.assertEqual(result['output_tokens'], 19)
        self.assertEqual(result['old_prefix_forward_tokens'], 0)
        self.assertEqual(result['loaded_file_bytes'], 0)

    def test_rejects_disk_load_even_when_question_forward_count_matches(self):
        r = runner()
        before = {'forward_tokens': 7231, 'prefill_tokens': 7231,
                  'loaded_file_bytes': 0, 'integrity_mismatches': 0}
        after = {'forward_tokens': 7231 + 47 + 19,
                 'prefill_tokens': 7231 + 47,
                 'loaded_file_bytes': 332_013_048, 'integrity_mismatches': 0}
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_question_work(before, after, question_token_ids=tuple(range(47)),
                                     output_token_ids=tuple(range(19)))

    def test_rejects_any_old_prefix_recomputation(self):
        r = runner()
        before = {'forward_tokens': 7231, 'prefill_tokens': 7231,
                  'loaded_file_bytes': 0, 'integrity_mismatches': 0}
        after = {'forward_tokens': 2 * 7231 + 47 + 19,
                 'prefill_tokens': 2 * 7231 + 47,
                 'loaded_file_bytes': 0, 'integrity_mismatches': 0}
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_question_work(before, after, question_token_ids=tuple(range(47)),
                                     output_token_ids=tuple(range(19)))

    def test_rejects_wrong_source_and_accumulated_question_history(self):
        r = runner()
        prefix = tuple(range(11))
        question = (21, 22, 23)
        answer = (31, 32)
        r.validate_question_lineage(source_snapshot='P', expected_root_snapshot='P',
                                    root_token_ids=prefix, branch_source_token_ids=prefix,
                                    branch_lineage_ids=prefix + question + answer,
                                    question_token_ids=question, output_token_ids=answer)
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_question_lineage(source_snapshot='A1', expected_root_snapshot='P',
                                        root_token_ids=prefix, branch_source_token_ids=prefix + question + answer,
                                        branch_lineage_ids=prefix + question + answer + (40, 41),
                                        question_token_ids=(40,), output_token_ids=(41,))
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_question_lineage(source_snapshot='P', expected_root_snapshot='P',
                                        root_token_ids=prefix, branch_source_token_ids=prefix,
                                        branch_lineage_ids=prefix + (8, 9, 10) + question + answer,
                                        question_token_ids=question, output_token_ids=answer)

    def test_requires_captured_root_handle_and_restore_for_later_questions(self):
        r = runner()
        capture = {'command': 'capture', 'response':
                   {'status': 'ok', 'handle': 17, 'consumed': 7231}}
        traces = [
            [{'command': 'decode 9', 'response': {'status': 'ok'}}],
            [{'command': 'restore 17', 'response': {'status': 'ok', 'consumed': 7231}},
             {'command': 'decode 9', 'response': {'status': 'ok'}}],
            [{'command': 'restore 17', 'response': {'status': 'ok', 'consumed': 7231}},
             {'command': 'decode 9', 'response': {'status': 'ok'}}],
        ]
        r.validate_resident_prefix(root_handle=17, root_tokens=7231,
                                   root_capture=capture, question_traces=traces)
        bad = [traces[0], [{'command': 'restore 99', 'response': {'status': 'ok', 'consumed': 7231}}], traces[2]]
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_resident_prefix(root_handle=17, root_tokens=7231,
                                       root_capture=capture, question_traces=bad)

    def test_rejects_root_handle_drop_before_last_question(self):
        r = runner()
        capture = {'command': 'capture', 'response':
                   {'status': 'ok', 'handle': 17, 'consumed': 7231}}
        traces = [
            [{'command': 'decode 9', 'response': {'status': 'ok'}}],
            [{'command': 'drop 17', 'response': {'status': 'ok'}},
             {'command': 'restore 17', 'response': {'status': 'ok', 'consumed': 7231}}],
            [{'command': 'restore 17', 'response': {'status': 'ok', 'consumed': 7231}}],
        ]
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_resident_prefix(root_handle=17, root_tokens=7231,
                                       root_capture=capture, question_traces=traces)

    def test_records_actual_eos_and_rejects_false_5000_token_claims(self):
        r = runner()
        short_stopped = r.validate_answer_completion(output_token_ids=(5, 6, 2),
            max_tokens=5000, controller_status='stopped', eos_tokens=(2,))
        self.assertEqual(short_stopped['actual_output_tokens'], 3)
        self.assertTrue(short_stopped['ended_by_eos'])
        full_active = r.validate_answer_completion(output_token_ids=tuple(range(5000)),
            max_tokens=5000, controller_status='active', eos_tokens=(5001,))
        self.assertTrue(full_active['reached_max_tokens'])
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_answer_completion(output_token_ids=(5, 6), max_tokens=5000,
                controller_status='active', eos_tokens=(2,))
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_answer_completion(output_token_ids=(5, 6), max_tokens=5000,
                controller_status='stopped', eos_tokens=(2,))
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_answer_completion(output_token_ids=tuple(range(5001)), max_tokens=5000,
                controller_status='active', eos_tokens=(2,))

    def test_root_heads_and_payload_fingerprint_remain_unchanged(self):
        r = runner()
        r.validate_root_stability(root_tip_before='P', root_tip_after='P',
                                  sibling_tips_before={'Q2':'P','Q3':'P'},
                                  sibling_tips_after={'Q2':'P','Q3':'P'},
                                  root_payload_sha_before='a'*64,
                                  root_payload_sha_after='a'*64)
        with self.assertRaises(r.ResidentQuestionsError):
            r.validate_root_stability(root_tip_before='P', root_tip_after='Q1',
                                      sibling_tips_before={'Q2':'P','Q3':'P'},
                                      sibling_tips_after={'Q2':'P','Q3':'P'},
                                      root_payload_sha_before='a'*64,
                                      root_payload_sha_after='a'*64)

    def test_answer_decoder_preserves_pending_utf8_at_ceiling_and_flushes_at_eos(self):
        r = runner()

        class TokenizerFixture:
            pieces = {1: b'\x82', 2: b'\xac', 3: b'!', 99: b''}
            def token_bytes(self, token):
                return self.pieces[token]

        tokenizer = TokenizerFixture()
        source = {'output_pending_hex': 'e2'}
        self.assertEqual(r._decoded_answer((1, 2, 3), source, tokenizer, eos_tokens=(99,)), '€!')
        self.assertEqual(r._decoded_answer((1,), source, tokenizer, eos_tokens=(99,)), '')
        self.assertEqual(r._decoded_answer((1, 99), source, tokenizer, eos_tokens=(99,)), '�')


if __name__ == '__main__':
    unittest.main()
