import unittest
from pathlib import Path
from unittest.mock import patch

from run_cli_seed_stop_raw_audit import (
    InterleavedCase, build_raw_command, expected_operator_schedule,
    expected_plain_operator_schedule, make_child_environment,
    validate_case_receipt, validate_schedule,
)
from build_cli_seed_stop_raw_oracle import _normalize_step, _rewrite_compile, _rewrite_link


class InterleavedRawAuditContractTests(unittest.TestCase):
    def setUp(self):
        self.case = InterleavedCase(18, (5, 11), 18, 1)
        self.expected = [
            {"operator": "batched-prefill", "purpose": "root", "position": 0, "count": 18},
            {"operator": "normal-verifier-decode", "purpose": "segment-1", "position": 18, "count": 5},
            {"operator": "batched-prefill", "purpose": "appended-user", "position": 23, "count": 18},
            {"operator": "normal-verifier-decode", "purpose": "segment-2", "position": 41, "count": 11},
        ]

    def test_exact_interleaved_schedule_and_positions(self):
        self.assertEqual(expected_operator_schedule(self.case), self.expected)

    def test_plain_schedule_binds_every_absolute_verifier_position(self):
        self.assertEqual(expected_plain_operator_schedule(18, (1, 16, 10)), [
            {"operator": "batched-prefill", "purpose": "root", "position": 0, "count": 18},
            {"operator": "normal-verifier-decode", "purpose": "segment-1", "position": 18, "count": 1},
            {"operator": "normal-verifier-decode", "purpose": "segment-2", "position": 19, "count": 16},
            {"operator": "normal-verifier-decode", "purpose": "segment-3", "position": 35, "count": 10},
        ])

    def test_plain_schedule_rejects_invalid_types_and_empty_segments(self):
        for prompt, segments in ((True, (1,)), (18, ()), (18, (1, False)), (0, (1,))):
            with self.subTest(prompt=prompt, segments=segments), self.assertRaises(ValueError):
                expected_plain_operator_schedule(prompt, segments)

    def test_interleaved_schedule_supports_append_after_later_segment(self):
        case = InterleavedCase(18, (1, 16, 10), 18, 2)
        self.assertEqual(expected_operator_schedule(case), [
            {"operator": "batched-prefill", "purpose": "root", "position": 0, "count": 18},
            {"operator": "normal-verifier-decode", "purpose": "segment-1", "position": 18, "count": 1},
            {"operator": "normal-verifier-decode", "purpose": "segment-2", "position": 19, "count": 16},
            {"operator": "batched-prefill", "purpose": "appended-user", "position": 35, "count": 18},
            {"operator": "normal-verifier-decode", "purpose": "segment-3", "position": 53, "count": 10},
        ])

    def test_plain_child_environment_clears_inherited_interleave_controls(self):
        profile = {"execution": {"environment": {
            "STRATA_PREFILL_AFTER_SEGMENT": "2",
            "STRATA_PREFILL_APPEND_TOKENS": "4,5",
        }}}
        with patch.dict("os.environ", {
            "STRATA_PREFILL_AFTER_SEGMENT": "1",
            "STRATA_PREFILL_APPEND_TOKENS": "7,8",
        }, clear=False):
            env = make_child_environment(profile, prompt_count=18, segments=(1, 16),
                                         forced_ids=(2, 3), append_ids=())
        self.assertNotIn("STRATA_PREFILL_AFTER_SEGMENT", env)
        self.assertNotIn("STRATA_PREFILL_APPEND_TOKENS", env)

    def test_schedule_rejects_verifier_before_appended_prefill(self):
        wrong = [self.expected[0], self.expected[1], self.expected[3], self.expected[2]]
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_schedule(wrong, self.expected)

    def test_schedule_rejects_wrong_absolute_position(self):
        wrong = [dict(row) for row in self.expected]
        wrong[3]["position"] += 1
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_schedule(wrong, self.expected)

    def test_schedule_rejects_missing_or_extra_operator(self):
        for wrong in (self.expected[:-1], self.expected + [dict(self.expected[-1])]):
            with self.subTest(length=len(wrong)), self.assertRaisesRegex(ValueError, "schedule"):
                validate_schedule(wrong, self.expected)

    def test_receipt_binds_both_prefills_and_all_verifier_work(self):
        receipt = {"status": "ok", "exit_code": 0, "pid_absent": True,
                   "generated": [1] * 16, "normal_forward_tokens": 16,
                   "normal_decode_windows": 16, "initial_prefill_tokens": 18,
                   "appended_prefill_tokens": 18, "prefill_tokens": 36,
                   "consumed": 52, "operator_schedule": self.expected}
        self.assertIs(validate_case_receipt(receipt, self.case, self.expected,
                                            receipt["generated"]), receipt)

    def test_receipt_rejects_missing_forward_work_and_nonterminal_child(self):
        receipt = {"status": "ok", "exit_code": 0, "pid_absent": True,
                   "generated": [1] * 16, "normal_forward_tokens": 16,
                   "normal_decode_windows": 16, "initial_prefill_tokens": 18,
                   "appended_prefill_tokens": 18, "prefill_tokens": 36,
                   "consumed": 52, "operator_schedule": self.expected}
        for changes in ({"normal_forward_tokens": 15}, {"pid_absent": False},
                        {"appended_prefill_tokens": 17}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_case_receipt({**receipt, **changes}, self.case, self.expected,
                                      receipt["generated"])

    def test_rejects_bool_as_count_or_boundary(self):
        for case in (InterleavedCase(True, (5, 11), 18, 1),
                     InterleavedCase(18, (5, 11), 18, True)):
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "exact integers"):
                expected_operator_schedule(case)

    def test_compile_and_link_rewriters_target_one_new_output_each(self):
        compile_argv = ["c++", "-c", "/old/fresh_prefill_segmented_forced.cpp", "-o", "/old/fresh_prefill_segmented_forced.o"]
        self.assertEqual(_rewrite_compile(compile_argv, "/new/fresh_prefill_segmented_forced.cpp",
                                          "/new/fresh_prefill_segmented_forced.o"),
                         ["c++", "-c", "/new/fresh_prefill_segmented_forced.cpp", "-o",
                          "/new/fresh_prefill_segmented_forced.o"])
        link_argv = ["c++", "/old/fresh_prefill_segmented_forced.o", "-o", "/old/audit"]
        self.assertEqual(_rewrite_link(link_argv, "/new/fresh_prefill_segmented_forced.o",
                                       "/new/audit"),
                         ["c++", "/new/fresh_prefill_segmented_forced.o", "-o", "/new/audit"])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _rewrite_link(link_argv + ["/also/fresh_prefill_segmented_forced.o"],
                          "/new/fresh_prefill_segmented_forced.o", "/new/audit")

    def test_recipe_comparison_ignores_only_source_object_and_output_paths(self):
        base = {"stage": "compile", "cwd": "/pinned/build",
                "argv": ["c++", "-c", "/old/a.cpp", "-o", "/old/a.o"]}
        relocated = {"stage": "compile", "cwd": "/pinned/build",
                     "argv": ["c++", "-c", "/new/b.cpp", "-o", "/new/b.o"]}
        self.assertEqual(_normalize_step(base), _normalize_step(relocated))
        wrong_flags = {**relocated, "argv": ["c++", "-O0", "-c", "/new/b.cpp", "-o", "/new/b.o"]}
        self.assertNotEqual(_normalize_step(base), _normalize_step(wrong_flags))

    def test_raw_command_rebinds_only_binary_decode_count_and_token_file(self):
        profile = {"execution": {"oracle_command": [
            "/pinned/old", "--max-new", "1", "--prompt-cache", "0",
            "--conversation-cache-mib", "0", "--seed", "0"]}}
        self.assertEqual(build_raw_command(profile, Path("/audit/raw"),
                                           Path("/run/prompt.tokens"), 16),
                         ["/audit/raw", "--max-new", "16", "--prompt-cache", "0",
                          "--conversation-cache-mib", "0", "--seed", "0",
                          "--tokens-file", "/run/prompt.tokens"])
        bad = {"execution": {"oracle_command": profile["execution"]["oracle_command"] +
                              ["--load-state", "checkpoint"]}}
        with self.assertRaisesRegex(ValueError, "checkpoint import"):
            build_raw_command(bad, Path("/audit/raw"), Path("/run/input"), 16)

    def test_interleaved_receipt_requires_exact_total_positions_and_forward_count(self):
        receipt = {"status": "ok", "exit_code": 0, "pid_absent": True,
                   "generated": [1] * 16, "normal_forward_tokens": 16,
                   "normal_decode_windows": 16, "initial_prefill_tokens": 18,
                   "appended_prefill_tokens": 18, "prefill_tokens": 36,
                   "consumed": 52, "operator_schedule": self.expected}
        bad = {**receipt, "consumed": 51}
        with self.assertRaisesRegex(ValueError, "work receipt"):
            validate_case_receipt(bad, self.case, self.expected, receipt["generated"])


if __name__ == "__main__":
    unittest.main()
