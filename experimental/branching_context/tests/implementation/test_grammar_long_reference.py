"""CPU-only validators for the independent long-grammar native reference."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "grammar_long_reference.py"
spec = importlib.util.spec_from_file_location("grammar_long_reference", SCRIPT)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class FixtureValidationTests(unittest.TestCase):
    def setUp(self):
        root = [101, 202, 303]
        self.case = {"root_token_ids": root, "branches": {
            "A": {"suffix_token_ids": [11, 12], "full_token_ids": root + [11, 12]},
            "B": {"suffix_token_ids": [21], "full_token_ids": root + [21]},
        }}

    def test_selected_branch_is_exact_root_plus_suffix(self):
        self.assertEqual(reference.validate_fixture(self.case, "A"), ((101, 202, 303), (11, 12)))
        self.assertEqual(reference.validate_fixture(self.case, "B"), ((101, 202, 303), (21,)))

    def test_rejects_fixture_full_history_mismatch(self):
        bad = {"root_token_ids": [101], "branches": {
            "A": {"suffix_token_ids": [11], "full_token_ids": [11, 101]}}}
        with self.assertRaisesRegex(ValueError, "must equal"):
            reference.validate_fixture(bad, "A")

    def test_rejects_boolean_negative_and_noninteger_fixture_ids(self):
        for invalid in (True, -1, "12"):
            bad = {"root_token_ids": [101], "branches": {
                "A": {"suffix_token_ids": [invalid], "full_token_ids": [101, invalid]}}}
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                reference.validate_fixture(bad, "A")

    def test_rejects_boolean_alias_in_full_token_ids(self):
        bad = {"root_token_ids": [1], "branches": {
            "A": {"suffix_token_ids": [2], "full_token_ids": [True, 2]}}}
        with self.assertRaisesRegex(ValueError, "full_token_ids"):
            reference.validate_fixture(bad, "A")

    def test_result_exposes_candidate_stopped_and_final_sample_fields(self):
        class Session:
            def is_stopped(self):
                return True

            def is_complete(self):
                return False

        final = {"path": "generated-8.f32", "sha256": "abc", "bytes": 12,
                 "consumed": 7751}
        record = {"samples": {"generated-8": final}, "final_sample_key": "generated-8"}
        reference.record_grammar_outcome(record, Session(), [3, 4])
        self.assertIs(record["stopped"], True)
        self.assertIs(record["grammar_stopped"], True)
        self.assertIs(record["grammar_complete"], False)
        self.assertEqual(record["output_count"], 2)
        self.assertIs(record["samples"]["final"], final)

    def test_result_rejects_nonboolean_stopped_state(self):
        class Session:
            def is_stopped(self):
                return 1

            def is_complete(self):
                return False

        with self.assertRaisesRegex(ValueError, "must be boolean"):
            reference.record_grammar_outcome({"samples": {}}, Session(), [])

    def test_branch_stream_and_output_cap_are_bound(self):
        reference.validate_request("A", "review-A", 128)
        for branch, stream, limit in (("A", "review-B", 8), ("C", "review-C", 8),
                                      ("B", "review-B", 0), ("B", "review-B", 129),
                                      ("B", "review-B", True)):
            with self.subTest(branch=branch, stream=stream, limit=limit), self.assertRaises(ValueError):
                reference.validate_request(branch, stream, limit)


class NativeReceiptTests(unittest.TestCase):
    def test_prefill_receipt_validates_two_absolute_chunks(self):
        response = {"status": "ok", "operator": "batched-prefill", "position": 8192,
                    "count": 9000, "consumed": 17192,
                    "chunks": [[8192, 8192], [16384, 808]]}
        self.assertEqual(reference.validate_feed_receipt(response, position=8192,
            count=9000, chunk_size=8192), response)

    def test_rejects_wrong_prefill_position_count_or_schedule(self):
        valid = {"status": "ok", "operator": "batched-prefill", "position": 10,
                 "count": 5, "consumed": 15, "chunks": [[10, 5]]}
        for field, value in (("position", True), ("position", 9), ("count", 6),
                             ("consumed", 14), ("chunks", [[10, 4], [14, 1]]),
                             ("operator", "normal-verifier-decode")):
            bad = dict(valid, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                reference.validate_feed_receipt(bad, position=10, count=5, chunk_size=8192)

    def test_decode_receipt_is_single_token_at_absolute_position(self):
        response = {"status": "ok", "operator": "normal-verifier-decode",
                    "position": 7750, "count": 1, "consumed": 7751,
                    "chunks": [[7750, 1]]}
        self.assertEqual(reference.validate_decode_receipt(response, position=7750), response)
        with self.assertRaisesRegex(ValueError, "decode receipt"):
            reference.validate_decode_receipt(dict(response, position=7749), position=7750)

    def test_native_command_guard_excludes_all_checkpoint_and_mask_commands(self):
        reference.validate_command_inventory([{"command": "feed"}, {"command": "decode"},
                                               {"command": "logits"}, {"command": "stats"},
                                               {"command": "quit"}])
        for command in ("select_mask", "capture", "restore", "load", "park", "activate", "write"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                reference.validate_command_inventory([{"command": command}])

    def test_logits_path_matches_actual_unquoted_native_grammar(self):
        self.assertEqual(reference.logits_command(Path("/tmp/frontier-1.f32")),
                         "logits /tmp/frontier-1.f32")
        with self.assertRaisesRegex(ValueError, "unquoted token"):
            reference.logits_command(Path("/tmp/frontier with-space.f32"))

    def test_counter_delta_requires_exact_forward_prefill_and_decode_work(self):
        before = {"forward_tokens": 0, "prefill_tokens": 0, "decode_timed_tokens": 0}
        after = {"forward_tokens": 7751, "prefill_tokens": 7743, "decode_timed_tokens": 8,
                 "consumed": 7751}
        self.assertEqual(reference.validate_counter_delta(before, after,
            forward=7751, prefill=7743, decode=8),
            {"forward_tokens": 7751, "prefill_tokens": 7743, "decode_timed_tokens": 8})
        with self.assertRaisesRegex(ValueError, "forward_tokens"):
            reference.validate_counter_delta(before, dict(after, forward_tokens=7750),
                forward=7751, prefill=7743, decode=8)


class NativeProcessLifecycleTests(unittest.TestCase):
    def test_native_child_stays_in_reference_worker_process_group(self):
        with tempfile.TemporaryDirectory(prefix="grammar-native-group-") as temp:
            output = Path(temp)
            code = ("import json,sys; "
                    "print(json.dumps({'status':'ready','vocab':8,'capacity':32,"
                    "'grammar_selection_protocol':'finite-mask-argmax-v1'}),flush=True); "
                    "line=sys.stdin.readline(); "
                    "print(json.dumps({'status':'closed'}),flush=True) if line.strip()=='quit' else None")
            pipe = reference.NativePipe([sys.executable, "-u", "-c", code],
                                        os.environ.copy(), output, timeout=5)
            try:
                self.assertIsNone(pipe.startup_error)
                self.assertEqual(os.getpgid(pipe.process.pid), os.getpgrp(),
                                 "native must remain in the reference worker's owned process group")
            finally:
                pipe.close()

    def test_forced_cleanup_targets_only_owned_native_process(self):
        with tempfile.TemporaryDirectory(prefix="grammar-native-cleanup-") as temp:
            output = Path(temp)
            code = ("import json,sys,time; "
                    "print(json.dumps({'status':'ready','vocab':8,'capacity':32,"
                    "'grammar_selection_protocol':'finite-mask-argmax-v1'}),flush=True); "
                    "sys.stdin.readline(); time.sleep(60)")
            pipe = reference.NativePipe([sys.executable, "-u", "-c", code],
                                        os.environ.copy(), output, timeout=5)
            sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            try:
                self.assertEqual(os.getpgid(pipe.process.pid), os.getpgrp())
                self.assertEqual(os.getpgid(sibling.pid), os.getpgrp())
                pipe._terminate_owned()
                self.assertIsNotNone(pipe.process.poll())
                self.assertIsNone(sibling.poll(), "cleanup must not signal a sibling owned by the outer group")
            finally:
                sibling.terminate()
                sibling.wait(timeout=5)
                pipe.close()


if __name__ == "__main__":
    unittest.main()
