"""CPU evidence guards for the Q4 post-commit lost-ACK harness."""

import unittest
import ast
import inspect
import sys
from unittest.mock import patch
import tempfile
import queue
import threading
from pathlib import Path

from run_q4_postcommit_restart import (
    PostcommitRestartError,
    validate_commit_witness,
    validate_idempotency_conflict,
    validate_replay_completion,
    validate_replay_move_unchanged,
    validate_single_move_delta,
    validate_no_ack_events,
    _proc_state,
    _drain_crashed_stream,
    _stop_and_record,
    main,
    run_probe,
)


class PostcommitEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.witness = {
            "event": "branch_committed_before_ack",
            "operation_id": "op-q4-test",
            "branch": "Q4Answer",
            "expected_revision": 4,
            "snapshot": "a" * 64,
            "pid": 321,
            "thread_id": 654,
        }
        self.receipt = {
            "branch": {"name": "Q4Answer", "revision": 5,
                       "tip": "a" * 64, "root": "b" * 64},
            "snapshot": "a" * 64,
            "payload_sha256": "c" * 64,
            "model_contract_id": "model-q4",
            "execution_profile_id": "profile-q4",
        }

    def test_commit_witness_binds_target_operation_branch_and_service(self):
        self.assertEqual(validate_commit_witness(
            self.witness, operation_id="op-q4-test", branch="Q4Answer",
            snapshot="a" * 64, expected_revision=4, server_pid=321), self.witness)

    def test_commit_witness_rejects_wrong_operation(self):
        with self.assertRaises(PostcommitRestartError):
            validate_commit_witness(self.witness, operation_id="op-other",
                branch="Q4Answer", snapshot="a" * 64,
                expected_revision=4, server_pid=321)

    def test_commit_witness_rejects_wrong_target_or_uncommitted_hook(self):
        for changes in ({"snapshot": "d" * 64}, {"event": "before_branch_publish"}):
            with self.subTest(changes=changes), self.assertRaises(PostcommitRestartError):
                validate_commit_witness(dict(self.witness, **changes),
                    operation_id="op-q4-test", branch="Q4Answer",
                    snapshot="a" * 64, expected_revision=4, server_pid=321)

    def test_replay_completion_must_be_recovered_and_identical_receipt(self):
        event = {"event": "complete",
                 "answer": {"snapshot": "a" * 64, "tokens": [1, 2], "recovered": True},
                 "receipt": self.receipt}
        self.assertEqual(validate_replay_completion([event], self.receipt,
            branch="Q4Answer", snapshot="a" * 64, revision=5), self.receipt)

    def test_replay_rejects_plausible_but_changed_receipt(self):
        changed = dict(self.receipt, payload_sha256="d" * 64)
        event = {"event": "complete",
                 "answer": {"snapshot": "a" * 64, "tokens": [1, 2], "recovered": True},
                 "receipt": changed}
        with self.assertRaises(PostcommitRestartError):
            validate_replay_completion([event], self.receipt,
                branch="Q4Answer", snapshot="a" * 64, revision=5)

    def test_replay_rejects_nonrecovered_or_wrong_revision_completion(self):
        for changes in ({"answer": {"snapshot": "a" * 64, "tokens": [1, 2],
                          "recovered": False}},
                        {"receipt": dict(self.receipt, branch=dict(
                            self.receipt["branch"], revision=6))}):
            event = {"event": "complete",
                     "answer": {"snapshot": "a" * 64, "tokens": [1, 2], "recovered": True},
                     "receipt": self.receipt}
            event.update(changes)
            with self.subTest(changes=changes), self.assertRaises(PostcommitRestartError):
                validate_replay_completion([event], self.receipt,
                    branch="Q4Answer", snapshot="a" * 64, revision=5)

    def test_move_delta_requires_exactly_one_original_operation(self):
        before = ["op-old"]
        self.assertEqual(validate_single_move_delta(before,
            ["op-old", "op-q4-test"], "op-q4-test"), ["op-q4-test"])
        for after in (["op-old"], ["op-old", "op-q4-test", "op-q4-test"],
                      ["op-old", "op-replacement"]):
            with self.subTest(after=after), self.assertRaises(PostcommitRestartError):
                validate_single_move_delta(before, after, "op-q4-test")

    def test_replay_move_log_must_remain_identical(self):
        before = ["op-create", "op-fork", "op-q4-test"]
        validate_replay_move_unchanged(before, list(before))
        with self.assertRaises(PostcommitRestartError):
            validate_replay_move_unchanged(before, before + ["op-q4-test"])

    def test_changed_same_id_rejection_uses_observed_http_contract(self):
        self.assertIn("Operation ID reused", validate_idempotency_conflict(
            400, {"error": "Operation ID reused for different input"}, "op-q4-test"))
        for status, body in ((409, {"error": "Operation ID reused for different input"}),
                             (400, {"error": "unrelated validation error"})):
            with self.subTest(status=status, body=body), self.assertRaises(PostcommitRestartError):
                validate_idempotency_conflict(status, body, "op-q4-test")

    def test_proc_state_distinguishes_absent_zombie_and_live(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            self.assertEqual(_proc_state(proc, 123), "absent")
            (proc / "123").mkdir()
            (proc / "123" / "stat").write_text("123 (test) S 1 2 3\n", encoding="ascii")
            self.assertEqual(_proc_state(proc, 123), "running")
            (proc / "123" / "stat").write_text("123 (test) Z 1 2 3\n", encoding="ascii")
            self.assertEqual(_proc_state(proc, 123), "zombie")

    def test_crash_drain_keeps_already_consumed_snapshot_once_and_no_complete(self):
        events = queue.Queue()
        events.put(("eof",))
        done = threading.Event()
        done.set()
        reader = threading.Thread(target=lambda: None)
        reader.start()
        snapshot = {"event": "snapshot", "snapshot": "a" * 64}
        rows, errors = _drain_crashed_stream(events, done, reader, 1, initial_events=[snapshot])
        self.assertEqual(rows, [snapshot])
        self.assertEqual(errors, [])

    def test_lost_ack_stream_rejects_either_durable_or_complete_event(self):
        validate_no_ack_events([{"event": "snapshot", "snapshot": "a" * 64}])
        for event in ("durable", "complete"):
            with self.subTest(event=event), self.assertRaises(PostcommitRestartError):
                validate_no_ack_events([{"event": "snapshot", "snapshot": "a" * 64},
                                        {"event": event, "snapshot": "a" * 64}])

    def test_cleanup_retains_stop_receipt_and_stop_error(self):
        class CleanService:
            def stop(self):
                return {"exit_code": 0, "pid": 123}

        class BrokenService:
            def stop(self):
                raise OSError("stop failed")

        receipts, errors = [], []
        _stop_and_record("clean", CleanService(), receipts, errors)
        _stop_and_record("broken", BrokenService(), receipts, errors)
        self.assertEqual(receipts, [{"service": "clean", "receipt": {"exit_code": 0, "pid": 123}}])
        self.assertEqual(errors, [{"service": "broken", "error": "OSError: stop failed"}])

    def test_main_returns_nonzero_when_probe_finishes_nonpass(self):
        argv = ["runner", "--profile", "p", "--store", "s", "--branch", "b",
            "--runtime-root", "r", "--api-src", "a", "--api-scripts", "as",
            "--expected-profile-sha256", "0" * 64, "--expected-native-sha256", "1" * 64,
            "--run-dir", "out"]
        with patch.object(sys, "argv", argv), patch(
                "run_q4_postcommit_restart.run_probe", return_value={"status": "FAIL"}):
            self.assertEqual(main(), 1)
        with patch.object(sys, "argv", argv), patch(
                "run_q4_postcommit_restart.run_probe", return_value={"status": "PASS"}):
            self.assertEqual(main(), 0)

    def test_probe_opens_store_before_first_branch_move_log_read(self):
        tree = ast.parse(inspect.getsource(run_probe))
        store_assignments = [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "store" for target in node.targets)]
        move_reads = [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_read_branch" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "store"]
        self.assertTrue(store_assignments, "run_probe must create its Store handle")
        self.assertTrue(move_reads, "run_probe must capture canonical branch moves")
        self.assertLess(min(store_assignments), min(move_reads))


if __name__ == "__main__":
    unittest.main()
