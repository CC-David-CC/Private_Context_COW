"""CPU contract checks for the real Q4 HTTP publication-fault harness.

These tests validate evidence classification and fail-closed guards only. They
do not emulate GPU work or qualify the kernel fault path.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import unittest
import run_q4_publication_fault as fault_probe

from run_q4_publication_fault import (
    PublicationFaultError,
    validate_efbig_writer_observation,
    validate_faulted_operation,
    validate_fault_stream,
    validate_retry_publication,
    validate_limit_isolation,
    _canonical_history_sha,
    validate_idle_metrics,
    validate_source_checkpoint_unchanged,
    validate_applied_limit,
    _children,
)


class PublicationFaultEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.writer = {
            "operation_id": "op-q4-test",
            "branch": "BranchA",
            "snapshot": "snap-next",
            "destination_kind": "checkpoint-payload-state",
            "errno": 27,
            "exception_type": "OSError",
            "message": "[Errno 27] File too large",
            "payload_name": "payload.state",
            "requested_payload_bytes": 67108865,
            "published_payload_exists": False,
            "published_manifest_exists": False,
            "staging_files_after": [],
        }
        self.operation = {
            "operation_id": "op-q4-test",
            "branch": "BranchA",
            "scope": "coordinator-process",
            "state": "failed",
            "durable": False,
            "publication_retryable": True,
        }
        self.head_before = {"name": "BranchA", "revision": 4, "tip": "snap-parent"}
        self.head_after_fault = dict(self.head_before)

    def test_children_finds_subprocess_spawned_by_non_main_thread(self):
        spawned = queue.Queue()
        ready = threading.Event()
        release_thread = threading.Event()

        def launch_child():
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            spawned.put(child)
            ready.set()
            release_thread.wait(5)

        thread = threading.Thread(target=launch_child)
        thread.start()
        self.assertTrue(ready.wait(5), 'worker thread did not launch child')
        child = spawned.get(timeout=1)
        try:
            self.assertIn(child.pid, _children(os.getpid()))
        finally:
            child.terminate()
            child.wait(timeout=5)
            release_thread.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_only_real_payload_writer_efbig_is_accepted(self):
        accepted = validate_efbig_writer_observation(self.writer, "op-q4-test", "BranchA")
        self.assertEqual(accepted["snapshot"], "snap-next")
        for change in (
            {"errno": 28},
            {"destination_kind": "native-export"},
            {"payload_name": "wrong.json"},
            {"requested_payload_bytes": 64 * 1024**2},
            {"published_payload_exists": True},
            {"staging_files_after": [".stage-leftover"]},
            {"operation_id": "other-op"},
        ):
            with self.subTest(change=change), self.assertRaises(PublicationFaultError):
                validate_efbig_writer_observation({**self.writer, **change}, "op-q4-test", "BranchA")

    def test_fault_stream_never_acknowledges_complete_and_allows_http_eof(self):
        validate_fault_stream([])
        validate_fault_stream([{"event": "error", "error": "writer failed"}])
        validate_fault_stream([{"event": "failure", "errno": 27}])
        for events in (
            [{"event": "complete", "receipt": {}}],
            [{"event": "error"}, {"event": "failure"}],
            ["not-an-event"],
        ):
            with self.subTest(events=events), self.assertRaises(PublicationFaultError):
                validate_fault_stream(events)

    def test_real_write_failure_requires_explicit_original_error(self):
        validate = getattr(fault_probe, 'validate_reported_write_failure', None)
        self.assertTrue(callable(validate), 'Real write failures need an explicit client-error guard')
        valid = [{'event': 'error', 'error': '[Errno 27] File too large', 'cancelled': False}]
        validate(valid, expected_errno=27)
        for events in ([], [{'event': 'error', 'error': 'unspecified', 'cancelled': False}],
                       [{**valid[0], 'cancelled': True}],
                       [{'event': 'error', 'error': '[Errno 28] No space left on device', 'cancelled': False}]):
            with self.subTest(events=events), self.assertRaises(PublicationFaultError):
                validate(events, expected_errno=27)

    def test_fault_must_be_retryable_and_leave_old_branch_head(self):
        accepted = validate_faulted_operation(
            self.operation, self.head_before, self.head_after_fault,
            operation_id="op-q4-test", branch="BranchA", parent_snapshot="snap-parent",
        )
        self.assertEqual(accepted["parent_snapshot"], "snap-parent")
        bad = dict(self.operation, durable=True, state="durable")
        with self.assertRaises(PublicationFaultError):
            validate_faulted_operation(bad, self.head_before, self.head_after_fault,
                operation_id="op-q4-test", branch="BranchA", parent_snapshot="snap-parent")
        moved = dict(self.head_after_fault, revision=5, tip="snap-next")
        with self.assertRaises(PublicationFaultError):
            validate_faulted_operation(self.operation, self.head_before, moved,
                operation_id="op-q4-test", branch="BranchA", parent_snapshot="snap-parent")

    def test_retry_is_bound_to_same_operation_snapshot_and_single_cas(self):
        retry = {
            "event": "complete",
            "receipt": {
                "operation_id": "op-q4-test",
                "snapshot": "snap-next",
                "branch": {"name": "BranchA", "revision": 5, "tip": "snap-next"},
            },
        }
        head_after = {"name": "BranchA", "revision": 5, "tip": "snap-next"}
        status = {"operation_id": "op-q4-test", "branch": "BranchA", "state": "durable",
                  "durable": True, "publication_retryable": False, "receipt": retry["receipt"]}
        accepted = validate_retry_publication(
            retry, "op-q4-test", "BranchA", "snap-next", self.head_before, head_after,
            moves_before=["old-op"], moves_after=["old-op", "op-q4-test"],
            operation_status=status,
        )
        self.assertEqual(accepted["revision"], 5)
        wrong = {**retry, "receipt": {**retry["receipt"], "snapshot": "another-snapshot"}}
        with self.assertRaises(PublicationFaultError):
            validate_retry_publication(wrong, "op-q4-test", "BranchA", "snap-next", self.head_before,
                head_after, moves_before=["old-op"], moves_after=["old-op", "op-q4-test"],
                operation_status=status)
        duplicate_move = validate_retry_publication
        with self.assertRaises(PublicationFaultError):
            duplicate_move(retry, "op-q4-test", "BranchA", "snap-next", self.head_before,
                head_after, moves_before=["old-op"], moves_after=["old-op", "op-q4-test", "op-q4-test"],
                operation_status=status)

    def test_publication_retry_does_not_add_native_work(self):
        before = {"forward_tokens": 128, "consumed_tokens": 128, "capture_count": 2}
        unchanged = dict(before)
        retry = {"event": "complete", "receipt": {"operation_id": "op-q4-test", "snapshot": "snap-next",
                  "branch": {"name": "BranchA", "revision": 5, "tip": "snap-next"}}}
        status = {"operation_id": "op-q4-test", "branch": "BranchA", "state": "durable",
                  "durable": True, "publication_retryable": False, "receipt": retry["receipt"]}
        validate_retry_publication(
            retry,
            "op-q4-test", "BranchA", "snap-next", self.head_before,
            {"name": "BranchA", "revision": 5, "tip": "snap-next"},
            moves_before=["old-op"], moves_after=["old-op", "op-q4-test"],
            operation_status=status,
            work_before=before, work_after=unchanged,
        )
        changed = dict(before, forward_tokens=136)
        with self.assertRaises(PublicationFaultError):
            validate_retry_publication(
                retry,
                "op-q4-test", "BranchA", "snap-next", self.head_before,
                {"name": "BranchA", "revision": 5, "tip": "snap-next"},
                moves_before=["old-op"], moves_after=["old-op", "op-q4-test"],
                operation_status=status,
                work_before=before, work_after=changed,
            )

    def test_limit_is_applied_only_to_owned_server_and_restored(self):
        valid = {
            "server": {"pid": 100, "start_time": "s1", "soft_fsize": 2**63 - 1,
                       "hard_fsize": 2**63 - 1, "sigxfsz_ignored": True},
            "native": {"pid": 101, "parent_pid": 100, "start_time": "n1",
                       "soft_fsize": 2**63 - 1, "hard_fsize": 2**63 - 1},
            "applied": {"pid": 100, "start_time": "s1", "soft_fsize": 64 * 1024**2,
                        "hard_fsize": 2**63 - 1},
            "restored": {"pid": 100, "start_time": "s1", "soft_fsize": 2**63 - 1,
                         "hard_fsize": 2**63 - 1},
        }
        validate_limit_isolation(valid, expected_server_pid=100, expected_native_pid=101)
        wrong_native = {**valid, "applied": {**valid["applied"], "pid": 101}}
        with self.assertRaises(PublicationFaultError):
            validate_limit_isolation(wrong_native, expected_server_pid=100, expected_native_pid=101)
        wrong_hard = {**valid, "applied": {**valid["applied"], "hard_fsize": 64 * 1024**2}}
        with self.assertRaises(PublicationFaultError):
            validate_limit_isolation(wrong_hard, expected_server_pid=100, expected_native_pid=101)


    def test_idle_metrics_requires_zero_active_and_queued_work(self):
        valid = {"GPU": [0], "active_per_gpu": {"0": 0}, "queued": 0}
        validate_idle_metrics(valid)
        for bad in (
            {"GPU": [0], "active_per_gpu": {"0": 1}, "queued": 0},
            {"GPU": [0], "active_per_gpu": {"0": 0}, "queued": 1},
            {"GPU": [0], "active_per_gpu": {"0": -1}, "queued": 0},
            {"GPU": [0], "active_per_gpu": {"1": 0}, "queued": 0},
            {"GPU": [0], "active_per_gpu": {}, "queued": 0},
        ):
            with self.subTest(bad=bad), self.assertRaises(PublicationFaultError):
                validate_idle_metrics(bad)

    def test_source_checkpoint_fingerprint_must_remain_exact(self):
        before = {"snapshot.json": {"bytes": 44, "sha256": "a" * 64},
                  f"{'b' * 64}.state": {"bytes": 100, "sha256": "b" * 64}}
        validate_source_checkpoint_unchanged(before, dict(before), phase='after retry')
        after = dict(before)
        after[f"{'b' * 64}.state"] = {"bytes": 99, "sha256": "c" * 64}
        with self.assertRaises(PublicationFaultError):
            validate_source_checkpoint_unchanged(before, after, phase='after retry')

    def test_applied_limit_is_checked_before_restoration(self):
        validate_applied_limit((64 * 1024**2, 2**63 - 1), (64 * 1024**2, 2**63 - 1))
        with self.assertRaises(PublicationFaultError):
            validate_applied_limit((64 * 1024**2, 2**63 - 1), (32 * 1024**2, 2**63 - 1))

    def test_history_digest_ignores_only_transient_pending_request(self):
        durable = {"messages": [{"id": "m1", "text": "kept"}], "moves": []}
        pending = {**durable, "pending_request": {"state": "persisting"}}
        self.assertEqual(_canonical_history_sha(durable), _canonical_history_sha(pending))
        changed = {**durable, "messages": [{"id": "m1", "text": "changed"}]}
        self.assertNotEqual(_canonical_history_sha(durable), _canonical_history_sha(changed))

    def test_limit_isolation_allows_distinct_original_soft_and_hard_limits(self):
        valid = {
            "server": {"pid": 100, "start_time": "s1", "soft_fsize": 4096,
                       "hard_fsize": 8192, "sigxfsz_ignored": True},
            "native": {"pid": 101, "parent_pid": 100, "start_time": "n1",
                       "soft_fsize": 4096, "hard_fsize": 8192},
            "applied": {"pid": 100, "start_time": "s1", "soft_fsize": 1024,
                        "hard_fsize": 8192},
            "restored": {"pid": 100, "start_time": "s1", "soft_fsize": 4096,
                         "hard_fsize": 8192},
        }
        validate_limit_isolation(valid, expected_server_pid=100, expected_native_pid=101, limit_bytes=1024)
        impossible = {**valid, "applied": {**valid["applied"], "soft_fsize": 64 * 1024**2}}
        with self.assertRaises(PublicationFaultError):
            validate_limit_isolation(impossible, expected_server_pid=100, expected_native_pid=101,
                                     limit_bytes=64 * 1024**2)
        bad = {**valid, "applied": {**valid["applied"], "hard_fsize": 1024}}
        with self.assertRaises(PublicationFaultError):
            validate_limit_isolation(bad, expected_server_pid=100, expected_native_pid=101)


if __name__ == "__main__":
    unittest.main()
