import json
import math
import struct
import unittest

from verify_cli_seed_stop_interleaved_numerics import (
    FRONTIER_BYTES, _append_input_ids, _frontier_bytes, _operation_id,
    _reconcile_raw_process, _validate_reseed_transition, compare_frontiers,
)


class SeedStopAuditOrchestratorTests(unittest.TestCase):
    def test_operation_id_is_bound_from_exact_campaign_command(self):
        campaign = {"commands": [{"label": "sampled-continue",
                                  "argv_redacted": ["continue", "--operation-id", "resume-17"]}]}
        self.assertEqual(_operation_id(campaign, "sampled-continue"), "resume-17")

    def test_operation_id_rejects_duplicate_or_missing_command(self):
        for campaign in (
            {"commands": []},
            {"commands": [{"label": "continue", "argv_redacted": ["continue"]}]},
            {"commands": [{"label": "continue", "argv_redacted": ["continue", "--operation-id", "a", "--operation-id", "b"]}]},
        ):
            with self.subTest(campaign=campaign), self.assertRaises(ValueError):
                _operation_id(campaign, "continue")

    def test_full_frontier_comparison_requires_exact_bytes_and_finite_values(self):
        zero = bytes(FRONTIER_BYTES)
        proof = compare_frontiers(zero, zero, "same")
        self.assertTrue(proof["exact"])
        self.assertEqual(proof["max_abs_error"], 0.0)
        changed = bytearray(zero)
        changed[0] = 1
        with self.assertRaisesRegex(ValueError, "full-frontier mismatch"):
            compare_frontiers(zero, bytes(changed), "different")

    def test_frontier_rejects_nonfinite_or_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _frontier_bytes(b"short", "short")
        nonfinite = struct.pack("<f", math.inf) + bytes(FRONTIER_BYTES - 4)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _frontier_bytes(nonfinite, "infinity")

    def test_reseed_is_a_zero_token_direct_child_changing_only_rng_binding(self):
        source_id = "source"
        target_id = "reseeded"
        source_controller = {
            "schema": 3, "status": "active", "accepted_in_scope": 1,
            "stream_key": "old-stream", "next_draw": 1,
            "grammar_state": {"state": 7}, "policy": {"temperature": 0.8},
        }
        target_controller = dict(source_controller, stream_key="new-stream", next_draw=0)

        class Store:
            def get_snapshot(self, snapshot):
                return {"parent": source_id, "controller": target_controller}

            def lineage(self, snapshot):
                return "root", [10, 11, 12]

        campaign = {"reseed": {
            "ack": {"tip": target_id},
            "controller": target_controller,
        }}
        proof = _validate_reseed_transition(
            campaign, Store(), "root", source_id, [10, 11, 12], source_controller)
        self.assertEqual(proof["stream_key"], "new-stream")
        self.assertEqual(proof["next_draw"], 0)
        for bad_target in (
            dict(target_controller, accepted_in_scope=0),
            dict(target_controller, next_draw=1),
            dict(target_controller, stream_key="old-stream"),
        ):
            bad_campaign = {"reseed": {
                "ack": {"tip": target_id}, "controller": bad_target,
            }}

            class BadStore(Store):
                def get_snapshot(self, snapshot):
                    return {"parent": source_id, "controller": bad_target}

            with self.subTest(controller=bad_target), self.assertRaises(ValueError):
                _validate_reseed_transition(
                    bad_campaign, BadStore(), "root", source_id, [10, 11, 12],
                    source_controller)

    def test_append_span_excludes_read_output_and_test_output(self):
        read_history = list(range(18))
        appended = list(range(100, 118))
        test_tokens = list(range(200, 211))
        test_history = read_history + appended + test_tokens
        self.assertEqual(_append_input_ids(
            read_history, test_history, test_tokens, {"start": 18, "count": 18}), appended)
        with self.assertRaisesRegex(ValueError, "only new input"):
            _append_input_ids(
                read_history, test_history, test_tokens, {"start": 18, "count": 29})

    def test_failed_raw_process_receipt_still_records_gpu_attempt(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            record = {"runtime_executed": False, "gpu_launched": False}
            (run_dir / "process.json").write_text(json.dumps({
                "status": "FAILED", "pid": 123, "exit_code": 1,
                "pid_absent": True, "gpu_launched_by_audit_runner": True,
            }), encoding="utf-8")
            _reconcile_raw_process(record, run_dir, "failed-child")
        self.assertTrue(record["runtime_executed"])
        self.assertTrue(record["gpu_launched"])
        self.assertEqual(record["raw_processes"][0]["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
