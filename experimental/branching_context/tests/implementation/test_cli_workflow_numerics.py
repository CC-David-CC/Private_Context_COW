"""Fail-closed validators for the independent G4 CLI numerical audit."""
from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path
import verify_cli_workflow_numerics as audit

from verify_cli_workflow_numerics import (
    validate_campaign_schedule,
    validate_controller_exact,
    validate_frontier_bytes,
    validate_operation_binding,
    validate_oracle_profile,
    validate_root_presentation,
    continuation_reference_plan,
    validate_continuation_plan,
    validate_continuation_reference_plan,
    validate_fresh_oracle_eos,
    validate_segmented_decode_receipt,
    validate_segment_eos,
)


def _campaign():
    labels = (
        "namespace", "prepare", "fork-a", "fork-b", "answer-a", "answer-b",
        "evaluate-a", "evaluate-b", "inspect-a", "inspect-b", "select", "adopt",
        "history", "continue",
    )
    return {
        "status": "CLI_FLOW_PASS_NUMERICAL_AUDIT_PENDING",
        "outputs": 8,
        "workflow": ["namespace", "prepare", "fork-a", "fork-b", "ask-a", "ask-b",
                     "evaluate-a", "evaluate-b", "inspect-a", "inspect-b", "select",
                     "adopt", "history", "continue"],
        "commands": [{"label": label, "exit_code": 0,
                      **({"argv_redacted": ["cow-cli", "continue", "--max-tokens", "8"]}
                         if label == "continue" else {})} for label in labels],
        "candidates": {
            "a": {"operation_id": "alt-answer", "source_snapshot": "root",
                  "snapshot": "snap-a", "output_token_ids": list(range(10, 18)),
                  "output_span": {"start": 5, "count": 8},
                  "receipt": {"snapshot": "snap-a", "branch": {"tip": "snap-a"}}},
            "b": {"operation_id": "resume-op", "source_snapshot": "root",
                  "snapshot": "snap-b", "output_token_ids": list(range(10, 18)),
                  "output_span": {"start": 5, "count": 8},
                  "receipt": {"snapshot": "snap-b", "branch": {"tip": "snap-b"}}},
        },
        "selection": {"snapshot": "snap-b"},
        "adoption": {
            "before": {"name": "main", "revision": 1, "tip": "root", "root": "root"},
            "after": {"name": "main", "revision": 2, "tip": "snap-b", "root": "root"},
            "ack": {"snapshot_id": "snap-b",
                    "branch": {"name": "main", "revision": 2, "tip": "snap-b", "root": "root"}},
        },
        "continuation_action": "continue",
        "continuation": {"snapshot": "snap-c", "branch": {
            "name": "main", "revision": 3, "tip": "snap-c", "root": "root"}},
        "services": [{"pid": 111, "stop": {"exit_code": 0},
                      "worker_shutdown": {"native_pid": 222, "native_exit_code": 0}}],
        "source": {"execution_profile_id": "profile-id", "model_contract_id": "model-id",
                   "native_sha256": "a" * 64},
    }


class CliWorkflowNumericsTests(unittest.TestCase):
    def test_prepared_root_uses_verified_root_annotation_not_move_presentation(self):
        annotation = {
            "kind": "root", "namespace_id": "chat-1",
            "source": {"snapshot": None, "revision": 0, "token_count": 0},
            "input_span": {"start": 0, "count": 181, "token_ids_sha256": "a" * 64},
            "output_span": None,
        }
        self.assertEqual(validate_root_presentation([annotation], namespace_id="chat-1",
                                                     token_count=181,
                                                     input_identity="a" * 64), annotation)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_root_presentation([], namespace_id="chat-1", token_count=181,
                                       input_identity="a" * 64)
        with self.assertRaisesRegex(ValueError, "namespace"):
            validate_root_presentation([annotation], namespace_id="other-chat", token_count=181,
                                       input_identity="a" * 64)
        with self.assertRaisesRegex(ValueError, "input span"):
            validate_root_presentation([dict(annotation, input_span={"start": 0, "count": 181,
                                       "token_ids_sha256": "b" * 64})], namespace_id="chat-1",
                                       token_count=181, input_identity="a" * 64)

    def test_oracle_prefill_chunks_match_sut_root_execution_receipt(self):
        validator = getattr(audit, "validate_prefill_chunks", None)
        self.assertTrue(callable(validator), "auditor must bind oracle chunks to SUT receipt")
        root_manifest = {"execution_schedule": [
            {"operator": "batched-prefill", "position": 0, "count": 181,
             "chunks": [[0, 181]]},
            {"operator": "normal-verifier-decode", "position": 181,
             "count": 8, "chunks": [[181, 1], [182, 1], [183, 1], [184, 1],
                                       [185, 1], [186, 1], [187, 1], [188, 1]]},
        ]}
        oracle_schedule = {"prefill_operator": "batched-prefill",
                           "prefill_requests": [181], "prefill_chunks": [[0, 181]]}
        validator(root_manifest, oracle_schedule)
        with self.assertRaisesRegex(ValueError, "differ from the SUT root"):
            validator(root_manifest, dict(oracle_schedule, prefill_chunks=[[0, 180], [180, 1]]))
        with self.assertRaisesRegex(ValueError, "differ from the SUT root"):
            validator(root_manifest, dict(oracle_schedule, prefill_chunks=[[0, 181], [181, 1]]))

    def test_raw_oracle_eos_contract_and_segment_receipt_are_exact(self):
        validate_fresh_oracle_eos([248044, 248046])
        validate_fresh_oracle_eos([248046, 248044])
        with self.assertRaisesRegex(ValueError, "EOS IDs are incompatible"):
            validate_fresh_oracle_eos([248044])
        with self.assertRaisesRegex(ValueError, "EOS IDs are incompatible"):
            validate_fresh_oracle_eos([248044, 248046, 9])

        schedule = {
            "decode_operator": "normal-verifier-decode",
            "requested_segments": [8, 8],
            "completed_segments": [
                {"generated_start": 0, "native_start_position": 5, "count": 8},
                {"generated_start": 8, "native_start_position": 13, "count": 8},
            ],
        }
        work = {"decode_tokens": 16, "normal_forward_tokens": 16,
                "consumed": 21, "prefill_tokens": 5}
        validate_segmented_decode_receipt(work, schedule, segments=[8, 8], prompt_tokens=5)
        malformed = dict(schedule, completed_segments=schedule["completed_segments"][:1])
        with self.assertRaisesRegex(ValueError, "positions/counts"):
            validate_segmented_decode_receipt(work, malformed, segments=[8, 8], prompt_tokens=5)
        with self.assertRaisesRegex(ValueError, "forward count"):
            validate_segmented_decode_receipt(dict(work, normal_forward_tokens=8), schedule,
                                              segments=[8, 8], prompt_tokens=5)
        wrong_position = dict(schedule, completed_segments=[
            schedule["completed_segments"][0],
            dict(schedule["completed_segments"][1], native_start_position=12),
        ])
        with self.assertRaisesRegex(ValueError, "positions/counts"):
            validate_segmented_decode_receipt(work, wrong_position, segments=[8, 8], prompt_tokens=5)

    def test_eos_must_be_terminal_and_selected_answer_must_continue(self):
        validate_segment_eos([1, 248044], [248044, 248046], must_continue=False)
        with self.assertRaisesRegex(ValueError, "before its final token"):
            validate_segment_eos([248044, 1], [248044, 248046], must_continue=False)
        with self.assertRaisesRegex(ValueError, "ended with EOS"):
            validate_segment_eos([1, 248046], [248044, 248046], must_continue=True)

    def test_campaign_requires_exact_prepare_two_answers_adoption_and_continuation_schedule(self):
        campaign = _campaign()
        validate_campaign_schedule(campaign, expected_outputs=8, expected_continue=8)

        malformed = _campaign()
        malformed["commands"] = malformed["commands"][:-1]
        with self.assertRaisesRegex(ValueError, "command schedule"):
            validate_campaign_schedule(malformed, expected_outputs=8, expected_continue=8)

        malformed = _campaign()
        malformed["candidates"]["b"]["output_token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "answer count"):
            validate_campaign_schedule(malformed, expected_outputs=8, expected_continue=8)

    def test_campaign_rejects_wrong_continuation_boundary_or_nonselected_parent(self):
        malformed = _campaign()
        malformed["continuation_action"] = "append-user"
        with self.assertRaisesRegex(ValueError, "continuation action"):
            validate_campaign_schedule(malformed, expected_outputs=8, expected_continue=8)

        malformed = _campaign()
        malformed["selection"]["snapshot"] = "snap-a"
        with self.assertRaisesRegex(ValueError, "selected snapshot"):
            validate_campaign_schedule(malformed, expected_outputs=8, expected_continue=8)

    def test_operation_binding_requires_receipt_source_and_target_identity(self):
        source = {"snapshot": "root", "controller": {"stream_key": "stream-root"}}
        target = {"parent": "root", "controller": {"stream_id": "stream-root"}}
        envelope = {"operation_id": "alt-answer", "result": {"tip": "snap-a"},
                    "presentation": {"kind": "answer", "source": {"snapshot": "root"},
                                     "input_span": {"count": 0},
                                     "output_span": {"start": 5, "count": 8}}}
        validate_operation_binding("alt-answer", envelope, "root", "snap-a", target,
                                   {"start": 5, "count": 8}, expected_kind="answer")
        envelope["presentation"]["source"]["snapshot"] = "other-root"
        with self.assertRaisesRegex(ValueError, "source"):
            validate_operation_binding("alt-answer", envelope, "root", "snap-a", target,
                                       {"start": 5, "count": 8}, expected_kind="answer")

    def test_complete_controller_comparison_rejects_stream_or_counter_drift(self):
        expected = {"stream_key": "a" * 64, "accepted_in_scope": 8,
                    "status": "active", "next_draw": 0}
        validate_controller_exact(dict(expected), expected, label="candidate")
        actual = dict(expected, accepted_in_scope=7)
        with self.assertRaisesRegex(ValueError, "controller differs"):
            validate_controller_exact(actual, expected, label="candidate")

    def test_oracle_must_be_distinct_and_prompt_cache_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            sut = Path(temporary) / "sut"
            oracle = Path(temporary) / "oracle"
            sut.write_bytes(b"sut")
            oracle.write_bytes(b"oracle")
            profile = {"execution": {
                "sut_command": [str(sut)],
                "oracle_command": [str(oracle), "--prompt-cache", "0",
                                   "--conversation-cache-mib", "0"],
            }}
            validate_oracle_profile(profile, sut_path=str(sut))
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_oracle_profile(profile, sut_path=str(oracle))
            profile["execution"]["oracle_command"][2] = "128"
            with self.assertRaisesRegex(ValueError, "prompt-cache"):
                validate_oracle_profile(profile, sut_path=str(sut))

    def test_full_frontier_requires_exact_size_and_finite_f32_values(self):
        width = 248320
        valid = struct.pack("<248320f", *([0.0] * width))
        self.assertEqual(validate_frontier_bytes(valid), width)
        with self.assertRaisesRegex(ValueError, "width"):
            validate_frontier_bytes(valid[:-4])
        nonfinite = struct.pack("<248320f", math.inf, *([0.0] * (width - 1)))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_frontier_bytes(nonfinite)

    def test_continuation_oracle_does_not_substitute_answer_tokens_as_prefill(self):
        root = [1, 2, 3, 4, 5]
        answer = [10, 11, 12, 13, 14, 15, 16, 17]
        plan = continuation_reference_plan(root, answer, 8)
        self.assertEqual(plan["prompt_ids"], root)
        self.assertEqual(plan["request_sizes"], [5])
        self.assertEqual(plan["generated"], 16)
        self.assertEqual(plan["decode_segments"], [8, 8])
        self.assertEqual(plan["answer_boundary"], 8)
        validate_continuation_plan(plan, root_tokens=root, answer_tokens=answer,
                                   continuation_outputs=8)
        validate_continuation_reference_plan(
            prompt_ids=root, request_sizes=[5], generated=16, decode_segments=[8, 8],
            root_tokens=root, answer_tokens=answer, continuation_outputs=8)

        # This old proposal feeds answer tokens through batched prefill, which
        # cannot stand in for the SUT's consumed verifier-decode answer.
        with self.assertRaisesRegex(ValueError, "cannot prefill answer"):
            validate_continuation_reference_plan(
                prompt_ids=root + answer, request_sizes=[5, 8], generated=8,
                decode_segments=[8],
                root_tokens=root, answer_tokens=answer, continuation_outputs=8)

        with self.assertRaisesRegex(ValueError, "two exact generation segments"):
            validate_continuation_reference_plan(
                prompt_ids=root, request_sizes=[5], generated=16, decode_segments=[16],
                root_tokens=root, answer_tokens=answer, continuation_outputs=8)
        malformed_plan = dict(plan, decode_segments=[16])
        with self.assertRaisesRegex(ValueError, "two exact generation segments"):
            validate_continuation_plan(malformed_plan, root_tokens=root, answer_tokens=answer,
                                       continuation_outputs=8)


if __name__ == "__main__":
    unittest.main()
