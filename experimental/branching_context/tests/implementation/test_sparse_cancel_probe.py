"""CPU-only evidence-shape tests for the sparse-prefix cancellation probe."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_sparse_cancel_probe.py"
SPEC = importlib.util.spec_from_file_location("sparse_cancel_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class SparseCancelEvidenceTests(unittest.TestCase):
    def test_profile_pinned_metadata_helper_is_checked_separately_from_api_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "declared-helper.py"
            path.write_bytes(b"pinned metadata helper\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            profile = {"gguf_reader": str(path), "native_build": {"sha256": {str(path): digest}}}
            module = types.ModuleType("ctt_strata_gguf")
            module.__file__ = str(path)
            with patch.dict(PROBE.sys.modules, {"ctt_strata_gguf": module}):
                proof = PROBE._verify_native_metadata_helper(profile)
                self.assertEqual(proof["actual_path"], str(path.resolve()))
                self.assertEqual(proof["sha256"], digest)
                self.assertNotIn(module, PROBE._loaded_runtime_modules())

    def test_metadata_helper_rejects_unpinned_digest_and_wrong_import_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "declared-helper.py"
            other = Path(temporary) / "undeclared-helper.py"
            path.write_bytes(b"same source\n")
            other.write_bytes(path.read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            profile = {"gguf_reader": str(path), "native_build": {"sha256": {str(path): digest}}}
            module = types.ModuleType("ctt_strata_gguf")
            module.__file__ = str(other)
            with patch.dict(PROBE.sys.modules, {"ctt_strata_gguf": module}):
                with self.assertRaisesRegex(AssertionError, "metadata helper.*path"):
                    PROBE._verify_native_metadata_helper(profile)
            module.__file__ = str(path)
            with patch.dict(PROBE.sys.modules, {"ctt_strata_gguf": module}):
                for hashes in ({}, {str(path): "0" * 64}):
                    altered = {**profile, "native_build": {"sha256": hashes}}
                    with self.subTest(hashes=hashes), self.assertRaises(AssertionError):
                        PROBE._verify_native_metadata_helper(altered)

    def test_profile_closure_maps_tools_relative_to_runtime_tools_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime-v1"
            api_src = runtime / "experimental" / "branching_context" / "src"
            api_scripts = runtime / "experimental" / "branching_context" / "scripts"
            tools = runtime / "tools"
            for directory in (api_src, api_scripts, tools):
                directory.mkdir(parents=True, exist_ok=True)
            file = tools / "strata_tokenizer.py"
            file.write_bytes(b"pinned tokenizer source\n")
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            declared = "/original/runtime/tools/strata_tokenizer.py"
            sources = {"tools.strata_tokenizer": {"path": str(file.resolve()), "sha256": digest}}
            bound = PROBE._verify_profile_closure(sources,
                {"files_sha256": {declared: digest}}, runtime, api_src, api_scripts)
            self.assertEqual(bound["files"][declared]["actual_path"], str(file.resolve()))
            self.assertEqual(bound["imported"]["tools.strata_tokenizer"]["declared_profile_path"], declared)

    def test_profile_closure_rejects_forged_imported_digest_with_unchanged_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime-v1"
            api_src = runtime / "experimental" / "branching_context" / "src"
            api_scripts = runtime / "experimental" / "branching_context" / "scripts"
            file = api_src / "context_time_travel" / "sample.py"
            file.parent.mkdir(parents=True)
            api_scripts.mkdir(parents=True)
            file.write_bytes(b"pinned source\n")
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            declared = "/original/runtime/experimental/branching_context/src/context_time_travel/sample.py"
            sources = {"context_time_travel.sample": {"path": str(file.resolve()), "sha256": "0" * 64}}
            with self.assertRaisesRegex(AssertionError, "Imported runtime bytes"):
                PROBE._verify_profile_closure(sources,
                    {"files_sha256": {declared: digest}}, runtime, api_src, api_scripts)

    def test_cancel_is_three_consumed_tokens_without_snapshot_or_ref_move(self):
        trace = {"callback_tokens": [11, 12, 13], "native_request": {
                 "status": "ok", "operator": "normal-verifier-decode", "generated": 3,
                 "position": 8704, "consumed": 8707, "forward_tokens": 8707,
                 "chunks": [[8704, 1], [8705, 1], [8706, 1]]},
                 "after": {"forward_tokens": 8707},
                 "protocol_failed": False, "forward_delta": 515, "consumed_delta": 515,
                 "loaded_file_bytes_delta": 0}
        events = ["request_started", "token", "token", "token", "cancelled"]
        self.assertTrue(PROBE.validate_cancel_evidence(
            state="cancelled", events=events, branch_before={"tip": "S", "revision": 0},
            branch_after={"tip": "S", "revision": 0}, snapshot_cancelled=True,
            durable_cancelled=True, operation_receipt_present=False,
            trace=trace, suffix_tokens=512))

    def test_cancel_rejects_publication_wrong_count_work_or_protocol_failure(self):
        valid = {"callback_tokens": [11, 12, 13], "native_request": {
                 "status": "ok", "operator": "normal-verifier-decode", "generated": 3,
                 "position": 8704, "consumed": 8707, "forward_tokens": 8707,
                 "chunks": [[8704, 1], [8705, 1], [8706, 1]]},
                 "after": {"forward_tokens": 8707},
                 "protocol_failed": False, "forward_delta": 515, "consumed_delta": 515,
                 "loaded_file_bytes_delta": 0}
        kwargs = dict(state="cancelled", events=["request_started", "token", "token", "token", "cancelled"],
            branch_before={"tip": "S", "revision": 0}, branch_after={"tip": "S", "revision": 0},
            snapshot_cancelled=True, durable_cancelled=True, operation_receipt_present=False,
            trace=valid, suffix_tokens=512)
        self.assertTrue(PROBE.validate_cancel_evidence(**kwargs))
        invalid = [
            {**kwargs, "events": kwargs["events"] + ["snapshot"]},
            {**kwargs, "events": kwargs["events"] + ["durable"]},
            {**kwargs, "branch_after": {"tip": "T", "revision": 1}},
            {**kwargs, "snapshot_cancelled": False},
            {**kwargs, "durable_cancelled": False},
            {**kwargs, "operation_receipt_present": True},
            {**kwargs, "trace": {**valid, "native_request": {**valid["native_request"], "generated": 4}}},
            {**kwargs, "trace": {**valid, "native_request": {**valid["native_request"], "chunks": [[8704, 3]]}}},
            {**kwargs, "trace": {**valid, "forward_delta": 514}},
            {**kwargs, "trace": {**valid, "loaded_file_bytes_delta": 4096}},
            {**kwargs, "trace": {**valid, "protocol_failed": True}},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(AssertionError):
                PROBE.validate_cancel_evidence(**values)

    def test_cancel_binds_cumulative_native_counter_to_worker_counter(self):
        trace = {"callback_tokens": [11, 12, 13], "native_request": {
                 "status": "ok", "operator": "normal-verifier-decode", "generated": 3,
                 "position": 8704, "consumed": 8707, "forward_tokens": 8707,
                 "chunks": [[8704, 1], [8705, 1], [8706, 1]]},
                 "after": {"forward_tokens": 8707},
                 "protocol_failed": False, "forward_delta": 515, "consumed_delta": 515,
                 "loaded_file_bytes_delta": 0}
        kwargs = dict(state="cancelled",
            events=["request_started", "token", "token", "token", "cancelled"],
            branch_before={"tip": "S", "revision": 0}, branch_after={"tip": "S", "revision": 0},
            snapshot_cancelled=True, durable_cancelled=True, operation_receipt_present=False,
            trace=trace, suffix_tokens=512)
        self.assertTrue(PROBE.validate_cancel_evidence(**kwargs))
        bad_trace = {**trace, "native_request": {**trace["native_request"], "forward_tokens": 8706}}
        with self.assertRaisesRegex(AssertionError, "cumulative forward"):
            PROBE.validate_cancel_evidence(**{**kwargs, "trace": bad_trace})

    def test_profile_closure_binds_relocated_source_bytes_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime-v1"
            api_src = runtime / "experimental" / "branching_context" / "src"
            api_scripts = runtime / "experimental" / "branching_context" / "scripts"
            module_file = api_src / "context_time_travel" / "sample.py"
            module_file.parent.mkdir(parents=True)
            api_scripts.mkdir(parents=True)
            runtime.mkdir(exist_ok=True)
            module_file.write_bytes(b"pinned source\n")
            digest = hashlib.sha256(module_file.read_bytes()).hexdigest()
            declared = "/original/runtime/experimental/branching_context/src/context_time_travel/sample.py"
            sources = {"context_time_travel.sample": {
                "path": str(module_file.resolve()), "sha256": digest}}
            closure = {"files_sha256": {declared: digest}}
            bound = PROBE._verify_profile_closure(sources, closure, runtime, api_src, api_scripts)
            self.assertEqual(bound["imported"]["context_time_travel.sample"]["sha256"], digest)
            self.assertEqual(bound["imported"]["context_time_travel.sample"]["declared_profile_path"], declared)
            with self.assertRaisesRegex(AssertionError, "differ from profile closure"):
                PROBE._verify_profile_closure(
                    sources, {"files_sha256": {declared: "0" * 64}},
                    runtime, api_src, api_scripts)

    def test_recovery_binds_receipt_history_controller_and_full_frontier(self):
        answer = {"tokens": [21, 22], "counter_deltas": {"forward_tokens": 514,
                  "consumed_tokens": 514,
                  "loaded_file_bytes": 0}, "snapshot": "T"}
        receipt = {"snapshot": "T", "branch": {"name": "b", "tip": "T", "revision": 1}}
        state = {"history": [1, 2, 3, 21, 22], "controller": {"status": "idle"}}
        self.assertTrue(PROBE.validate_recovery_answer(
            answer, receipt, state, {"tip": "T", "revision": 1}, source_revision=0,
            expected_tokens=[21, 22], suffix_tokens=512, expected_history=[1, 2, 3, 21, 22],
            actual_frontier=b"full-vector", expected_frontier=b"full-vector",
            actual_controller={"status": "idle"}, expected_controller={"status": "idle"}))

    def test_recovery_rejects_wrong_token_work_load_frontier_controller_or_branch(self):
        answer = {"tokens": [21, 22], "counter_deltas": {"forward_tokens": 514,
                  "consumed_tokens": 514,
                  "loaded_file_bytes": 0}, "snapshot": "T"}
        receipt = {"snapshot": "T", "branch": {"name": "b", "tip": "T", "revision": 1}}
        state = {"history": [1, 2, 3, 21, 22], "controller": {"status": "idle"}}
        expected = dict(answer=answer, receipt=receipt, state=state,
            branch_head={"tip": "T", "revision": 1}, source_revision=0,
            expected_tokens=[21, 22], suffix_tokens=512, expected_history=[1, 2, 3, 21, 22],
            actual_frontier=b"full-vector", expected_frontier=b"full-vector",
            actual_controller={"status": "idle"}, expected_controller={"status": "idle"})
        self.assertTrue(PROBE.validate_recovery_answer(**expected))
        invalid = []
        invalid.append({**expected, "answer": {**answer, "tokens": [21, 23]}})
        invalid.append({**expected, "answer": {**answer, "counter_deltas": {
            "forward_tokens": 513, "loaded_file_bytes": 0}}})
        invalid.append({**expected, "answer": {**answer, "counter_deltas": {
            "forward_tokens": 514, "loaded_file_bytes": 1}}})
        invalid.append({**expected, "answer": {**answer, "counter_deltas": {
            "forward_tokens": 514, "consumed_tokens": 513, "loaded_file_bytes": 0}}})
        invalid.append({**expected, "receipt": {"snapshot": "U"}})
        invalid.append({**expected, "receipt": {"snapshot": "T", "branch": "b"}})
        invalid.append({**expected, "receipt": {"snapshot": "T", "branch": {
            "name": "b", "tip": "T", "revision": 2}}})
        invalid.append({**expected, "branch_head": {"tip": "U", "revision": 1}})
        invalid.append({**expected, "actual_frontier": b"different-vector"})
        invalid.append({**expected, "actual_controller": {"status": "stopped"}})
        invalid.append({**expected, "state": {**state, "history": [1, 2, 3, 21, 99]}})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(AssertionError):
                PROBE.validate_recovery_answer(**values)


if __name__ == "__main__":
    unittest.main()
