"""CPU-only guards for the independent raw GBNF CLI numerical audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_cli_grammar_numerical_audit as audit


class GrammarAuditInputTests(unittest.TestCase):
    def test_exact_cli_outputs_and_segment_boundaries_are_bound(self):
        read_partial = [4754]
        read_finish = [13766, 3147, 851, 2378, 2129, 2267,
                       3147, 3431, 14905, 6971, 8934, 248046]
        test = [4754, 13766, 3147, 5917, 31436, 2129,
                48310, 3147, 3715, 8934, 248046]
        plan = audit.validate_trajectories(
            root_token_ids=list(range(22)),
            read_partial_ids=read_partial,
            read_finish_ids=read_finish,
            test_ids=test,
            read_segments=[1, 12],
            test_segments=[11],
            eos_tokens=[248044, 248046],
            vocab_size=248320,
        )
        self.assertEqual(plan["read"]["outputs"], read_partial + read_finish)
        self.assertEqual(plan["read"]["segments"], [1, 12])
        self.assertEqual(plan["test"]["segments"], [11])

    def test_rejects_wrong_schedule_sum_and_nonterminal_eos(self):
        common = dict(root_token_ids=list(range(22)), read_partial_ids=[4754],
                      read_finish_ids=[13766, 3147, 851, 2378, 2129, 2267,
                                       3147, 3431, 14905, 6971, 8934, 248046],
                      test_ids=[4754, 13766, 3147, 5917, 31436, 2129,
                                48310, 3147, 3715, 8934, 248046],
                      test_segments=[11], eos_tokens=[248044, 248046], vocab_size=248320)
        with self.assertRaisesRegex(ValueError, "partition"):
            audit.validate_trajectories(**common, read_segments=[1, 11])
        with self.assertRaisesRegex(ValueError, "EOS"):
            audit.validate_trajectories(**{**common, "read_finish_ids": [248046] + common["read_finish_ids"][1:]},
                                         read_segments=[1, 12])

    def test_rejects_bool_or_out_of_vocab_token_ids(self):
        tokens = [1, True]
        with self.assertRaisesRegex(ValueError, "token"):
            audit.validate_token_ids(tokens, expected_count=2, vocab_size=16)
        with self.assertRaisesRegex(ValueError, "token"):
            audit.validate_token_ids([1, 16], expected_count=2, vocab_size=16)


class FrontierAuditTests(unittest.TestCase):
    def test_exact_frontier_passes_and_single_bit_change_fails(self):
        raw = struct.pack("<4f", 1.0, -2.0, 0.0, 4.5)
        self.assertEqual(audit.compare_frontiers(raw, raw, "control"), 0.0)
        adjacent_f32 = struct.unpack("<f", struct.pack("<I", 0x40900001))[0]
        altered = struct.pack("<4f", 1.0, -2.0, 0.0, adjacent_f32)
        with self.assertRaisesRegex(AssertionError, "differs"):
            audit.compare_frontiers(altered, raw, "adversarial")

    def test_rejects_wrong_width_and_nonfinite_frontier(self):
        with self.assertRaisesRegex(ValueError, "width"):
            audit.validate_frontier_bytes(b"\0\0\0")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit.validate_frontier_bytes(struct.pack("<2f", 1.0, math.nan), vocab_size=2)

    def test_boundary_map_requires_all_four_sut_frontiers(self):
        ids = {name: b"raw-frontier" for name in ("root", "read_partial", "read_final", "test_final")}
        self.assertEqual(audit.validate_frontier_map(ids), ids)
        with self.assertRaisesRegex(ValueError, "frontier"):
            audit.validate_frontier_map({"root": "root", "read_final": "rf", "test_final": "tf"})
        with self.assertRaisesRegex(ValueError, "immutable bytes"):
            audit.validate_frontier_map({**ids, "read_partial": "text"})

    def test_store_lineage_roots_must_match_shared_source_root(self):
        self.assertTrue(audit.validate_lineage_roots("root", {
            "READ partial": "root", "READ final": "root", "TEST final": "root"}))
        with self.assertRaisesRegex(ValueError, "canonical source root"):
            audit.validate_lineage_roots("root", {"READ final": "read-final-snapshot"})

    def test_snapshot_summary_uses_requested_store_id_not_snapshot_object_id(self):
        row = {"snapshot_id": "a" * 64,
               "snapshot": {"kind": "snapshot", "parent": "b" * 64,
                            "controller": {"constraint": {"engine_id": "e",
                             "spec_sha256": "s", "tokenizer_sha256": "t",
                             "agent_state": "READ", "scope_start": 22},
                             "accepted_in_scope": 1, "status": "active",
                             "seen": {"1": 1}, "output_pending_hex": ""}},
               "payload_sha256": "c" * 64,
               "controller": {"constraint": {"engine_id": "e", "spec_sha256": "s",
                            "tokenizer_sha256": "t", "agent_state": "READ",
                            "scope_start": 22}, "accepted_in_scope": 1,
                            "status": "active", "seen": {"1": 1},
                            "output_pending_hex": ""}}
        summary = audit.snapshot_audit_summary(row)
        self.assertEqual(summary["snapshot"], "a" * 64)
        self.assertEqual(summary["parent"], "b" * 64)
        self.assertEqual(summary["payload_sha256"], "c" * 64)

    def test_independent_argmax_applies_grammar_mask_specials_and_accepting_eos(self):
        # Legal bits: token 0, token 2, forbidden special 3, and EOS 4.
        mask = bytes([0b00011101])
        logits = [1.0, 99.0, 1.0, 100.0, 50.0]
        selected, legal = audit.greedy_legal_argmax(
            logits, mask, special_ids={3, 4}, eos_tokens={4}, accepting=False)
        self.assertEqual(legal, [0, 2])
        self.assertEqual(selected, 0)  # ID-ascending tie break.
        selected, legal = audit.greedy_legal_argmax(
            logits, mask, special_ids={3, 4}, eos_tokens={4}, accepting=True)
        self.assertEqual(legal, [0, 2, 4])
        self.assertEqual(selected, 4)

    def test_independent_argmax_clears_only_out_of_vocab_mask_padding(self):
        raw = bytes([0b11100001]) + b"\xff\xff\xff\xff"
        selected, legal = audit.greedy_legal_argmax(
            [4.0, 9.0, 8.0, 7.0, 6.0], raw,
            special_ids=set(), eos_tokens=set(), accepting=False)
        self.assertEqual((selected, legal), (0, [0]))
        self.assertEqual(audit.normalize_grammar_mask(raw, 5), bytes([1]))
        with self.assertRaisesRegex(ValueError, "no legal"):
            audit.greedy_legal_argmax([0.0] * 5, bytes([0b00001000]),
                                      special_ids={3}, eos_tokens=set(), accepting=False)

    def test_pinned_llguidance_matcher_word_padding_is_normalized_without_changing_vocab_bits(self):
        base = Path("/home/dflanag3/spatial-artifacts/cli-grammar-probe-20261016-v1")
        runtime = base / "source-v3-green6"
        profile_path = base / "profile-v2/profile.json"
        registry_path = base / "runs/cli-grammar-q4-tiny-v2/campaign/grammar-registry.json"
        deps = Path("/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps")
        child = r'''import json
import sys
from pathlib import Path

script, runtime, profile_path, registry_path, deps = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
import run_cli_grammar_numerical_audit as audit

profile = json.loads(Path(profile_path).read_bytes())
registry, dependency = audit.load_pinned_grammar_registry(
    runtime, profile, registry_path, deps)
vocab = len(registry.engine._adapter.tokens)
initial = registry.start("READ", "mask-padding-regression", [], vocab)
session = registry.restore(initial, [])
raw = session._matcher.compute_bitmask()
normalized = audit.normalize_grammar_mask(raw, vocab)
print(json.dumps({
    "vocab": vocab,
    "raw_width": len(raw),
    "normalized_width": len(normalized),
    "raw_prefix_hex": raw[:len(normalized)].hex(),
    "normalized_hex": normalized.hex(),
    "native_sha256": dependency["llguidance_native_sha256"],
    "llguidance_version": dependency["llguidance_version"],
    "llguidance_path": dependency["llguidance_python_path"],
    "tokenizer_path": str(Path(sys.modules["tools.strata_tokenizer"].__file__).resolve()),
    "grammar_path": str(Path(sys.modules["serve.cow_grammar"].__file__).resolve()),
}))
'''
        before_path = tuple(sys.path)
        before_modules = set(sys.modules)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.run(
                [sys.executable, "-B", "-c", child,
                 str(Path(audit.__file__).resolve()), str(runtime), str(profile_path),
                 str(registry_path), str(deps)],
                cwd=temporary, env=env, capture_output=True, text=True, check=False,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        width = (audit.VOCAB_SIZE + 7) // 8
        self.assertEqual(result["vocab"], audit.VOCAB_SIZE)
        self.assertEqual(result["native_sha256"], audit.LLGUIDANCE_NATIVE_SHA256)
        self.assertEqual(result["llguidance_version"], "1.9.1")
        self.assertEqual(result["raw_width"], width + 4)  # one pinned matcher word pad
        self.assertEqual(result["normalized_width"], width)
        self.assertEqual(result["normalized_hex"], result["raw_prefix_hex"])
        self.assertTrue(Path(result["llguidance_path"]).is_relative_to(deps.resolve()))
        self.assertTrue(Path(result["tokenizer_path"]).is_relative_to(runtime.resolve()))
        self.assertTrue(Path(result["grammar_path"]).is_relative_to(runtime.resolve()))
        # The actual-dependency probe runs in a child so its sys.path and imports
        # cannot shadow modules used by the rest of a combined suite.
        self.assertEqual(tuple(sys.path), before_path)
        self.assertEqual(set(sys.modules), before_modules)

    def test_controller_progress_projection_rejects_wrong_parser_or_consumed_count(self):
        state = {"constraint": {"engine_id": "e", "spec_sha256": "s",
                   "tokenizer_sha256": "t", "agent_state": "READ", "scope_start": 2},
                 "accepted_in_scope": 1, "status": "active", "seen": {"7": 1},
                 "output_pending_hex": ""}
        audit.validate_controller_progress(dict(state, stream_key="different request"), state, "control")
        for changed in (
            {**state, "accepted_in_scope": 2},
            {**state, "constraint": {**state["constraint"], "agent_state": "TEST"}},
            {**state, "seen": {"8": 1}},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, "progress differs"):
                audit.validate_controller_progress(changed, state, "adversarial")

    def test_complete_controller_comparison_binds_rng_policy_and_cursor(self):
        state = {"schema": 2, "status": "active", "stream_key": "a" * 64,
                 "next_draw": 7, "policy": {"temperature": 0.0},
                 "constraint": {"agent_state": "READ"}}
        digest = audit.validate_controller_equal(state, dict(state), "same")
        self.assertEqual(len(digest), 64)
        for changed in (
            {**state, "next_draw": 8},
            {**state, "stream_key": "b" * 64},
            {**state, "policy": {"temperature": 0.5}},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, "complete canonical"):
                audit.validate_controller_equal(state, changed, "changed")

    def test_campaign_stream_is_bound_to_initial_generation_operation(self):
        campaign = {"commands": [
            {"label": "read-partial", "argv_redacted": ["answer", "--operation-id", "read-op"]},
            {"label": "read-finish", "argv_redacted": ["continue", "--operation-id", "continue-op"]},
        ]}
        self.assertEqual(audit.campaign_operation_id(campaign, "read-partial"), "read-op")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit.campaign_operation_id(campaign, "missing")

    def test_repeated_raw_trajectory_requires_every_frontier_schedule_and_work_match(self):
        one = struct.pack("<f", 1.0)
        two = struct.pack("<f", 2.0)
        run = {"prompt_frontier": one, "final_prefill_frontier": one,
               "decode_frontiers": [one, two], "segment_frontiers": [two],
               "schedule": {"segments": [2]},
               "work": {"consumed": 4, "prompt_consumed": 2, "decode_tokens": 2,
                        "normal_forward_tokens": 2, "normal_decode_windows": 2,
                        "stopping_reason": "eos", "prefill_tokens": 2,
                        "prefill_chunks": 1, "generated": [7, 9],
                        "prefill_seconds": 0.1, "decode_seconds": 0.2}}
        second = {**run, "work": {**run["work"],
                                  "prefill_seconds": 0.3, "decode_seconds": 0.4}}
        comparisons = audit.compare_raw_trajectories(run, second, "READ")
        self.assertEqual(len(comparisons), 5)
        self.assertEqual(max(row["max_abs_error"] for row in comparisons), 0.0)
        changed = {**run, "decode_frontiers": [one, one]}
        with self.assertRaisesRegex(AssertionError, "differs"):
            audit.compare_raw_trajectories(run, changed, "READ adversarial")
        for field, value in (("consumed", 5), ("prompt_consumed", 3),
                             ("decode_tokens", 3), ("normal_forward_tokens", 3),
                             ("normal_decode_windows", 3), ("stopping_reason", "limit"),
                             ("prefill_tokens", 3), ("prefill_chunks", 2),
                             ("generated", [7, 10])):
            altered = {**run, "work": {**run["work"], field: value}}
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, "work receipt"):
                audit.compare_raw_trajectories(run, altered, "READ deterministic work")

    def test_raw_work_receipt_requires_deterministic_fields_and_finite_timings(self):
        work = {"consumed": 4, "prompt_consumed": 2, "decode_tokens": 2,
                "normal_forward_tokens": 2, "normal_decode_windows": 2,
                "stopping_reason": "eos", "prefill_tokens": 2,
                "prefill_chunks": 1, "generated": [7, 9],
                "prefill_seconds": 0.1, "decode_seconds": 0.2}
        audit.validate_raw_work_receipt(work, expected_generated=[7, 9])
        for altered in ({**work, "consumed": True}, {**work, "generated": [7]},
                        {**work, "decode_seconds": math.inf},
                        {key: value for key, value in work.items() if key != "stopping_reason"}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                audit.validate_raw_work_receipt(altered, expected_generated=[7, 9])

    def test_controller_receipt_hashes_full_state_without_copying_seen_map(self):
        state = {"constraint": {"engine_id": "e", "spec_sha256": "s",
                   "tokenizer_sha256": "t", "agent_state": "READ", "scope_start": 2},
                 "accepted_in_scope": 2, "status": "active",
                 "seen": {str(i): 1 for i in range(1000)}, "output_pending_hex": ""}
        receipt = audit.controller_audit_receipt(state)
        changed = {**state, "seen": {**state["seen"], "1000": 1}}
        changed_receipt = audit.controller_audit_receipt(changed)
        self.assertEqual(receipt["seen_token_count"], 1000)
        self.assertEqual(len(receipt["controller_sha256"]), 64)
        self.assertNotIn("seen", receipt["progress"])
        self.assertNotEqual(receipt["controller_sha256"],
                            changed_receipt["controller_sha256"])

    def test_non_grammar_root_controller_receipt_is_compact_and_hash_bound(self):
        root_controller = {"schema": 2, "status": "active", "seen": {"12": 2},
                           "output_pending_hex": ""}
        receipt = audit.controller_audit_receipt(root_controller)
        self.assertIsNone(receipt["progress"]["constraint"])
        self.assertEqual(receipt["seen_token_count"], 1)


class ProvenanceAndSourceTests(unittest.TestCase):
    def test_refusing_existing_audit_output_never_rewrites_prior_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "already-used"
            output_dir.mkdir()
            progress_path = output_dir / "audit-progress.json"
            original = b'{"status":"PASS","phase":"complete","audit_run_id":"prior"}\n'
            progress_path.write_bytes(original)
            for name in ("campaign.json", "profile.json", "registry.json", "oracle",
                         "build.json"):
                (root / name).write_bytes(b"placeholder")
            (root / "deps").mkdir()
            (root / "runtime").mkdir()
            argv = ["audit", "--campaign", str(root / "campaign.json"),
                    "--profile", str(root / "profile.json"), "--registry", str(root / "registry.json"),
                    "--grammar-deps", str(root / "deps"), "--runtime-root", str(root / "runtime"),
                    "--oracle-binary", str(root / "oracle"), "--build-receipt", str(root / "build.json"),
                    "--output-dir", str(output_dir)]
            with self.assertRaises(FileExistsError):
                audit.main(argv)
            self.assertEqual(progress_path.read_bytes(), original)

    def test_audit_failure_writes_terminal_progress_phase_and_cause(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "audit-output"
            argv = ["audit", "--campaign", "campaign.json", "--profile", "profile.json",
                    "--registry", "registry.json", "--grammar-deps", "deps",
                    "--runtime-root", "runtime", "--oracle-binary", "oracle",
                    "--build-receipt", "build.json", "--output-dir", str(output_dir)]

            def fail_after_claim(args, *, audit_run_id):
                claimed = audit._claim_audit_output(args.output_dir, audit_run_id)
                audit._atomic_json(claimed / "audit-progress.json",
                                   {"audit_run_id": audit_run_id, "status": "RUNNING",
                                    "phase": "read-1", "completed_raw_processes": 1})
                raise ValueError("mask width mismatch")

            with mock.patch.object(audit, "run_audit", side_effect=fail_after_claim):
                with self.assertRaisesRegex(ValueError, "mask width mismatch"):
                    audit.main(argv)
            result = json.loads((output_dir / "audit-progress.json").read_bytes())
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(result["failed_phase"], "read-1")
        self.assertEqual(result["completed_raw_processes"], 1)
        self.assertEqual(result["failure_type"], "ValueError")
        self.assertEqual(result["failure"], "mask width mismatch")

    def test_grammar_preflight_requires_explicit_dependency_path(self):
        for mode in ("store-preflight", "audit"):
            with self.subTest(mode=mode), self.assertRaises(SystemExit):
                audit._parser().parse_args([mode])

    def test_pinned_registry_loader_rejects_wrong_native_dependency_hash_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deps = root / "deps"
            native = deps / "llguidance" / "_lib.abi3.so"
            native.parent.mkdir(parents=True)
            native.write_bytes(b"not the pinned extension")
            registry_path = root / "registry.json"
            registry_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(audit, "sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "native extension"):
                    audit.load_pinned_grammar_registry(root, {}, registry_path, deps)

    def test_manifest_input_hashes_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.cpp"
            path.write_bytes(b"pinned source\n")
            rows = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
            self.assertEqual(audit.validate_input_rows(rows, expected_count=1), rows)
            path.write_bytes(b"corrupted source\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                audit.validate_input_rows(rows, expected_count=1)

    def test_final_prefill_file_uses_exact_environment_path_without_double_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "read.f32"
            frontier = bytes(audit.FRONTIER_WIDTH)
            path.write_bytes(frontier)
            self.assertEqual(audit.read_final_prefill_frontier(path, frontier), frontier)
            with self.assertRaisesRegex(ValueError, "frontier differs"):
                audit.read_final_prefill_frontier(path, b"different")

    def test_rejects_wrong_or_incomplete_53_input_provenance(self):
        with self.assertRaisesRegex(ValueError, "53"):
            audit.validate_input_rows([], expected_count=53)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit.validate_build_provenance({"status": "BUILT_NOT_EXECUTED",
                "runtime_executed": False, "gpu_launched": False,
                "verified_input_count": 52, "verified_input_mismatches": []}, 53)

    def test_forced_oracle_transform_has_unique_guarded_anchors(self):
        source = """#include <iostream>\n#include <vector>\nint main() {\n"""
        with self.assertRaisesRegex(ValueError, "source anchor"):
            audit.instrument_forced_tokens(source)

    def test_forced_oracle_transform_rejects_corrupt_or_already_modified_source(self):
        source = audit.FORCED_TOKEN_SOURCE_FIXTURE
        transformed = audit.instrument_forced_tokens(source)
        self.assertIn("STRATA_FORCE_GENERATED_TOKENS", transformed)
        self.assertIn("forced_tokens[static_cast<size_t>(i)]", transformed)
        self.assertNotIn("BranchSession", transformed)
        with self.assertRaisesRegex(ValueError, "anchor"):
            audit.instrument_forced_tokens(source.replace("logits[t] > logits[token]", "corrupt"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
