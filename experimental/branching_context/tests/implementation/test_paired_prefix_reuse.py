"""CPU-only timing-table validation tests; these do not qualify a GPU path."""
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "paired_prefix_reuse.py"
SPEC = importlib.util.spec_from_file_location("paired_prefix_reuse", SCRIPT)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def rows():
    result = []
    for phase, count in (("warmup", 1), ("measured", 3)):
        for repeat in range(count):
            for suffix in ("A", "B"):
                base = 1.0 + repeat + (suffix == "B")
                result.append({"phase": phase, "repeat": repeat, "suffix": suffix,
                    "actual_output_tokens": 8, "suffix_tokens": 512,
                    "forward_tokens": 520, "loaded_file_bytes": 0,
                    "submit_to_first_token_seconds": base,
                    "worker_to_first_token_seconds": base / 2,
                    "median_inter_token_seconds": 0.01,
                    "submit_to_durable_seconds": base + 1,
                    "generation_seconds": base / 2,
                    "request_seconds": base,
                    "capture_seconds": 0.1})
    return result


class PairedPrefixMetricTests(unittest.TestCase):
    def test_summary_requires_one_warmup_and_three_measured_a_b_pairs(self):
        summary = HARNESS.summarize_arm(rows())
        self.assertEqual(summary["measured_request_count"], 6)
        self.assertEqual(summary["by_suffix"]["A"]["request_count"], 3)
        self.assertEqual(summary["by_suffix"]["B"]["request_count"], 3)
        self.assertEqual(summary["by_suffix"]["A"]["medians_seconds"]["submit_to_first_token_seconds"], 2.0)

    def test_summary_rejects_missing_or_duplicate_measured_pair(self):
        missing = rows()
        missing.pop()
        with self.assertRaisesRegex(ValueError, "missing request records"):
            HARNESS.summarize_arm(missing)
        duplicate = rows()
        duplicate.append(dict(duplicate[1]))
        with self.assertRaisesRegex(ValueError, "duplicate request record"):
            HARNESS.summarize_arm(duplicate)

    def test_summary_rejects_nonresident_work_or_inconsistent_forward_count(self):
        nonresident = rows()
        nonresident[1]["loaded_file_bytes"] = 4096
        with self.assertRaisesRegex(ValueError, "resident source path"):
            HARNESS.summarize_arm(nonresident)
        wrong_work = rows()
        wrong_work[1]["forward_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "suffix plus actual output"):
            HARNESS.summarize_arm(wrong_work)

    def test_summary_rejects_output_count_and_invalid_timing(self):
        zero_output = rows()
        zero_output[1]["actual_output_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "no actual generated tokens"):
            HARNESS.summarize_arm(zero_output)
        bad_time = rows()
        bad_time[1]["submit_to_durable_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid submit_to_durable_seconds"):
            HARNESS.summarize_arm(bad_time)

    def test_summary_rejects_negative_or_nonfinite_inter_token_intervals(self):
        for value in (-0.1, float("inf"), float("nan")):
            bad = rows()
            bad[1]["inter_token_intervals_seconds"] = [0.01, value]
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid inter-token interval"):
                HARNESS.summarize_arm(bad)

    def test_request_wide_backend_counters_require_same_complete_integer_schema(self):
        before = {"forward_tokens": 100, "loaded_file_bytes": 0, "checkpoint_import_bytes": 0}
        after = {"forward_tokens": 620, "loaded_file_bytes": 0, "checkpoint_import_bytes": 0}
        delta = HARNESS._backend_counter_delta(before, after)
        self.assertEqual(delta["numeric_deltas"]["forward_tokens"], 520)
        self.assertEqual(delta["numeric_deltas"]["loaded_file_bytes"], 0)
        for invalid_before, invalid_after in (
                (before, {**after, "new_counter": 0}),
                ({"forward_tokens": 100}, {"forward_tokens": 620}),
                ({**before, "forward_tokens": True}, {**after, "forward_tokens": 620})):
            with self.subTest(before=invalid_before), self.assertRaises(AssertionError):
                HARNESS._backend_counter_delta(invalid_before, invalid_after)

    def test_request_wide_stats_preserve_text_diagnostics_separately(self):
        before = {"forward_tokens": 100, "loaded_file_bytes": 0,
                  "vmm_last_failed_operation": "", "host_heap_allocated_bytes": None}
        after = {"forward_tokens": 620, "loaded_file_bytes": 0,
                 "vmm_last_failed_operation": "", "host_heap_allocated_bytes": 4000}
        snapshot = HARNESS._backend_counter_delta(before, after)
        self.assertEqual(snapshot["numeric_deltas"], {"forward_tokens": 520, "loaded_file_bytes": 0})
        self.assertEqual(snapshot["diagnostics_before"], {
            "vmm_last_failed_operation": "", "host_heap_allocated_bytes": None})
        self.assertEqual(snapshot["diagnostics_after"], {
            "vmm_last_failed_operation": "", "host_heap_allocated_bytes": 4000})

    def test_numeric_backend_fields_cannot_fall_back_to_diagnostics(self):
        before = {"forward_tokens": 100, "loaded_file_bytes": 0,
                  "checkpoint_import_bytes": 10}
        invalid = (
            {"forward_tokens": 620, "loaded_file_bytes": 0, "checkpoint_import_bytes": None},
            {"forward_tokens": 620, "loaded_file_bytes": 0, "checkpoint_import_bytes": "10"},
            {"forward_tokens": 620, "loaded_file_bytes": float("nan"), "checkpoint_import_bytes": 11},
            {"forward_tokens": "620", "loaded_file_bytes": 0, "checkpoint_import_bytes": 11},
        )
        for after in invalid:
            with self.subTest(after=after), self.assertRaisesRegex(AssertionError, "malformed"):
                HARNESS._backend_counter_delta(before, after)

    def test_coordinator_single_gpu_metrics_use_integer_worker_keys(self):
        valid = {"active_per_gpu": {0: 0}, "max_active_per_gpu": {0: 1}}
        HARNESS._validate_single_gpu_metrics(valid, 0)
        for invalid in (
                {"active_per_gpu": {0: 0}, "max_active_per_gpu": {"0": 1}},
                {"active_per_gpu": {1: 0}, "max_active_per_gpu": {1: 1}},
                {"active_per_gpu": {0: 0, 1: 0}, "max_active_per_gpu": {0: 1, 1: 1}},
                {"active_per_gpu": {0: 1}, "max_active_per_gpu": {0: 1}},
                {"active_per_gpu": {0: 0}, "max_active_per_gpu": {0: 2}},
        ):
            with self.subTest(metrics=invalid), self.assertRaises(AssertionError):
                HARNESS._validate_single_gpu_metrics(invalid, 0)

    def test_runtime_imports_must_be_rooted_and_match_both_profile_closures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            api = root / "api"
            api.mkdir()
            source = api / "module.py"
            source.write_text("pinned source\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            module = SimpleNamespace(__file__=str(source))
            profile = {"native_build": {"selected_runtime_source_closure": {
                "files_sha256": {str(source.resolve()): digest}}}}
            observed = HARNESS.validate_runtime_sources({"sample": module}, [root, api],
                                                         [profile, profile])
            self.assertEqual(observed["sample"]["sha256"], digest)

            stale = {"native_build": {"selected_runtime_source_closure": {
                "files_sha256": {str(source.resolve()): "0" * 64}}}}
            with self.assertRaisesRegex(ValueError, "absent or mismatched"):
                HARNESS.validate_runtime_sources({"sample": module}, [root, api],
                                                  [profile, stale])
            with self.assertRaisesRegex(ValueError, "outside explicit"):
                HARNESS.validate_runtime_sources({"sample": module}, [root / "elsewhere"],
                                                  [profile, profile])
            with self.assertRaisesRegex(ValueError, "files_sha256"):
                HARNESS.validate_runtime_sources({"sample": module}, [root, api],
                    [{"native_build": {}}, profile])

    def test_runtime_inventory_explicitly_imports_lazy_receipt_module(self):
        # A populated runtime dictionary does not imply every package module is
        # already in sys.modules; execution_receipt is imported lazily by some
        # coordinator paths and must be loaded before its source is pinned.
        fake_module = SimpleNamespace(__file__=__file__)
        fake_rt = {
            "ContextCoordinator": SimpleNamespace(__module__="context_time_travel.coordinator"),
            "controller": fake_module,
            "native_probe": fake_module,
            "get_state": SimpleNamespace(__module__="pool_campaign"),
        }
        with patch.object(HARNESS.importlib, "import_module", return_value=fake_module) as importer:
            inventory = HARNESS._runtime_module_inventory(fake_rt)
        self.assertIn("execution_receipt", inventory)
        imported = [call.args[0] for call in importer.call_args_list]
        self.assertIn("context_time_travel.execution_receipt", imported)


if __name__ == "__main__":
    unittest.main()
