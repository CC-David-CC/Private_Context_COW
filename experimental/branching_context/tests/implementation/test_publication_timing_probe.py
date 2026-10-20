"""CPU contract tests for test-only checkpoint publication timing wrappers."""
from pathlib import Path
import importlib.util
import json
import hashlib
import stat
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publication_timing_probe.py"
SPEC = importlib.util.spec_from_file_location("publication_timing_probe_test", SCRIPT) if SCRIPT.is_file() else None
PROBE = importlib.util.module_from_spec(SPEC) if SPEC and SPEC.loader else None
if SPEC is not None and SPEC.loader is not None:
    SPEC.loader.exec_module(PROBE)
RUNNER_PATH = SCRIPT.with_name("run_cli_publication_timing_probe.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("publication_timing_runner_test", RUNNER_PATH) if RUNNER_PATH.is_file() else None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC) if RUNNER_SPEC and RUNNER_SPEC.loader else None
if RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None:
    RUNNER_SPEC.loader.exec_module(RUNNER)


class TimingWrapperTests(unittest.TestCase):
    def probe(self):
        if PROBE is None:
            self.fail("publication timing wrapper module missing (expected initial RED)")
        return PROBE

    def make_modules(self, fake_os):
        class Model:
            def prepare_checkpoint(self, snapshot):
                return types.SimpleNamespace(snapshot=snapshot)
            def _prepared_envelope(self, snapshot, checkpoint, history):
                return b"envelope"
            def _publish_checkpoint_parts(self, destination, parts):
                fake_os.fsync(1)
                fake_os.replace("stage", destination)
                return "payload-ok"
            def _publish_checkpoint_file(self, destination, data):
                return "manifest-ok"
            def validate_checkpoint(self, snapshot):
                return {"snapshot": snapshot}
            def publish_prepared_checkpoint(self, prepared):
                self._prepared_envelope(prepared.snapshot, None, ())
                self._publish_checkpoint_parts("payload", (b"x",))
                self._publish_checkpoint_file("manifest", b"x")
                self.validate_checkpoint(prepared.snapshot)
                return {"ok": prepared.snapshot}
            def save_prepared_branch(self, name, revision, prepared, operation):
                return {"receipt": prepared.snapshot}

        class Coordinator:
            def _run(self, model, job):
                return {"snapshot": job.snapshot}
            def _publish(self, model, job, prepared, key, pin, lease):
                return model.publish_prepared_checkpoint(prepared)

        class Request:
            def emit(self, event, **fields):
                return (event, fields.get("receipt"))

        class Store:
            def move(self, name, revision, snapshot, operation):
                return {"tip": snapshot, "revision": revision + 1}

        class Pool:
            def reserve(self, size, *, timeout=0, payload_sha256=None):
                return _Context("reservation", size)

        class SharedCheckpointPool(Pool):
            pass
        class DiskCheckpointPool(Pool):
            pass

        real = types.SimpleNamespace(BranchingModel=Model, os=fake_os,
                                     _sync_dir=lambda path: None)
        coordinator = types.SimpleNamespace(ContextCoordinator=Coordinator, Request=Request)
        store = types.SimpleNamespace(Store=Store, os=fake_os)
        pools = types.SimpleNamespace(SharedCheckpointPool=SharedCheckpointPool,
                                      DiskCheckpointPool=DiskCheckpointPool)
        return real, coordinator, store, pools, Model, Coordinator, Request, Store, SharedCheckpointPool

    def test_wrappers_preserve_nested_results_and_redact_correlation_values(self):
        p = self.probe()
        snapshot_id = "already-content-addressed-snapshot-id"
        fake_os = _FakeOS()
        real, coordinator, store, pools, Model, Coordinator, Request, Store, _ = self.make_modules(fake_os)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            recorder = p.install_timing_wrappers(real, coordinator, store, pools, output)
            model = Model()
            prepared = model.prepare_checkpoint(snapshot_id)
            result = Coordinator()._publish(model, types.SimpleNamespace(snapshot=snapshot_id),
                                            prepared, None, None, None)
            self.assertEqual(result, {"ok": snapshot_id})
            moved = Store().move("branch", 3, snapshot_id, "operation-secret")
            self.assertEqual(moved["revision"], 4)
            request = Request()
            request.emit("snapshot", snapshot=snapshot_id)
            request.emit("durable", receipt={"snapshot": snapshot_id})
            recorder.close()
            status = json.loads(output.with_suffix(".status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["instrumentation_error"], False)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            phases = [row["phase"] for row in rows]
            self.assertIn("worker.prepare_checkpoint", phases)
            self.assertIn("writer.publish_total", phases)
            self.assertIn("checkpoint.envelope_validation", phases)
            self.assertIn("store.move_receipt", phases)
            self.assertIn("request.snapshot_emit", phases)
            self.assertIn("request.durable_emit", phases)
            self.assertTrue(all("operation-secret" not in str(row) for row in rows))
            self.assertTrue(all(row["correlation_id"] == snapshot_id for row in rows))
            for row in rows:
                self.assertGreaterEqual(row["end_monotonic_ns"], row["start_monotonic_ns"])
                self.assertGreaterEqual(row["depth"], 0)

    def test_context_reservation_records_wait_and_hold_and_preserves_exceptions(self):
        p = self.probe()
        fake_os = _FakeOS()
        real, coordinator, store, pools, Model, _, _, _, SharedPool = self.make_modules(fake_os)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            recorder = p.install_timing_wrappers(real, coordinator, store, pools, output)
            pool = SharedPool()
            with pool.reserve(19) as value:
                self.assertEqual(value, ("reservation", 19))
            with self.assertRaisesRegex(ValueError, "inner failure"):
                with pool.reserve(5):
                    raise ValueError("inner failure")
            recorder.close()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            events = [row for row in rows if row["phase"] == "memory_pool.reserve"]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(type(row["wait_ns"]) is int and row["wait_ns"] >= 0 for row in events))
            self.assertTrue(all(type(row["held_ns"]) is int and row["held_ns"] >= 0 for row in events))
            self.assertEqual(events[-1]["outcome"], "exception:ValueError")

    def test_filesystem_wrappers_measure_only_inside_publication_scope(self):
        p = self.probe()
        fake_os = _FakeOS()
        real, coordinator, store, pools, Model, _, _, _, _ = self.make_modules(fake_os)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            recorder = p.install_timing_wrappers(real, coordinator, store, pools, output)
            self.assertEqual(Model().publish_prepared_checkpoint(
                types.SimpleNamespace(snapshot="snapshot-id")), {"ok": "snapshot-id"})
            fake_os.fsync(1)  # outside a publication span: deliberately not recorded
            recorder.close()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            phases = [row["phase"] for row in rows]
            self.assertIn("checkpoint.payload_write_fsync_rename_dirsync", phases)
            self.assertIn("filesystem.fsync_file", phases)
            self.assertIn("filesystem.atomic_replace", phases)
            self.assertEqual(phases.count("filesystem.fsync_file"), 1)

    def test_timeline_joins_observer_and_server_monotonic_boundaries(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        import base64
        snapshot = "snapshot-id"
        phases = [
            ("worker.prepare_checkpoint", 100, 200),
            ("request.snapshot_emit", 220, 600),
            ("writer.publish_total", 610, 870),
            ("checkpoint.publish_prepared_checkpoint", 620, 860),
            ("checkpoint.payload_write_fsync_rename_dirsync", 700, 720),
            ("checkpoint.manifest_write_fsync_rename_dirsync", 721, 730),
            ("checkpoint.saved_validation", 740, 760),
            ("store.move_receipt", 770, 780),
            ("request.durable_emit", 800, 850),
        ]
        rows = [{"phase": label, "correlation_id": snapshot, "span_id": index,
                 "thread_id": 1, "depth": 0, "start_monotonic_ns": start,
                 "end_monotonic_ns": end, "duration_ns": end-start, "outcome": "ok"}
                for index, (label, start, end) in enumerate(phases, 1)]
        observer = {"events": []}
        for stamp, event in ((400, {"event": "text", "token_id": 11, "text": "A"}),
                             (410, {"event": "text", "token_id": 12, "text": "B"}),
                             (500, {"event": "snapshot", "snapshot": snapshot, "tokens": [11, 12]}),
                             (820, {"event": "durable", "receipt": {"snapshot": snapshot}}),
                             (830, {"event": "complete", "answer": {"tokens": [11, 12], "text": "AB"},
                                    "receipt": {"snapshot": snapshot}})):
            observer["events"].append({"arrival_monotonic_ns": stamp,
                "raw_line_b64": base64.b64encode(json.dumps(event).encode()).decode()})
        timeline = RUNNER.build_question_timeline(rows, snapshot=snapshot,
                                                   observer_request=observer,
                                                   expected_token_ids=[11, 12],
                                                   expected_answer={"tokens": [11, 12], "text": "AB"},
                                                   expected_receipt={"snapshot": snapshot})
        self.assertEqual(timeline["prepare_start_minus_last_token_arrival_ns"], -310)
        self.assertEqual(timeline["client_durable_arrival_minus_server_emit_end_ns"], -30)
        self.assertEqual(timeline["writer_publish_duration_ns"], 260)
        self.assertEqual(timeline["text_event_token_ids"], [11, 12])
        self.assertEqual(timeline["text_event_count"], 2)
        with self.assertRaises(RUNNER.TimingProbeError):
            RUNNER.build_question_timeline(rows[:-2], snapshot=snapshot,
                                           observer_request=observer,
                                           expected_token_ids=[11, 12],
                                           expected_answer={"tokens": [11, 12], "text": "AB"},
                                           expected_receipt={"snapshot": snapshot})

    def test_text_events_must_match_the_canonical_answer_tokens_and_text(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        import base64
        snapshot = "snapshot-id"
        phases = [
            ("worker.prepare_checkpoint", 100, 200),
            ("request.snapshot_emit", 220, 600),
            ("writer.publish_total", 610, 870),
            ("checkpoint.publish_prepared_checkpoint", 620, 860),
            ("checkpoint.payload_write_fsync_rename_dirsync", 700, 720),
            ("checkpoint.manifest_write_fsync_rename_dirsync", 721, 730),
            ("checkpoint.saved_validation", 740, 760),
            ("store.move_receipt", 770, 780),
            ("request.durable_emit", 800, 850),
        ]
        rows = [{"phase": label, "correlation_id": snapshot, "span_id": index,
                 "thread_id": 1, "depth": 0, "start_monotonic_ns": start,
                 "end_monotonic_ns": end, "duration_ns": end-start, "outcome": "ok"}
                for index, (label, start, end) in enumerate(phases, 1)]
        events = [
            (400, {"event": "text", "token_id": 7, "text": "hé"}),
            (410, {"event": "text", "token_id": 8, "text": "🙂"}),
            (500, {"event": "snapshot", "snapshot": snapshot, "tokens": [7, 8]}),
            (820, {"event": "durable", "receipt": {"snapshot": snapshot}}),
            (830, {"event": "complete", "answer": {"tokens": [7, 8], "text": "hé🙂"},
                   "receipt": {"snapshot": snapshot}}),
        ]
        observer = {"events": [{"arrival_monotonic_ns": stamp,
                    "raw_line_b64": base64.b64encode(json.dumps(event, ensure_ascii=False).encode()).decode()}
                    for stamp, event in events]}
        expected_answer = {"tokens": [7, 8], "text": "hé🙂"}
        expected_receipt = {"snapshot": snapshot}
        valid = RUNNER.build_question_timeline(rows, snapshot=snapshot,
            observer_request=observer, expected_token_ids=[7, 8],
            expected_answer=expected_answer, expected_receipt=expected_receipt)
        self.assertEqual(valid["text_event_token_ids"], [7, 8])
        self.assertEqual(valid["text_event_text_sha256"],
                         __import__("hashlib").sha256("hé🙂".encode()).hexdigest())
        for bad_observer in (
            {"events": observer["events"][:-3] + [
                {"arrival_monotonic_ns": 410, "raw_line_b64": base64.b64encode(
                    json.dumps({"event": "text", "token_id": True, "text": "🙂"}).encode()).decode()},
                *observer["events"][-3:]]},
            {"events": observer["events"][:-3] + [
                {"arrival_monotonic_ns": 410, "raw_line_b64": base64.b64encode(
                    json.dumps({"event": "text", "token_id": 9, "text": "🙂"}).encode()).decode()},
                *observer["events"][-3:]]},
        ):
            with self.subTest(bad_observer=bad_observer):
                with self.assertRaises(RUNNER.TimingProbeError):
                    RUNNER.build_question_timeline(rows, snapshot=snapshot,
                        observer_request=bad_observer, expected_token_ids=[7, 8],
                        expected_answer=expected_answer, expected_receipt=expected_receipt)

    def test_offline_summarizer_writes_new_output_without_running_campaign(self):
        if RUNNER is None or not hasattr(RUNNER, "summarize_existing"):
            self.fail("offline summarizer missing (expected RED before implementation)")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "campaign-result.json"
            phases = root / "phases.jsonl"
            status = root / "status.json"
            baseline = root / "baseline.json"
            out = root / "new-summary"
            instrumented_store = root / "instrumented-store"
            baseline_store = root / "baseline-store"
            snapshot_ids = ["root", "snapshot-1", "snapshot-2", "snapshot-3"]

            def write_store(store):
                (store / "objects").mkdir(parents=True)
                (store / "checkpoints").mkdir()
                for snapshot_id in snapshot_ids:
                    controller = {"status": "idle", "cursor": snapshot_ids.index(snapshot_id)}
                    object_record = {"kind": "snapshot", "schema": 1, "parent": None,
                                     "model_contract_id": "model", "inputs": [],
                                     "controller": controller}
                    object_bytes = json.dumps(object_record, sort_keys=True, separators=(",", ":")).encode()
                    (store / "objects" / f"{snapshot_id}.json").write_bytes(object_bytes)
                    state = f"state-{snapshot_id}".encode()
                    payload_hash = hashlib.sha256(state).hexdigest()
                    (store / "checkpoints" / f"{payload_hash}.state").write_bytes(state)
                    manifest = {"checksum": "checksum", "manifest": {"snapshot": snapshot_id,
                        "payload_sha256": payload_hash, "state_bytes": len(state)}}
                    (store / "checkpoints" / f"{snapshot_id}.json").write_text(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                root_manifest = store / "checkpoints" / "root.json"
                root_data = json.loads(root_manifest.read_text(encoding="utf-8"))
                root_payload = root_data["manifest"]["payload_sha256"]
                root_state = store / "checkpoints" / f"{root_payload}.state"
                return {"snapshot": "root", "payload_sha256": root_payload,
                    "files": {
                        root_manifest.name: {"bytes": root_manifest.stat().st_size,
                            "mode": stat.S_IMODE(root_manifest.stat().st_mode),
                            "sha256": hashlib.sha256(root_manifest.read_bytes()).hexdigest(),
                            "inode": root_manifest.stat().st_ino},
                        root_state.name: {"bytes": root_state.stat().st_size,
                            "mode": stat.S_IMODE(root_state.stat().st_mode),
                            "sha256": root_payload, "inode": root_state.stat().st_ino}}}

            instrumented_fingerprint = write_store(instrumented_store)
            baseline_fingerprint = write_store(baseline_store)
            import base64
            phase_rows = []
            questions = []
            for number in range(1, 4):
                snapshot = f"snapshot-{number}"
                receipt = {"snapshot": snapshot}
                observer_events = []
                for stamp, event in (
                    (400, {"event": "text", "token_id": 11, "text": "A"}),
                    (410, {"event": "text", "token_id": 12, "text": "B"}),
                    (500, {"event": "snapshot", "snapshot": snapshot, "tokens": [11, 12]}),
                    (820, {"event": "durable", "receipt": receipt}),
                    (830, {"event": "complete", "answer": {"tokens": [11, 12], "text": "AB"},
                           "receipt": receipt}),
                ):
                    observer_events.append({"arrival_monotonic_ns": stamp,
                        "raw_line_b64": base64.b64encode(json.dumps(event).encode()).decode()})
                base = (number - 1) * 1000
                for index, (phase, begin, end) in enumerate((
                    ("worker.prepare_checkpoint", 100, 200),
                    ("request.snapshot_emit", 220, 600),
                    ("writer.publish_total", 610, 870),
                    ("checkpoint.publish_prepared_checkpoint", 620, 860),
                    ("checkpoint.payload_write_fsync_rename_dirsync", 700, 720),
                    ("checkpoint.manifest_write_fsync_rename_dirsync", 721, 730),
                    ("checkpoint.saved_validation", 740, 760),
                    ("store.move_receipt", 770, 780),
                    ("request.durable_emit", 800, 850),
                ), 1):
                    phase_rows.append({"phase": phase, "correlation_id": snapshot,
                        "span_id": base + index, "thread_id": 1, "depth": 0,
                        "start_monotonic_ns": base + begin, "end_monotonic_ns": base + end,
                        "duration_ns": end - begin, "outcome": "ok"})
                answer = {"tokens": [11, 12], "text": "AB"}
                questions.append({"index": number, "question_sha256": "q" * 64,
                    "source_snapshot": "root", "snapshot": snapshot, "output_count": 2,
                    "output_token_ids": [11, 12], "machine_answer_sha256": "a" * 64,
                    "controller_exact": True, "completion": {"actual_output_tokens": 2},
                    "receipt": receipt, "lineage": ["root", snapshot],
                    "inactive_sibling_tips_unchanged": True,
                    "work": {"forward_tokens": 2, "old_prefix_forward_tokens": 0},
                    "request": {"answer": answer, "receipt": receipt,
                                "machine_envelope": {"answer": {
                                    "tokens": [11, 12], "text": "AB",
                                    "capture_seconds": 1.0, "generation_seconds": 2.0,
                                    "request_seconds": 3.0,
                                    "counter_deltas": {
                                        "decode_compute_ns": 10, "decode_finish_ns": 11,
                                        "decode_prepare_ns": 12, "integrity_plan_upload_ns": 13,
                                        "integrity_seal_compare_ns": 14,
                                        "integrity_seal_result_download_ns": 15,
                                        "host_heap_allocated_bytes": 16}}},
                                "observation": {"requests": [{"events": observer_events}]}}})
            campaign_record = {"status": "PASS", "execution_profile_id": "profile",
                "model_contract_id": "model", "root_snapshot": "root",
                "root_payload_sha256": instrumented_fingerprint["payload_sha256"], "root_token_count": 100,
                "final_root_fingerprint": instrumented_fingerprint, "context_file_sha256": "c" * 64,
                "fixture_root_text_sha256": "t" * 64, "total_actual_output_tokens": 6,
                "root_stability": {"root_payload_sha256_unchanged": True,
                    "root_tip_unchanged": True, "sibling_tips_unchanged": True},
                "questions": questions}
            result.write_text(json.dumps(campaign_record), encoding="utf-8")
            baseline_record = dict(campaign_record)
            baseline_record["final_root_fingerprint"] = baseline_fingerprint
            baseline_record["root_payload_sha256"] = baseline_fingerprint["payload_sha256"]
            baseline_record["questions"] = json.loads(json.dumps(questions))
            for row in baseline_record["questions"]:
                answer = row["request"]["machine_envelope"]["answer"]
                answer["capture_seconds"] += 0.25
                answer["generation_seconds"] += 0.5
                answer["request_seconds"] += 0.75
                for name in answer["counter_deltas"]:
                    answer["counter_deltas"][name] += 1
                row["machine_answer_sha256"] = "b" * 64
            baseline.write_text(json.dumps(baseline_record), encoding="utf-8")
            phases.write_text("".join(json.dumps(row) + "\n" for row in phase_rows), encoding="utf-8")
            status.write_text(json.dumps({"schema": "cow-publication-phase-status-v1",
                                          "instrumentation_error": False}), encoding="utf-8")
            with patch.object(RUNNER, "_load", side_effect=AssertionError("must not launch campaign")):
                summary = RUNNER.summarize_existing(
                    result_path=result, phase_path=phases, status_path=status,
                    baseline_path=baseline, instrumented_store=instrumented_store,
                    baseline_store=baseline_store, output_dir=out,
                    expected_hashes={name: RUNNER._sha(path) for name, path in
                        (("result", result), ("phases", phases),
                         ("status", status), ("baseline", baseline))},
                    instrumentation_path=SCRIPT)
            self.assertTrue((out / "publication-timing-summary.json").is_file())
            self.assertTrue(summary["offline_only"])
            self.assertEqual(len(summary["baseline_comparison"]["questions"]), 3)
            store_rows = summary["baseline_comparison"]["store_snapshot_comparisons"]
            self.assertEqual(len(store_rows), 4)
            self.assertTrue(all(change["different"] for change in
                store_rows[0]["inode_changes_across_store_copies"].values()))
            self.assertEqual(len(summary["baseline_comparison"]["questions"][0][
                "explicit_machine_measurement_differences"]), 10)
            self.assertFalse(summary["baseline_comparison"]["questions"][0][
                "machine_answer_stdout_hash_equal"])
            existing = root / "preexisting-output"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(RUNNER.TimingProbeError):
                RUNNER.summarize_existing(
                    result_path=result, phase_path=phases, status_path=status,
                    baseline_path=baseline, instrumented_store=instrumented_store,
                    baseline_store=baseline_store, output_dir=existing,
                    expected_hashes={name: RUNNER._sha(path) for name, path in
                        (("result", result), ("phases", phases),
                         ("status", status), ("baseline", baseline))},
                    instrumentation_path=SCRIPT)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_machine_comparison_rejects_unlisted_answer_changes(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        measurement = {"tokens": [11], "text": "A", "capture_seconds": 1.0,
            "generation_seconds": 2.0, "request_seconds": 3.0,
            "counter_deltas": {"decode_compute_ns": 1, "decode_finish_ns": 2,
                "decode_prepare_ns": 3, "integrity_plan_upload_ns": 4,
                "integrity_seal_compare_ns": 5, "integrity_seal_result_download_ns": 6,
                "host_heap_allocated_bytes": 7}}
        actual = {"request": {"machine_envelope": {"answer": json.loads(json.dumps(measurement))}}}
        expected = {"request": {"machine_envelope": {"answer": json.loads(json.dumps(measurement))}}}
        actual["request"]["machine_envelope"]["answer"]["answer_id"] = "changed"
        with self.assertRaisesRegex(RUNNER.TimingProbeError, "outside explicit measurement"):
            RUNNER._compare_machine_envelopes(actual, expected, 1)

    def test_store_comparison_rejects_controller_or_manifest_change(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stores = [root / "left", root / "right"]
            ids = ["root", "q1", "q2", "q3"]
            for store in stores:
                (store / "objects").mkdir(parents=True)
                (store / "checkpoints").mkdir()
                for index, snapshot in enumerate(ids):
                    obj = {"kind": "snapshot", "schema": 1, "parent": None,
                           "model_contract_id": "m", "inputs": [],
                           "controller": {"status": "idle", "next_draw": index}}
                    (store / "objects" / f"{snapshot}.json").write_text(
                        json.dumps(obj, sort_keys=True), encoding="utf-8")
                    payload = f"payload-{snapshot}".encode()
                    digest = hashlib.sha256(payload).hexdigest()
                    (store / "checkpoints" / f"{digest}.state").write_bytes(payload)
                    manifest = {"checksum": "c", "manifest": {"snapshot": snapshot,
                        "payload_sha256": digest, "state_bytes": len(payload) - 1}}
                    (store / "checkpoints" / f"{snapshot}.json").write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8")
            candidate = {"root_snapshot": "root", "questions": [{"snapshot": sid} for sid in ids[1:]]}
            baseline = json.loads(json.dumps(candidate))
            self.assertEqual(len(RUNNER._compare_store_snapshots(
                candidate, baseline, stores[0], stores[1])), 4)
            changed = json.loads((stores[1] / "objects" / "q1.json").read_text(encoding="utf-8"))
            changed["controller"]["next_draw"] += 1
            (stores[1] / "objects" / "q1.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.TimingProbeError, "controller"):
                RUNNER._compare_store_snapshots(candidate, baseline, stores[0], stores[1])
            (stores[1] / "objects" / "q1.json").write_text(
                (stores[0] / "objects" / "q1.json").read_text(encoding="utf-8"), encoding="utf-8")
            manifest_path = stores[1] / "checkpoints" / "q1.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checksum"] = "altered"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.TimingProbeError, "manifest"):
                RUNNER._compare_store_snapshots(candidate, baseline, stores[0], stores[1])

    def test_store_comparison_rejects_controller_or_manifest_change(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stores = [root / "left", root / "right"]
            ids = ["root", "q1", "q2", "q3"]
            for store in stores:
                (store / "objects").mkdir(parents=True)
                (store / "checkpoints").mkdir()
                for index, snapshot in enumerate(ids):
                    obj = {"kind": "snapshot", "schema": 1, "parent": None,
                           "model_contract_id": "m", "inputs": [],
                           "controller": {"status": "idle", "next_draw": index}}
                    (store / "objects" / f"{snapshot}.json").write_text(
                        json.dumps(obj, sort_keys=True), encoding="utf-8")
                    payload = f"payload-{snapshot}".encode()
                    digest = hashlib.sha256(payload).hexdigest()
                    (store / "checkpoints" / f"{digest}.state").write_bytes(payload)
                    manifest = {"checksum": "c", "manifest": {"snapshot": snapshot,
                        "payload_sha256": digest, "state_bytes": len(payload)}}
                    (store / "checkpoints" / f"{snapshot}.json").write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8")
            questions = [{"snapshot": snapshot} for snapshot in ids[1:]]
            candidate = {"root_snapshot": "root", "questions": questions}
            baseline = json.loads(json.dumps(candidate))
            passed = RUNNER._compare_store_snapshots(candidate, baseline, stores[0], stores[1])
            self.assertEqual(len(passed), 4)
            changed = json.loads((stores[1] / "objects" / "q1.json").read_text(encoding="utf-8"))
            changed["controller"]["next_draw"] += 1
            (stores[1] / "objects" / "q1.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.TimingProbeError, "controller"):
                RUNNER._compare_store_snapshots(candidate, baseline, stores[0], stores[1])

    def test_bootstrap_hook_wraps_the_pinned_service_entrypoint_once(self):
        if RUNNER is None:
            self.fail("publication timing runner missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "owned-run"
            run_dir.mkdir()
            instrumentation = root / "publication_timing_probe.py"
            instrumentation.write_text("# pinned probe module\n", encoding="utf-8")

            class Campaign:
                def __init__(self):
                    self.run_dir = run_dir
                def _bootstrap(self, _ordinal):
                    target = run_dir / "server_bootstrap.py"
                    target.write_text(
                        "importlib.import_module('serve.cow_http').main()\n",
                        encoding="utf-8")
                    return target

            helper = types.SimpleNamespace(_Campaign=Campaign)
            RUNNER._install_bootstrap_hook(helper, instrumentation)
            bootstrap = Campaign()._bootstrap(1)
            text = bootstrap.read_text(encoding="utf-8")
            self.assertEqual(text.count("install_from_loaded("), 1)
            self.assertLess(text.index("install_from_loaded("),
                            text.index("importlib.import_module('serve.cow_http').main()"))
            self.assertEqual((run_dir / "publication_timing_probe.py").read_text(encoding="utf-8"),
                             "# pinned probe module\n")

    def test_trace_write_failure_never_changes_wrapped_return_or_exception(self):
        p = self.probe()
        fake_os = _FakeOS()
        real, coordinator, store, pools, Model, _, _, _, _ = self.make_modules(fake_os)

        class BrokenSink:
            def write(self, _value):
                raise OSError("diagnostic sink failed")
            def flush(self):
                raise OSError("diagnostic sink failed")
            def close(self):
                raise OSError("diagnostic sink failed")

        with tempfile.TemporaryDirectory() as directory:
            recorder = p.install_timing_wrappers(real, coordinator, store, pools,
                                                  Path(directory) / "events.jsonl")
            recorder._stream.close()
            recorder._stream = BrokenSink()
            result = Model().prepare_checkpoint("opaque-snapshot-id")
            self.assertEqual(result.snapshot, "opaque-snapshot-id")
            recorder.close()
            self.assertTrue(recorder._instrumentation_error)
            status = json.loads(recorder.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["instrumentation_error"], True)

        class ExplodingModel(Model):
            def prepare_checkpoint(self, snapshot):
                raise LookupError("original service exception")

        real.BranchingModel = ExplodingModel
        with tempfile.TemporaryDirectory() as directory:
            recorder = p.install_timing_wrappers(real, coordinator, store, pools,
                                                  Path(directory) / "events.jsonl")
            recorder._stream.close()
            recorder._stream = BrokenSink()
            with self.assertRaisesRegex(LookupError, "original service exception"):
                ExplodingModel().prepare_checkpoint("opaque-snapshot-id")
            recorder.close()
            self.assertTrue(recorder._instrumentation_error)
            status = json.loads(recorder.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["instrumentation_error"], True)

    def test_correlation_and_span_begin_failures_do_not_block_service_call(self):
        p = self.probe()

        class Owner:
            def call(self):
                return 42

        with tempfile.TemporaryDirectory() as directory:
            recorder = p.TimingRecorder(Path(directory) / "events.jsonl")
            recorder.wrap_method(Owner, "call", "test.call",
                lambda *_args: (_ for _ in ()).throw(RuntimeError("bad correlation")))
            self.assertEqual(Owner().call(), 42)
            with patch.object(p.time, "monotonic_ns", side_effect=RuntimeError("clock failed")):
                self.assertEqual(Owner().call(), 42)
            recorder.close()
            self.assertTrue(recorder._instrumentation_error)
            status = json.loads(recorder.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["instrumentation_error"], True)


class _Context:
    def __init__(self, name, value):
        self.name, self.value = name, value
    def __enter__(self):
        return (self.name, self.value)
    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeOS:
    O_WRONLY = 1
    O_CREAT = 2
    O_EXCL = 4
    def __init__(self):
        self.calls = []
    def fsync(self, fd):
        self.calls.append(("fsync", fd))
    def replace(self, source, destination):
        self.calls.append(("replace", source, destination))
    def fstat(self, fd):
        return types.SimpleNamespace(st_mode=0o100600)


if __name__ == "__main__":
    unittest.main()
