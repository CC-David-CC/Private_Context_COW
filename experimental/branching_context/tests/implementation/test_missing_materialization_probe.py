"""CPU-only guard tests for the real missing-materialization GPU probe."""
import importlib.util
import hashlib
import json
import os
import contextlib
import io
import copy
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import time
from types import SimpleNamespace
from unittest import mock
import unittest

API = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(API / "src"))
API_VALUE = os.environ.get("MISSING_MATERIALIZATION_API_SRC")
if API_VALUE:
    # Select one package universe before the flow fixture imports Store and
    # BranchHead. Re-importing these modules later creates incompatible classes.
    sys.path.insert(0, str(Path(API_VALUE).resolve(strict=True)))
    importlib.invalidate_caches()
SCRIPT = API / "scripts" / "run_missing_materialization_probe.py"
SPEC = importlib.util.spec_from_file_location("missing_materialization_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.store import IntegrityError
import serve.test_cow_presentation_flow as flow_fixture


def manifest(snapshot, payload):
    value = {"snapshot": snapshot, "payload_sha256": payload}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    return {"manifest": value, "checksum": hashlib.sha256(canonical).hexdigest()}


ACTION_RUNTIME = {}


def coordinator_class_for_action_test():
    """Load the coordinator overlay without replacing its shared API modules."""
    package = importlib.import_module("context_time_travel")
    coordinator_module = importlib.import_module("context_time_travel.coordinator")
    store_module = importlib.import_module("context_time_travel.store")
    pools_module = importlib.import_module("context_time_travel.pools")
    ACTION_RUNTIME.update(Store=store_module.Store, DiskCheckpointPool=pools_module.DiskCheckpointPool)
    overlay_value = os.environ.get("MISSING_MATERIALIZATION_COORDINATOR_OVERLAY")
    if not overlay_value:
        return coordinator_module.ContextCoordinator
    overlay = Path(overlay_value).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("context_time_travel.coordinator", overlay)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned candidate coordinator overlay")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    package.coordinator = candidate
    return candidate.ContextCoordinator


class MissingMaterializationProbeTests(unittest.TestCase):
    def test_native_counter_diagnostic_preserves_changed_integer_fields(self):
        before = {"forward_tokens": 8, "capture_count": 2, "active": False,
                  "nested": {"loads": 1}}
        after = {"forward_tokens": 8, "capture_count": 3, "active": False,
                 "nested": {"loads": 2}}
        self.assertEqual(PROBE.native_stats_delta(before, after),
                         {"capture_count": 1, "forward_tokens": 0})

    def make_store(self, root):
        (root / "checkpoints").mkdir(parents=True)
        (root / "objects").mkdir()
        (root / "objects" / "keep.json").write_text("canonical", encoding="utf-8")
        rows = []
        for snapshot in ("snapshot-p", "snapshot-s"):
            payload_bytes = (snapshot + " state").encode()
            payload = hashlib.sha256(payload_bytes).hexdigest()
            (root / "checkpoints" / f"{snapshot}.json").write_text(
                json.dumps(manifest(snapshot, payload)), encoding="utf-8")
            (root / "checkpoints" / f"{payload}.state").write_bytes(payload_bytes)
            rows.append(snapshot)
        return set(rows)

    def test_inventory_binds_all_manifests_and_payload_hash_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            rows = PROBE.plan_checkpoint_eviction(root, expected)
            self.assertEqual(len(rows), 4)
            names = {Path(row["relative_path"]).name for row in rows}
            self.assertEqual({name for name in names if name.endswith(".json")},
                             {"snapshot-p.json", "snapshot-s.json"})
            self.assertEqual({name for name in names if name.endswith(".state")}, {
                hashlib.sha256(b"snapshot-p state").hexdigest() + ".state",
                hashlib.sha256(b"snapshot-s state").hexdigest() + ".state"})
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))

    def test_present_manifest_missing_payload_is_incomplete_not_rebuildable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            payload = json.loads((root / "checkpoints" / "snapshot-s.json").read_text())
            (root / "checkpoints" / (payload["manifest"]["payload_sha256"] + ".state")).unlink()
            with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
                PROBE.plan_checkpoint_eviction(root, expected)

    def test_manifest_checksum_and_identity_are_verified_before_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            path = root / "checkpoints" / "snapshot-p.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["manifest"]["snapshot"] = "snapshot-s"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                PROBE.plan_checkpoint_eviction(root, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            path = root / "checkpoints" / "snapshot-p.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["manifest"]["consumed"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum is invalid"):
                PROBE.plan_checkpoint_eviction(root, expected)

    def test_orphan_payload_and_unexpected_snapshot_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            (root / "checkpoints" / (("c" * 64) + ".state")).write_bytes(b"orphan")
            with self.assertRaisesRegex(ValueError, "incomplete or has extras"):
                PROBE.plan_checkpoint_eviction(root, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            extra = "snapshot-extra"
            (root / "checkpoints" / f"{extra}.json").write_text(
                json.dumps(manifest(extra, "c" * 64)), encoding="utf-8")
            (root / "checkpoints" / ("c" * 64 + ".state")).write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "Unexpected numerical checkpoint"):
                PROBE.plan_checkpoint_eviction(root, expected)

    def test_unrecognized_checkpoint_entry_is_not_silently_left_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            (root / "checkpoints" / ".stage-abandoned").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "Unexpected entry"):
                PROBE.plan_checkpoint_eviction(root, expected)

    def test_eviction_removes_only_pinned_checkpoint_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            inventory = PROBE.plan_checkpoint_eviction(root, expected)
            result = PROBE.evict_checkpoint_files(root, expected, inventory)
            self.assertEqual(result["removed_bytes"], sum(row["bytes"] for row in inventory))
            self.assertEqual(len(result["removed"]), 4)
            self.assertEqual(PROBE.numerical_checkpoint_files(root), [])
            self.assertEqual((root / "objects" / "keep.json").read_text(encoding="utf-8"), "canonical")

    def test_inventory_change_after_freeze_prevents_unlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            inventory = PROBE.plan_checkpoint_eviction(root, expected)
            payload_path = next(root / "checkpoints" / Path(row["relative_path"]).name
                                for row in inventory if row["relative_path"].endswith(".state"))
            payload_path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed after it was frozen"):
                PROBE.evict_checkpoint_files(root, expected, inventory)
            self.assertTrue((root / "checkpoints" / "snapshot-p.json").exists())

    def test_payload_bytes_are_bound_to_manifest_content_address(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_store(root)
            state = next(root / "checkpoints" / Path(row["relative_path"]).name
                         for row in PROBE.plan_checkpoint_eviction(root, expected)
                         if row["relative_path"].endswith(".state"))
            state.write_bytes(b"wrong bytes under a valid-looking payload name")
            with self.assertRaisesRegex(ValueError, "do not match their SHA-256 filename"):
                PROBE.plan_checkpoint_eviction(root, expected)

    def test_rebuild_evidence_requires_exact_source_and_work_truth(self):
        good = {"source_kind": "recomputed-history", "durable": False,
                "prior_computation_preserved": False, "prefix_history_tokens": 132,
                "prefix_forward_tokens": 132, "prefix_consumed_tokens": 132}
        verified = PROBE.verify_rebuild_evidence(good, snapshot="s", history_tokens=132,
            model_contract_id="m", execution_profile_id="p")
        self.assertEqual(verified["snapshot"], "s")
        for field, value in (("source_kind", "disk-restore"), ("durable", True),
                             ("prior_computation_preserved", True), ("prefix_forward_tokens", 131),
                             ("prefix_consumed_tokens", 133), ("prefix_history_tokens", True),
                             ("snapshot", "other"), ("model_contract_id", "wrong")):
            with self.subTest(field=field):
                bad = dict(good, **{field: value})
                with self.assertRaises(ValueError):
                    PROBE.verify_rebuild_evidence(bad, snapshot="s", history_tokens=132,
                        model_contract_id="m", execution_profile_id="p")

    def test_missing_failure_classifier_requires_enoent_in_exception_chain(self):
        import errno
        missing = RuntimeError("invalid checkpoint")
        missing.__cause__ = FileNotFoundError(errno.ENOENT, "missing")
        self.assertTrue(PROBE.is_missing_checkpoint_error(missing))
        self.assertTrue(PROBE.is_missing_checkpoint_error(FileNotFoundError("missing")))
        self.assertFalse(PROBE.is_missing_checkpoint_error(PermissionError("denied")))
        self.assertFalse(PROBE.is_missing_checkpoint_error(RuntimeError("bad checksum")))

    def test_detached_incomplete_checkpoint_error_matches_only_pinned_integrity_error(self):
        wrapped = IntegrityError("Invalid or incomplete checkpoint")
        try:
            raise wrapped from FileNotFoundError("missing checkpoint payload")
        except IntegrityError as error:
            detached = ContextCoordinator._detached_error(error)

        self.assertIs(type(detached), IntegrityError)
        self.assertEqual(str(detached), "Invalid or incomplete checkpoint")
        self.assertIsNone(detached.__cause__)
        self.assertIsNone(detached.__context__)
        self.assertFalse(PROBE.is_missing_checkpoint_error(detached))
        self.assertTrue(PROBE.is_detached_incomplete_checkpoint_error(detached, IntegrityError))
        self.assertFalse(PROBE.is_detached_incomplete_checkpoint_error(
            IntegrityError("Invalid checkpoint checksum"), IntegrityError))
        self.assertFalse(PROBE.is_detached_incomplete_checkpoint_error(
            RuntimeError("Invalid or incomplete checkpoint"), IntegrityError))

    def test_detached_incomplete_checkpoint_classifier_is_not_a_general_missing_error(self):
        import errno
        for error in (
            IntegrityError("Invalid checkpoint checksum"),
            IntegrityError("checkpoint payload missing"),
            RuntimeError("Invalid or incomplete checkpoint"),
            PermissionError(errno.EACCES, "denied"),
        ):
            with self.subTest(error=error):
                self.assertFalse(PROBE.is_detached_incomplete_checkpoint_error(error, IntegrityError))
        # Strict restore without the manifest is still classified only by the
        # preserved ENOENT chain; the detached negative-control diagnostic is
        # not silently accepted by that separate check.
        self.assertFalse(PROBE.is_missing_checkpoint_error(
            IntegrityError("Invalid or incomplete checkpoint")))

    def test_work_split_separates_rebuild_from_suffix_and_outputs(self):
        result = PROBE.validate_work_split(history_tokens=132, suffix_tokens=4, output_tokens=7,
            prefix_forward=132, prefix_consumed=132, request_forward=11, total_forward_delta=143)
        self.assertEqual(result["total_forward_tokens"], 143)
        for kwargs in (
            dict(history_tokens=132, suffix_tokens=4, output_tokens=7, prefix_forward=131,
                 prefix_consumed=132, request_forward=11, total_forward_delta=142),
            dict(history_tokens=132, suffix_tokens=4, output_tokens=7, prefix_forward=132,
                 prefix_consumed=132, request_forward=12, total_forward_delta=144),
            dict(history_tokens=132, suffix_tokens=4, output_tokens=7, prefix_forward=True,
                 prefix_consumed=132, request_forward=11, total_forward_delta=143),
            dict(history_tokens=132, suffix_tokens=4, output_tokens=7, prefix_forward=132,
                 prefix_consumed=132, request_forward=11, total_forward_delta=142),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PROBE.validate_work_split(**kwargs)


    def test_real_fork_and_idempotent_rewinds_do_not_need_numeric_checkpoint_files(self):
        from context_time_travel.pools import PoolConfig

        class CheckedFlowModel(flow_fixture.FlowModel):
            plans = dict(flow_fixture.FlowModel.plans, **{"op-source": (257, 299)})

            def prepare_checkpoint(self, sid):
                checkpoint = self.cache[sid]
                manifest_value = dict(snapshot=sid, state_bytes=len(checkpoint.blob),
                    logits_count=len(checkpoint.frontier), payload_sha256=checkpoint.checksum)
                canonical = json.dumps(manifest_value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
                envelope = dict(manifest=manifest_value,
                    checksum=hashlib.sha256(canonical).hexdigest())
                return flow_fixture.PreparedCheckpoint(sid, checkpoint,
                    json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())

        coordinator_type = coordinator_class_for_action_test()
        with tempfile.TemporaryDirectory() as directory:
            trace = []
            gate = threading.Event()
            gate.set()
            models = {}

            def factory(gpu):
                model = CheckedFlowModel(directory, gpu, (0,), trace, gate)
                models[gpu] = model
                return model

            coordinator = coordinator_type(directory,
                PoolConfig(gpus=(0,), ram_bytes=1024 * 1024, disk_bytes=8 * 1024 * 1024,
                           min_available_bytes=0), factory)
            try:
                coordinator.ready(5)
                root_result = coordinator.create("source", (10, 11), "op-root").durable.result(5)
                root_snapshot = root_result["snapshot"]
                source_request = coordinator.submit("source", (12,), 2, "op-source")
                source_result = source_request.durable.result(5)
                source_snapshot = source_result["snapshot"]
                with coordinator._condition:
                    self.assertTrue(coordinator._condition.wait_for(
                        lambda: sum(coordinator.metrics()["active_per_gpu"].values()) == 0, timeout=5))

                store = coordinator.store
                source_head_before = store.head("source")
                source_record_before = store.get_snapshot(source_snapshot)
                source_history_before = tuple(store.lineage(source_snapshot)[1])
                with store._locked():
                    old_receipt = copy.deepcopy(store._read_refs()["receipts"].get("op-source"))
                self.assertIsNotNone(old_receipt)
                inventory = PROBE.plan_checkpoint_eviction(store.root, {root_snapshot, source_snapshot})
                PROBE.evict_checkpoint_files(store.root, {root_snapshot, source_snapshot}, inventory)
                self.assertEqual(PROBE.numerical_checkpoint_files(store.root), [])

                model = models[0]
                native_before = dict(model.stats)
                with store._locked():
                    source_receipt = copy.deepcopy(store._read_refs()["receipts"].get("op-source"))
                actions = PROBE.exercise_evicted_checkpoint_branch_actions(
                    coordinator, model, source_branch="source", source_snapshot=source_snapshot,
                    root_snapshot=root_snapshot, answer_branch="answer", source_head=source_head_before,
                    source_receipt=source_receipt)
                self.assertEqual(actions["fork"], {
                    "branch": "answer", "snapshot": source_snapshot, "revision": 0})
                self.assertEqual(store.head("answer").revision, 2)
                self.assertEqual(actions["fork"]["revision"], 0)
                self.assertEqual(store.head("answer").tip, source_snapshot)
                self.assertEqual(len(actions["rewinds"]), 2)
                self.assertTrue(all(item["idempotent_retry_equal"] for item in actions["rewinds"]))

                self.assertEqual(store.head("source"), source_head_before)
                self.assertEqual(store.get_snapshot(source_snapshot), source_record_before)
                self.assertEqual(tuple(store.lineage(source_snapshot)[1]), source_history_before)
                with store._locked():
                    self.assertEqual(store._read_refs()["receipts"].get("op-source"), old_receipt)
                self.assertEqual(model.stats, native_before)
                self.assertEqual(PROBE.numerical_checkpoint_files(store.root), [])
            finally:
                coordinator.close()

    def test_disk_pool_reservation_evicts_the_checked_inventory_without_canonical_changes(self):
        from context_time_travel.pools import PoolConfig

        class CheckedFlowModel(flow_fixture.FlowModel):
            plans = dict(flow_fixture.FlowModel.plans, **{"op-source": (257, 299)})

            def prepare_checkpoint(self, sid):
                checkpoint = self.cache[sid]
                manifest_value = dict(snapshot=sid, state_bytes=len(checkpoint.blob),
                    logits_count=len(checkpoint.frontier), payload_sha256=checkpoint.checksum)
                canonical = json.dumps(manifest_value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
                envelope = dict(manifest=manifest_value,
                    checksum=hashlib.sha256(canonical).hexdigest())
                return flow_fixture.PreparedCheckpoint(sid, checkpoint,
                    json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())

        coordinator_type = coordinator_class_for_action_test()
        with tempfile.TemporaryDirectory() as directory:
            trace = []
            gate = threading.Event(); gate.set()
            models = {}

            def factory(gpu):
                model = CheckedFlowModel(directory, gpu, (0,), trace, gate)
                models[gpu] = model
                return model

            coordinator = coordinator_type(directory,
                PoolConfig(gpus=(0,), ram_bytes=1024 * 1024, disk_bytes=8 * 1024 * 1024,
                           min_available_bytes=0), factory)
            try:
                coordinator.ready(5)
                root_snapshot = coordinator.create("source", (10, 11), "op-root").durable.result(5)["snapshot"]
                coordinator.submit("source", (12,), 2, "op-source").durable.result(5)
                with coordinator._condition:
                    self.assertTrue(coordinator._condition.wait_for(
                        lambda: sum(coordinator.metrics()["active_per_gpu"].values()) == 0, timeout=5))
                source_snapshot = coordinator.store.head("source").tip
                coordinator.close()  # Wait for the writer lease to end before pool admission.
                store = ACTION_RUNTIME["Store"](directory)
                expected = {root_snapshot, source_snapshot}
                inventory = PROBE.plan_checkpoint_eviction(store.root, expected)
                self.assertTrue(inventory)

                result = PROBE.evict_via_disk_pool(store,
                    ACTION_RUNTIME["DiskCheckpointPool"], expected, inventory)

                self.assertEqual(result["pressure_budget_bytes"], sum(row["bytes"] for row in inventory))
                self.assertEqual(result["metrics_before"]["file_bytes"], result["pressure_budget_bytes"])
                self.assertEqual(result["metrics_during_reservation"]["reserved_bytes"],
                                 result["pressure_budget_bytes"])
                self.assertEqual(result["metrics_during_reservation"]["files_evicted"], len(inventory))
                self.assertEqual(result["metrics_after_release"]["reserved_bytes"], 0)
                self.assertEqual(PROBE.numerical_checkpoint_files(store.root), [])
            finally:
                coordinator.close()

    def test_progress_serializes_concurrent_heartbeat_and_phase_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = PROBE.Progress(Path(directory))
            progress.start()
            failures = []
            def write_phase(index):
                try:
                    for step in range(40):
                        progress.set(f"phase-{index}", child_pid=1000 + index,
                                     last_completed_step=str(step))
                        progress.write()
                except BaseException as exc:
                    failures.append(exc)
            workers = [threading.Thread(target=write_phase, args=(index,)) for index in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            progress.close()
            self.assertEqual(failures, [])
            final = json.loads(progress.path.read_text(encoding="utf-8"))
            self.assertIn(final["phase"], {"starting", "phase-0", "phase-1", "phase-2", "phase-3"})
            self.assertGreaterEqual(final["elapsed_seconds"], 0)

    def test_progress_records_explicit_terminal_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = PROBE.Progress(Path(directory))
            progress.start()
            progress.set("complete", status="PASS")
            progress.close()
            value = json.loads(progress.path.read_text(encoding="utf-8"))
            self.assertEqual((value["phase"], value["status"]), ("complete", "PASS"))

    def test_observed_rebuild_receipt_binds_saved_schedule_controller_and_work(self):
        controller = {"status": "idle", "cursor": 9}
        schedule = "a" * 64
        controller_sha = PROBE.canonical_json_sha256(controller)
        evidence = {"schedule_sha256": schedule, "controller_sha256": controller_sha}
        receipt = {"snapshot": "s", "source_kind": "recomputed-history", "durable": False,
                   "prior_computation_preserved": False, "consumed": 12, "forward_tokens": 12,
                   "model_contract_id": "m", "execution_profile_id": "p",
                   "schedule_sha256": schedule, "controller_sha256": controller_sha,
                   "counter_deltas": {"forward_tokens": 12}}
        verified = PROBE.verify_observed_rebuild_receipt(receipt, evidence, snapshot="s",
            history_tokens=12, model_contract_id="m", execution_profile_id="p",
            controller=controller, expected_schedule_sha256=schedule)
        self.assertEqual(verified["forward_tokens"], 12)
        for field, value in (("snapshot", "wrong"), ("consumed", 11), ("forward_tokens", 13),
                             ("schedule_sha256", "b" * 64), ("controller_sha256", "c" * 64),
                             ("durable", 0)):
            with self.subTest(field=field):
                bad = dict(receipt, **{field: value})
                with self.assertRaises(ValueError):
                    PROBE.verify_observed_rebuild_receipt(bad, evidence, snapshot="s",
                        history_tokens=12, model_contract_id="m", execution_profile_id="p",
                        controller=controller, expected_schedule_sha256=schedule)

    def test_child_exit_status_is_collected_after_polling_completion(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        self.assertEqual(child.wait(timeout=5), 0)
        self.assertEqual(PROBE.wait_child_exit(child), 0)

    def test_requested_fixture_dimensions_are_accepted_and_forwarded_to_each_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            captured = {}
            args = ["--profile", str(root / "profile.json"),
                    "--coordinator-overlay", str(root / "coordinator.py"),
                    "--store", str(output / "store"), "--output", str(output),
                    "--prefix", "1024", "--suffix", "64", "--generated", "16"]
            with mock.patch.object(PROBE, "run", side_effect=lambda value: captured.setdefault("args", value)):
                self.assertEqual(PROBE.main(args), 0)
            parsed = captured["args"]
            self.assertEqual((parsed.prefix, parsed.suffix, parsed.generated), (1024, 64, 16))

            class Child:
                pid = 4242
                def poll(self):
                    return 0
                def wait(self, timeout=None):
                    return 0

            for phase, phase_args in (("producer", {"root_branch": "source-A"}),
                                      ("consumer", {"source_branch": "source-A",
                                                     "source_snapshot": "snap-S",
                                                     "root_snapshot": "snap-R",
                                                     "answer_branch": "answer-B"})):
                with self.subTest(phase=phase):
                    output.mkdir(parents=True, exist_ok=True)
                    (output / f"{phase}-worker.json").write_text(
                        json.dumps({"status": "PASS", "native_exit_success": True}), encoding="utf-8")
                    command = []
                    args_object = SimpleNamespace(profile=root / "profile.json", api_src=root / "api",
                        coordinator_overlay=root / "coordinator.py", store=output / "store", output=output,
                        gpu=0, ram_gib=96, disk_gib=128, min_available_gib=16,
                        prefix=parsed.prefix, suffix=parsed.suffix, generated=parsed.generated)
                    with mock.patch.object(PROBE.subprocess, "Popen",
                                           side_effect=lambda cmd, **_kwargs: command.extend(cmd) or Child()):
                        worker = PROBE._run_child(args_object, phase, phase_args, output,
                                                  SimpleNamespace(set=lambda *_a, **_k: None))
                    self.assertEqual(worker["status"], "PASS")
                    for flag, expected in (("--prefix", "1024"), ("--suffix", "64"),
                                           ("--generated", "16")):
                        self.assertIn(flag, command)
                        self.assertEqual(command[command.index(flag) + 1], expected)

    def test_nonpositive_fixture_dimensions_fail_before_output_or_model_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flag, value in (("--prefix", "0"), ("--suffix", "-1"),
                                ("--generated", "0")):
                with self.subTest(flag=flag), tempfile.TemporaryDirectory(dir=root) as case:
                    case_root = Path(case)
                    output = case_root / "must-not-exist"
                    argv = ["--profile", str(case_root / "profile.json"),
                            "--coordinator-overlay", str(case_root / "coordinator.py"),
                            "--store", str(output / "store"), "--output", str(output),
                            "--prefix", "128", "--suffix", "4", "--generated", "8"]
                    argv[argv.index(flag) + 1] = value
                    with mock.patch.object(PROBE, "run") as run_mock:
                        stderr = io.StringIO()
                        with contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit) as raised:
                                PROBE.main(argv)
                    self.assertNotEqual(raised.exception.code, 0)
                    self.assertIn(flag[2:], stderr.getvalue())
                    self.assertIn("positive", stderr.getvalue().lower())
                    self.assertFalse(output.exists())
                    run_mock.assert_not_called()

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(),
                         "process-group cleanup requires Linux procfs")
    def test_owned_process_group_cleanup_kills_ignored_term_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "grandchild.pid"
            grandchild_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
            leader_code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                           + repr(grandchild_code) + "]); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)")
            leader = subprocess.Popen([sys.executable, "-c", leader_code, str(ready)],
                                      start_new_session=True, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "leader did not start its child")
            grandchild_pid = int(ready.read_text())
            self.assertEqual(os.getpgid(grandchild_pid), leader.pid)
            PROBE._stop_child(leader, grace_seconds=0.1)
            self.assertIsNotNone(leader.poll())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = Path(f"/proc/{grandchild_pid}/status")
                if not status.exists():
                    break
                state = next((line for line in status.read_text().splitlines()
                              if line.startswith("State:")), "")
                if "Z" in state:
                    break
                time.sleep(0.02)
            else:
                self.fail("native-like grandchild survived owned process-group cleanup")

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(),
                         "process-group cleanup requires Linux procfs")
    def test_run_child_launcher_receipt_failure_cleans_owned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ready = output / "grandchild.pid"
            grandchild_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
            leader_code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                           + repr(grandchild_code) + "]); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)")
            real_popen = subprocess.Popen
            spawned = []
            def spawn_group(_command, **kwargs):
                child = real_popen([sys.executable, "-c", leader_code, str(ready)],
                    stdout=kwargs["stdout"], stderr=kwargs["stderr"], start_new_session=True)
                spawned.append(child)
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not ready.exists():
                    PROBE._stop_child(child)
                    raise AssertionError("test leader did not start its grandchild")
                return child
            real_atomic_json = PROBE.atomic_json
            failed = []
            def fail_first_process_record(path, value):
                if path.name == "producer.process.json" and not failed:
                    failed.append(True)
                    raise OSError("injected first process-record write failure")
                return real_atomic_json(path, value)
            args = SimpleNamespace(profile=output / "profile.json", api_src=output / "api",
                coordinator_overlay=output / "coordinator.py", store=output / "store",
                output=output, gpu=0, ram_gib=1, disk_gib=1, min_available_gib=1,
                prefix=128, suffix=4, generated=8)
            class ProgressStub:
                def set(self, *_args, **_kwargs):
                    pass
            with mock.patch.object(PROBE.subprocess, "Popen", side_effect=spawn_group), \
                 mock.patch.object(PROBE, "atomic_json", side_effect=fail_first_process_record):
                with self.assertRaisesRegex(OSError, "injected first process-record"):
                    PROBE._run_child(args, "producer", {"root_branch": "source-A"}, output, ProgressStub())
            self.assertTrue(ready.exists())
            grandchild_pid = int(ready.read_text())
            process_record = json.loads((output / "producer.process.json").read_text())
            self.assertEqual(process_record["status"], "FAILED")
            self.assertEqual(process_record["process_group_id"], spawned[0].pid)
            self.assertIsNotNone(spawned[0].poll())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = Path(f"/proc/{grandchild_pid}/status")
                if not status.exists():
                    break
                state = next((line for line in status.read_text().splitlines()
                              if line.startswith("State:")), "")
                if "Z" in state:
                    break
                time.sleep(0.02)
            else:
                self.fail("_run_child exception path left native-like grandchild alive")

    def test_strict_restore_rejections_require_zero_native_counter_delta(self):
        PROBE.require_no_native_work({"forward_tokens": 12, "loaded_file_bytes": 0},
                                     {"forward_tokens": 12, "loaded_file_bytes": 0}, "strict restore")
        with self.assertRaisesRegex(ValueError, "strict restore"):
            PROBE.require_no_native_work({"forward_tokens": 12, "loaded_file_bytes": 0},
                                         {"forward_tokens": 12, "loaded_file_bytes": 4096},
                                         "strict restore")

    def test_oracle_receipt_binds_observed_pid_exit_and_actual_return_shape(self):
        value = {"tokens": [7, 8], "prompt_frontiers": [b"p1", b"p2"],
                 "generated_frontiers": [b"g1", b"g2"], "work": {"consumed": 5},
                 "seconds": 0.25, "command": ["oracle", "--max-new", "2"]}
        receipt = PROBE.bind_oracle_result(value, pid=1234, exit_code=0,
            requested_frontiers=2, generated_limit=8)
        self.assertEqual((receipt["pid"], receipt["exit_code"]), (1234, 0))
        self.assertEqual(receipt["tokens"], [7, 8])
        for bad, pid, code in (({}, 1234, 0), (value, 0, 0), (value, 1234, 2),
                               (dict(value, generated_frontiers=[b"g1"]), 1234, 0),
                               (dict(value, tokens=[True]), 1234, 0)):
            with self.subTest(bad=bad, pid=pid, code=code), self.assertRaises(ValueError):
                PROBE.bind_oracle_result(bad, pid=pid, exit_code=code,
                    requested_frontiers=2, generated_limit=8)

    def test_oracle_receipt_preserves_actual_early_stop_count(self):
        value = {"tokens": [7, 299], "prompt_frontiers": [b"p"],
                 "generated_frontiers": [b"g1", b"g2"], "work": {"consumed": 5},
                 "seconds": 0.25, "command": ["oracle", "--max-new", "16"]}
        receipt = PROBE.bind_oracle_result(value, pid=1234, exit_code=0,
            requested_frontiers=1, generated_limit=16)
        self.assertEqual(len(receipt["tokens"]), 2)
        self.assertEqual(len(receipt["generated_frontiers"]), 2)
        self.assertEqual(receipt["tokens"], [7, 299])

    def test_runtime_pins_must_match_and_be_present_across_workers(self):
        pins = {"profile_sha256": "a" * 64, "profile_id": "p",
                "model_contract_id": "m", "model_identity": {"weights": "w"},
                "coordinator_overlay_sha256": "b" * 64, "api_src_sha256": {"x.py": "c" * 64}}
        PROBE.require_matching_worker_pins(dict(pins), dict(pins))
        with self.assertRaisesRegex(ValueError, "api_src_sha256"):
            PROBE.require_matching_worker_pins(dict(pins), dict(pins, api_src_sha256={"x.py": "d" * 64}))
        with self.assertRaisesRegex(ValueError, "profile_sha256"):
            PROBE.require_matching_worker_pins(dict(pins, profile_sha256=None), dict(pins))

    def test_evicted_source_branch_setup_uses_store_without_coordinator_restore(self):
        class Store:
            def __init__(self):
                self.calls = []
            def fork(self, branch, snapshot):
                self.calls.append((branch, snapshot))
                return "canonical-ref"
        class Coordinator:
            def __init__(self):
                self.store = Store()
            def fork(self, *_args, **_kwargs):
                raise AssertionError("Coordinator fork would restore the missing checkpoint")
        coordinator = Coordinator()
        self.assertEqual(PROBE.create_canonical_only_fork(coordinator, "answer-B", "snapshot-S"),
                         "canonical-ref")
        self.assertEqual(coordinator.store.calls, [("answer-B", "snapshot-S")])


if __name__ == "__main__":
    unittest.main()
