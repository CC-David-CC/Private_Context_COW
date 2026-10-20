"""Run the frozen resident CLI campaign with test-only publication phase timers."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import stat
from pathlib import Path
import sys


class TimingProbeError(RuntimeError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TimingProbeError(f"could not load pinned module: {Path(path).name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_rows(observer_request):
    rows = observer_request.get("events") if isinstance(observer_request, dict) else None
    if not isinstance(rows, list):
        raise TimingProbeError("observer request has no event list")
    decoded = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("arrival_monotonic_ns")) is not int:
            raise TimingProbeError("observer event lacks its monotonic arrival time")
        try:
            event = json.loads(base64.b64decode(row["raw_line_b64"], validate=True).decode("utf-8"))
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise TimingProbeError("observer event is malformed") from None
        decoded.append((row["arrival_monotonic_ns"], event))
    return decoded


def build_question_timeline(rows, *, snapshot, observer_request, expected_token_ids,
                            expected_answer, expected_receipt):
    """Join existing server spans to one observer request using its snapshot ID."""
    selected = [row for row in rows if row.get("correlation_id") == snapshot]
    if not selected:
        raise TimingProbeError("no publication spans are bound to the canonical answer snapshot")
    by_phase = {}
    for row in selected:
        by_phase.setdefault(row.get("phase"), []).append(row)
    required_single = (
        "worker.prepare_checkpoint", "writer.publish_total",
        "checkpoint.publish_prepared_checkpoint", "checkpoint.saved_validation",
        "store.move_receipt", "request.snapshot_emit", "request.durable_emit",
    )
    chosen = {}
    for phase in required_single:
        candidates = by_phase.get(phase, [])
        if len(candidates) != 1:
            raise TimingProbeError(f"expected one {phase} span for the answer snapshot")
        chosen[phase] = candidates[0]
    for phase in ("checkpoint.payload_write_fsync_rename_dirsync",
                  "checkpoint.manifest_write_fsync_rename_dirsync"):
        if len(by_phase.get(phase, [])) != 1:
            raise TimingProbeError(f"expected one {phase} span for the answer snapshot")

    events = _event_rows(observer_request)
    text_events = [(stamp, event) for stamp, event in events if event.get("event") == "text"]
    durable_events = [(stamp, event) for stamp, event in events if event.get("event") == "durable"]
    snapshot_events = [(stamp, event) for stamp, event in events if event.get("event") == "snapshot"]
    complete_events = [(stamp, event) for stamp, event in events if event.get("event") == "complete"]
    if (not text_events or len(durable_events) != 1
            or len(snapshot_events) != 1 or len(complete_events) != 1):
        raise TimingProbeError("observer trace lacks token, snapshot, or durable boundary")
    token_ids, token_text, token_times = [], [], []
    for stamp, event in text_events:
        token_id = event.get("token_id")
        text = event.get("text")
        if type(token_id) is not int or token_id < 0 or not isinstance(text, str):
            raise TimingProbeError("observer text event lacks a valid token_id or text")
        token_ids.append(token_id)
        token_text.append(text)
        token_times.append(stamp)
    if (not isinstance(expected_token_ids, list)
            or any(type(token_id) is not int or token_id < 0 for token_id in expected_token_ids)
            or token_ids != expected_token_ids):
        raise TimingProbeError("observer text token_ids differ from the canonical answer")
    streamed_text = "".join(token_text)
    if (not isinstance(expected_answer, dict)
            or expected_answer.get("tokens") != expected_token_ids
            or expected_answer.get("text") != streamed_text):
        raise TimingProbeError("observer text events differ from the exact canonical answer")
    snapshot_stamp, snapshot_event = snapshot_events[0]
    if snapshot_event.get("snapshot") != snapshot or snapshot_event.get("tokens") != expected_token_ids:
        raise TimingProbeError("observer snapshot event differs from the canonical answer")
    complete_stamp, complete_event = complete_events[0]
    complete_answer = complete_event.get("answer")
    if (not isinstance(complete_answer, dict)
            or complete_answer.get("tokens") != expected_token_ids
            or complete_answer.get("text") != streamed_text
            or complete_event.get("receipt") != expected_receipt
            or durable_events[0][1].get("receipt") != expected_receipt):
        raise TimingProbeError("observer complete/durable events differ from the canonical receipt")
    last_token = token_times[-1]
    durable_server = chosen["request.durable_emit"]
    prepare = chosen["worker.prepare_checkpoint"]
    snapshot_server = chosen["request.snapshot_emit"]
    publish = chosen["writer.publish_total"]
    if (token_times != sorted(token_times)
            or not last_token <= snapshot_stamp <= durable_events[0][0] <= complete_stamp):
        raise TimingProbeError("observer client-side token/snapshot/durable order is invalid")
    publication = chosen["checkpoint.publish_prepared_checkpoint"]
    saved_validation = chosen["checkpoint.saved_validation"]
    move = chosen["store.move_receipt"]
    if not (prepare["start_monotonic_ns"] <= prepare["end_monotonic_ns"]
            <= snapshot_server["start_monotonic_ns"] <= snapshot_server["end_monotonic_ns"]
            <= publish["start_monotonic_ns"] <= publication["start_monotonic_ns"]
            <= publication["end_monotonic_ns"] <= publish["end_monotonic_ns"]
            and publication["start_monotonic_ns"] <= saved_validation["start_monotonic_ns"]
            <= saved_validation["end_monotonic_ns"] <= move["start_monotonic_ns"]
            <= move["end_monotonic_ns"] <= durable_server["start_monotonic_ns"]
            <= durable_server["end_monotonic_ns"] <= publish["end_monotonic_ns"]):
        raise TimingProbeError("server-internal publication phase order is invalid")
    for phase in ("checkpoint.payload_write_fsync_rename_dirsync",
                  "checkpoint.manifest_write_fsync_rename_dirsync"):
        item = by_phase[phase][0]
        if not (publication["start_monotonic_ns"] <= item["start_monotonic_ns"]
                <= item["end_monotonic_ns"] <= publication["end_monotonic_ns"]):
            raise TimingProbeError("checkpoint filesystem phase escaped its publication span")

    phases = []
    for row in sorted(selected, key=lambda item: (item["start_monotonic_ns"], item["span_id"])):
        phases.append({key: row.get(key) for key in (
            "span_id", "parent_span_id", "phase", "thread_id", "depth",
            "start_monotonic_ns", "end_monotonic_ns", "duration_ns", "outcome",
            "wait_ns", "held_ns") if key in row})
    return {
        "snapshot": snapshot,
        "last_token_arrival_monotonic_ns": last_token,
        "prepare_start_monotonic_ns": prepare["start_monotonic_ns"],
        "prepare_end_monotonic_ns": prepare["end_monotonic_ns"],
        "snapshot_event_server_end_monotonic_ns": snapshot_server["end_monotonic_ns"],
        "snapshot_event_client_arrival_monotonic_ns": snapshot_stamp,
        "writer_publish_start_monotonic_ns": publish["start_monotonic_ns"],
        "writer_publish_end_monotonic_ns": publish["end_monotonic_ns"],
        "durable_event_server_start_monotonic_ns": durable_server["start_monotonic_ns"],
        "durable_event_server_end_monotonic_ns": durable_server["end_monotonic_ns"],
        "durable_event_client_arrival_monotonic_ns": durable_events[0][0],
        "text_event_count": len(text_events),
        "text_event_token_ids": token_ids,
        "text_event_text_sha256": hashlib.sha256(streamed_text.encode("utf-8")).hexdigest(),
        # Client arrival timestamps and server method timestamps may overlap:
        # transport buffering can delay token observation, and a durable line
        # can arrive while Request.emit is still returning. Keep signed deltas.
        "prepare_start_minus_last_token_arrival_ns": prepare["start_monotonic_ns"] - last_token,
        "prepare_duration_ns": prepare["duration_ns"],
        "prepare_to_snapshot_emit_ns": snapshot_server["start_monotonic_ns"] - prepare["end_monotonic_ns"],
        "snapshot_emit_to_writer_start_ns": publish["start_monotonic_ns"] - snapshot_server["end_monotonic_ns"],
        "writer_publish_duration_ns": publish["duration_ns"],
        "client_durable_arrival_minus_server_emit_end_ns": durable_events[0][0] - durable_server["end_monotonic_ns"],
        "phase_spans": phases,
    }


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        raise TimingProbeError(f"{label} is not readable JSON") from None


def _store_snapshot_receipt(store_root, snapshot):
    """Read small canonical records and stat payloads; never reread checkpoint bytes."""
    root = Path(store_root)
    object_path = root / "objects" / f"{snapshot}.json"
    manifest_path = root / "checkpoints" / f"{snapshot}.json"
    try:
        object_bytes = object_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        object_record = json.loads(object_bytes)
        manifest_record = json.loads(manifest_bytes)
    except (OSError, ValueError):
        raise TimingProbeError(f"canonical records are missing or malformed for {snapshot}") from None
    if (not isinstance(object_record, dict) or object_record.get("kind") != "snapshot"
            or not isinstance(object_record.get("controller"), dict)
            or not isinstance(manifest_record, dict)
            or not isinstance(manifest_record.get("manifest"), dict)):
        raise TimingProbeError(f"canonical snapshot/controller receipt is malformed for {snapshot}")
    payload_hash = manifest_record["manifest"].get("payload_sha256")
    if (not isinstance(payload_hash, str) or len(payload_hash) != 64
            or manifest_record["manifest"].get("snapshot") != snapshot):
        raise TimingProbeError(f"checkpoint payload digest is malformed for {snapshot}")
    payload_path = root / "checkpoints" / f"{payload_hash}.state"
    try:
        object_stat = object_path.stat()
        manifest_stat = manifest_path.stat()
        payload_stat = payload_path.stat()
    except OSError:
        raise TimingProbeError(f"canonical checkpoint file is missing for {snapshot}") from None
    controller_bytes = json.dumps(object_record["controller"], sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
    return {
        "snapshot": snapshot,
        "controller_sha256": hashlib.sha256(controller_bytes).hexdigest(),
        "object_sha256": hashlib.sha256(object_bytes).hexdigest(),
        "object_file": {"bytes": object_stat.st_size, "mode": stat.S_IMODE(object_stat.st_mode),
                        "sha256": hashlib.sha256(object_bytes).hexdigest(), "inode": object_stat.st_ino},
        "object_record": object_record,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_state_bytes": manifest_record["manifest"].get("state_bytes"),
        "manifest_record": manifest_record,
        "files": {
            manifest_path.name: {"bytes": manifest_stat.st_size, "mode": stat.S_IMODE(manifest_stat.st_mode),
                                 "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "inode": manifest_stat.st_ino},
            payload_path.name: {"bytes": payload_stat.st_size, "mode": stat.S_IMODE(payload_stat.st_mode),
                                "sha256": payload_hash, "inode": payload_stat.st_ino},
        },
    }


def _compare_store_snapshots(candidate, baseline, candidate_store, baseline_store):
    snapshot_ids = [candidate.get("root_snapshot")]
    snapshot_ids.extend(row.get("snapshot") for row in candidate.get("questions", []))
    if (len(snapshot_ids) != 4 or any(not isinstance(item, str) for item in snapshot_ids)
            or snapshot_ids != [baseline.get("root_snapshot"),
                               *[row.get("snapshot") for row in baseline.get("questions", [])]]):
        raise TimingProbeError("baseline and instrumented snapshot inventories differ")
    comparisons = []
    for snapshot in snapshot_ids:
        actual = _store_snapshot_receipt(candidate_store, snapshot)
        expected = _store_snapshot_receipt(baseline_store, snapshot)
        if actual["object_record"] != expected["object_record"]:
            raise TimingProbeError(f"canonical snapshot/controller differs from resident baseline at {snapshot}")
        if actual["manifest_record"] != expected["manifest_record"]:
            raise TimingProbeError(f"checkpoint manifest differs from resident baseline at {snapshot}")
        if actual["controller_sha256"] != expected["controller_sha256"]:
            raise TimingProbeError(f"canonical controller digest differs from resident baseline at {snapshot}")
        inode_changes = {}
        if set(actual["files"]) != set(expected["files"]):
            raise TimingProbeError(f"canonical checkpoint filenames differ at {snapshot}")
        for name, actual_file in actual["files"].items():
            expected_file = expected["files"][name]
            for field in ("bytes", "mode", "sha256"):
                if actual_file[field] != expected_file[field]:
                    raise TimingProbeError(f"canonical file differs at {snapshot}/{name}:{field}")
            inode_changes[name] = {"instrumented": actual_file["inode"],
                                   "baseline": expected_file["inode"],
                                   "different": actual_file["inode"] != expected_file["inode"]}
        inode_changes[f"objects/{snapshot}.json"] = {
            "instrumented": actual["object_file"]["inode"],
            "baseline": expected["object_file"]["inode"],
            "different": actual["object_file"]["inode"] != expected["object_file"]["inode"]}
        comparisons.append({"snapshot": snapshot,
                            "controller_sha256": actual["controller_sha256"],
                            "object_sha256": actual["object_sha256"],
                            "object_file": {key: value for key, value in actual["object_file"].items()
                                            if key != "inode"},
                            "manifest_sha256": actual["manifest_sha256"],
                            "manifest_state_bytes": actual["manifest_state_bytes"],
                            "files": {name: {key: value for key, value in row.items() if key != "inode"}
                                      for name, row in actual["files"].items()},
                            "inode_changes_across_store_copies": inode_changes})
    return comparisons


_MACHINE_MEASUREMENT_PATHS = (
    ("answer", "capture_seconds"),
    ("answer", "generation_seconds"),
    ("answer", "request_seconds"),
    ("answer", "counter_deltas", "decode_compute_ns"),
    ("answer", "counter_deltas", "decode_finish_ns"),
    ("answer", "counter_deltas", "decode_prepare_ns"),
    ("answer", "counter_deltas", "integrity_plan_upload_ns"),
    ("answer", "counter_deltas", "integrity_seal_compare_ns"),
    ("answer", "counter_deltas", "integrity_seal_result_download_ns"),
    ("answer", "counter_deltas", "host_heap_allocated_bytes"),
)


def _compare_machine_envelopes(actual, expected, number):
    left = actual.get("request", {}).get("machine_envelope")
    right = expected.get("request", {}).get("machine_envelope")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise TimingProbeError(f"question {number} machine answer envelope is missing")
    left_stable, right_stable = copy.deepcopy(left), copy.deepcopy(right)
    measurement_differences = []
    for path in _MACHINE_MEASUREMENT_PATHS:
        lvalue = left_stable
        rvalue = right_stable
        for key in path[:-1]:
            lvalue = lvalue.get(key) if isinstance(lvalue, dict) else None
            rvalue = rvalue.get(key) if isinstance(rvalue, dict) else None
        key = path[-1]
        if not isinstance(lvalue, dict) or not isinstance(rvalue, dict) or key not in lvalue or key not in rvalue:
            raise TimingProbeError(f"question {number} machine measurement field is missing: {'.'.join(path)}")
        actual_value, expected_value = lvalue.pop(key), rvalue.pop(key)
        if actual_value != expected_value:
            measurement_differences.append({"field": ".".join(path),
                                            "instrumented": actual_value,
                                            "baseline": expected_value})
    if left_stable != right_stable:
        raise TimingProbeError(f"question {number} machine answer differs outside explicit measurement fields")
    return measurement_differences


def _compare_to_resident_baseline(candidate, baseline, *, candidate_store, baseline_store):
    if candidate.get("status") != "PASS" or baseline.get("status") != "PASS":
        raise TimingProbeError("instrumented and baseline campaigns must both report PASS")
    stable_root_fields = (
        "execution_profile_id", "model_contract_id", "root_snapshot", "root_payload_sha256",
        "root_token_count", "context_file_sha256",
        "fixture_root_text_sha256", "total_actual_output_tokens", "root_stability")
    for field in stable_root_fields:
        if candidate.get(field) != baseline.get(field):
            raise TimingProbeError(f"instrumented campaign differs from resident baseline at {field}")
    actual_fingerprint = candidate.get("final_root_fingerprint")
    expected_fingerprint = baseline.get("final_root_fingerprint")
    if not isinstance(actual_fingerprint, dict) or not isinstance(expected_fingerprint, dict):
        raise TimingProbeError("final root checkpoint fingerprint is missing")
    for field in ("snapshot", "payload_sha256"):
        if actual_fingerprint.get(field) != expected_fingerprint.get(field):
            raise TimingProbeError(f"final root checkpoint differs from baseline at {field}")
    actual_files = actual_fingerprint.get("files")
    expected_files = expected_fingerprint.get("files")
    if not isinstance(actual_files, dict) or not isinstance(expected_files, dict) or set(actual_files) != set(expected_files):
        raise TimingProbeError("final root checkpoint file inventory differs from baseline")
    inode_changes = {}
    for name in actual_files:
        left, right = actual_files[name], expected_files[name]
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise TimingProbeError("final root checkpoint file receipt is malformed")
        for field in ("bytes", "mode", "sha256"):
            if left.get(field) != right.get(field):
                raise TimingProbeError(f"final root checkpoint file differs at {name}:{field}")
        inode_changes[name] = {"instrumented": left.get("inode"), "baseline": right.get("inode"),
                               "different": left.get("inode") != right.get("inode")}
    current = candidate.get("questions")
    original = baseline.get("questions")
    if not isinstance(current, list) or not isinstance(original, list) or len(current) != 3 or len(original) != 3:
        raise TimingProbeError("instrumented and baseline campaigns must each contain three questions")
    stable_question_fields = (
        "index", "question_sha256", "source_snapshot", "snapshot", "output_count",
        "output_token_ids", "controller_exact", "completion",
        "receipt", "lineage", "inactive_sibling_tips_unchanged", "work")
    rows = []
    for number, (actual, expected) in enumerate(zip(current, original), 1):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise TimingProbeError("campaign question record is malformed")
        if actual.get("controller_exact") is not True or expected.get("controller_exact") is not True:
            raise TimingProbeError(f"question {number} lacks exact canonical controller evidence")
        for field in stable_question_fields:
            if actual.get(field) != expected.get(field):
                raise TimingProbeError(f"question {number} differs from resident baseline at {field}")
        actual_answer = actual.get("request", {}).get("answer")
        expected_answer = expected.get("request", {}).get("answer")
        if not isinstance(actual_answer, dict) or not isinstance(expected_answer, dict):
            raise TimingProbeError(f"question {number} canonical answer is missing")
        for question, answer in ((actual, actual_answer), (expected, expected_answer)):
            if answer.get("tokens") != question.get("output_token_ids") or not isinstance(answer.get("text"), str):
                raise TimingProbeError(f"question {number} answer text/tokens do not match its output record")
        actual_text_sha = hashlib.sha256(actual_answer["text"].encode("utf-8")).hexdigest()
        expected_text_sha = hashlib.sha256(expected_answer["text"].encode("utf-8")).hexdigest()
        if actual_text_sha != expected_text_sha:
            raise TimingProbeError(f"question {number} canonical answer text differs from resident baseline")
        measurement_differences = _compare_machine_envelopes(actual, expected, number)
        rows.append({"question": number, "snapshot": actual["snapshot"],
                     "output_tokens": len(actual["output_token_ids"]),
                     "answer_text_sha256": actual_text_sha,
                     "machine_answer_stdout_hash_equal":
                         actual.get("machine_answer_sha256") == expected.get("machine_answer_sha256"),
                     "explicit_machine_measurement_differences": measurement_differences,
                     "controller_exact": True, "matched_fields": list(stable_question_fields)})
    store_comparisons = _compare_store_snapshots(candidate, baseline, candidate_store, baseline_store)
    for campaign, store_comparison in ((candidate, store_comparisons[0]),
                                       (baseline, store_comparisons[0])):
        fingerprint = campaign["final_root_fingerprint"]
        recorded_files = fingerprint.get("files")
        if not isinstance(recorded_files, dict):
            raise TimingProbeError("campaign root checkpoint file fingerprint is malformed")
        expected_names = set(store_comparison["files"])
        if set(recorded_files) != expected_names:
            raise TimingProbeError("campaign root checkpoint file names differ from the retained Store")
        for name in expected_names:
            for field in ("bytes", "mode", "sha256"):
                if recorded_files[name].get(field) != store_comparison["files"][name].get(field):
                    raise TimingProbeError(f"campaign root checkpoint receipt differs from Store at {name}:{field}")
    return {"questions": rows, "store_snapshot_comparisons": store_comparisons,
            "final_root_file_inode_changes": inode_changes,
            "excluded_volatile_fields": {
                "question.machine_answer_sha256": "hashes full machine stdout including timing/counter data",
                "checkpoint_file.inode": "independent Store copies necessarily allocate distinct inodes"}}


def _make_summary(result, result_hash, phase_hash, status_hash, baseline_hash,
                  phase_rows, instrumentation_path, baseline_comparison):
    timelines = []
    for index, question in enumerate(result.get("questions", []), 1):
        request = question.get("request", {}).get("observation", {}).get("requests", [])
        answer = question.get("request", {}).get("answer")
        if len(request) != 1 or not isinstance(answer, dict):
            raise TimingProbeError(f"question {index} observer request or canonical answer is missing")
        timelines.append(build_question_timeline(
            phase_rows, snapshot=question["snapshot"], observer_request=request[0],
            expected_token_ids=question.get("output_token_ids"), expected_answer=answer,
            expected_receipt=question.get("receipt")))
    if len(timelines) != 3:
        raise TimingProbeError("resident campaign did not return exactly three question timelines")
    instrumentation_path = Path(instrumentation_path)
    return {
        "schema": "cow-publication-timing-attribution-v1",
        "status": "PASS",
        "diagnostic_only": True,
        "offline_only": True,
        "production_source_modified": False,
        "instrumentation_sha256": _sha(instrumentation_path),
        "instrumentation_targets": [
            "BranchingModel.prepare_checkpoint", "BranchingModel._prepared_envelope",
            "BranchingModel._publish_checkpoint_parts", "BranchingModel._publish_checkpoint_file",
            "BranchingModel.publish_prepared_checkpoint", "BranchingModel.validate_checkpoint",
            "BranchingModel.save_prepared_branch", "ContextCoordinator._run",
            "ContextCoordinator._publish", "Request.emit(snapshot|durable)", "Store.move",
            "SharedCheckpointPool.reserve", "DiskCheckpointPool.reserve", "real._sync_dir",
            "os.fsync(publication scope only)", "os.replace(publication scope only)"],
        "trace_write_policy": "per-span JSONL append+flush; no fsync added by instrumentation",
        "runtime_root": result.get("runtime_root"),
        "runtime_cli_sha256": result.get("cli_sha256"),
        "profile_sha256": result.get("profile_sha256"),
        "baseline_campaign_result_sha256": baseline_hash,
        "baseline_comparison": baseline_comparison,
        "campaign_result_sha256": result_hash,
        "publication_phase_trace_sha256": phase_hash,
        "publication_phase_status_sha256": status_hash,
        "question_timelines": timelines,
        "limitations": [
            "Method wrappers add diagnostic overhead; timings are attribution data, not a stable benchmark.",
            "Nested span durations overlap and must not be summed.",
            "Raw observer text token IDs and exact canonical answer text were cross-checked offline.",
            "Full object/controller records were independently read from both retained Stores and compared; controller payloads are represented by canonical hashes."]}


def summarize_existing(*, result_path, phase_path, status_path, baseline_path,
                       instrumented_store, baseline_store,
                       output_dir, expected_hashes, instrumentation_path):
    """Summarize retained artifacts without starting the service or model campaign."""
    paths = {"result": Path(result_path), "phases": Path(phase_path),
             "status": Path(status_path), "baseline": Path(baseline_path)}
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(paths):
        raise TimingProbeError("expected SHA-256 values are required for all four input artifacts")
    hashes = {name: _sha(path) for name, path in paths.items()}
    if any(hashes[name] != expected_hashes[name] for name in paths):
        raise TimingProbeError("one or more retained input artifact hashes do not match")
    result = _read_json(paths["result"], "instrumented campaign result")
    baseline = _read_json(paths["baseline"], "resident baseline result")
    status = _read_json(paths["status"], "publication timing status")
    if (not isinstance(status, dict) or status.get("schema") != "cow-publication-phase-status-v1"
            or status.get("instrumentation_error") is not False):
        raise TimingProbeError("publication timing status is not a clean terminal receipt")
    baseline_comparison = _compare_to_resident_baseline(
        result, baseline, candidate_store=instrumented_store, baseline_store=baseline_store)
    try:
        phase_rows = [json.loads(line) for line in paths["phases"].read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        raise TimingProbeError("publication timing trace is malformed") from None
    summary = _make_summary(result, hashes["result"], hashes["phases"], hashes["status"],
                            hashes["baseline"], phase_rows, instrumentation_path,
                            baseline_comparison)
    output = Path(output_dir)
    if output.exists():
        raise TimingProbeError("offline summary output directory already exists")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "publication-timing-summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return summary


def _install_bootstrap_hook(helper, local_instrumentation):
    original = helper._Campaign._bootstrap
    expected_marker = "importlib.import_module('serve.cow_http').main()"

    def patched(self, ordinal):
        bootstrap = original(self, ordinal)
        run_dir = Path(self.run_dir)
        target = run_dir / "publication_timing_probe.py"
        target.write_bytes(Path(local_instrumentation).read_bytes())
        insertion = (
            "import publication_timing_probe\n"
            "publication_timing_probe.install_from_loaded(" + repr(str(run_dir / "publication-phases.jsonl")) + ")\n"
            + expected_marker
        )
        text = bootstrap.read_text(encoding="utf-8")
        if text.count(expected_marker) != 1:
            raise TimingProbeError("generated server bootstrap hook anchor is not unique")
        bootstrap.write_text(text.replace(expected_marker, insertion), encoding="utf-8", newline="\n")
        return bootstrap

    helper._Campaign._bootstrap = patched


def run(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "summarize-existing":
        offline = argparse.ArgumentParser(prog="summarize-existing")
        offline.add_argument("--campaign-result", type=Path, required=True)
        offline.add_argument("--phase-trace", type=Path, required=True)
        offline.add_argument("--recorder-status", type=Path, required=True)
        offline.add_argument("--baseline-result", type=Path, required=True)
        offline.add_argument("--instrumented-store", type=Path, required=True)
        offline.add_argument("--baseline-store", type=Path, required=True)
        offline.add_argument("--output-dir", type=Path, required=True)
        offline.add_argument("--instrumentation", type=Path,
                             default=Path(__file__).with_name("publication_timing_probe.py"))
        offline.add_argument("--expected-campaign-sha256", required=True)
        offline.add_argument("--expected-phase-sha256", required=True)
        offline.add_argument("--expected-status-sha256", required=True)
        offline.add_argument("--expected-baseline-sha256", required=True)
        args = offline.parse_args(argv[1:])
        summarize_existing(
            result_path=args.campaign_result, phase_path=args.phase_trace,
            status_path=args.recorder_status, baseline_path=args.baseline_result,
            instrumented_store=args.instrumented_store, baseline_store=args.baseline_store,
            output_dir=args.output_dir,
            expected_hashes={"result": args.expected_campaign_sha256,
                             "phases": args.expected_phase_sha256,
                             "status": args.expected_status_sha256,
                             "baseline": args.expected_baseline_sha256},
            instrumentation_path=args.instrumentation)
        return 0

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--campaign-script", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    runtime = args.runtime_root.resolve(strict=True)
    scripts = runtime / "experimental/branching_context/scripts"
    sys.path[:0] = [str(runtime), str(runtime / "tools"),
                    str(runtime / "experimental/branching_context/src"), str(scripts)]
    helper = importlib.import_module("run_cli_workflow_probe")
    _install_bootstrap_hook(helper, Path(__file__).with_name("publication_timing_probe.py"))
    campaign_script = args.campaign_script.resolve(strict=True)
    campaign = _load(campaign_script, "_pinned_resident_questions_campaign")
    code = campaign.main([*rest, "--runtime-root", str(runtime), "--run-dir", str(args.run_dir)])
    if code != 0:
        return code
    output = args.run_dir.resolve(strict=True)
    phase_path = output / "publication-phases.jsonl"
    recorder_status_path = output / "publication-phases.status.json"
    result_path = output / "result.json"
    result = json.loads(result_path.read_bytes())
    if result.get("status") != "PASS":
        raise TimingProbeError("frozen resident campaign did not report PASS")
    try:
        recorder_status = json.loads(recorder_status_path.read_bytes())
    except (OSError, ValueError):
        raise TimingProbeError("service timing recorder did not publish a valid status sidecar") from None
    if (not isinstance(recorder_status, dict)
            or recorder_status.get("schema") != "cow-publication-phase-status-v1"
            or recorder_status.get("instrumentation_error") is not False):
        raise TimingProbeError("service timing recorder reported an instrumentation failure")
    phase_rows = [json.loads(line) for line in phase_path.read_text(encoding="utf-8").splitlines() if line]
    instrumentation = Path(__file__).with_name("publication_timing_probe.py")
    status = json.loads(recorder_status_path.read_bytes())
    if (not isinstance(status, dict) or status.get("schema") != "cow-publication-phase-status-v1"
            or status.get("instrumentation_error") is not False):
        raise TimingProbeError("service timing recorder reported an instrumentation failure")
    baseline_comparison = []
    summary = _make_summary(result, _sha(result_path), _sha(phase_path),
                            _sha(recorder_status_path), None, phase_rows,
                            instrumentation, baseline_comparison)
    summary["resident_campaign_script"] = str(campaign_script)
    summary["resident_campaign_script_sha256"] = _sha(campaign_script)
    summary_path = output / "publication-timing-summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
