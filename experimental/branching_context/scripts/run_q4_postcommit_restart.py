"""Q4 HTTP lost-ack crash/restart probe after durable branch publication.

This is a process-crash test, not a power-loss test. A test-only observer
blocks the real BranchingModel hook immediately after Store.move commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback
import uuid


class PostcommitRestartError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_commit_witness(witness, *, operation_id, branch, snapshot,
                            expected_revision, server_pid):
    if (type(witness) is not dict or witness.get("event") != "branch_committed_before_ack"
            or witness.get("operation_id") != operation_id or witness.get("branch") != branch
            or witness.get("snapshot") != snapshot
            or type(witness.get("expected_revision")) is not int
            or witness["expected_revision"] != expected_revision
            or type(witness.get("pid")) is not int or witness["pid"] != server_pid
            or type(witness.get("thread_id")) is not int or witness["thread_id"] <= 0):
        raise PostcommitRestartError("Fault witness is not bound to the expected committed operation")
    return witness


def validate_replay_completion(events, original_receipt, *, branch, snapshot, revision):
    complete = [row for row in events if type(row) is dict and row.get("event") == "complete"]
    if len(complete) != 1:
        raise PostcommitRestartError("Replay did not produce exactly one complete event")
    event = complete[0]
    answer, receipt = event.get("answer"), event.get("receipt")
    branch_row = receipt.get("branch") if type(receipt) is dict else None
    if (type(answer) is not dict or answer.get("recovered") is not True
            or answer.get("snapshot") != snapshot or type(answer.get("tokens")) is not list
            or any(type(token) is not int for token in answer["tokens"])
            or type(receipt) is not dict or receipt != original_receipt
            or type(branch_row) is not dict or branch_row.get("name") != branch
            or branch_row.get("tip") != snapshot
            or type(branch_row.get("revision")) is not int or branch_row["revision"] != revision):
        raise PostcommitRestartError("Replay differs from the original durable receipt")
    return receipt


def validate_single_move_delta(before, after, operation_id):
    if type(before) is not list or type(after) is not list or after != before + [operation_id]:
        raise PostcommitRestartError("Expected exactly one move under the original operation ID")
    return [operation_id]


def validate_replay_move_unchanged(before, after):
    if type(before) is not list or type(after) is not list or after != before:
        raise PostcommitRestartError("Replay changed the canonical branch move log")


def validate_idempotency_conflict(status, body, operation_id):
    message = body.get("error") if type(body) is dict else None
    if type(status) is not int or status != 400 or not isinstance(message, str) \
            or "Operation ID reused" not in message:
        raise PostcommitRestartError(f"Changed request with reused operation ID was not rejected: {operation_id}")
    return message


def validate_owned_kill_scope(server, native, *, server_pid, server_start,
                              native_pid, native_start, process_group, expected_native_exe):
    if (type(server) is not dict or type(native) is not dict
            or server.get("pid") != server_pid or server.get("start_time") != server_start
            or server.get("process_group") != process_group or process_group != server_pid
            or native.get("pid") != native_pid or native.get("start_time") != native_start
            or native.get("parent_pid") != server_pid or native.get("process_group") != process_group
            or Path(str(native.get("exe", ""))).resolve() != Path(expected_native_exe).resolve()):
        raise PostcommitRestartError("Refusing SIGKILL outside the exact owned service/native process group")
    return dict(server_pid=server_pid, native_pid=native_pid, process_group=process_group,
                server_start_time=server_start, native_start_time=native_start)


def validate_crash_terminal(server_exit, native_absent, signal_number=signal.SIGKILL):
    if type(server_exit) is not int or server_exit != -signal_number or native_absent is not True:
        raise PostcommitRestartError("Crash evidence does not show owned SIGKILL and native-process absence")


def validate_no_ack_events(events):
    if type(events) is not list:
        raise PostcommitRestartError("Lost-ACK event stream is malformed")
    if any(type(row) is dict and row.get("event") in ("durable", "complete") for row in events):
        raise PostcommitRestartError("Durable or complete acknowledgment arrived before the simulated crash")


def _proc_state(proc_root: Path, pid: int):
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return "absent"
    end = raw.rfind(")")
    fields = raw[end + 2:].split() if end >= 0 else []
    if not fields:
        raise PostcommitRestartError("Cannot read process state after crash")
    return "zombie" if fields[0] == "Z" else "running"


def validate_store_commit(store, operation_id, branch, source_head, target_snapshot):
    head = store.head(branch)
    record = store._read_branch(branch)
    receipt = record.get("receipts", {}).get(operation_id)
    checkpoint = store.checkpoint_receipt(operation_id)
    events = [row for row in record.get("events", [])
              if row.get("type") == "move" and row.get("operation_id") == operation_id]
    revision = source_head["revision"] + 1
    target = store.get_snapshot(target_snapshot)
    result = receipt.get("result") if type(receipt) is dict else None
    descriptor = checkpoint.get("checkpoint_receipt") if type(checkpoint) is dict else None
    if (head.name != branch or head.revision != revision or head.tip != target_snapshot
            or type(result) is not dict or result.get("name") != branch
            or result.get("revision") != revision or result.get("tip") != target_snapshot
            or checkpoint.get("result") != result or type(descriptor) is not dict
            or descriptor.get("snapshot") != target_snapshot
            or target.get("parent") != source_head.get("tip") or len(events) != 1
            or events[0].get("from") != source_head.get("tip")
            or events[0].get("to") != target_snapshot
            or events[0].get("from_revision") != source_head["revision"]
            or events[0].get("to_revision") != revision):
        raise PostcommitRestartError("Store lacks the exact durable branch/checkpoint operation receipt")
    http_receipt = dict(branch=result, snapshot=target_snapshot,
        payload_sha256=descriptor.get("payload_sha256"),
        model_contract_id=descriptor.get("model_contract_id"),
        execution_profile_id=descriptor.get("execution_profile_id"))
    return dict(branch=dict(name=head.name, revision=head.revision, tip=head.tip, root=head.root),
                store_receipt=checkpoint, http_receipt=http_receipt, move_events=events)


def _write_wrapper(run_dir: Path, operation_id: str, witness_path: Path) -> Path:
    path = run_dir / "postcommit_hook_wrapper.py"
    source = r'''import json, os, sys, threading, time
from context_time_travel.real import BranchingModel
operation_id, witness_path = sys.argv[1], sys.argv[2]
server_args = sys.argv[3:]
local = threading.local()
original_init = BranchingModel.__init__
original_save = BranchingModel.save_prepared_branch
def append_witness(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(witness_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw): offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(os.path.dirname(witness_path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(dfd)
    finally: os.close(dfd)
def observed_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_fault = self._checkpoint_fault
    blocked = threading.Event()
    def fault(point):
        if point == "branch_committed_before_ack" and getattr(local, "operation_id", None) == operation_id:
            append_witness({"event": point, "operation_id": operation_id, "branch": local.branch,
                "snapshot": local.snapshot, "expected_revision": local.expected_revision,
                "pid": os.getpid(), "thread_id": threading.get_ident(), "observed_ns": time.time_ns()})
            blocked.wait()
        return original_fault(point)
    self._checkpoint_fault = fault
def observed_save(self, name, revision, prepared, op, *, presentation=None):
    prior = {key: getattr(local, key, None) for key in ("operation_id", "branch", "snapshot", "expected_revision")}
    local.operation_id, local.branch = op, name
    local.snapshot, local.expected_revision = prepared.snapshot, revision
    try: return original_save(self, name, revision, prepared, op, presentation=presentation)
    finally:
        for key, value in prior.items():
            if value is None:
                try: delattr(local, key)
                except AttributeError: pass
            else: setattr(local, key, value)
BranchingModel.__init__ = observed_init
BranchingModel.save_prepared_branch = observed_save
sys.argv = ["serve.cow_http", *server_args]
from serve.cow_http import main
main()
'''
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return path


def _read_proc(proc_root: Path, pid: int) -> dict:
    base = proc_root / str(pid)
    raw = (base / "stat").read_text(encoding="ascii")
    end = raw.rfind(")")
    fields = raw[end + 2:].split() if end >= 0 else []
    if len(fields) < 20:
        raise PostcommitRestartError("Cannot parse process identity")
    return dict(pid=pid, state=fields[0], parent_pid=int(fields[1]), process_group=int(fields[2]),
                session=int(fields[3]), start_time=fields[19], exe=os.readlink(base / "exe"),
                argv=[x.decode("utf-8", "replace") for x in (base / "cmdline").read_bytes().split(b"\0") if x])


def _group_processes(proc_root: Path, pgid: int) -> list[dict]:
    rows = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            try:
                row = _read_proc(proc_root, int(entry.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if row["process_group"] == pgid:
                rows.append(row)
    return sorted(rows, key=lambda row: row["pid"])


def _kill_owned_group(service, native_info, proc_root: Path):
    process = service.process
    if process is None or process.poll() is not None:
        raise PostcommitRestartError("HTTP leader exited before the authorized crash point")
    server_pid = process.pid
    server = _read_proc(proc_root, server_pid)
    native = _read_proc(proc_root, native_info["pid"])
    group = server["process_group"]
    members = _group_processes(proc_root, group)
    validate_owned_kill_scope(server, native, server_pid=server_pid,
        server_start=server["start_time"], native_pid=native_info["pid"],
        native_start=native_info["start_time"], process_group=group,
        expected_native_exe=native_info["exe"])
    if {row["pid"] for row in members} != {server_pid, native_info["pid"]}:
        raise PostcommitRestartError("Refusing SIGKILL because the process group has unexpected members")
    if (_read_proc(proc_root, server_pid)["start_time"] != server["start_time"]
            or _read_proc(proc_root, native_info["pid"])["start_time"] != native_info["start_time"]):
        raise PostcommitRestartError("Owned PID identity changed immediately before SIGKILL")
    os.killpg(group, signal.SIGKILL)
    server_exit = process.wait(timeout=10)
    if service.reader is not None:
        service.reader.join(timeout=5)
        if service.reader.is_alive():
            raise PostcommitRestartError("Service stdout reader did not terminate after crash")
    if process.stdout is not None:
        process.stdout.close()
    if service._stdout_stream is not None:
        service._stdout_stream.close()
    deadline = time.monotonic() + 10
    native_state = "running"
    while time.monotonic() < deadline:
        native_state = _proc_state(proc_root, native_info["pid"])
        if native_state in ("absent", "zombie"):
            break
        time.sleep(.05)
    validate_crash_terminal(server_exit, native_state in ("absent", "zombie"))
    return dict(server_pid=server_pid, server_exit_code=server_exit, native_pid=native_info["pid"],
        native_exit_code=None, native_state=native_state,
        native_absent=native_state == "absent", native_not_running=native_state in ("absent", "zombie"),
        process_group=group, prekill_members=members)


def _stream_until_snapshot(client, route, body, progress):
    output, done = queue.Queue(), threading.Event()
    def read():
        conn = client._connection()
        response = None
        try:
            raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            conn.request("POST", client.prefix + route, body=raw,
                        headers={"Content-Type": "application/json", "Connection": "close"})
            response = conn.getresponse()
            output.put(("status", response.status, response.reason))
            if response.status != 200:
                output.put(("body", response.read().decode("utf-8", "replace")))
                return
            while True:
                line = response.fp.readline()
                if not line:
                    output.put(("eof",))
                    return
                event = json.loads(line)
                if type(event) is not dict or not isinstance(event.get("event"), str):
                    output.put(("malformed", repr(event)))
                    return
                progress.event(body["operation_id"], event)
                output.put(("event", event))
                if event.get("event") in ("complete", "error"):
                    return
        except BaseException as error:
            output.put(("exception", type(error).__name__, str(error)))
        finally:
            if response is not None:
                response.close()
            conn.close()
            done.set()
    thread = threading.Thread(target=read, name="q4-postcommit-http-reader", daemon=True)
    thread.start()
    return output, done, thread


def _load_helpers(args):
    runtime_root = Path(args.runtime_root).resolve(strict=True)
    api_src, api_scripts = Path(args.api_src).resolve(strict=True), Path(args.api_scripts).resolve(strict=True)
    if not api_src.is_relative_to(runtime_root) or not api_scripts.is_relative_to(runtime_root):
        raise PostcommitRestartError("API source/scripts must be beneath explicit runtime root")
    sys.path[:0] = [str(api_scripts), str(api_src), str(runtime_root)]
    sys.path.insert(0, str(runtime_root / "experimental" / "branching_context" / "scripts"))
    harness_scripts = Path(__file__).resolve().parent
    required_helpers = (harness_scripts / "run_cow_client_recovery.py",
                        harness_scripts / "run_q4_publication_fault.py")
    if any(not path.is_file() or path.is_symlink() for path in required_helpers):
        raise PostcommitRestartError("Required frozen harness helper scripts are missing beside this runner")
    sys.path.insert(0, str(harness_scripts))
    import run_cow_client_recovery as recovery_module
    import run_q4_publication_fault as publication_module
    if (Path(recovery_module.__file__).resolve() != required_helpers[0].resolve()
            or Path(publication_module.__file__).resolve() != required_helpers[1].resolve()):
        raise PostcommitRestartError("Imported recovery helper modules do not match the adjacent pinned files")
    from context_time_travel.store import Store
    from run_cow_client_recovery import (
        HttpClient, ProgressTracker, ServiceProcess, _fork, _ready_url, _trace_worker,
    )
    from run_q4_publication_fault import (
        _canonical_history_sha, _children, _complete_receipt, _generation_body, _native_trace_rows,
        _parse_head_and_history, _runtime_closure, _sha256, _snapshot_file_fingerprint,
        _snapshot_record_fingerprint, _wait_http_idle,
        validate_source_checkpoint_unchanged,
    )
    return locals()


def _parse_store_head_history(store, branch):
    head = store.head(branch)
    tokens = tuple(store.lineage(head.tip)[1])
    snapshot_ids = []
    cursor = head.tip
    while cursor is not None:
        snapshot_ids.append(cursor)
        cursor = store.get_snapshot(cursor).get("parent")
    snapshot_ids.reverse()
    records = [dict(snapshot=snapshot, record=store.get_snapshot(snapshot)) for snapshot in snapshot_ids]
    tip_record = store.get_snapshot(head.tip)
    return dict(head=dict(name=head.name, revision=head.revision, tip=head.tip, root=head.root),
        history_sha256=hashlib.sha256(_canonical(records)).hexdigest(),
        controller_status=tip_record.get("controller", {}).get("status"), token_count=len(tokens))


def _wait_witness(path: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if type(value) is dict:
                    return value
            except (OSError, ValueError):
                pass
        time.sleep(.05)
    raise PostcommitRestartError("Timed out waiting for post-CAS hook witness")


def _drain_crashed_stream(events, done, thread, timeout, initial_events=()):
    thread.join(timeout)
    if thread.is_alive() or not done.is_set():
        raise PostcommitRestartError("HTTP stream reader remained live after SIGKILL")
    rows, errors = list(initial_events), []
    while True:
        try:
            item = events.get_nowait()
        except queue.Empty:
            break
        if item[0] == "event":
            rows.append(item[1])
        elif item[0] == "exception":
            errors.append(item[1:])
    if any(row.get("event") == "complete" for row in rows):
        raise PostcommitRestartError("Old socket delivered a completion acknowledgment before crash")
    validate_no_ack_events(rows)
    if not any(row.get("event") == "snapshot" for row in rows):
        raise PostcommitRestartError("Client did not receive a provisional snapshot before crash")
    return rows, errors


def _stop_and_record(label, service, records, errors):
    try:
        records.append({"service": label, "receipt": service.stop()})
    except BaseException as error:
        errors.append({"service": label, "error": f"{type(error).__name__}: {error}"})


def run_probe(args):
    if not sys.platform.startswith("linux"):
        raise PostcommitRestartError("Process-group crash qualification requires Linux")
    run_dir = Path(args.run_dir).expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    api = _load_helpers(args)
    profile_path, store_path = Path(args.profile).resolve(strict=True), Path(args.store).resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    api["_runtime_closure"](profile, Path(args.runtime_root).resolve(strict=True))
    native = Path(profile["execution"]["sut_command"][0]).resolve(strict=True)
    if api["_sha256"](native) != args.expected_native_sha256:
        raise PostcommitRestartError("Native binary SHA differs from selected Q4 build")
    if api["_sha256"](profile_path) != args.expected_profile_sha256:
        raise PostcommitRestartError("Profile SHA differs from selected Q4 profile")
    execution = profile.get("execution", {})
    if execution.get("GPU") != [0] or execution.get("capacity") != 98304 or execution.get("prefill_chunk") != 8192:
        raise PostcommitRestartError("Profile is not the pinned Q4 GPU0 capacity/chunk configuration")
    if store_path.is_symlink() or not (store_path / "references").is_dir() or not (store_path / "checkpoints").is_dir():
        raise PostcommitRestartError("Input must be a private copied Store with canonical refs and checkpoints")

    result_path = run_dir / "result.json"
    progress = api["ProgressTracker"](run_dir, "q4-postcommit-lost-ack")
    progress.start()
    result = dict(schema="q4-postcommit-lost-ack-v1", status="RUNNING",
        scope="real Q4 COW HTTP; SIGKILL after durable Store CAS and before ACK; process crash, not power loss",
        run_directory=str(run_dir), profile_path=str(profile_path), profile_sha256=api["_sha256"](profile_path),
        native_binary=str(native), native_sha256=api["_sha256"](native), store_path=str(store_path),
        runtime_root=str(Path(args.runtime_root).resolve()), api_src=str(Path(args.api_src).resolve()),
        api_scripts=str(Path(args.api_scripts).resolve()), harness_sha256=_sha(Path(__file__).resolve()),
        helper_sources={
            "run_cow_client_recovery.py": dict(path=str(Path(api["recovery_module"].__file__).resolve()),
                sha256=_sha(Path(api["recovery_module"].__file__).resolve())),
            "run_q4_publication_fault.py": dict(path=str(Path(api["publication_module"].__file__).resolve()),
                sha256=_sha(Path(api["publication_module"].__file__).resolve())),
        }, steps=[])
    service = service2 = None
    stream_thread = None
    try:
        nonce = uuid.uuid4().hex
        target_branch = "Q4Crash" + nonce[:16]
        target_operation = "op-q4-crash-" + nonce
        followup_operation = "op-q4-after-" + nonce
        witness_path = run_dir / "commit-witness.json"
        wrapper = _write_wrapper(run_dir, target_operation, witness_path)
        progress.phase("start_fault_observer_service")
        trace1, native_trace1 = run_dir / "pool-trace-crash.json", run_dir / "native-trace-crash.jsonl"
        native_trace1.touch(mode=0o600, exist_ok=False)
        server_args = ["--profile", str(profile_path), "--store", str(store_path),
            "--host", "127.0.0.1", "--port", "0", "--text", "--ram-gib", str(args.ram_gib),
            "--disk-gib", str(args.disk_gib), "--pool-trace-output", str(trace1)]
        env1 = os.environ.copy()
        env1["PYTHONPATH"] = os.pathsep.join((str(Path(args.runtime_root).resolve()),
            str(Path(args.api_src).resolve()), str(Path(args.api_scripts).resolve())))
        env1["CTT_NATIVE_TRACE"] = str(native_trace1)
        service = api["ServiceProcess"]([sys.executable, str(wrapper), target_operation,
            str(witness_path), *server_args], cwd=args.runtime_root, env=env1,
            stdout_path=run_dir / "service-crash.stdout.log", stderr_path=run_dir / "service-crash.stderr.log",
            timeout=args.timeout, progress=progress)
        ready = service.start()
        server_pid = service.process.pid
        progress.server_started(server_pid)
        client = api["HttpClient"](api["_ready_url"](ready), timeout=args.timeout, progress=progress)
        status, health = client.get("/health")
        if status != 200 or health.get("service") != "context-cow":
            raise PostcommitRestartError("Fault observer did not start the COW HTTP service")
        if ready.get("execution_profile_id") != profile.get("execution_profile_id"):
            raise PostcommitRestartError("First service profile differs from the selected profile")
        source_head, source_history = api["_parse_head_and_history"](client, args.branch)
        source_snapshot = source_head["tip"]
        store = api["Store"](store_path)
        source_store_state = _parse_store_head_history(store, args.branch)
        if source_store_state["head"] != source_head:
            raise PostcommitRestartError("Canonical Store and HTTP source heads differ at startup")
        source_checkpoint = api["_snapshot_file_fingerprint"](store_path, source_snapshot)
        source_record = api["_snapshot_record_fingerprint"](store_path, source_snapshot)
        source_history_sha = api["_canonical_history_sha"](source_history)
        result["source_before"] = dict(head=source_head, checkpoint_files=source_checkpoint,
            object_sha256=source_record, history_sha256=source_history_sha)

        children = api["_children"](server_pid)
        if len(children) != 1:
            raise PostcommitRestartError(f"Expected one native worker child after service ready, got {children}")
        native_pid = children[0]
        native_info, server_info = _read_proc(Path("/proc"), native_pid), _read_proc(Path("/proc"), server_pid)
        progress.native_started(native_pid)
        if Path(native_info["exe"]).resolve() != native or native_info["parent_pid"] != server_pid:
            raise PostcommitRestartError("Native child PID/executable/parent differs from the selected service")
        if native_info["process_group"] != server_info["process_group"] or server_info["process_group"] != server_pid:
            raise PostcommitRestartError("Service and native worker do not share the owned process group")
        result["initial_processes"] = dict(server=server_info, native=native_info)

        progress.phase("fork_isolated_answer_branch")
        fork_head = api["_fork"](client, args.branch, source_snapshot, target_branch, "op-q4-fork-" + nonce)
        if fork_head.get("tip") != source_snapshot:
            raise PostcommitRestartError("Fork does not retain the selected source snapshot")
        warm_head, warm_history = api["_parse_head_and_history"](client, target_branch)
        warm_route, warm_body = api["_generation_body"](warm_history, warm_head, target_branch,
            "op-q4-warm-" + nonce, 1, "Warm the isolated Q4 crash-test branch.")
        warm_events, warm_disconnected = client.post_stream(warm_route, warm_body)
        if warm_disconnected:
            raise PostcommitRestartError("Warmup stream disconnected")
        warm_complete = api["_complete_receipt"](warm_events, branch=target_branch,
            expected_revision=warm_head["revision"])
        warm_receipt = warm_complete.get("receipt")
        if type(warm_receipt) is not dict:
            raise PostcommitRestartError("Warmup returned no durable receipt")
        warm_snapshot = warm_receipt["snapshot"]
        parent_head, parent_history = api["_parse_head_and_history"](client, target_branch)
        api["validate_source_checkpoint_unchanged"](source_checkpoint,
            api["_snapshot_file_fingerprint"](store_path, source_snapshot), phase="after warmup")
        source_after, history_after = api["_parse_head_and_history"](client, args.branch)
        if source_after != source_head or api["_canonical_history_sha"](history_after) != source_history_sha:
            raise PostcommitRestartError("Warmup mutated the original branch/history")
        result["warmup"] = dict(route=warm_route, request=warm_body, receipt=warm_receipt,
            checkpoint_files=api["_snapshot_file_fingerprint"](store_path, warm_snapshot),
            event_names=[row.get("event") for row in warm_events])

        api["_wait_http_idle"](client, args.timeout, progress)
        parent_head, parent_history = api["_parse_head_and_history"](client, target_branch)
        pre_target_record = store._read_branch(target_branch)
        moves_before_target = [row.get("operation_id") for row in pre_target_record.get("events", [])
            if row.get("type") == "move"]
        route, body = api["_generation_body"](parent_history, parent_head, target_branch,
            target_operation, args.max_tokens, "Continue the Q4 isolated answer.")
        body_digest = hashlib.sha256(_canonical(dict(route=route, body=body))).hexdigest()
        result["lost_ack_request"] = dict(route=route, body=body, canonical_sha256=body_digest,
            parent_head=parent_head, parent_history_sha256=api["_canonical_history_sha"](parent_history))
        result["steps"].append("fork_and_warm_completed")
        _atomic_json(result_path, result)

        progress.phase("request_until_postcommit_hook")
        event_queue, stream_done, stream_thread = _stream_until_snapshot(client, route, body, progress)
        status_item = event_queue.get(timeout=args.timeout)
        if status_item[0] != "status" or status_item[1] != 200:
            raise PostcommitRestartError(f"Target request was not accepted: {status_item}")
        provisional = []
        while True:
            item = event_queue.get(timeout=args.timeout)
            if item[0] != "event":
                raise PostcommitRestartError(f"HTTP stream ended before committed snapshot: {item}")
            provisional.append(item[1])
            validate_no_ack_events(provisional)
            if item[1].get("event") == "snapshot":
                target_snapshot = item[1].get("snapshot")
                break
        witness = _wait_witness(witness_path, args.timeout)
        validate_commit_witness(witness, operation_id=target_operation, branch=target_branch,
            snapshot=target_snapshot, expected_revision=parent_head["revision"], server_pid=server_pid)
        commit = validate_store_commit(store, target_operation, target_branch, parent_head, target_snapshot)
        committed_record = store._read_branch(target_branch)
        moves_after_commit = [row.get("operation_id") for row in committed_record.get("events", [])
            if row.get("type") == "move"]
        validate_single_move_delta(moves_before_target, moves_after_commit, target_operation)
        checkpoint_files = api["_snapshot_file_fingerprint"](store_path, target_snapshot)
        target_object = api["_snapshot_record_fingerprint"](store_path, target_snapshot)
        source_tokens = tuple(store.lineage(parent_head["tip"])[1])
        target_tokens = tuple(store.lineage(target_snapshot)[1])
        if target_tokens[:len(source_tokens)] != source_tokens:
            raise PostcommitRestartError("Committed target lineage does not preserve its parent token history")
        status, process_status = client.get("/cow/operation?operation=" + target_operation)
        if (status != 200 or process_status.get("scope") != "coordinator-process"
                or process_status.get("durable") is not False or "receipt" in process_status):
            raise PostcommitRestartError("Old HTTP service did not expose the request as unacknowledged")
        result["durable_before_crash"] = dict(witness=witness, provisional_events=provisional,
            process_status=process_status, store_commit=commit, checkpoint_files=checkpoint_files,
            snapshot_object_sha256=target_object, source_token_count=len(source_tokens),
            target_token_count=len(target_tokens))
        result["steps"].append("durable_cas_and_checkpoint_observed_before_ack")
        _atomic_json(result_path, result)

        progress.phase("sigkill_verified_owned_process_group")
        crash = _kill_owned_group(service, native_info, Path("/proc"))
        service = None
        crashed_events, crashed_errors = _drain_crashed_stream(
            event_queue, stream_done, stream_thread, args.timeout, initial_events=provisional)
        stream_thread = None
        source_final = _parse_store_head_history(store, args.branch)
        if (source_final["head"] != source_head
                or source_final["history_sha256"] != source_store_state["history_sha256"]):
            raise PostcommitRestartError("Original branch/history changed across process crash")
        api["validate_source_checkpoint_unchanged"](source_checkpoint,
            api["_snapshot_file_fingerprint"](store_path, source_snapshot), phase="across process crash")
        result["crash"] = dict(processes=crash, client_events=crashed_events,
            client_reader_errors=crashed_errors, original_source_unchanged=True,
            pool_trace_written=trace1.is_file())
        result["steps"].append("http_and_native_process_group_killed_after_durable_commit")
        _atomic_json(result_path, result)

        progress.phase("restart_new_service_same_store")
        trace2, native_trace2 = run_dir / "pool-trace-restart.json", run_dir / "native-trace-restart.jsonl"
        native_trace2.touch(mode=0o600, exist_ok=False)
        env2 = os.environ.copy()
        env2["PYTHONPATH"] = env1["PYTHONPATH"]
        env2["CTT_NATIVE_TRACE"] = str(native_trace2)
        command2 = [sys.executable, "-m", "serve.cow_http", "--profile", str(profile_path),
            "--store", str(store_path), "--host", "127.0.0.1", "--port", "0", "--text",
            "--ram-gib", str(args.ram_gib), "--disk-gib", str(args.disk_gib),
            "--pool-trace-output", str(trace2)]
        service2 = api["ServiceProcess"](command2, cwd=args.runtime_root, env=env2,
            stdout_path=run_dir / "service-restart.stdout.log", stderr_path=run_dir / "service-restart.stderr.log",
            timeout=args.timeout, progress=progress)
        ready2 = service2.start()
        progress.server_started(service2.process.pid)
        client2 = api["HttpClient"](api["_ready_url"](ready2), timeout=args.timeout, progress=progress)
        status, health2 = client2.get("/health")
        if status != 200 or health2.get("service") != "context-cow" \
                or ready2.get("execution_profile_id") != profile.get("execution_profile_id"):
            raise PostcommitRestartError("Restarted service identity/profile is invalid")
        status, operation_status = client2.get("/cow/operation?operation=" + target_operation)
        if status != 404 or type(operation_status) is not dict:
            raise PostcommitRestartError("Coordinator-scoped operation status incorrectly survived restart")
        native_trace_before_replay = native_trace2.read_bytes()
        if hashlib.sha256(_canonical(dict(route=route, body=body))).hexdigest() != body_digest:
            raise PostcommitRestartError("Replay body or route differs from the original request")

        progress.phase("exact_same_id_request_replay")
        replay_events, replay_disconnected = client2.post_stream(route, body)
        if replay_disconnected:
            raise PostcommitRestartError("Exact replay disconnected")
        replay_completion = _one_complete(replay_events)
        original_http_receipt = commit["http_receipt"]
        validate_replay_completion(replay_events, original_http_receipt, branch=target_branch,
            snapshot=target_snapshot, revision=parent_head["revision"] + 1)
        native_trace_after_replay = native_trace2.read_bytes()
        if native_trace_after_replay != native_trace_before_replay:
            raise PostcommitRestartError("Same-ID replay issued native work after restart")
        replay_store = api["Store"](store_path)
        replay_head = replay_store.head(target_branch)
        replay_record = replay_store._read_branch(target_branch)
        replay_moves = [row.get("operation_id") for row in replay_record.get("events", [])
                        if row.get("type") == "move"]
        validate_replay_move_unchanged(moves_after_commit, replay_moves)
        if replay_head.tip != target_snapshot or replay_head.revision != parent_head["revision"] + 1:
            raise PostcommitRestartError("Replay changed the already-committed branch head")
        if api["_snapshot_file_fingerprint"](store_path, target_snapshot) != checkpoint_files:
            raise PostcommitRestartError("Replay changed the already-committed numerical checkpoint")
        result["replay"] = dict(events=replay_events, receipt=replay_completion["receipt"],
            native_trace_unchanged=True, native_trace_sha256=hashlib.sha256(native_trace_after_replay).hexdigest(),
            move_operation_ids=replay_moves, head=dict(name=replay_head.name, revision=replay_head.revision,
                tip=replay_head.tip, root=replay_head.root))
        result["steps"].append("replayed_same_operation_without_native_work_or_second_move")
        _atomic_json(result_path, result)

        progress.phase("reject_changed_body_same_operation_id")
        changed_body = dict(body)
        changed_body["max_tokens"] = body["max_tokens"] + 1
        status, conflict = client2.post_json(route, changed_body)
        conflict_message = validate_idempotency_conflict(status, conflict, target_operation)
        if native_trace2.read_bytes() != native_trace_after_replay:
            raise PostcommitRestartError("Changed same-ID request performed native work")
        result["same_id_changed_body"] = dict(status=status, error=conflict_message, no_native_work=True)
        result["steps"].append("same_operation_id_changed_arguments_rejected")
        _atomic_json(result_path, result)

        progress.phase("serve_following_new_operation")
        next_head, next_history = api["_parse_head_and_history"](client2, target_branch)
        next_route, next_body = api["_generation_body"](next_history, next_head, target_branch,
            followup_operation, args.followup_tokens, "Verify the restarted branch accepts another request.")
        trace_before_followup = native_trace2.read_bytes()
        follow_events, disconnected = client2.post_stream(next_route, next_body)
        if disconnected:
            raise PostcommitRestartError("Following operation disconnected after restart replay")
        follow_complete = api["_complete_receipt"](follow_events, branch=target_branch,
            expected_revision=next_head["revision"])
        api["_wait_http_idle"](client2, args.timeout, progress)
        trace_after_followup = native_trace2.read_bytes()
        if trace_after_followup == trace_before_followup:
            raise PostcommitRestartError("Following operation produced no native trace activity")
        source_after = _parse_store_head_history(api["Store"](store_path), args.branch)
        if (source_after["head"] != source_head
                or source_after["history_sha256"] != source_store_state["history_sha256"]):
            raise PostcommitRestartError("Following request changed the original source branch/history")
        api["validate_source_checkpoint_unchanged"](source_checkpoint,
            api["_snapshot_file_fingerprint"](store_path, source_snapshot), phase="after following request")
        if api["_snapshot_file_fingerprint"](store_path, target_snapshot) != checkpoint_files:
            raise PostcommitRestartError("Following request changed its committed parent checkpoint")
        result["followup"] = dict(route=next_route, request=next_body, events=follow_events,
            receipt=follow_complete.get("receipt"), native_trace_changed=True,
            original_source_unchanged=True, committed_parent_unchanged=True)
        result["steps"].append("following_request_succeeded_after_replay")

        stopped = service2.stop()
        service2 = None
        if not stopped.get("graceful") or stopped.get("exit_code") != 0:
            raise PostcommitRestartError("Restarted service did not close successfully")
        result["restart_service_shutdown"] = stopped
        trace_record = api["_trace_worker"](trace2, stopped["pid"])
        result["restart_native_shutdown_trace"] = trace_record
        result["status"] = "PASS"
        progress.finish("PASS")
        _atomic_json(result_path, result)
        return result
    except BaseException as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        try:
            progress.finish("FAIL", result["error"])
            _atomic_json(result_path, result)
        except BaseException:
            pass
        raise
    finally:
        stop_receipts, stop_errors = [], []
        if stream_thread is not None and stream_thread.is_alive() and service is not None:
            stopping, service = service, None
            _stop_and_record("fault-observer", stopping, stop_receipts, stop_errors)
            stream_thread.join(timeout=2)
        for label, running in (("restart", service2), ("fault-observer", service)):
            if running is not None:
                if label == "restart": service2 = None
                else: service = None
                _stop_and_record(label, running, stop_receipts, stop_errors)
        try:
            progress.close()
        except BaseException as error:
            stop_errors.append({"service": "progress", "error": f"{type(error).__name__}: {error}"})
        result["cleanup"] = dict(stop_receipts=stop_receipts, errors=stop_errors)
        if stop_errors:
            result["status"] = "FAIL"
            result.setdefault("error", "Cleanup did not complete cleanly")
        try:
            _atomic_json(result_path, result)
        except BaseException:
            # Earlier phase writes remain available; a terminal cleanup write
            # failure cannot safely be represented by a fabricated receipt.
            pass


def _one_complete(events):
    complete = [row for row in events if row.get("event") == "complete"]
    if len(complete) != 1:
        raise PostcommitRestartError("Expected exactly one complete event")
    return complete[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--api-src", required=True)
    parser.add_argument("--api-scripts", required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--followup-tokens", type=int, default=4)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.max_tokens <= 0 or args.followup_tokens <= 0:
            raise PostcommitRestartError("Output bounds must be positive")
        result = run_probe(args)
        if type(result) is not dict or result.get("status") != "PASS":
            print("postcommit restart FAIL: run completed without PASS status", file=sys.stderr)
            return 1
        return 0
    except BaseException as error:
        print(f"postcommit restart FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
