"""Real-GPU consumed-token cancellation and same-root recovery probe.

This harness consumes already-executed, profile-matched A/B root-reuse
references. It does not create a fresh numerical oracle or make a gate claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
import time
import traceback


FRONTIER_WIDTH = 4 * 248320


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_cancel_evidence(*, state, events, branch_before, branch_after,
                             snapshot_cancelled, durable_cancelled,
                             operation_receipt_present, trace, suffix_tokens):
    """Validate that cancellation drained exactly three consumed tokens."""
    _require(state == "cancelled", "request did not reach cancelled state")
    names = [event.get("event") if isinstance(event, dict) else event for event in events]
    _require(names.count("token") == 3, "cancel stream did not expose exactly three tokens")
    _require("cancelled" in names, "cancel acknowledgement event is missing")
    _require("snapshot" not in names and "durable" not in names,
             "cancelled request emitted a successful publication event")
    _require(branch_before == branch_after, "cancelled request moved its branch")
    _require(snapshot_cancelled and durable_cancelled,
             "cancelled request futures were not both cancelled")
    _require(not operation_receipt_present, "cancelled operation has a canonical receipt")
    _require(isinstance(trace, dict) and trace.get("protocol_failed") is False,
             "native consumed-token protocol failed")
    callbacks = trace.get("callback_tokens")
    _require(isinstance(callbacks, list) and len(callbacks) == 3,
             "worker callback did not observe exactly three consumed tokens")
    native = trace.get("native_request")
    _require(isinstance(native, dict), "native cancellation acknowledgement is absent")
    _require(native.get("status") == "ok", "native cancellation acknowledgement is not successful")
    _require(native.get("operator") == "normal-verifier-decode",
             "native cancellation used an unpinned generated operator")
    _require(type(native.get("generated")) is int and native["generated"] == 3,
             "native acknowledgement generated count differs")
    _require(type(native.get("position")) is int and native["position"] >= suffix_tokens,
             "native acknowledgement position is invalid")
    _require(type(native.get("consumed")) is int and
             native["consumed"] == native["position"] + 3,
             "native acknowledgement consumed position differs")
    after_stats = trace.get("after")
    _require(isinstance(after_stats, dict) and type(after_stats.get("forward_tokens")) is int and
             type(native.get("forward_tokens")) is int and
             native["forward_tokens"] == after_stats["forward_tokens"],
             "native acknowledgement cumulative forward count differs from worker stats")
    chunks = native.get("chunks")
    _require(isinstance(chunks, list) and bool(chunks),
             "native acknowledgement omitted verifier chunk schedule")
    cursor = native["position"]
    for chunk in chunks:
        _require(isinstance(chunk, list) and len(chunk) == 2 and
                 type(chunk[0]) is int and type(chunk[1]) is int and
                 chunk[0] == cursor and chunk[1] == 1,
                 "native acknowledgement verifier chunks are not contiguous single-token steps")
        cursor += chunk[1]
    _require(cursor == native["consumed"], "native acknowledgement chunk schedule does not cover consumed tokens")
    _require(type(trace.get("forward_delta")) is int and
             trace["forward_delta"] == suffix_tokens + 3,
             "cancel request forward work is not suffix plus three outputs")
    _require(type(trace.get("consumed_delta")) is int and
             trace["consumed_delta"] == suffix_tokens + 3,
             "cancel request consumed work is not suffix plus three outputs")
    _require(type(trace.get("loaded_file_bytes_delta")) is int and
             trace["loaded_file_bytes_delta"] == 0,
             "cancel request reloaded a serialized prefix")
    return {"events": names, "generated": 3, "forward_delta": trace["forward_delta"],
            "consumed_delta": trace["consumed_delta"], "loaded_file_bytes_delta": 0,
            "native_request": native}


def validate_recovery_answer(answer, receipt, state, branch_head, *, source_revision,
                             expected_tokens, suffix_tokens, expected_history,
                             actual_frontier, expected_frontier, actual_controller,
                             expected_controller):
    """Bind a recovered answer to tokens, work, canonical state, and receipt."""
    _require(isinstance(answer, dict) and answer.get("tokens") == list(expected_tokens),
             "recovery tokens differ from the executed oracle")
    _require(bool(expected_tokens), "executed oracle has no output tokens")
    deltas = answer.get("counter_deltas")
    _require(isinstance(deltas, dict), "recovery omitted coordinator work counters")
    _require(type(deltas.get("forward_tokens")) is int and
             deltas["forward_tokens"] == suffix_tokens + len(expected_tokens),
             "recovery forward work is not suffix plus actual output")
    _require(type(deltas.get("consumed_tokens")) is int and
             deltas["consumed_tokens"] == suffix_tokens + len(expected_tokens),
             "recovery consumed work is not suffix plus actual output")
    _require(type(deltas.get("loaded_file_bytes")) is int and deltas["loaded_file_bytes"] == 0,
             "recovery reloaded the serialized prefix instead of retaining it")
    snapshot = answer.get("snapshot")
    _require(isinstance(snapshot, str) and snapshot, "recovery answer has no snapshot")
    _require(isinstance(receipt, dict) and receipt.get("snapshot") == snapshot,
             "durable receipt is not bound to the answer snapshot")
    branch = receipt.get("branch")
    _require(isinstance(branch, dict), "durable receipt omitted its canonical branch record")
    _require(branch.get("tip") == snapshot, "durable receipt branch tip differs")
    _require(branch.get("revision") == source_revision + 1,
             "durable receipt revision is not one CAS advance")
    _require(branch_head.get("tip") == snapshot and
             branch_head.get("revision") == source_revision + 1,
             "canonical branch head does not match the single durable advance")
    _require(state.get("history") == list(expected_history), "recovery canonical history differs")
    _require(actual_frontier == expected_frontier, "recovery full frontier differs from oracle")
    _require(actual_controller == expected_controller, "recovery generation controller differs")
    return {"snapshot": snapshot, "revision": branch_head["revision"],
            "tokens": len(expected_tokens), "forward_tokens": deltas["forward_tokens"],
            "consumed_tokens": deltas["consumed_tokens"], "loaded_file_bytes": 0,
            "history_tokens": len(expected_history), "full_frontier_exact": True,
            "controller_exact": True}


def _json_atomic(path: Path, value, lock: threading.Lock | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = lock if lock is not None else threading.Lock()
    with guard:
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_path(module):
    path = getattr(module, "__file__", None)
    if not path:
        raise RuntimeError(f"Imported module {module.__name__} has no source path")
    return Path(path).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _install_runtime_paths(runtime_root: Path, api_src: Path, api_scripts: Path):
    runtime_root = runtime_root.resolve(strict=True)
    api_src = api_src.resolve(strict=True)
    api_scripts = api_scripts.resolve(strict=True)
    _require(_inside(api_src, runtime_root) and _inside(api_scripts, runtime_root),
             "API source/script paths must be inside the explicit runtime root")
    for path in (str(api_src), str(api_scripts), str(runtime_root)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [str(api_src), str(api_scripts), str(runtime_root)]
    return runtime_root, api_src, api_scripts


def _verify_runtime_modules(runtime_root, api_src, api_scripts, modules):
    rows = {}
    for module in modules:
        path = _module_path(module)
        if module.__name__.startswith("context_time_travel"):
            root = api_src
        elif module.__name__ in ("native_request_probe", "pool_campaign"):
            root = api_scripts
        else:
            root = runtime_root
        _require(_inside(path, root), f"Runtime import escaped explicit root: {module.__name__}: {path}")
        rows[module.__name__] = {"path": str(path), "sha256": _sha(path)}
    if len(rows) != len(modules):
        raise AssertionError("Runtime module inventory contains duplicate module names")
    return rows


def _verify_profile_closure(runtime_sources, closure, runtime_root, api_src, api_scripts):
    """Bind relocated byte-identical runtime files to profile closure entries.

    A staging copy may live at a different absolute path from the path used to
    produce the profile. Compare the declared source-relative path and digest,
    while retaining both paths in the report.
    """
    files = closure.get("files_sha256") if isinstance(closure, dict) else None
    _require(isinstance(files, dict) and bool(files), "Profile runtime closure is absent")
    resolved = {}
    roots = ((("experimental", "branching_context", "src"), api_src),
             (("experimental", "branching_context", "scripts"), api_scripts),
             (("tools",), runtime_root / "tools"))
    for declared_path, declared_sha in files.items():
        parts = Path(str(declared_path)).parts
        candidates = []
        for marker, root in roots:
            for index in range(len(parts) - len(marker) + 1):
                if parts[index:index + len(marker)] == marker:
                    candidate = root.joinpath(*parts[index + len(marker):]).resolve()
                    if _inside(candidate, root) and candidate.is_file():
                        candidates.append(candidate)
        unique = {str(candidate): candidate for candidate in candidates}
        _require(len(unique) == 1,
                 f"Cannot map profile closure path to one staged file: {declared_path}")
        actual = next(iter(unique.values()))
        actual_sha = _sha(actual)
        _require(actual_sha == declared_sha,
                 f"Staged source bytes differ from profile closure: {declared_path}")
        resolved[str(declared_path)] = {"actual_path": str(actual), "sha256": actual_sha}
    bound = {}
    for name, row in runtime_sources.items():
        actual = Path(row["path"])
        if name.startswith("context_time_travel"):
            root = api_src
        elif name in ("native_request_probe", "pool_campaign"):
            root = api_scripts
        else:
            root = runtime_root
        relative = actual.relative_to(root).as_posix()
        expected_prefix = "experimental/branching_context/src/" if root == api_src else (
            "experimental/branching_context/scripts/" if root == api_scripts else "")
        matches = [(declared, files[declared]) for declared in files
                   if str(declared).replace("\\", "/").endswith(expected_prefix + relative)]
        _require(len(matches) == 1,
                 f"Profile closure has {len(matches)} source matches for {name} ({relative})")
        declared_path, declared_sha = matches[0]
        _require(declared_sha == row["sha256"],
                 f"Imported runtime bytes differ from profile closure: {name}")
        bound[name] = {"actual_path": row["path"], "declared_profile_path": declared_path,
                       "sha256": row["sha256"]}
    if runtime_sources:
        _require(bool(bound), "No imported runtime files were bound to the profile closure")
    return {"files": resolved, "imported": bound}


def _import_profile_api_modules(closure):
    """Eagerly import all Python API modules listed in the pinned closure."""
    modules = []
    for declared_path in closure["files_sha256"]:
        parts = Path(str(declared_path)).parts
        marker = ("experimental", "branching_context", "src")
        index = next((i for i in range(len(parts) - len(marker) + 1)
                      if parts[i:i + len(marker)] == marker), None)
        if index is None:
            continue
        suffix = list(parts[index + len(marker):])
        if not suffix or suffix[-1] == "__init__.py" or not suffix[-1].endswith(".py"):
            continue
        suffix[-1] = suffix[-1][:-3]
        modules.append(".".join(suffix))
    return [importlib.import_module(name) for name in sorted(set(modules))]


def _loaded_runtime_modules():
    selected = []
    for name, module in tuple(sys.modules.items()):
        if (name.startswith("context_time_travel") or name.startswith("tools.strata_")
                or name in ("native_request_probe", "pool_campaign")):
            if getattr(module, "__file__", None):
                selected.append(module)
    return selected


def _verify_native_metadata_helper(profile, *, require_loaded=True):
    """Check the separately declared native metadata loader, not an API import."""
    declared = profile.get("gguf_reader")
    _require(isinstance(declared, str) and bool(declared), "metadata helper path is missing")
    expected = profile.get("native_build", {}).get("sha256", {}).get(declared)
    _require(isinstance(expected, str) and len(expected) == 64,
             "metadata helper digest is not pinned in the native build")
    path = Path(declared).resolve(strict=True)
    _require(_sha(path) == expected, "metadata helper digest differs from the pinned build")
    module = sys.modules.get("ctt_strata_gguf")
    _require(module is not None or not require_loaded, "metadata helper was not loaded")
    actual = None
    if module is not None:
        actual = _module_path(module)
        _require(actual == path, "metadata helper imported from an undeclared path")
        _require(_sha(actual) == expected, "metadata helper imported bytes differ")
    return {"declared_path": declared, "actual_path": str(actual) if actual else None,
            "sha256": expected, "loaded": module is not None}


def _wait_idle(coordinator, timeout=120):
    deadline = time.monotonic() + timeout
    with coordinator._condition:
        while any(worker["active"] for worker in coordinator._workers.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Coordinator worker did not become idle")
            coordinator._condition.wait(remaining)


def _event_list(request):
    events = []
    while not request.events.empty():
        events.append(request.events.get())
    return events


def _frontier_bytes(path: Path, index: int, count: int):
    raw = path.read_bytes()
    if len(raw) != FRONTIER_WIDTH * count:
        raise ValueError(f"Reference frontier file has {len(raw)} bytes, expected {FRONTIER_WIDTH*count}")
    value = raw[index * FRONTIER_WIDTH:(index + 1) * FRONTIER_WIDTH]
    floats = struct.unpack("<248320f", value)
    if not all(math.isfinite(item) for item in floats):
        raise ValueError("Reference frontier contains a nonfinite value")
    return value


def _load_case(case_dir: Path, profile_path: Path, args):
    result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    _require(result.get("status") == "PASS", "Reference case is not terminal PASS")
    _require(result.get("execution_profile_id") == profile.get("execution_profile_id"),
             "Case/profile execution identity differs")
    _require(result.get("model_contract_id") == profile.get("model_contract_id"),
             "Case/profile model contract differs")
    _require(result.get("model_weights_sha256") == profile.get("model_weights_sha256"),
             "Case/profile model weight identity differs")
    execution = profile.get("execution", {})
    _require(execution.get("resume_contract") == "identical-native-operators-v1" and
             execution.get("generated_operator") == "normal-verifier-decode" and
             execution.get("native_request_protocol") == "consumed-greedy-ack-v1",
             "Profile does not pin the current schema-3 mixed operator/ack contract")
    _require(type(args.cancel_after) is int and args.cancel_after == 3,
             "This probe is pinned to cancellation after exactly three consumed outputs")
    sizes = result.get("sizes", {})
    _require(sizes.get("prefix") == 8192 and sizes.get("suffix") == 512 and
             sizes.get("generated_limit") == 128,
             "Reference case must be the executed 8K/512/128 fixture")
    prefix = result.get("prefix_token_ids")
    suffix_a = result.get("suffix_a_token_ids")
    suffix_b = result.get("suffix_b_token_ids")
    _require(all(isinstance(x, list) for x in (prefix, suffix_a, suffix_b)),
             "Case is missing its exact token fixtures")
    _require((len(prefix), len(suffix_a), len(suffix_b)) == (8192, 512, 512),
             "Case token fixture lengths differ")
    refs = {row.get("name"): row for row in result.get("references", [])}
    expected = {}
    for label in ("A", "B"):
        row = refs.get("reference-" + label)
        _require(isinstance(row, dict), f"Case lacks executed reference {label}")
        tokens = row.get("tokens")
        _require(isinstance(tokens, list) and 0 < len(tokens) <= 128 and
                 all(type(token) is int for token in tokens),
                 f"Reference {label} output IDs are invalid")
        frontiers = row.get("generated_frontier_sha256")
        _require(isinstance(frontiers, list) and len(frontiers) == len(tokens),
                 f"Reference {label} frontier schedule differs from actual early stop")
        files = case_dir / "references"
        base = files / f"reference-{label}.f32"
        last = _frontier_bytes(Path(str(base) + ".decode.f32"), len(tokens) - 1, len(tokens))
        _require(hashlib.sha256(last).hexdigest() == frontiers[-1],
                 f"Reference {label} final frontier digest differs")
        prompt_path = Path(str(base) + ".requests.f32")
        _require(prompt_path.stat().st_size == FRONTIER_WIDTH * 2,
                 f"Reference {label} omitted prefix/suffix prompt frontiers")
        expected[label] = {"tokens": tokens, "final_frontier": last,
                           "root_frontier": _frontier_bytes(prompt_path, 0, 2),
                           "frontier_count": len(frontiers)}
    return result, profile, tuple(prefix), tuple(suffix_a), tuple(suffix_b), expected


def _controller_for(operation, history_before_output, tokens, tokenizer, execution):
    from context_time_travel import controller
    state = controller.start(dict(temperature=0, eos_tokens=execution["eos_tokens"]),
        operation, tuple(history_before_output), len(tokenizer.tokens))
    return controller.consume_many(state, ((token, tokenizer.token_bytes(token),
        token in execution["eos_tokens"]) for token in tokens))


def _counter_delta(before, after, key):
    left, right = before.get(key), after.get(key)
    if type(left) is not int or type(right) is not int:
        raise AssertionError(f"Missing integer native counter {key}")
    return right - left


def _copy_store(source: Path, destination: Path):
    _require(source.is_dir() and not source.is_symlink(), "Case store is not a regular directory")
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        _require(not path.is_symlink(), f"Refusing symlink in copied source Store: {path}")
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copy2)


def _branch_record_hash(store_path: Path, branch: str):
    path = store_path / "references" / (branch + ".json")
    return _sha(path), json.loads(path.read_text(encoding="utf-8"))


def _start_heartbeat(output, progress, lock, stop, started):
    def run():
        while not stop.wait(30):
            with lock:
                value = dict(progress)
                value.update(event="heartbeat", elapsed_seconds=time.monotonic() - started,
                             python_pid=os.getpid(), heartbeat_time=time.time())
                _json_atomic(output / "progress.json", value)
    thread = threading.Thread(target=run, name="sparse-cancel-heartbeat", daemon=True)
    thread.start()
    return thread


def run(args):
    if os.name != "posix":
        raise RuntimeError("The real native probe is Linux-only; Windows is the controller")
    runtime_root, api_src, api_scripts = _install_runtime_paths(
        args.runtime_root, args.api_src, args.api_scripts)
    case, profile, prefix, suffix_a, suffix_b, refs = _load_case(args.case_dir, args.profile, args)
    closure = profile.get("native_build", {}).get("selected_runtime_source_closure")
    _require(isinstance(closure, dict) and isinstance(closure.get("files_sha256"), dict),
             "Profile lacks a pinned runtime source closure")
    _verify_profile_closure({}, closure, runtime_root, api_src, api_scripts)
    _import_profile_api_modules(closure)
    # Explicit imports keep lazy modules in the source-closure inventory.
    import native_request_probe as native_probe_module
    import pool_campaign as pool_campaign_module
    from context_time_travel import controller
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig, PoolTransitionTrace
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _validate_execution, _validate_model
    from tools.strata_tokenizer import Tokenizer

    runtime_modules = [native_probe_module, pool_campaign_module, controller,
        sys.modules[ContextCoordinator.__module__], sys.modules[BranchingModel.__module__],
        sys.modules[StrataBackend.__module__], sys.modules[Tokenizer.__module__]]
    if "context_time_travel.execution_receipt" in sys.modules:
        runtime_modules.append(sys.modules["context_time_travel.execution_receipt"])
    runtime_sources = _verify_runtime_modules(runtime_root, api_src, api_scripts, runtime_modules)
    closure_bindings = _verify_profile_closure(runtime_sources, closure, runtime_root, api_src, api_scripts)
    _validate_execution(profile)
    model_identity = _validate_model(profile)
    metadata_helper_before = _verify_native_metadata_helper(profile, require_loaded=False)
    args.output.mkdir(parents=True, exist_ok=False)
    store_path = args.output / "store"
    _copy_store(args.case_dir / "store", store_path)
    profile_copy = args.output / "profile.json"
    shutil.copyfile(args.profile, profile_copy)
    record = {"schema": "sparse-prefix-consumed-cancel-probe-v1", "status": "RUNNING",
        "scope": "real native consumed-boundary cancellation, same-root recovery and sibling recovery; not a gate",
        "python_pid": os.getpid(), "python_executable": sys.executable,
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": _sha(Path(__file__).resolve()),
        "runtime_root": str(runtime_root), "api_src": str(api_src), "api_scripts": str(api_scripts),
        "runtime_sources": runtime_sources, "profile_path": str(args.profile.resolve()),
        "profile_sha256": _sha(profile_copy), "case_dir": str(args.case_dir.resolve()),
        "case_result_sha256": _sha(args.case_dir / "result.json"),
        "case_profile_id": case["execution_profile_id"], "model_contract_id": profile["model_contract_id"],
        "model_weights_sha256": profile["model_weights_sha256"], "native_build_sha256": profile["native_build"]["sha256"],
        "model_identity": model_identity, "gpu": args.gpu,
        "native_metadata_helper_before_startup": metadata_helper_before,
        "operators": {key: profile["execution"][key] for key in
            ("resume_contract", "generated_operator", "native_request_protocol")},
        "dimensions": {"prefix": len(prefix), "suffix_a": len(suffix_a),
            "suffix_b": len(suffix_b), "output_limit": 128, "cancel_after_consumed": 3},
        "checks": [], "native_processes": {}, "failures": []}
    destination = args.output / "result.json"
    lock = threading.RLock()
    progress = {"status": "RUNNING", "phase": "validated-inputs", "native_pids": {},
                "last_completed_step": None}
    _json_atomic(destination, record)
    _json_atomic(args.output / "progress.json", progress, lock)
    started = time.monotonic()
    stop = threading.Event()
    heartbeat = _start_heartbeat(args.output, progress, lock, stop, started)
    trace_by_operation = {}
    cancellation_event = threading.Event()
    coordinator = None
    workers = {}
    try:
        from context_time_travel.strata_backend import StrataBackend
        pool = PoolConfig(gpus=(args.gpu,), ram_bytes=args.ram_gib * 1024**3,
            disk_bytes=args.disk_gib * 1024**3, min_available_bytes=args.min_available_gib * 1024**3)
        transition_trace = PoolTransitionTrace("sparse-cancel-" + str(os.getpid()), max_events=4096)

        def factory(gpu):
            model = BranchingModel(store_path, StrataBackend(profile, gpu_id=gpu), profile)
            original = model.run_greedy_request
            def observed(snapshot, ids, max_steps, *, stream, on_token=None, cancelled=None):
                before = model.backend.work_counters()
                callback_tokens = []
                def callback(token):
                    if on_token is not None:
                        on_token(token)
                    callback_tokens.append(token)
                    if stream == cancel_operation and len(callback_tokens) == 3:
                        cancellation_event.set()
                options = dict(stream=stream, on_token=callback, cancelled=cancelled)
                try:
                    return original(snapshot, ids, max_steps, **options)
                finally:
                    after = model.backend.work_counters()
                    trace_by_operation[stream] = {"callback_tokens": list(callback_tokens),
                        "before": before, "after": after,
                        "forward_delta": _counter_delta(before, after, "forward_tokens"),
                        "consumed_delta": _counter_delta(before, after, "consumed_tokens"),
                        "loaded_file_bytes_delta": _counter_delta(before, after, "loaded_file_bytes"),
                        "native_request": getattr(model.backend, "last_native_request", None),
                        "native_execution_receipt": getattr(model.backend, "last_execution_receipt", None),
                        "protocol_failed": bool(getattr(model.backend, "_protocol_failed", False))}
            model.run_greedy_request = observed
            return model

        # The closure's operation ID is chosen before coordinator startup and is
        # captured by the worker wrapper above.
        cancel_operation = "cancel-" + os.urandom(12).hex()
        coordinator = ContextCoordinator(store_path, pool, factory, transition_trace=transition_trace)
        coordinator.ready()
        # Keep process handles before any post-startup assertion can fail.
        workers = {str(gpu): worker["model"].backend.process
                   for gpu, worker in coordinator._workers.items()}
        record["native_processes"] = {key: proc.pid for key, proc in workers.items()}
        progress.update(phase="native-ready", native_pids=record["native_processes"])
        model = coordinator._workers[args.gpu]["model"]
        record["native_metadata_helper_after_startup"] = _verify_native_metadata_helper(profile)
        loaded_sources = _verify_runtime_modules(runtime_root, api_src, api_scripts,
                                                  _loaded_runtime_modules())
        loaded_closure = _verify_profile_closure(loaded_sources, closure,
                                                  runtime_root, api_src, api_scripts)
        _require(loaded_closure["files"] == closure_bindings["files"],
                 "Pinned runtime source closure changed after native worker startup")
        record["runtime_sources_after_native_startup"] = loaded_sources
        record["runtime_import_bindings_after_native_startup"] = loaded_closure["imported"]
        _json_atomic(destination, record, lock)
        _json_atomic(args.output / "progress.json", progress, lock)
        if coordinator.metrics()["GPU"] != [args.gpu]:
            raise AssertionError("Coordinator GPU placement differs from requested single device")

        # Materialize the existing prefix once, then fork all measured work
        # from that same immutable root. This keeps source activation outside
        # the cancellation and recovery request work counters.
        root_operation = "root-" + os.urandom(12).hex()
        root_request = coordinator.create("sparse-root", prefix, root_operation)
        root_receipt = root_request.durable.result(timeout=1800)
        root_answer = root_request.snapshot.result(timeout=1800)
        _wait_idle(coordinator)
        root_snapshot = root_receipt["snapshot"]
        _require(root_answer.get("exact_materialization_reused") is True,
                 "Prepared case root was not reused exactly")
        _require(root_answer.get("counter_deltas", {}).get("forward_tokens") == 0,
                 "Root activation unexpectedly recomputed the prefix")
        root_state = pool_campaign_module.get_state(coordinator, root_snapshot)
        root_frontier = Path(root_state["frontier_file"]).read_bytes()
        _require(root_frontier == refs["A"]["root_frontier"] == refs["B"]["root_frontier"],
                 "Prepared root full frontier differs from executed A/B references")
        root_payload = Path(root_state["payload_file"])
        root_payload_sha = _sha(root_payload)
        _require(root_payload_sha == root_state["manifest"]["payload_sha256"],
                 "Prepared root payload hash differs from its manifest")

        branch_a, branch_b = "sparse-A-" + os.urandom(6).hex(), "sparse-B-" + os.urandom(6).hex()
        coordinator.fork(branch_a, root_snapshot)
        coordinator.fork(branch_b, root_snapshot)
        cancel_before = coordinator.store.head(branch_a)
        cancel_branch_file_sha, cancel_branch_record = _branch_record_hash(store_path, branch_a)
        objects_before = sorted(path.name for path in (store_path / "objects").glob("*.json"))
        receipts_before = dict(coordinator.store._read_refs()["receipts"])
        progress.update(phase="cancel-request-running", last_completed_step="root-and-siblings-ready")
        _json_atomic(args.output / "progress.json", progress, lock)
        request = coordinator.submit(branch_a, suffix_a, 128, cancel_operation,
                                     expected_revision=cancel_before.revision,
                                     cancelled=cancellation_event.is_set)
        try:
            request.durable.result(timeout=1800)
        except CancelledError:
            pass
        else:
            raise AssertionError("Three-token cancellation returned a durable receipt")
        try:
            request.snapshot.result(timeout=1)
        except CancelledError:
            pass
        else:
            raise AssertionError("Three-token cancellation returned a snapshot")
        _wait_idle(coordinator)
        cancel_events = _event_list(request)
        cancel_after = coordinator.store.head(branch_a)
        after_branch_sha, after_branch_record = _branch_record_hash(store_path, branch_a)
        objects_after = sorted(path.name for path in (store_path / "objects").glob("*.json"))
        receipts_after = dict(coordinator.store._read_refs()["receipts"])
        cancellation = validate_cancel_evidence(state=request.state, events=cancel_events,
            branch_before=asdict(cancel_before), branch_after=asdict(cancel_after),
            snapshot_cancelled=request.snapshot.cancelled(), durable_cancelled=request.durable.cancelled(),
            operation_receipt_present=cancel_operation in receipts_after,
            trace=trace_by_operation.get(cancel_operation), suffix_tokens=len(suffix_a))
        cancel_native = cancellation["native_request"]
        _require(cancel_native.get("position") == len(prefix) + len(suffix_a),
                 "Cancelled native request began at a position other than prefix plus suffix")
        native_receipt = trace_by_operation[cancel_operation].get("native_execution_receipt")
        callback_tokens = trace_by_operation[cancel_operation]["callback_tokens"]
        _require(callback_tokens == refs["A"]["tokens"][:3],
                 "Cancelled request tokens differ from the executed reference prefix")
        _require([event["token"] for event in cancel_events if event.get("event") == "token"] == callback_tokens,
                 "Streamed provisional tokens differ from consumed native callbacks")
        from context_time_travel.execution_receipt import tokens_digest
        _require(isinstance(native_receipt, dict) and
                 native_receipt.get("operator") == "normal-verifier-decode" and
                 native_receipt.get("position") == len(prefix) + len(suffix_a) and
                 native_receipt.get("count") == 3 and
                 native_receipt.get("tokens_sha256") == tokens_digest(callback_tokens),
                 "Cancelled native operator receipt does not bind the three consumed callback tokens")
        cancellation["native_execution_receipt"] = native_receipt
        _require(cancel_branch_file_sha == after_branch_sha and cancel_branch_record == after_branch_record,
                 "Cancellation changed the branch reference record")
        _require(objects_before == objects_after, "Cancellation published or removed canonical history objects")
        _require(receipts_before == receipts_after, "Cancellation changed canonical operation receipts")
        _require(_sha(root_payload) == root_payload_sha, "Cancellation mutated the prepared source payload")
        _require(coordinator.store.lineage(root_snapshot)[1] == list(prefix),
                 "Cancellation changed immutable source history")
        _require(coordinator.store.get_snapshot(root_snapshot)["controller"] == root_state["controller"],
                 "Cancellation changed source controller state")
        record["cancel"] = {"operation": cancel_operation, "branch": branch_a,
            "source_snapshot": root_snapshot, "source_payload_sha256": root_payload_sha,
            "source_branch_before_sha256": cancel_branch_file_sha,
            "source_branch_after_sha256": after_branch_sha, "source_history_unchanged": True,
            "source_controller_unchanged": True, "canonical_objects_unchanged": True,
            "canonical_receipts_unchanged": True, "evidence": cancellation,
            "native_trace": trace_by_operation[cancel_operation],
            "oracle_prefix_tokens": refs["A"]["tokens"][:3],
            "events": cancel_events}
        record["checks"].append("SPARSE-CANCEL-001: three consumed tokens, native cancel/ack, no publication or source mutation")
        progress.update(phase="cancel-verified", last_completed_step="cancelled-without-publication")
        _json_atomic(destination, record, lock)
        _json_atomic(args.output / "progress.json", progress, lock)

        execution = profile["execution"]
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        recovery_rows = []
        for label, branch, suffix in (("A", branch_a, suffix_a), ("B", branch_b, suffix_b)):
            source_head = coordinator.store.head(branch)
            _require(source_head.tip == root_snapshot and source_head.revision == 0,
                     f"Recovery branch {label} no longer points at the prepared root")
            operation = "recover-" + label.lower() + "-" + os.urandom(12).hex()
            progress.update(phase="recovery-" + label.lower() + "-running",
                            last_completed_step="recovery-" + ("A" if label == "A" else "A-and-B-forked"))
            _json_atomic(args.output / "progress.json", progress, lock)
            future = coordinator.submit(branch, suffix, 128, operation,
                                        expected_revision=source_head.revision)
            receipt = future.durable.result(timeout=1800)
            answer = future.snapshot.result(timeout=1800)
            _wait_idle(coordinator)
            events = _event_list(future)
            state = pool_campaign_module.get_state(coordinator, receipt["snapshot"])
            final_frontier = Path(state["frontier_file"]).read_bytes()
            expected = refs[label]
            _require(final_frontier == expected["final_frontier"],
                     f"Recovery {label} final full frontier differs from executed reference")
            before_output = list(prefix + suffix)
            expected_control = _controller_for(operation, before_output, expected["tokens"], tokenizer, execution)
            head = coordinator.store.head(branch)
            proof = validate_recovery_answer(answer, receipt, state, asdict(head),
                source_revision=source_head.revision, expected_tokens=expected["tokens"],
                suffix_tokens=len(suffix), expected_history=before_output + expected["tokens"],
                actual_frontier=final_frontier, expected_frontier=expected["final_frontier"],
                actual_controller=state["controller"], expected_controller=expected_control)
            trace = trace_by_operation.get(operation)
            _require(isinstance(trace, dict), f"Recovery {label} lacks worker-owned native counters")
            _require(trace["forward_delta"] == len(suffix) + len(expected["tokens"]),
                     f"Recovery {label} native forward work includes unexpected prefix work")
            _require(trace["consumed_delta"] == len(suffix) + len(expected["tokens"]),
                     f"Recovery {label} native consumed work differs")
            _require(trace["loaded_file_bytes_delta"] == 0,
                     f"Recovery {label} loaded serialized bytes instead of reusing resident prefix")
            native = trace.get("native_request")
            _require(isinstance(native, dict) and native.get("status") == "ok" and
                     native.get("operator") == "normal-verifier-decode" and
                     native.get("generated") == len(expected["tokens"]),
                     f"Recovery {label} native request receipt differs from output")
            _require(native.get("position") == len(prefix) + len(suffix),
                     f"Recovery {label} native request position differs from prompt history")
            _require(type(native.get("consumed")) is int and
                     native.get("consumed") == native.get("position") + len(expected["tokens"]),
                     f"Recovery {label} native consumed position differs")
            receipt_native = trace.get("native_execution_receipt")
            from context_time_travel.execution_receipt import tokens_digest
            _require(isinstance(receipt_native, dict) and
                     receipt_native.get("position") == len(prefix) + len(suffix) and
                     receipt_native.get("count") == len(expected["tokens"]) and
                     receipt_native.get("tokens_sha256") == tokens_digest(expected["tokens"]),
                     f"Recovery {label} operator schedule receipt differs")
            _require(receipt.get("branch") == asdict(head),
                     f"Recovery {label} receipt does not bind the exact canonical branch head")
            expected_operation_receipt = coordinator.store._read_refs()["receipts"].get(operation)
            _require(isinstance(expected_operation_receipt, dict) and
                     expected_operation_receipt.get("result", {}).get("tip") == receipt["snapshot"],
                     f"Recovery {label} operation receipt was not durably recorded once")
            sibling_before = coordinator.store.head(branch_b if label == "A" else branch_a)
            if label == "A":
                _require(sibling_before.tip == root_snapshot and sibling_before.revision == 0,
                         "A recovery moved its B sibling branch")
            else:
                _require(sibling_before.tip == recovery_rows[0]["receipt"]["snapshot"] and
                         sibling_before.revision == 1,
                         "B recovery changed the completed A sibling branch")
            row = {"label": label, "operation": operation, "source_head": asdict(source_head),
                "answer": answer, "receipt": receipt, "state": state, "events": events,
                "native_trace": trace, "proof": proof,
                "frontier_sha256": hashlib.sha256(final_frontier).hexdigest(),
                "oracle_tokens": expected["tokens"], "oracle_frontier_sha256": hashlib.sha256(expected["final_frontier"]).hexdigest()}
            recovery_rows.append(row)
            record["recoveries"] = recovery_rows
            record["checks"].append(f"SPARSE-CANCEL-00{len(recovery_rows)+1}: recovery {label} exact oracle, retained prefix, durable once")
            progress.update(phase="recovery-" + label.lower() + "-verified",
                            last_completed_step="recovery-" + label.lower() + "-durable-and-verified")
            _json_atomic(destination, record, lock)
            _json_atomic(args.output / "progress.json", progress, lock)

        sibling_head = coordinator.store.head(branch_b)
        _require(sibling_head.tip == recovery_rows[1]["receipt"]["snapshot"] and
                 coordinator.store.get_snapshot(recovery_rows[1]["receipt"]["snapshot"])["parent"] == root_snapshot,
                 "B recovery ancestry does not descend directly from the original prepared root")
        _require(coordinator.store.get_snapshot(recovery_rows[0]["receipt"]["snapshot"])["parent"] == root_snapshot,
                 "A recovery ancestry does not descend directly from the original prepared root")
        metrics = coordinator.metrics()
        _require(metrics["max_active_per_gpu"] == {args.gpu: 1},
                 "Coordinator did not preserve exactly one active worker on the selected GPU")
        runtime_sources_after = _verify_runtime_modules(runtime_root, api_src, api_scripts,
            _loaded_runtime_modules())
        closure_bindings_after = _verify_profile_closure(runtime_sources_after, closure,
                                                         runtime_root, api_src, api_scripts)
        _require(closure_bindings_after["files"] == closure_bindings["files"],
                 "Runtime source provenance changed after native startup")
        record["native_metadata_helper_after_requests"] = _verify_native_metadata_helper(profile)
        record.update(runtime_sources_after_native_startup=runtime_sources_after,
            runtime_import_bindings_after_native_startup=closure_bindings_after["imported"],
            metrics=metrics, root_snapshot=root_snapshot,
            profile_runtime_bindings=closure_bindings["imported"],
            root_payload_sha256=root_payload_sha, root_frontier_sha256=hashlib.sha256(root_frontier).hexdigest(),
            transition_trace=transition_trace.snapshot(), status="PASS", elapsed_seconds=time.monotonic()-started,
            executed_checks=["SPARSE-CANCEL-001", "SPARSE-CANCEL-002", "SPARSE-CANCEL-003"],
            skips=[], xfails=[], numerical_max_error=0.)
        record["checks"].append("SPARSE-CANCEL-003: independent A/B suffix recoveries preserve sibling and exact ancestry")
        _json_atomic(destination, record, lock)
        progress.update(status="PASS", phase="complete", last_completed_step="all-three-checks-passed")
        _json_atomic(args.output / "progress.json", progress, lock)
    except BaseException as error:
        with lock:
            record.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
            record["failures"].append(repr(error))
            progress.update(status="FAIL", phase="failure", error=repr(error))
            _json_atomic(destination, record, lock)
            _json_atomic(args.output / "progress.json", progress, lock)
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=5)
        native_probe_module._record_native_shutdown(record, workers,
            coordinator.close if coordinator is not None else lambda: None)
        record["elapsed_seconds"] = time.monotonic() - started
        if record.get("native_exit_success") is not True and record.get("status") == "PASS":
            record["status"] = "FAIL"
            record["failures"].append("Native process did not exit with numeric code zero")
        _json_atomic(destination, record, lock)
        if record.get("native_exit_success") is True:
            progress.update(native_exit_codes=record.get("native_exit_codes"))
        if record.get("status") != "PASS" or record.get("native_exit_success") is not True:
            progress.update(status="FAIL", phase="terminal-failure")
        _json_atomic(args.output / "progress.json", progress, lock)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--api-src", type=Path, required=True)
    parser.add_argument("--api-scripts", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True,
                        help="Executed root-reuse case directory containing result.json and store/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    parser.add_argument("--min-available-gib", type=int, default=16)
    parser.add_argument("--cancel-after", type=int, default=3)
    args = parser.parse_args(argv)
    if args.gpu < 0 or min(args.ram_gib, args.disk_gib, args.min_available_gib) <= 0:
        parser.error("GPU and pool budgets must be positive nonnegative values")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
