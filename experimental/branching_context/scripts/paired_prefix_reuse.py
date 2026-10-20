"""Paired, same-root old-vs-lazy COW timing harness (execution requires Linux/GPU)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid

FRONTIER_FLOATS = 248320
FRONTIER_BYTES = FRONTIER_FLOATS * 4
WARMUP_PAIRS = 1
MEASURED_REPEATS = 3
SUFFIX_NAMES = ("A", "B")
TEXT_DIAGNOSTIC_FIELDS = {"vmm_last_failed_operation"}
NULLABLE_NUMERIC_DIAGNOSTIC_FIELDS = {"host_heap_allocated_bytes"}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values):
    rows = list(values)
    return statistics.median(rows) if rows else None


def validate_request_matrix(rows, *, measured_repeats=MEASURED_REPEATS):
    """Validate the timing table without claiming anything about GPU numerics."""
    expected = {(phase, repeat, suffix)
                for phase, count in (("warmup", WARMUP_PAIRS), ("measured", measured_repeats))
                for repeat in range(count) for suffix in SUFFIX_NAMES}
    seen = set()
    for row in rows:
        key = (row.get("phase"), row.get("repeat"), row.get("suffix"))
        if key in seen:
            raise ValueError(f"duplicate request record: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"unexpected request record: {key}")
        if type(row.get("actual_output_tokens")) is not int or row["actual_output_tokens"] <= 0:
            raise ValueError(f"request {key} has no actual generated tokens")
        if type(row.get("forward_tokens")) is not int or row["forward_tokens"] != row.get("suffix_tokens", -1) + row["actual_output_tokens"]:
            raise ValueError(f"request {key} forward work is not suffix plus actual output")
        if row.get("loaded_file_bytes") != 0:
            raise ValueError(f"request {key} did not use the resident source path")
        for field in ("submit_to_first_token_seconds", "worker_to_first_token_seconds",
                      "submit_to_durable_seconds", "generation_seconds", "request_seconds",
                      "capture_seconds"):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"request {key} has invalid {field}")
        intervals = row.get("inter_token_intervals_seconds")
        if intervals is not None and (not isinstance(intervals, list) or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in intervals)):
            raise ValueError(f"request {key} has invalid inter-token interval")
        median_interval = row.get("median_inter_token_seconds")
        if median_interval is not None and (type(median_interval) not in (int, float)
                or not math.isfinite(median_interval) or median_interval < 0):
            raise ValueError(f"request {key} has invalid median inter-token interval")
    missing = expected - seen
    if missing:
        raise ValueError(f"missing request records: {sorted(missing)}")
    return True


def summarize_arm(rows):
    validate_request_matrix(rows)
    measured = [row for row in rows if row["phase"] == "measured"]
    fields = ("submit_to_first_token_seconds", "worker_to_first_token_seconds",
              "median_inter_token_seconds", "submit_to_durable_seconds",
              "generation_seconds", "request_seconds", "capture_seconds")
    def summarize(selected):
        return {"request_count": len(selected),
                "actual_output_token_counts": [row["actual_output_tokens"] for row in selected],
                "medians_seconds": {field: median(row[field] for row in selected) for field in fields}}
    return {"measured_request_count": len(measured),
            "actual_output_token_counts": [row["actual_output_tokens"] for row in measured],
            "medians_seconds": {field: median(row[field] for row in measured) for field in fields},
            "by_suffix": {suffix: summarize([row for row in measured if row["suffix"] == suffix])
                          for suffix in SUFFIX_NAMES},
            "overall": summarize(measured)}


def compare_arm_summaries(old, lazy):
    """Report lazy-minus-old medians without imposing a speed pass threshold."""
    result = {}
    for scope in ("overall", "by_suffix"):
        names = SUFFIX_NAMES if scope == "by_suffix" else ("overall",)
        result[scope] = {}
        for name in names:
            left = old[scope][name]["medians_seconds"] if scope == "by_suffix" else old[scope]["medians_seconds"]
            right = lazy[scope][name]["medians_seconds"] if scope == "by_suffix" else lazy[scope]["medians_seconds"]
            result[scope][name] = {
                metric: (right[metric] - left[metric]
                    if type(left[metric]) in (int, float) and type(right[metric]) in (int, float) else None)
                for metric in left}
    return result


def _runtime(args):
    for path in (args.runtime_root, args.api_src, args.api_scripts):
        if not path.is_dir():
            raise FileNotFoundError(f"runtime path is not a directory: {path}")
    for path in (args.runtime_root, args.api_src, args.api_scripts):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    import native_request_probe as native_probe
    from context_time_travel import controller
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel import execution_receipt
    from context_time_travel.pools import PoolConfig, PoolTransitionTrace
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, StrataImage
    from pool_campaign import get_state
    from tools.strata_tokenizer import Tokenizer
    return locals()


def _runtime_module_inventory(rt):
    """Return the exact imported modules whose source files are recorded."""
    receipt_module = rt.get("execution_receipt")
    if receipt_module is None:
        receipt_module = importlib.import_module("context_time_travel.execution_receipt")
    return {
        "coordinator": importlib.import_module("context_time_travel.coordinator"),
        "real": importlib.import_module("context_time_travel.real"),
        "backend": importlib.import_module("context_time_travel.strata_backend"),
        "store": importlib.import_module("context_time_travel.store"),
        "pools": importlib.import_module("context_time_travel.pools"),
        "controller": rt["controller"],
        "execution_receipt": receipt_module,
        "native_request_probe": rt["native_probe"],
        "pool_campaign": importlib.import_module("pool_campaign"),
        "tokenizer": importlib.import_module("tools.strata_tokenizer"),
    }


def _runtime_closure(profile):
    closure = profile.get("native_build", {}).get("selected_runtime_source_closure")
    if not isinstance(closure, dict) or not isinstance(closure.get("files_sha256"), dict):
        raise ValueError("profile lacks native_build.selected_runtime_source_closure.files_sha256")
    files = closure["files_sha256"]
    if not files:
        raise ValueError("selected runtime source closure is empty")
    normalized = {}
    for path, digest in files.items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("selected runtime closure contains a non-absolute path")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"selected runtime closure has invalid hash for {path}")
        normalized[str(Path(path).resolve())] = digest
    return normalized


def validate_runtime_sources(runtime_modules, roots, profiles):
    """Fail closed unless imports are in the explicit roots and match both profiles."""
    resolved_roots = [Path(root).resolve() for root in roots]
    closures = [_runtime_closure(profile) for profile in profiles]
    observed = {}
    for name, module in runtime_modules.items():
        filename = getattr(module, "__file__", None)
        if not filename:
            raise ValueError(f"runtime module {name} has no source file")
        path = Path(filename).resolve()
        if not any(path == root or root in path.parents for root in resolved_roots):
            raise ValueError(f"runtime module {name} loaded outside explicit runtime/API roots: {path}")
        digest = sha256_file(path)
        if any(closure.get(str(path)) != digest for closure in closures):
            raise ValueError(f"runtime module {name} is absent or mismatched in selected profile closure: {path}")
        observed[name] = {"path": str(path), "sha256": digest}
    return observed


def _clear_native_environment():
    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            del os.environ[key]


def _load_case(result_path: Path, ref_dir: Path, profile: dict, label: str):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("schema") not in (
            "duplicate-root-reuse-probe-v3", "duplicate-root-reuse-probe-v4"):
        raise ValueError(f"{label} reference result is not a completed root-reuse case")
    if result.get("execution_profile_id") != profile.get("execution_profile_id"):
        raise ValueError(f"{label} reference does not match the pinned execution profile")
    if result.get("model_contract_id") != profile.get("model_contract_id"):
        raise ValueError(f"{label} reference does not match the model contract")
    sizes = result.get("sizes", {})
    if (sizes.get("prefix") != 8192 or sizes.get("suffix") != 512
            or sizes.get("generated_limit") != 128):
        raise ValueError(f"{label} case must be exactly 8192/512/128")
    prefix = tuple(result.get("prefix_token_ids", ()))
    suffixes = {"A": tuple(result.get("suffix_a_token_ids", ())),
                "B": tuple(result.get("suffix_b_token_ids", ()))}
    if len(prefix) != 8192 or any(len(ids) != 512 for ids in suffixes.values()):
        raise ValueError(f"{label} reference token input lengths do not match its receipt")
    references = {row.get("name", "").removeprefix("reference-"): row
                  for row in result.get("references", [])}
    if set(references) != set(SUFFIX_NAMES):
        raise ValueError(f"{label} case must contain references A and B")
    raw = {}
    for suffix in SUFFIX_NAMES:
        row = references[suffix]
        if not row.get("tokens") or len(row["tokens"]) > 128:
            raise ValueError(f"{label} reference {suffix} has an invalid actual output")
        stem = ref_dir / f"reference-{suffix}.f32"
        decode_path = Path(str(stem) + ".decode.f32")
        prompt_path = Path(str(stem) + ".requests.f32")
        decode = decode_path.read_bytes()
        prompt = prompt_path.read_bytes()
        if len(decode) != FRONTIER_BYTES * len(row["tokens"]):
            raise ValueError(f"{label} reference {suffix} decode frontier size is invalid")
        if len(prompt) != FRONTIER_BYTES * 2:
            raise ValueError(f"{label} reference {suffix} prompt frontier size is invalid")
        generated_hashes = row.get("generated_frontier_sha256")
        prompt_hashes = row.get("prompt_frontier_sha256")
        actual_generated = [hashlib.sha256(decode[i:i + FRONTIER_BYTES]).hexdigest()
                            for i in range(0, len(decode), FRONTIER_BYTES)]
        actual_prompt = [hashlib.sha256(prompt[i:i + FRONTIER_BYTES]).hexdigest()
                         for i in range(0, len(prompt), FRONTIER_BYTES)]
        if actual_generated != generated_hashes or actual_prompt != prompt_hashes:
            raise ValueError(f"{label} reference {suffix} frontier bytes do not match its result receipt")
        raw[suffix] = {"tokens": tuple(row["tokens"]),
                       "root_frontier": prompt[:FRONTIER_BYTES],
                       "final_frontier": decode[-FRONTIER_BYTES:],
                       "frontier_sha256": actual_generated[-1],
                       "prompt_frontiers_sha256": actual_prompt,
                       "source_result": str(result_path.resolve()),
                       "decode_file_sha256": sha256_file(decode_path),
                       "prompt_file_sha256": sha256_file(prompt_path)}
    return {"result": result, "prefix": prefix, "suffixes": suffixes, "references": raw}


def _profile_pair(old, lazy):
    if old.get("model_contract_id") != lazy.get("model_contract_id"):
        raise ValueError("old/lazy arms use different model contracts")
    if old.get("model_weights_sha256") != lazy.get("model_weights_sha256"):
        raise ValueError("old/lazy arms use different model weights")
    if old.get("hardware") != lazy.get("hardware"):
        raise ValueError("old/lazy arms use different hardware pins")
    def common_execution(profile):
        value = json.loads(json.dumps(profile.get("execution")))
        for command_name in ("sut_command", "oracle_command"):
            command = value.get(command_name)
            if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
                raise ValueError(f"Pinned profile has invalid {command_name}")
            command[0] = "<arm-pinned-native-binary>"
        return value
    if common_execution(old) != common_execution(lazy):
        raise ValueError("old/lazy arms use different execution/operator settings")
    # The native binary closure is intentionally allowed to differ between arms.
    if old.get("native_build") == lazy.get("native_build"):
        raise ValueError("old and lazy arms do not have distinct native build pins")


def _gpu_memory(gpu):
    result = subprocess.run(["nvidia-smi", "--id=" + str(gpu),
        "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    values = [int(piece.strip()) for piece in result.stdout.strip().split(",")]
    if len(values) != 2:
        raise ValueError("unexpected nvidia-smi memory output")
    return {"used_mib": values[0], "free_mib": values[1], "sampled_monotonic": time.monotonic()}


def _controller_for(operation, history, tokens, tokenizer, eos_tokens):
    from context_time_travel import controller
    state = controller.start(dict(temperature=0, eos_tokens=eos_tokens), operation,
                             tuple(history), len(tokenizer.tokens))
    return controller.consume_many(state,
        ((token, tokenizer.token_bytes(token), token in eos_tokens) for token in tokens))


def _drain_events(request):
    events = []
    while True:
        try:
            events.append(request.events.get_nowait())
        except queue.Empty:
            return events


def _timing(events, submit_started, durable_event, answer):
    starts = [event for event in events if event.get("event") == "request_started"]
    if len(starts) != 1:
        raise AssertionError("Native request did not emit exactly one request_started event")
    started = starts[0]
    token_events = [event for event in events if event.get("event") == "token"]
    if not token_events:
        raise AssertionError("Native request emitted no token timestamp")
    if [event.get("token") for event in token_events] != list(answer.get("tokens", ())):
        raise AssertionError("Token timing events do not match the acknowledged token sequence")
    token_times = [event.get("time") for event in token_events]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in token_times):
        raise AssertionError("Native token event has no monotonic timestamp")
    intervals = [b - a for a, b in zip(token_times, token_times[1:])]
    if any(not math.isfinite(value) or value < 0 for value in intervals):
        raise AssertionError("Native token timestamps are not monotonic")
    return {"submit_to_first_token_seconds": token_times[0] - submit_started,
            "worker_to_first_token_seconds": token_times[0] - started["time"],
            "median_inter_token_seconds": median(intervals),
            "inter_token_intervals_seconds": intervals,
            "submit_to_durable_seconds": durable_event["time"] - submit_started,
            "generation_seconds": answer.get("generation_seconds"),
            "request_seconds": answer.get("request_seconds"),
            "capture_seconds": answer.get("capture_seconds")}


def _wait_idle(coordinator, timeout=60):
    deadline = time.monotonic() + timeout
    with coordinator._condition:
        while any(worker["active"] for worker in coordinator._workers.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Coordinator worker did not become idle")
            coordinator._condition.wait(remaining)


def _backend_counter_delta(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict) or before.keys() != after.keys():
        raise AssertionError("Backend counter keys changed during the request")
    delta, diagnostics_before, diagnostics_after = {}, {}, {}
    for key in before:
        left, right = before[key], after[key]
        if key in TEXT_DIAGNOSTIC_FIELDS:
            if not isinstance(left, str) or not isinstance(right, str):
                raise AssertionError(f"Diagnostic field {key} must remain textual")
            diagnostics_before[key], diagnostics_after[key] = left, right
        elif key in NULLABLE_NUMERIC_DIAGNOSTIC_FIELDS and (left is None or right is None):
            diagnostics_before[key], diagnostics_after[key] = left, right
        elif (type(left) in (int, float) and type(right) in (int, float)
                and math.isfinite(left) and math.isfinite(right)):
            delta[key] = right - left
        else:
            raise AssertionError(f"Numeric backend counter/gauge {key} is malformed")
    for required in ("forward_tokens", "loaded_file_bytes"):
        if required not in delta:
            raise AssertionError(f"Backend request-wide counter is missing {required}")
    return {"numeric_deltas": delta,
            "diagnostics_before": diagnostics_before,
            "diagnostics_after": diagnostics_after}


def _validate_single_gpu_metrics(metrics, gpu):
    """Validate coordinator.metrics() in-process keys (integer for one GPU)."""
    active = metrics.get("active_per_gpu")
    maximum = metrics.get("max_active_per_gpu")
    expected = {gpu}
    if not isinstance(active, dict) or set(active) != expected:
        raise AssertionError("Coordinator active-per-GPU metrics do not match selected GPU")
    if not isinstance(maximum, dict) or set(maximum) != expected:
        raise AssertionError("Coordinator max-active-per-GPU metrics do not match selected GPU")
    if type(active[gpu]) is not int or active[gpu] != 0:
        raise AssertionError("Coordinator still has an active GPU request after durable completion")
    if type(maximum[gpu]) is not int or maximum[gpu] != 1:
        raise AssertionError("Coordinator did not prove exactly one active request on the selected GPU")


def _one_request(rt, args, profile, arm, model, tokenizer, coordinator, root_snapshot, root_branch,
                 case, suffix_name, phase, repeat, run_nonce):
    suffix = case["suffixes"][suffix_name]
    operation = f"op-{arm}-{phase}-{repeat}-{suffix_name.lower()}-{run_nonce}"
    branch = f"b-{arm}-{phase}-{repeat}-{suffix_name.lower()}-{run_nonce}"
    head = coordinator.fork(branch, root_snapshot)
    if head.tip != root_snapshot or head.root != coordinator.store._root_for(root_snapshot):
        raise AssertionError("Fork did not start at the retained common root")
    if model.cache.get(root_snapshot) is None or not isinstance(model.cache[root_snapshot].blob, rt["StrataImage"]):
        raise AssertionError("The common root numerical image is not resident before request")
    memory_before = _gpu_memory(args.gpu)
    _wait_idle(coordinator)
    backend_counters_before = dict(model.backend.stats)
    submit_started = time.monotonic()
    request = coordinator.submit(branch, suffix, 128, operation,
                                 expected_revision=head.revision)
    receipt = request.durable.result(timeout=1800)
    durable_observed = time.monotonic()
    _wait_idle(coordinator)
    backend_counters_after = dict(model.backend.stats)
    request_wide_deltas = _backend_counter_delta(backend_counters_before, backend_counters_after)
    answer = request.snapshot.result(timeout=1800)
    events = _drain_events(request)
    durable_event = next((event for event in events if event.get("event") == "durable"), None)
    if durable_event is None:
        raise AssertionError("Durability future completed without durable event")
    tokens = tuple(answer.get("tokens", ()))
    expected = case["references"][suffix_name]
    if tokens != expected["tokens"]:
        raise AssertionError(f"{arm}/{suffix_name} output IDs differ from matching executed case")
    deltas = answer.get("counter_deltas")
    expected_forward = len(suffix) + len(tokens)
    if not isinstance(deltas, dict) or deltas.get("forward_tokens") != expected_forward:
        raise AssertionError("Request forward work is not suffix plus actual output count")
    if deltas.get("loaded_file_bytes") != 0:
        raise AssertionError("Resident root path loaded numerical checkpoint bytes from disk")
    whole_request_numeric = request_wide_deltas["numeric_deltas"]
    if whole_request_numeric["forward_tokens"] != expected_forward:
        raise AssertionError("Whole-request backend forward work is not suffix plus actual output count")
    if whole_request_numeric["loaded_file_bytes"] != 0:
        raise AssertionError("Whole-request backend stats include numerical checkpoint file loads")
    if answer.get("snapshot") != receipt.get("snapshot"):
        raise AssertionError("Snapshot result and durable receipt are not bound")
    if durable_event.get("receipt", {}).get("snapshot") != receipt.get("snapshot"):
        raise AssertionError("Durable event receipt does not match the durable future")
    started_event = next((event for event in events if event.get("event") == "request_started"), None)
    if started_event is None or started_event.get("source") != root_snapshot:
        raise AssertionError("Native request did not execute from the common root snapshot")
    state = rt["get_state"](coordinator, receipt["snapshot"])
    history = tuple(state.get("history", ()))
    expected_history = case["prefix"] + suffix + tokens
    if history != expected_history:
        raise AssertionError("Canonical saved history differs from the supplied root-reuse case")
    frontier = Path(state["frontier_file"]).read_bytes()
    if frontier != expected["final_frontier"]:
        raise AssertionError("Final full frontier differs from the matching executed case")
    root_state = rt["get_state"](coordinator, root_snapshot)
    if tuple(root_state.get("history", ())) != case["prefix"]:
        raise AssertionError("The retained root history changed during a suffix request")
    root_frontier = Path(root_state["frontier_file"]).read_bytes()
    if root_frontier != expected["root_frontier"]:
        raise AssertionError("Root full frontier differs from the matching executed case")
    expected_controller = _controller_for(operation, case["prefix"] + suffix, tokens,
                                          tokenizer,
                                          profile["execution"]["eos_tokens"])
    if state.get("controller") != expected_controller:
        raise AssertionError("Complete saved controller differs from exact output/controller reconstruction")
    current_head = coordinator.store.head(branch)
    if current_head.tip != receipt["snapshot"] or current_head.revision != head.revision + 1:
        raise AssertionError("Branch did not advance exactly once to the durable snapshot")
    if coordinator.store.head(root_branch).tip != root_snapshot:
        raise AssertionError("Common root branch moved during a suffix request")
    if model.cache.get(root_snapshot) is None or not isinstance(model.cache[root_snapshot].blob, rt["StrataImage"]):
        raise AssertionError("The common root image was not kept resident across the request")
    stats = dict(model.backend.stats)
    metrics = coordinator.metrics()
    row = {"arm": arm, "phase": phase, "repeat": repeat, "suffix": suffix_name,
           "operation_id": operation, "branch": branch, "root_snapshot": root_snapshot,
           "source_revision": head.revision, "durable_revision": current_head.revision,
           "snapshot": receipt["snapshot"], "receipt_sha256": hashlib.sha256(
               json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
           "actual_output_tokens": len(tokens), "output_ids_sha256": hashlib.sha256(
               ",".join(map(str, tokens)).encode("ascii")).hexdigest(),
           "suffix_tokens": len(suffix), "forward_tokens": deltas["forward_tokens"],
           "loaded_file_bytes": deltas.get("loaded_file_bytes"),
           "counter_deltas": deltas,
           "backend_request_wide_counter_deltas": request_wide_deltas["numeric_deltas"],
           "backend_request_diagnostics_before_submit": request_wide_deltas["diagnostics_before"],
           "backend_request_diagnostics_after_durable_idle": request_wide_deltas["diagnostics_after"],
           "backend_counters_before_submit": backend_counters_before,
           "backend_counters_after_durable_idle": backend_counters_after,
           "backend_cumulative_counters": stats,
           "capture_reservation_bytes": answer.get("ram_reservation_bytes"),
           "root_frontier_sha256": hashlib.sha256(root_frontier).hexdigest(),
           "final_frontier_sha256": hashlib.sha256(frontier).hexdigest(),
           "controller_exact": True, "history_exact": True,
           "memory_before_submit": memory_before, "memory_after_durable": _gpu_memory(args.gpu),
           "pool_metrics_after_durable": metrics,
           "durable_wait_observed_seconds": durable_observed - submit_started,
           **_timing(events, submit_started, durable_event, answer)}
    _validate_single_gpu_metrics(metrics, args.gpu)
    # A completed branch checkpoint remains canonical on disk. Drop only its
    # in-memory image after ack so the common prefix image stays resident.
    model.cache.pop(receipt["snapshot"], None)
    return row


def _run_arm(rt, args, arm, profile, case, record, nonce):
    output = args.output / arm
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    store_path = output / "store"
    trace = rt["PoolTransitionTrace"](f"paired-prefix-{arm}-{nonce}", max_events=4096)
    pool = rt["PoolConfig"](gpus=(args.gpu,), ram_bytes=args.ram_gib * 1024**3,
        disk_bytes=args.disk_gib * 1024**3, min_available_bytes=args.min_available_gib * 1024**3)
    workers = {}
    coordinator = None
    gpu_memory_before_arm = _gpu_memory(args.gpu)
    arm_record = {"status": "RUNNING", "arm": arm,
        "execution_profile_id": profile["execution_profile_id"],
        "model_contract_id": profile["model_contract_id"],
        "native_build": profile["native_build"], "gpu": args.gpu,
        "gpu_memory_before_arm": gpu_memory_before_arm,
        "pool_budgets": {"ram_bytes": pool.ram_bytes, "disk_bytes": pool.disk_bytes,
                         "min_available_bytes": pool.min_available_bytes},
        "requests": [], "failures": []}
    atomic_json(output / "arm.json", arm_record)
    started = time.monotonic()
    try:
        if profile.get("execution", {}).get("GPU") != [args.gpu]:
            raise ValueError(f"{arm} profile is not pinned to the selected single GPU")
        def factory(gpu):
            return rt["BranchingModel"](store_path,
                rt["StrataBackend"](profile, gpu_id=gpu), profile)
        coordinator = rt["ContextCoordinator"](store_path, pool, factory,
                                               transition_trace=trace)
        coordinator.ready()
        workers = {str(gpu): worker["model"].backend.process
                   for gpu, worker in coordinator._workers.items()}
        model = next(worker["model"] for worker in coordinator._workers.values())
        if len(workers) != 1 or set(workers) != {str(args.gpu)}:
            raise AssertionError("Arm did not start exactly one worker on its selected GPU")
        arm_record["native_pids"] = {gpu: process.pid for gpu, process in workers.items()}
        tokenizer = rt["Tokenizer"].from_gguf(profile["model_path"])
        arm_record["startup_seconds"] = time.monotonic() - started
        arm_record["gpu_memory_after_startup"] = _gpu_memory(args.gpu)
        arm_record["pool_before_root"] = coordinator.metrics()
        atomic_json(output / "arm.json", arm_record)

        run_id = uuid.uuid4().hex[:10]
        root_branch = f"b-root-{arm}-{run_id}"
        root_operation = f"op-root-{arm}-{run_id}"
        root_started = time.monotonic()
        root_request = coordinator.create(root_branch, case["prefix"], root_operation)
        root_receipt = root_request.durable.result(timeout=1800)
        root_answer = root_request.snapshot.result(timeout=1800)
        root_events = _drain_events(root_request)
        _wait_idle(coordinator)
        root_snapshot = root_receipt["snapshot"]
        root_state = rt["get_state"](coordinator, root_snapshot)
        if tuple(root_state["history"]) != case["prefix"]:
            raise AssertionError("Created root history differs from pinned 8K fixture")
        root_frontier = Path(root_state["frontier_file"]).read_bytes()
        for name in SUFFIX_NAMES:
            if root_frontier != case["references"][name]["root_frontier"]:
                raise AssertionError(f"Created root frontier differs from executed case {name}")
        root_deltas = root_answer.get("counter_deltas", {})
        if root_answer.get("exact_materialization_reused") is not False:
            raise AssertionError("Fresh arm root unexpectedly reused a prior materialization")
        if root_deltas.get("forward_tokens") != len(case["prefix"]):
            raise AssertionError("Root creation forward work differs from the exact prefix size")
        root_row = {"operation_id": root_operation, "branch": root_branch,
            "snapshot": root_snapshot, "revision": coordinator.store.head(root_branch).revision,
            "root_frontier_sha256": hashlib.sha256(root_frontier).hexdigest(),
            "payload_sha256": sha256_file(Path(root_state["payload_file"])),
            "prefix_tokens": len(case["prefix"]), "counter_deltas": root_deltas,
            "root_create_seconds": time.monotonic() - root_started,
            "request_event_names": [event.get("event") for event in root_events]}
        arm_record["root"] = root_row
        record["arms"][arm] = arm_record
        atomic_json(args.output / "result.json", record)

        for phase, count in (("warmup", WARMUP_PAIRS), ("measured", MEASURED_REPEATS)):
            for repeat in range(count):
                for suffix in SUFFIX_NAMES:
                    row = _one_request(rt, args, profile, arm, model, tokenizer, coordinator,
                        root_snapshot, root_branch, case, suffix, phase, repeat, run_id)
                    arm_record["requests"].append(row)
                    atomic_json(output / "arm.json", arm_record)
                    atomic_json(args.output / "result.json", record)
        arm_record["summary"] = summarize_arm(arm_record["requests"])
        arm_record["pool_final"] = coordinator.metrics()
        arm_record["pool_transition_trace"] = trace.snapshot()
        arm_record["status"] = "PASS"
        arm_record["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "arm.json", arm_record)
        atomic_json(args.output / "result.json", record)
    except BaseException as exc:
        arm_record.update(status="FAIL", error=str(exc), traceback=traceback.format_exc(),
                          elapsed_seconds=time.monotonic() - started)
        arm_record["failures"].append(str(exc))
        atomic_json(output / "arm.json", arm_record)
        raise
    finally:
        if coordinator is not None:
            try:
                coordinator.close()
            except Exception as exc:
                arm_record.setdefault("failures", []).append(f"coordinator close: {exc}")
                arm_record["status"] = "FAIL"
            exits = {}
            for gpu, process in workers.items():
                code = process.poll()
                exits[gpu] = code
            arm_record["native_exit_codes"] = exits
            if any(code != 0 for code in exits.values()):
                arm_record["status"] = "FAIL"
                arm_record.setdefault("failures", []).append("native worker did not exit zero")
            arm_record["pool_transition_trace"] = trace.snapshot()
            arm_record["gpu_memory_after_shutdown"] = _gpu_memory(args.gpu)
            atomic_json(output / "arm.json", arm_record)
            atomic_json(args.output / "result.json", record)
        record["arms"][arm] = arm_record
        if arm_record.get("status") != "PASS":
            record["status"] = "FAIL"
            record["failures"].append(f"arm {arm} did not pass clean shutdown")
            atomic_json(args.output / "result.json", record)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    old_profile = json.loads(args.old_profile.read_text(encoding="utf-8"))
    lazy_profile = json.loads(args.lazy_profile.read_text(encoding="utf-8"))
    old_case = _load_case(args.old_case, args.old_case_dir, old_profile, "old")
    lazy_case = _load_case(args.lazy_case, args.lazy_case_dir, lazy_profile, "lazy")
    _profile_pair(old_profile, lazy_profile)
    if (old_case["prefix"], old_case["suffixes"]) != (lazy_case["prefix"], lazy_case["suffixes"]):
        raise ValueError("Old/lazy case token inputs differ")
    for suffix in SUFFIX_NAMES:
        if old_case["references"][suffix]["tokens"] != lazy_case["references"][suffix]["tokens"]:
            raise ValueError("Old/lazy executed cases have different expected output IDs")
        if old_case["references"][suffix]["final_frontier"] != lazy_case["references"][suffix]["final_frontier"]:
            raise ValueError("Old/lazy executed cases have different expected final frontier bytes")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {"schema": "paired-prefix-reuse-performance-v1", "status": "RUNNING",
        "qualification": "performance-only; not a COW correctness or full-gate claim",
        "design": {"same_root_per_arm": True, "fork_suffixes": ["A", "B"],
                   "warmup_pairs": WARMUP_PAIRS, "measured_pairs": MEASURED_REPEATS,
                   "requests_per_arm": 2 * (WARMUP_PAIRS + MEASURED_REPEATS),
                   "one_gpu_worker_at_a_time": True, "prefix_resident_during_suffixes": True,
                   "per_token_fsync": False, "fresh_store_per_arm": True},
        "inputs": {"old_profile": str(args.old_profile.resolve()),
            "old_profile_sha256": sha256_file(args.old_profile),
            "lazy_profile": str(args.lazy_profile.resolve()),
            "lazy_profile_sha256": sha256_file(args.lazy_profile),
            "old_case": str(args.old_case.resolve()), "old_case_sha256": sha256_file(args.old_case),
            "lazy_case": str(args.lazy_case.resolve()), "lazy_case_sha256": sha256_file(args.lazy_case),
            "runtime_root": str(args.runtime_root.resolve()), "api_src": str(args.api_src.resolve()),
            "api_scripts": str(args.api_scripts.resolve()),
            "prefix_tokens": len(old_case["prefix"]),
            "suffix_tokens": {key: len(value) for key, value in old_case["suffixes"].items()},
            "requested_output_limit": 128,
            "old_reference_frontier_hashes": {k: v["frontier_sha256"] for k, v in old_case["references"].items()},
            "lazy_reference_frontier_hashes": {k: v["frontier_sha256"] for k, v in lazy_case["references"].items()},
            "reference_files": {arm: {suffix: {"decode_sha256": row["decode_file_sha256"],
                "prompt_sha256": row["prompt_file_sha256"]} for suffix, row in case["references"].items()}
                for arm, case in (("old", old_case), ("lazy", lazy_case))}},
        "arms": {}, "failures": []}
    atomic_json(args.output / "result.json", record)
    started = time.monotonic()
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            atomic_json(args.output / "heartbeat.json", {"event": "heartbeat",
                "elapsed_seconds": time.monotonic() - started,
                "current_arm": record.get("current_arm", "preflight")})
    thread = threading.Thread(target=heartbeat, name="paired-prefix-heartbeat", daemon=True)
    thread.start()
    try:
        rt = _runtime(args)
        record["runner_sha256"] = sha256_file(Path(__file__).resolve())
        runtime_modules = _runtime_module_inventory(rt)
        runtime_roots = (args.runtime_root, args.api_src, args.api_scripts)
        runtime_sources = validate_runtime_sources(runtime_modules, runtime_roots,
                                                   (old_profile, lazy_profile))
        record["runtime_sources"] = runtime_sources
        record["runtime_source_sha256"] = {name: row["sha256"] for name, row in runtime_sources.items()}
        atomic_json(args.output / "result.json", record)
        for arm, profile, case in (("old", old_profile, old_case), ("lazy", lazy_profile, lazy_case)):
            record["current_arm"] = arm
            atomic_json(args.output / "result.json", record)
            _clear_native_environment()
            os.environ.update(profile.get("execution", {}).get("environment", {}))
            rt["native_probe"]._validate_execution(profile)
            rt["native_probe"]._validate_model(profile)
            _run_arm(rt, args, arm, profile, case, record, uuid.uuid4().hex[:10])
            if record["arms"][arm].get("status") != "PASS":
                raise RuntimeError(f"{arm} arm was not a clean PASS")
        summaries = {arm: value["summary"] for arm, value in record["arms"].items()}
        record.update(status="PASS", current_arm=None, summaries=summaries,
                      lazy_minus_old_median_seconds=compare_arm_summaries(
                          summaries["old"], summaries["lazy"]),
                      elapsed_seconds=time.monotonic() - started,
                      comparison_scope="Sequential old then lazy arms; report medians per suffix and overall. No randomized arm order.")
        atomic_json(args.output / "result.json", record)
        return 0
    except BaseException as exc:
        record.update(status="FAIL", error=str(exc), traceback=traceback.format_exc(),
                      failures=record.get("failures", []) + [str(exc)],
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output / "result.json", record)
        raise
    finally:
        stop.set()
        thread.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--api-src", type=Path, required=True)
    parser.add_argument("--api-scripts", type=Path, required=True)
    parser.add_argument("--old-profile", type=Path, required=True)
    parser.add_argument("--lazy-profile", type=Path, required=True)
    parser.add_argument("--old-case", type=Path, required=True,
                        help="PASS result.json from matching old root-reuse qualification")
    parser.add_argument("--old-case-dir", type=Path, required=True,
                        help="case/references directory containing reference-A/B.f32 dump files")
    parser.add_argument("--lazy-case", type=Path, required=True,
                        help="PASS result.json from matching lazy root-reuse qualification")
    parser.add_argument("--lazy-case-dir", type=Path, required=True,
                        help="case/references directory containing reference-A/B.f32 dump files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    parser.add_argument("--min-available-gib", type=int, default=16)
    args = parser.parse_args(argv)
    if min(args.ram_gib, args.disk_gib, args.min_available_gib) <= 0:
        parser.error("pool budgets must be positive")
    try:
        return run(args)
    except BaseException as exc:
        print(f"paired prefix harness failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
