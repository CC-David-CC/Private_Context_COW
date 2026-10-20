"""No-hold mixed-topology timing probe for the native model-free fixture.

This records fixture scheduling and persistence latency only; it is not a model
performance benchmark. Run only on a Linux GPU host with fixture_backend.
"""
import argparse
from concurrent.futures import TimeoutError
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
BRANCHING = HERE.parent.parent / "experimental" / "branching_context"
sys.path.insert(0, str(BRANCHING / "src"))
sys.path.insert(0, str(HERE))

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from model_free_backend import make_profile
from test_model_free_resume import _cfb1_devices, _durable_blob, build_fixture_model, verify_cfb1


GROUPS = ((0, 1), 2)
PREFIX = tuple(i % 64 for i in range(128))
INPUTS = {"A": (1, 2, 3, 4), "B": (5, 6, 7, 8), "C": (9, 10, 11, 12)}


def _write(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(target.suffix + ".tmp")
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(target)


def _availability(coordinator, worker, snapshot):
    model = coordinator._workers[worker]["model"]
    if snapshot in model.cache:
        return "gpu_resident"
    key = coordinator._key(model, snapshot)
    if key in coordinator.pool:
        return "shared_ram_resident"
    if (coordinator.store.root / "checkpoints" / (snapshot + ".json")).is_file():
        return "durable_file_native_cold"
    return "missing"


def _track_completion(handle):
    events = {"snapshot_ns": None, "durable_ns": None}
    lock = threading.Lock()
    recorded = threading.Event()

    def stamp(key):
        def done(_future):
            with lock:
                events[key] = time.perf_counter_ns()
                if events["snapshot_ns"] is not None and events["durable_ns"] is not None:
                    recorded.set()
        return done

    handle.snapshot.add_done_callback(stamp("snapshot_ns"))
    handle.durable.add_done_callback(stamp("durable_ns"))
    return events, lock, recorded


def _time_request(label, handle, submitted_ns, completion):
    answer = handle.snapshot.result(600)
    receipt = handle.durable.result(600)
    events, lock, recorded = completion
    if not recorded.wait(10):
        raise TimeoutError(f"{label} completion callback timestamps were not recorded")
    with lock:
        snapshot_ns, durable_ns = events["snapshot_ns"], events["durable_ns"]
    return answer, receipt, {
        "label": label,
        "submit_to_snapshot_ms": (snapshot_ns - submitted_ns) / 1e6,
        "submit_to_durable_ms": (durable_ns - submitted_ns) / 1e6,
        "generation_ms": answer["generation_seconds"] * 1000,
        "request_ms": answer["request_seconds"] * 1000,
        "capture_ms": answer["capture_seconds"] * 1000,
        "forward_tokens": answer["counter_deltas"]["forward_tokens"],
        "checkpoint_import_bytes": answer["counter_deltas"]["checkpoint_import_bytes"],
        "checkpoint_export_bytes": answer["counter_deltas"]["checkpoint_export_bytes"],
        "native_counters": answer["counter_deltas"],
        "snapshot": answer["snapshot"],
        "tip": receipt["branch"]["tip"],
    }


def run_timing(binary, hardware_identity, output):
    profile = make_profile(binary, [[0, 1], 2], 12288,
                           hardware_identity=hardware_identity)
    config = PoolConfig(gpus=GROUPS, ram_bytes=2 * 1024**3,
                        disk_bytes=8 * 1024**3, min_available_bytes=0)
    report = {
        "status": "RUNNING", "probe": "mixed-affinity-no-hold-timing-v1",
        "purpose": "synthetic fixture scheduling/persistence timing, not model performance",
        "filesystem_cache_state": "not flushed or measured; may be warm",
        "binary": str(Path(binary).resolve()),
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "hardware_identity": hardware_identity, "worker_groups": [[0, 1], [2]],
        "capacity": 12288, "prefix_tokens": len(PREFIX), "suffix_tokens": 4,
        "max_steps": 8, "warmup_repetitions": 1, "measured_repetitions": 3,
        "repetitions": [], "failure": None,
    }
    _write(output, report)
    with tempfile.TemporaryDirectory(prefix="mf-affinity-timing-") as directory:
        def new_coordinator():
            return ContextCoordinator(
                directory, config,
                lambda gpu: build_fixture_model(directory, profile, gpu))

        # Make durable roots, then throw away their GPU and shared-RAM images.
        # Each timing repetition uses new native workers and empty RAM pools:
        # A/C import from durable checkpoints, then B reuses A's source image.
        bootstrap = new_coordinator()
        roots = {}
        try:
            bootstrap.ready()
            root_inputs = (("split", PREFIX, (0, 1)),
                           ("single", tuple((i + 17) % 64 for i in PREFIX), (2,)))
            root_handles = {
                label: bootstrap.create(f"timing-root-{label}", ids,
                                        f"timing-create-{label}")
                for label, ids, _expected in root_inputs
            }
            for label, ids, expected in root_inputs:
                handle = root_handles[label]
                snapshot = handle.snapshot.result(600)
                handle.durable.result(600)
                blob, _frontier = _durable_blob(bootstrap.store.root / "checkpoints",
                                                snapshot["snapshot"])
                devices = _cfb1_devices(blob)
                audit = verify_cfb1(blob, ids, seed=profile["execution"]["seed"],
                                    capacity=12288, physical_devices=devices)
                if devices != expected:
                    raise AssertionError(f"{label} root placed on unexpected devices {devices}")
                roots[label] = {"snapshot": snapshot["snapshot"], "ids": ids,
                                "devices": devices, "audit": audit}
        finally:
            bootstrap.close()
        report["roots"] = {key: {"snapshot": value["snapshot"],
                                 "devices": list(value["devices"]),
                                 "audit": value["audit"]}
                           for key, value in roots.items()}

        for repetition in range(4):
            warmup = repetition == 0
            suffix = "warmup" if warmup else f"measured-{repetition}"
            coordinator = new_coordinator()
            try:
                coordinator.ready()
                worker_pids = {
                    str(group): coordinator._workers[group]["model"].backend.process.pid
                    for group in GROUPS
                }
                branches = {label: f"timing-{label}-{suffix}" for label in INPUTS}
                sources = {"A": roots["split"], "B": roots["split"],
                           "C": roots["single"]}
                workers = {"A": (0, 1), "B": (0, 1), "C": 2}
                for label in INPUTS:
                    coordinator.fork(branches[label], sources[label]["snapshot"])
                source_classes = {
                    label: _availability(coordinator, workers[label], sources[label]["snapshot"])
                    for label in INPUTS
                }
                if any(kind != "durable_file_native_cold" for kind in source_classes.values()):
                    raise AssertionError(f"fresh workers did not start with durable-file sources: {source_classes}")
                submitted = {}
                handles = {}
                completions = {}
                # No injected barrier, sleep, or lease hook: A and C race on
                # separate groups, then B queues naturally behind A on split.
                for label in ("A", "C", "B"):
                    submitted[label] = time.perf_counter_ns()
                    handles[label] = coordinator.submit(
                        branches[label], INPUTS[label], 8,
                        f"timing-operation-{label}-{suffix}")
                    completions[label] = _track_completion(handles[label])
                records = {}
                for label in ("A", "C", "B"):
                    answer, receipt, timings = _time_request(
                        label, handles[label], submitted[label], completions[label])
                    blob, _frontier = _durable_blob(
                        coordinator.store.root / "checkpoints", answer["snapshot"])
                    devices = _cfb1_devices(blob)
                    expected_devices = sources[label]["devices"]
                    ids = tuple(sources[label]["ids"]) + INPUTS[label] + tuple(answer["tokens"])
                    audit = verify_cfb1(blob, ids, seed=profile["execution"]["seed"],
                                        capacity=12288, physical_devices=devices)
                    if devices != expected_devices:
                        raise AssertionError(f"{label} landed on {devices}, expected {expected_devices}")
                    if answer["counter_deltas"]["forward_tokens"] != len(INPUTS[label]) + len(answer["tokens"]):
                        raise AssertionError(f"{label} forward count indicates prefix replay or omitted work")
                    imported = answer["counter_deltas"]["checkpoint_import_bytes"]
                    if label in ("A", "C") and imported <= 0:
                        raise AssertionError(f"cold {label} source did not import native checkpoint bytes")
                    if label == "B" and imported != 0:
                        raise AssertionError("B did not reuse A's resident source checkpoint")
                    timings["source_residency_at_submit"] = source_classes[label]
                    timings["source_residency_at_execution"] = (
                        "gpu_resident_reused_by_A" if label == "B" else "durable_file_native_import")
                    timings["actual_devices"] = list(devices)
                    timings["payload_audit"] = audit
                    records[label] = timings
                branches_once = all(
                    coordinator.store.head(branches[label]).revision == 1
                    for label in INPUTS)
                report["repetitions"].append({
                    "kind": "warmup" if warmup else "measured",
                    "index": repetition, "worker_pids": worker_pids,
                    "submit_order": ["A", "C", "B"],
                    "source_residency": source_classes,
                    "requests": records,
                    "all_branches_advanced_once": branches_once,
                    "process_pool": coordinator.metrics(),
                })
                if not branches_once:
                    raise AssertionError("request branch did not advance exactly once")
            except BaseException as exc:
                report["status"] = "FAIL"
                report["failure"] = {"type": type(exc).__name__, "message": str(exc),
                                     "repetition": repetition}
                report["process_pool_after"] = coordinator.metrics()
                _write(output, report)
                raise
            finally:
                coordinator.close()
        report["status"] = "PASS"
        report["process_pool_after"] = report["repetitions"][-1]["process_pool"]
        _write(output, report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--hardware-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_timing(args.binary, args.hardware_identity, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
