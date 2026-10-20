"""Three independent single-GPU workers each execute a real fixture branch.

The caller supplies a built native fixture binary, selected physical GPU IDs,
the pinned hardware identity, and a JSON report destination. This module has
no Python model fallback: native placement is read from CFB1 after execution.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BRANCHING = HERE.parent.parent / "experimental" / "branching_context"
sys.path.insert(0, str(BRANCHING / "src"))
sys.path.insert(0, str(HERE))

from model_free_backend import make_profile
from test_model_free_resume import _cfb1_devices, _durable_blob, build_fixture_model, verify_cfb1
from model_free_backend import cpu_frontier
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig


def _write_report(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def _snapshot_event_worker(handle):
    worker = None
    while True:
        try:
            event = handle.events.get_nowait()
        except queue.Empty:
            break
        if event.get("event") == "snapshot":
            worker = event.get("gpu")
    if worker is None:
        raise AssertionError("coordinator emitted no snapshot event with the actual worker ID")
    return worker


def run_independent_workers(binary, gpu_selection, hardware_identity, output):
    """Exercise 3 independent GPU workers and audit all persisted bytes on CPU.

    Three distinct canonical roots are created concurrently. Each root is
    forked and receives one suffix-4/output-8 request, concurrently with the
    other two. The test requires actual CFB1 payloads to identify all selected
    devices, one accepted branch revision per request, and exactly 12 native
    forward tokens per request (the four supplied plus eight generated).
    """
    from model_free_backend import DEFAULT_SEED

    devices = tuple(gpu_selection)
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("independent worker test requires three distinct GPU IDs")
    groups = tuple((device,) for device in devices)
    profile = make_profile(binary, list(devices), 12288,
                           hardware_identity=hardware_identity)
    config = PoolConfig(gpus=devices, ram_bytes=2 * 1024**3,
                        disk_bytes=8 * 1024**3, min_available_bytes=0)
    report = {
        "status": "RUNNING",
        "case": "three-independent-workers-tiny",
        "qualification": "model-free-cuda-fixture-only",
        "binary": str(Path(binary).resolve()),
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "hardware_identity": hardware_identity,
        "selected_worker_groups": [list(group) for group in groups],
        "capacity": 12288,
        "prefix_tokens_per_root": 128,
        "suffix_tokens": 4,
        "max_output_tokens": 8,
        "expected_forward_tokens_each": 12,
        "roots": [],
        "requests": [],
        "max_active_per_gpu": {},
        "failure": None,
    }
    _write_report(output, report)

    with tempfile.TemporaryDirectory(prefix="mf-independent-workers-") as directory:
        def factory(gpu):
            return build_fixture_model(directory, profile, gpu)

        coordinator = ContextCoordinator(directory, config, factory)
        try:
            coordinator.ready()
            roots_to_create = []
            for index, device in enumerate(devices):
                branch = f"indroot{index}"
                tokens = tuple((index * 17 + i * 3) % 64 for i in range(128))
                handle = coordinator.create(branch, tokens, f"ind-create-{index}")
                roots_to_create.append((index, device, branch, tokens, handle))

            # All creates are submitted before awaiting any completion, giving
            # the dispatcher an opportunity to use every ready worker.
            root_records = []
            for index, device, branch, tokens, handle in roots_to_create:
                answer = handle.snapshot.result(300)
                handle.durable.result(300)
                executed_worker = _snapshot_event_worker(handle)
                blob, frontier = _durable_blob(
                    coordinator.store.root / "checkpoints", answer["snapshot"])
                actual = _cfb1_devices(blob)
                if actual != (device,):
                    raise AssertionError(
                        f"root {index} was not placed on selected worker {device}: {actual}")
                if executed_worker != device:
                    raise AssertionError(
                        f"root {index} executed on coordinator worker {executed_worker}, expected {device}")
                if tuple(coordinator.store.lineage(answer["snapshot"])[1]) != tokens:
                    raise AssertionError("root canonical history differs from submitted tokens")
                audit = verify_cfb1(blob, tokens, seed=DEFAULT_SEED,
                                    capacity=12288, physical_devices=list(actual))
                expected_frontier = cpu_frontier(DEFAULT_SEED, tokens)
                if tuple(frontier) != expected_frontier:
                    raise AssertionError("root durable frontier differs from CPU reference")
                root_records.append({
                    "index": index, "device": device, "branch": branch,
                    "snapshot": answer["snapshot"], "token_ids": list(tokens),
                    "cfb1_devices": list(actual), "executed_worker": executed_worker,
                    "audit": audit,
                })

            if {record["device"] for record in root_records} != set(devices):
                raise AssertionError("three roots did not cover all selected physical workers")

            jobs = []
            for index, root in enumerate(root_records):
                branch = f"indchild{index}"
                coordinator.fork(branch, root["snapshot"])
                before = coordinator.store.head(branch)
                suffix = tuple((index * 13 + 41 + offset) % 64 for offset in range(4))
                handle = coordinator.submit(branch, suffix, 8, f"ind-request-{index}")
                jobs.append((index, root, branch, suffix, before, handle))

            # The coordinator owns all native work on its existing worker
            # threads; this pool only waits for the three independent requests.
            with ThreadPoolExecutor(max_workers=3) as waiters:
                answers = list(waiters.map(
                    lambda item: (item[0], item[1], item[2], item[3], item[4],
                                  item[5].snapshot.result(300), item[5]),
                    jobs))

            request_records = []
            for index, root, branch, suffix, before, answer, handle in answers:
                handle.durable.result(300)
                executed_worker = _snapshot_event_worker(handle)
                expected_device = root["device"]
                blob, frontier = _durable_blob(
                    coordinator.store.root / "checkpoints", answer["snapshot"])
                actual = _cfb1_devices(blob)
                if actual != (expected_device,):
                    raise AssertionError(
                        f"request {index} ran/restored on {actual}, expected {(expected_device,)}")
                if executed_worker != expected_device:
                    raise AssertionError(
                        f"request {index} executed on worker {executed_worker}, expected {expected_device}")
                tokens = tuple(root["token_ids"]) + suffix + tuple(answer["tokens"])
                delta = answer.get("counter_deltas", {}).get("forward_tokens")
                if delta != 12:
                    raise AssertionError(f"request {index} forward counter was {delta}, expected 12")
                if tuple(coordinator.store.lineage(answer["snapshot"])[1]) != tokens:
                    raise AssertionError("branch canonical history differs from root, suffix, and output")
                controller = coordinator.store.get_snapshot(answer["snapshot"])["controller"]
                if controller.get("accepted_in_scope") != len(answer["tokens"]):
                    raise AssertionError("controller did not accept exactly the generated output")
                audit = verify_cfb1(blob, tokens, seed=DEFAULT_SEED,
                                    capacity=12288, physical_devices=list(actual))
                expected_frontier = cpu_frontier(DEFAULT_SEED, tokens)
                if tuple(frontier) != expected_frontier:
                    raise AssertionError("durable frontier differs from independent CPU reference")
                after = coordinator.store.head(branch)
                if after.revision != before.revision + 1:
                    raise AssertionError("branch did not advance exactly one accepted revision")
                request_records.append({
                    "index": index, "branch": branch,
                    "source_worker": root["device"],
                    "executed_worker": executed_worker,
                    "cfb1_devices": list(actual),
                    "supplied_suffix": list(suffix),
                    "generated_tokens": list(answer["tokens"]),
                    "forward_tokens": delta,
                    "accepted_output_tokens": controller["accepted_in_scope"],
                    "branch_revision_before": before.revision,
                    "branch_revision_after": after.revision,
                    "snapshot": answer["snapshot"],
                    "audit": audit,
                    "frontier_matches_cpu": True,
                })

            metrics = coordinator.metrics()
            report["roots"] = root_records
            report["requests"] = request_records
            report["max_active_per_gpu"] = metrics["max_active_per_gpu"]
            report["process_pool_after"] = metrics
            if len(request_records) != 3:
                raise AssertionError("not all three independent requests completed")
            if {item["executed_worker"] for item in request_records} != set(devices):
                raise AssertionError("request execution did not cover all three selected workers")
            if any(value != 1 for value in metrics["max_active_per_gpu"].values()):
                raise AssertionError("worker concurrency was not exactly one active lease per worker")
            report["status"] = "PASS"
            _write_report(output, report)
            return report
        except BaseException as exc:
            report["status"] = "FAIL"
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report["process_pool_after"] = coordinator.metrics()
            _write_report(output, report)
            raise
        finally:
            coordinator.close()
