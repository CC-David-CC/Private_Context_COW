"""Remote-only native resume pressure harness and fixture contract checks.

Run on a Linux GPU host after building fixture_backend.cu. This file does not
provide a Python model fallback; without the native binary the pressure run is
unavailable and must be recorded as UNRUN.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
BRANCHING = HERE.parent.parent / "experimental" / "branching_context"
sys.path.insert(0, str(BRANCHING / "src"))
sys.path.insert(0, str(HERE))

from model_free_backend import (DEFAULT_SEED, FixtureModel, cpu_frontier,
                               make_profile, recurrence_final, ModelFreeBackend)
import fixture_reference


def verify_cfb1(payload, token_ids, *, seed, capacity, physical_devices):
    """Independently validate serialized history, recurrence and every KV byte."""
    view = memoryview(payload)
    offset = 0

    def take(fmt):
        nonlocal offset
        width = struct.calcsize(fmt)
        if offset + width > len(view):
            raise AssertionError("truncated CFB1 header/partition")
        values = struct.unpack_from(fmt, view, offset)
        offset += width
        return values[0] if len(values) == 1 else values

    if bytes(view[:4]) != b"CFB1":
        raise AssertionError("serialized state does not have CFB1 magic")
    offset = 4
    if take("<Q") != seed or take("<Q") != capacity:
        raise AssertionError("CFB1 seed or capacity differs from pinned profile")
    parts = take("<I")
    consumed = take("<Q")
    if parts != len(physical_devices) or consumed != len(token_ids):
        raise AssertionError("CFB1 part count or consumed count differs")
    devices = tuple(take("<i") for _ in range(parts))
    if devices != tuple(physical_devices):
        raise AssertionError("CFB1 device topology differs from selected devices")
    history = tuple(take("<i") for _ in range(consumed))
    if history != tuple(token_ids):
        raise AssertionError("CFB1 canonical token history differs")
    final_state = recurrence_final(seed, token_ids)
    expected_row_bytes = len(token_ids) * 32
    compared = 0
    for layer in range(parts):
        mapped_bytes = take("<Q")
        running = take("<I")
        if running != final_state or mapped_bytes < expected_row_bytes:
            raise AssertionError("CFB1 partition recurrence or mapped extent differs")
        if offset + mapped_bytes > len(view):
            raise AssertionError("truncated CFB1 mapped page bytes")
        for position, token in enumerate(token_ids):
            start = offset + position * 32
            expected = bytes(
                __import__("fixture_reference").byte_at(
                    seed, 0, layer, token, channel, position * 32 + channel)
                for channel in range(32))
            if bytes(view[start:start + 32]) != expected:
                raise AssertionError(f"CFB1 row mismatch at layer={layer}, position={position}")
            compared += 32
        if any(view[offset + expected_row_bytes:offset + mapped_bytes]):
            raise AssertionError("CFB1 unused mapped tail is not zero")
        offset += mapped_bytes
    if offset != len(view):
        raise AssertionError("CFB1 file has unaccounted trailing bytes")
    frontier = cpu_frontier(seed, token_ids)
    frontier_sha = hashlib.sha256(struct.pack("<" + "f" * len(frontier), *frontier)).hexdigest()
    return dict(consumed=consumed, partitions=parts, kv_bytes_compared=compared,
                frontier_sha256=frontier_sha, serialized_bytes=len(payload))


class FixtureContractTests(unittest.TestCase):
    def test_integer_recipe_is_deterministic_and_token_sensitive(self):
        left = cpu_frontier(DEFAULT_SEED, (2, 7, 11))
        self.assertEqual(left, cpu_frontier(DEFAULT_SEED, (2, 7, 11)))
        self.assertNotEqual(left, cpu_frontier(DEFAULT_SEED, (2, 7, 12)))
        self.assertEqual(len(set(left)), 64)

    def test_profile_requires_pinned_binary_and_hardware_identity(self):
        with tempfile.NamedTemporaryFile() as binary:
            binary.write(b"native-fixture")
            binary.flush()
            profile = make_profile(binary.name, [0, 1], 128, hardware_identity="gpu-test")
            self.assertEqual(profile["backend"], "model-free-cuda")
            self.assertEqual(profile["execution"]["GPU"], [0, 1])
            self.assertNotEqual(profile["execution"]["resume_contract"],
                                "identical-native-operators-v1")

    def test_cfb1_oracle_catches_byte_and_cell_permutations(self):
        seed, capacity, tokens, devices = DEFAULT_SEED, 2, (5, 9), (0, 1)
        state = recurrence_final(seed, tokens)
        payload = bytearray(b"CFB1" + struct.pack("<QQIQ", seed, capacity, 2, len(tokens)))
        payload.extend(struct.pack("<ii", *devices))
        payload.extend(struct.pack("<ii", *tokens))
        for layer in range(2):
            page = bytearray(4096)
            for position, token in enumerate(tokens):
                row = bytes(fixture_reference.byte_at(seed, 0, layer, token, channel,
                                                       position * 32 + channel)
                            for channel in range(32))
                page[position * 32:(position + 1) * 32] = row
            payload.extend(struct.pack("<QI", len(page), state))
            payload.extend(page)
        oracle = verify_cfb1(payload, tokens, seed=seed, capacity=capacity,
                              physical_devices=devices)
        self.assertEqual(oracle["kv_bytes_compared"], 128)
        with self.assertRaises(AssertionError):
            verify_cfb1(payload, (9, 5), seed=seed, capacity=capacity,
                        physical_devices=devices)
        corrupt = bytearray(payload)
        # Swap the first two logical rows in one device partition.
        header_bytes = 4 + 8 + 8 + 4 + 8 + 8 + 8
        first_page = header_bytes + 8 + 4
        corrupt[first_page:first_page + 64] = corrupt[first_page + 32:first_page + 64] + corrupt[first_page:first_page + 32]
        with self.assertRaises(AssertionError):
            verify_cfb1(corrupt, tokens, seed=seed, capacity=capacity,
                        physical_devices=devices)

    def test_fixture_cases_use_reserved_capacity_geometry(self):
        cases = {name: (capacity, prompt, suffix, steps)
                 for name, capacity, prompt, suffix, steps in FIXTURE_CASES}
        self.assertEqual(cases["tiny"], (12288, 128, 4, 8))
        self.assertEqual(cases["medium"], (12288, 1024, 64, 16))
        self.assertEqual(cases["long"], (12288, 8192, 512, 128))
        self.assertEqual(profile_capacity("q4"), 12288)
        self.assertEqual(profile_capacity("q8"), 98304)

    def test_fixture_model_factory_uses_production_constructor_order(self):
        from unittest.mock import patch
        profile = {"execution": {"GPU": [0]}}
        with patch(__name__ + ".ModelFreeBackend", return_value="native") as backend:
            with patch(__name__ + ".FixtureModel", return_value="model") as model:
                self.assertEqual(build_fixture_model("dir", profile, 0), "model")
        backend.assert_called_once_with(profile, gpu_id=0)
        model.assert_called_once_with("dir", "native", profile)


FIXTURE_CASES = (
    ("tiny", 12288, 128, 4, 8),
    ("medium", 12288, 1024, 64, 16),
    ("long", 12288, 8192, 512, 128),
)


def profile_capacity(profile_name):
    if profile_name == "q4":
        return 12288
    if profile_name == "q8":
        return 98304
    raise ValueError("profile must be q4 or q8")


def build_fixture_model(directory, profile, gpu_id):
    return FixtureModel(directory, ModelFreeBackend(profile, gpu_id=gpu_id), profile)


def _write_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(target)


def _durable_blob(checkpoint_directory, snapshot):
    envelope = json.loads((checkpoint_directory / (snapshot + ".json")).read_text())
    manifest = envelope["manifest"]
    data = (checkpoint_directory / (manifest["payload_sha256"] + ".state")).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["payload_sha256"]:
        raise AssertionError("durable checkpoint envelope digest mismatch")
    size = manifest["state_bytes"]
    if len(data) != size + 4 * manifest["logits_count"]:
        raise AssertionError("durable checkpoint payload extent mismatch")
    frontier = struct.unpack("<" + "f" * manifest["logits_count"], data[size:])
    return data[:size], frontier


def _cfb1_devices(payload):
    if len(payload) < 32 or payload[:4] != b"CFB1":
        raise AssertionError("durable blob lacks CFB1 header")
    parts = struct.unpack_from("<I", payload, 20)[0]
    if parts <= 0 or 32 + 4 * parts > len(payload):
        raise AssertionError("durable CFB1 part count is invalid")
    return tuple(struct.unpack_from("<" + "i" * parts, payload, 32))


def _future_record(future, timeout=1200):
    if not future.done() and timeout <= 0:
        return dict(state="pending")
    try:
        if not future.done() and timeout > 0:
            future.result(timeout)
        value = future.result(0)
        return dict(state="resolved", value=value if isinstance(value, (dict, list, str, int, float, bool)) else str(value))
    except BaseException as error:
        return dict(state="exception", type=type(error).__name__, message=str(error))


def _hardware_fingerprint():
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except BaseException as error:
        gpu_info = [f"nvidia-smi-error:{type(error).__name__}:{error}"]
    facts = dict(hostname=platform.node(), platform=platform.platform(), gpu_info=gpu_info)
    facts["sha256"] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return facts


def run_pressure(binary, gpu_selection, hardware_identity, output, profile_name="q4",
                 case_filter="all"):
    """Run the actual resident native backend through coordinator-owned branches."""
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig

    capacity_override = profile_capacity(profile_name)
    cases = [(name, capacity_override, prompt, suffix, steps)
             for name, _, prompt, suffix, steps in FIXTURE_CASES]
    if case_filter != "all":
        cases = [case for case in cases if case[0] == case_filter]
        if not cases:
            raise ValueError("case must be all, tiny, medium, or long")
    suite = []
    report = dict(status="RUNNING", qualification="model-free-cuda-only",
                  profile=profile_name, binary_sha256=_sha_file(binary), cases=suite)
    _write_report(output, report)
    for name, capacity, prompt, suffix, steps in cases:
        profile = make_profile(binary, gpu_selection, capacity,
                               hardware_identity=hardware_identity)
        config = PoolConfig(gpus=tuple(gpu_selection), ram_bytes=2 * 1024**3,
                             disk_bytes=8 * 1024**3, min_available_bytes=0)

        def factory(gpu):
            return build_fixture_model(directory, profile, gpu)

        with tempfile.TemporaryDirectory(prefix="mf-resume-") as directory:
            coordinator = ContextCoordinator(directory, config, factory)
            coordinator.ready()
            try:
                roots = []
                for root_index in range(2):
                    branch = f"root-{name}-{root_index}"
                    prompt_ids = tuple((i + root_index * 13) % 64 for i in range(prompt))
                    handle = coordinator.create(branch, prompt_ids, f"create-{name}-{root_index}")
                    created = handle.snapshot.result(600)
                    handle.durable.result(600)
                    roots.append((branch, created["snapshot"], prompt_ids))

                jobs = []
                for root_index, (root_branch, root_snapshot, root_ids) in enumerate(roots):
                    for child in range(4):
                        branch = f"tree-{root_index}-{child}"
                        coordinator.fork(branch, root_snapshot)
                        suffix_ids = tuple((child * 7 + i) % 64 for i in range(suffix))
                        operation = f"{name}-r{root_index}-b{child}"
                        jobs.append((branch, suffix_ids, operation, root_ids))

                def submit(job):
                    branch, suffix_ids, operation, root_ids = job
                    handle = coordinator.submit(branch, suffix_ids, steps, operation)
                    answer = handle.snapshot.result(1200)
                    handle.durable.result(1200)
                    expected_forward = len(suffix_ids) + len(answer["tokens"])
                    if answer["counter_deltas"].get("forward_tokens") != expected_forward:
                        raise AssertionError("native worker replayed or omitted prefix/request work")
                    final_ids = root_ids + suffix_ids + tuple(answer["tokens"])
                    frontier = cpu_frontier(DEFAULT_SEED, final_ids)
                    return dict(branch=branch, snapshot=answer["snapshot"], tokens=answer["tokens"],
                                final_ids=final_ids, frontier=frontier,
                                forward_tokens=answer["counter_deltas"]["forward_tokens"])

                with ThreadPoolExecutor(max_workers=4) as clients:
                    results = list(clients.map(submit, jobs))

                comparisons = []
                checkpoint_directory = coordinator.store.root / "checkpoints"
                device_groups = [(worker,) if type(worker) is int else tuple(worker)
                                 for worker in gpu_selection]
                for result in results:
                    canonical_history = tuple(coordinator.store.lineage(result["snapshot"])[1])
                    if canonical_history != tuple(result["final_ids"]):
                        raise AssertionError("published canonical inputs differ from request branches")
                    controller = coordinator.store.get_snapshot(result["snapshot"])["controller"]
                    if controller.get("accepted_in_scope") != len(result["tokens"]):
                        raise AssertionError("published controller did not consume exactly generated outputs")
                    blob, durable_frontier = _durable_blob(checkpoint_directory, result["snapshot"])
                    expected = cpu_frontier(DEFAULT_SEED, result["final_ids"])
                    if tuple(durable_frontier) != expected or tuple(result["frontier"]) != expected:
                        raise AssertionError("durable frontier differs from independent CPU oracle")
                    actual_devices = _cfb1_devices(blob)
                    if actual_devices not in device_groups:
                        raise AssertionError("durable CFB1 devices do not match any selected worker")
                    physical = list(actual_devices)
                    comparisons.append(verify_cfb1(blob, result["final_ids"], seed=DEFAULT_SEED,
                                                   capacity=capacity, physical_devices=physical))
                # Stop every native process, then reopen the exact canonical store.
                coordinator.close()
                coordinator = ContextCoordinator(directory, config, factory)
                coordinator.ready()
                resume_handle = coordinator.submit(roots[0][0], (3, 4), 2,
                                                   f"restart-{name}")
                resumed = resume_handle.snapshot.result(1200)
                resume_handle.durable.result(1200)
                if resumed["counter_deltas"].get("forward_tokens") != 4:
                    raise AssertionError("fresh-process request replayed saved root prefix")
                resumed_ids = roots[0][2] + (3, 4) + tuple(resumed["tokens"])
                if tuple(coordinator.store.lineage(resumed["snapshot"])[1]) != resumed_ids:
                    raise AssertionError("fresh-process canonical history differs from resumed branch")
                resumed_controller = coordinator.store.get_snapshot(resumed["snapshot"])["controller"]
                if resumed_controller.get("accepted_in_scope") != len(resumed["tokens"]):
                    raise AssertionError("fresh-process controller did not consume resumed outputs")
                new_blob, new_frontier = _durable_blob(checkpoint_directory,
                                                       resumed["snapshot"])
                expected_frontier = cpu_frontier(DEFAULT_SEED, resumed_ids)
                if tuple(new_frontier) != expected_frontier:
                    raise AssertionError("fresh-process resumed frontier differs from CPU oracle")
                resumed_devices = _cfb1_devices(new_blob)
                if resumed_devices not in device_groups:
                    raise AssertionError("fresh-process CFB1 devices do not match any selected worker")
                verify_cfb1(new_blob, resumed_ids, seed=DEFAULT_SEED, capacity=capacity,
                            physical_devices=list(resumed_devices))
                suite.append(dict(case=name, roots=2, branch_requests=len(results),
                                  queued_clients=4, snapshots=len(comparisons),
                                  native_forward_tokens=[result["forward_tokens"] for result in results],
                                  fresh_process_resume=True, cpu_state=comparisons))
                _write_report(output, report)
            finally:
                coordinator.close()
    report["status"] = "PASS"
    _write_report(output, report)
    return report


def run_mixed_probe(binary, gpu_selection, hardware_identity, output):
    """Force same-root work onto the other worker and retain topology failure evidence."""
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig

    expected_groups = ((0, 1), 2)
    if tuple(gpu_selection) != expected_groups:
        raise ValueError("mixed probe requires worker topology ((0, 1), 2)")
    fingerprint = _hardware_fingerprint()
    report = dict(status="RUNNING", case="mixed-tiny", probe="mixed-cfb1-worker-affinity-v1",
                  binary=str(Path(binary).resolve()), binary_sha256=_sha_file(binary),
                  hardware_identity=hardware_identity, hardware_fingerprint=fingerprint,
                  topology=[[0, 1], 2], capacity=12288, prefix_tokens=128,
                  suffix_tokens=4, max_steps=8, process_pool_before=None,
                  roots=[], futures={}, branch_heads={}, failure=None)
    _write_report(output, report)
    profile = make_profile(binary, gpu_selection, 12288,
                           hardware_identity=hardware_identity)
    config = PoolConfig(gpus=expected_groups, ram_bytes=2 * 1024**3,
                        disk_bytes=8 * 1024**3, min_available_bytes=0)
    with tempfile.TemporaryDirectory(prefix="mf-mixed-") as directory:
        def factory(gpu):
            return build_fixture_model(directory, profile, gpu)

        coordinator = ContextCoordinator(directory, config, factory)
        try:
            coordinator.ready()
            report["process_pool_before"] = dict(
                coordinator=coordinator.metrics(),
                workers=[dict(worker=str(gpu), pid=worker["model"].backend.process.pid,
                              selected=list(gpu) if isinstance(gpu, tuple) else [gpu])
                         for gpu, worker in coordinator._workers.items()])
            roots = []
            # Submit both roots before waiting, so each worker receives one root.
            root_handles = []
            for index in range(2):
                branch = f"mixed-root-{index}"
                ids = tuple((index * 19 + i) % 64 for i in range(128))
                root_handles.append((branch, ids,
                    coordinator.create(branch, ids, f"mixed-create-{index}")))
            for branch, ids, handle in root_handles:
                snapshot = handle.snapshot.result(300)
                handle.durable.result(300)
                state, frontier = _durable_blob(coordinator.store.root / "checkpoints",
                                                snapshot["snapshot"])
                devices = _cfb1_devices(state)
                roots.append(dict(branch=branch, snapshot=snapshot["snapshot"], ids=ids,
                                  devices=devices, frontier_sha256=hashlib.sha256(
                                      struct.pack("<" + "f" * len(frontier), *frontier)).hexdigest()))
            report["roots"] = [dict(branch=root["branch"], snapshot=root["snapshot"],
                                    token_ids=list(root["ids"]), devices=list(root["devices"]))
                               for root in roots]
            groups = { (0, 1): (0, 1), (2,): 2 }
            home_devices = roots[0]["devices"]
            home_id = groups.get(home_devices)
            if home_id not in coordinator._workers:
                raise AssertionError(f"root checkpoint topology has no worker: {home_devices}")
            other_id = 2 if home_id == (0, 1) else (0, 1)
            report["source_worker"] = str(home_id)
            report["target_worker"] = str(other_id)
            home_model = coordinator._workers[home_id]["model"]
            entered = threading.Event()
            release = threading.Event()

            def hold_owner_thread(point):
                if point == "after_token_consume" and not entered.is_set():
                    entered.set()
                    if not release.wait(120):
                        raise TimeoutError("mixed probe owner-thread hold timed out")

            # This only installs a test hook. All native operations remain on
            # their existing coordinator worker threads.
            home_model._lease_fault = hold_owner_thread
            source = roots[0]
            branch_a, branch_b = "mixed-child-a", "mixed-child-b"
            coordinator.fork(branch_a, source["snapshot"])
            coordinator.fork(branch_b, source["snapshot"])
            before_a, before_b = coordinator.store.head(branch_a), coordinator.store.head(branch_b)
            report["branch_heads"]["before"] = {
                branch_a: dict(tip=before_a.tip, revision=before_a.revision),
                branch_b: dict(tip=before_b.tip, revision=before_b.revision)}
            report["branch_inputs"] = {
                branch_a: [1, 2, 3, 4], branch_b: [5, 6, 7, 8],
                "max_steps": 8}
            first = coordinator.submit(branch_a, (1, 2, 3, 4), 8, "mixed-operation-a")
            if not entered.wait(120):
                raise TimeoutError("source worker did not reach the owner-thread hold")
            second = coordinator.submit(branch_b, (5, 6, 7, 8), 8, "mixed-operation-b")
            # Wait until the dispatcher has selected the only idle worker.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with coordinator._condition:
                    target_busy = coordinator._workers[other_id]["busy"]
                if target_busy:
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("coordinator did not dispatch queued request to other topology")
            report["futures"]["operation_a_snapshot_while_held"] = _future_record(first.snapshot, 0)
            report["futures"]["operation_b_snapshot"] = _future_record(second.snapshot, 300)
            report["futures"]["operation_b_durable"] = _future_record(second.durable, 300)
            release.set()
            report["futures"]["operation_a_snapshot"] = _future_record(first.snapshot, 300)
            report["futures"]["operation_a_durable"] = _future_record(first.durable, 300)
            after_a, after_b = coordinator.store.head(branch_a), coordinator.store.head(branch_b)
            report["branch_heads"]["after"] = {
                branch_a: dict(tip=after_a.tip, revision=after_a.revision),
                branch_b: dict(tip=after_b.tip, revision=after_b.revision)}
            report["branch_unchanged_after_failed_request"] = (
                after_b.tip == before_b.tip and after_b.revision == before_b.revision)
            b_snapshot = report["futures"]["operation_b_snapshot"]
            if b_snapshot["state"] == "exception":
                report["status"] = "FAIL"
                report["failure"] = b_snapshot
            else:
                report["status"] = "UNEXPECTED_PASS"
                report["failure"] = "cross-topology request unexpectedly resolved"
            report["process_pool_after"] = dict(
                coordinator=coordinator.metrics(),
                workers=[dict(worker=str(gpu), pid=worker["model"].backend.process.pid,
                              alive=worker["model"].backend.process.poll() is None)
                         for gpu, worker in coordinator._workers.items()])
            _write_report(output, report)
            return report
        except BaseException as error:
            report["status"] = "ERROR"
            report["failure"] = dict(type=type(error).__name__, message=str(error))
            report["process_pool_after"] = coordinator.metrics()
            _write_report(output, report)
            raise
        finally:
            coordinator.close()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="compiled fixture_backend native server")
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--hardware-identity", required=True)
    parser.add_argument("--profile", choices=("q4", "q8"), default="q4")
    parser.add_argument("--case", choices=("all", "tiny", "medium", "long"), default="all")
    parser.add_argument("--mixed-probe", action="store_true",
                        help="run the forced cross-topology negative regression for --gpus 0+1;2")
    parser.add_argument("--output", default="model-free-resume-report.json")
    args = parser.parse_args()
    gpu_selection = tuple(
        tuple(int(device) for device in worker.split("+"))
        if "+" in worker else int(worker)
        for worker in args.gpus.split(";"))
    if not Path(args.binary).is_file():
        parser.error("native fixture binary is missing; runtime qualification is UNRUN")
    try:
        if args.mixed_probe:
            report = run_mixed_probe(args.binary, gpu_selection, args.hardware_identity, args.output)
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("hardware_fingerprint", "process_pool_before",
                                             "process_pool_after")}, indent=2))
            if report["status"] != "PASS":
                raise RuntimeError("mixed topology regression failed; inspect the retained JSON report")
            return
        report = run_pressure(args.binary, gpu_selection, args.hardware_identity, args.output,
                              args.profile, args.case)
    except BaseException as error:
        target = Path(args.output)
        if target.is_file():
            retained = json.loads(target.read_text())
            if retained.get("status") == "RUNNING":
                retained["status"] = "FAIL"
                retained["failure"] = f"{type(error).__name__}: {error}"
            else:
                retained["runner_exception"] = f"{type(error).__name__}: {error}"
            _write_report(target, retained)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--binary":
        main()
    else:
        unittest.main()
