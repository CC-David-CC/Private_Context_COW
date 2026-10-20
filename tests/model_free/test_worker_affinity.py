"""Remote native regression for checkpoint-compatible worker dispatch.

Run on a Linux GPU host with the pinned fixture backend. The test deliberately
holds the source worker while a sibling request and unrelated compatible work
are queued; it never mocks or bypasses native IPC.
"""
import argparse
from concurrent.futures import TimeoutError
import hashlib
import json
from pathlib import Path
import struct
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
from model_free_backend import ModelFreeBackend, make_profile, FixtureModel
from test_model_free_resume import (_cfb1_devices, _durable_blob,
                                    build_fixture_model, verify_cfb1)


def _future_state(future):
    if not future.done():
        return {"state": "pending"}
    try:
        return {"state": "resolved", "value": future.result(0)}
    except BaseException as exc:
        return {"state": "exception", "type": type(exc).__name__, "message": str(exc)}


def _record(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(target)


def run_worker_affinity_probe(binary, hardware_identity, output):
    """Require compatible dispatch while proving another worker stays useful."""
    groups = ((0, 1), 2)
    profile = make_profile(binary, [[0, 1], 2], 12288,
                           hardware_identity=hardware_identity)
    config = PoolConfig(gpus=groups, ram_bytes=2 * 1024**3,
                        disk_bytes=8 * 1024**3, min_available_bytes=0)
    report = {"status": "RUNNING", "case": "worker-affinity-tiny",
              "probe": "compatible-checkpoint-worker-affinity-v1",
              "binary": str(Path(binary).resolve()),
              "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
              "hardware_identity": hardware_identity, "worker_groups": [[0, 1], [2]],
              "capacity": 12288, "prefix_tokens": 128, "suffix_tokens": 4,
              "max_steps": 8, "roots": [], "futures": {}, "branch_heads": {},
              "native_audits": {}, "failure": None}
    _record(output, report)
    with tempfile.TemporaryDirectory(prefix="mf-affinity-") as directory:
        def factory(gpu):
            return build_fixture_model(directory, profile, gpu)

        coordinator = ContextCoordinator(directory, config, factory)
        try:
            coordinator.ready()
            # Concurrent root creation gives each idle worker one canonical
            # checkpoint. Placement is read from native CFB1, never inferred.
            creations = []
            for index in range(2):
                ids = tuple((index * 19 + i) % 64 for i in range(128))
                handle = coordinator.create(f"affinity-root-{index}", ids,
                                            f"affinity-create-{index}")
                creations.append((index, ids, handle))
            roots = []
            for index, ids, handle in creations:
                snapshot = handle.snapshot.result(300)
                handle.durable.result(300)
                blob, frontier = _durable_blob(coordinator.store.root / "checkpoints",
                                               snapshot["snapshot"])
                devices = _cfb1_devices(blob)
                roots.append({"index": index, "snapshot": snapshot["snapshot"],
                              "token_ids": ids, "devices": devices,
                              "creation": snapshot,
                              "audit": verify_cfb1(blob, ids, seed=profile["execution"]["seed"],
                                                   capacity=12288, physical_devices=devices)})
            source = next(root for root in roots if root["devices"] == (0, 1))
            unrelated = next(root for root in roots if root["devices"] == (2,))
            report["roots"] = [{"index": root["index"], "snapshot": root["snapshot"],
                                "devices": list(root["devices"]), "audit": root["audit"]}
                               for root in roots]
            report["source_worker"] = "(0, 1)"
            report["unrelated_worker"] = "2"
            a, b, c = "affinity-child-a", "affinity-child-b", "affinity-unrelated-c"
            for branch in (a, b, c):
                coordinator.fork(branch, source["snapshot"] if branch != c else unrelated["snapshot"])
            before = {branch: coordinator.store.head(branch) for branch in (a, b, c)}
            report["branch_heads"]["before"] = {
                branch: {"tip": head.tip, "revision": head.revision}
                for branch, head in before.items()}

            owner = coordinator._workers[(0, 1)]["model"]
            entered, release = threading.Event(), threading.Event()

            def hold_source_owner(point):
                if point == "after_token_consume" and not entered.is_set():
                    entered.set()
                    if not release.wait(180):
                        raise TimeoutError("held source worker timed out")

            owner._lease_fault = hold_source_owner
            inputs = {a: (1, 2, 3, 4), b: (5, 6, 7, 8), c: (9, 10, 11, 12)}
            first = coordinator.submit(a, inputs[a], 8, "affinity-operation-a")
            if not entered.wait(120):
                raise TimeoutError("source worker did not reach owner-thread hold")
            third = coordinator.submit(c, inputs[c], 8, "affinity-operation-c")

            # Run C first on GPU 2 while A remains held. Then leave GPU 2 idle
            # and submit B; only compatible dispatch may keep B queued for the
            # (0,1) source worker instead of trying and poisoning GPU 2.
            deadline = time.monotonic() + 300
            while not third.durable.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            report["futures"]["A_snapshot_while_held"] = _future_state(first.snapshot)
            report["futures"]["C_snapshot_while_A_held"] = _future_state(third.snapshot)
            report["futures"]["C_durable_while_A_held"] = _future_state(third.durable)
            if not third.durable.done():
                raise TimeoutError("compatible unrelated work did not finish on GPU 2")
            c_answer = third.snapshot.result(0)
            c_blob, _c_frontier = _durable_blob(coordinator.store.root / "checkpoints",
                                                c_answer["snapshot"])
            c_devices = _cfb1_devices(c_blob)
            report["native_audits"]["C"] = verify_cfb1(
                c_blob, unrelated["token_ids"] + inputs[c] + tuple(c_answer["tokens"]),
                seed=profile["execution"]["seed"], capacity=12288,
                physical_devices=c_devices)
            report["native_audits"]["C"]["devices"] = list(c_devices)
            report["native_audits"]["C"]["forward_tokens"] = c_answer["counter_deltas"]["forward_tokens"]
            if c_devices != (2,):
                raise AssertionError(f"unrelated C ran on wrong native devices: {c_devices}")
            if c_answer["counter_deltas"]["forward_tokens"] != len(inputs[c]) + len(c_answer["tokens"]):
                raise AssertionError("C native counter shows prefix replay or missing work")

            second = coordinator.submit(b, inputs[b], 8, "affinity-operation-b")
            time.sleep(0.5)
            report["futures"]["B_snapshot_while_A_held"] = _future_state(second.snapshot)
            report["futures"]["B_durable_while_A_held"] = _future_state(second.durable)
            if report["futures"]["B_snapshot_while_A_held"]["state"] != "pending":
                raise AssertionError("B was assigned to an incompatible idle worker")

            release.set()
            a_answer = first.snapshot.result(300)
            first.durable.result(300)
            b_answer = second.snapshot.result(300)
            second.durable.result(300)
            report["futures"]["A_snapshot"] = _future_state(first.snapshot)
            report["futures"]["A_durable"] = _future_state(first.durable)
            report["futures"]["B_snapshot_after_A"] = _future_state(second.snapshot)
            report["futures"]["B_durable_after_A"] = _future_state(second.durable)
            for label, answer, branch, expected_devices, prefix, suffix in (
                    ("A", a_answer, a, (0, 1), source["token_ids"], inputs[a]),
                    ("B", b_answer, b, (0, 1), source["token_ids"], inputs[b])):
                blob, _frontier = _durable_blob(coordinator.store.root / "checkpoints",
                                                answer["snapshot"])
                devices = _cfb1_devices(blob)
                report["native_audits"][label] = verify_cfb1(
                    blob, prefix + suffix + tuple(answer["tokens"]),
                    seed=profile["execution"]["seed"], capacity=12288,
                    physical_devices=devices)
                report["native_audits"][label]["devices"] = list(devices)
                report["native_audits"][label]["forward_tokens"] = answer["counter_deltas"]["forward_tokens"]
                if devices != expected_devices:
                    raise AssertionError(f"{label} restored/ran on wrong devices: {devices}")
                if answer["counter_deltas"]["forward_tokens"] != len(suffix) + len(answer["tokens"]):
                    raise AssertionError(f"{label} native counter shows prefix replay or missing work")
            for branch, answer in ((a, a_answer), (b, b_answer), (c, c_answer)):
                coordinator.store.head(branch)
            after = {branch: coordinator.store.head(branch) for branch in (a, b, c)}
            report["branch_heads"]["after"] = {
                branch: {"tip": head.tip, "revision": head.revision}
                for branch, head in after.items()}
            report["branch_heads"]["all_advanced_once"] = all(
                after[name].revision == before[name].revision + 1 for name in (a, b, c))
            if not report["branch_heads"]["all_advanced_once"]:
                raise AssertionError("one of the request branches failed to publish exactly once")
            report["process_pool_after"] = coordinator.metrics()
            report["status"] = "PASS"
            _record(output, report)
            return report
        except BaseException as exc:
            report["status"] = "FAIL"
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report["process_pool_after"] = coordinator.metrics()
            _record(output, report)
            raise
        finally:
            release.set() if "release" in locals() else None
            coordinator.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--hardware-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_worker_affinity_probe(args.binary, args.hardware_identity, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
