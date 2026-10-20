"""Probe explicit canonical-history rebuild after all checkpoint files are evicted.

This is a hardware probe, not a CPU/mock qualification. It keeps the canonical
Store and operation receipts, removes only numerical checkpoint files from its
own newly-created Store, then measures an explicitly reported source rebuild.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def canonical_json_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def numerical_checkpoint_files(store_root: Path) -> list[Path]:
    """Return only direct regular manifest/state files in this Store's checkpoints."""
    store_root = store_root.resolve(strict=True)
    checkpoint_dir = store_root / "checkpoints"
    if checkpoint_dir.is_symlink() or checkpoint_dir.resolve(strict=True) != checkpoint_dir:
        raise ValueError("Checkpoint directory must be a direct, non-symlink Store child")
    files = []
    for path in sorted(checkpoint_dir.iterdir()):
        if path.suffix not in (".json", ".state"):
            raise ValueError(f"Unexpected entry in numerical checkpoint directory: {path.name}")
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True).parent != checkpoint_dir:
            raise ValueError(f"Refusing non-regular or redirected checkpoint path: {path}")
        files.append(path)
    return files


def plan_checkpoint_eviction(store_root: Path, expected_snapshots: set[str]) -> list[dict]:
    """Describe a strict all-checkpoint eviction without touching canonical Store data."""
    if not expected_snapshots or any(not isinstance(item, str) or not item for item in expected_snapshots):
        raise ValueError("Expected snapshot set must be nonempty")
    if any(Path(item).name != item or item in (".", "..") for item in expected_snapshots):
        raise ValueError("Snapshot identifiers may not redirect checkpoint paths")
    files = numerical_checkpoint_files(store_root)
    rows = []
    manifests = {}
    states = []
    for path in files:
        if path.suffix == ".json":
            if path.stem not in expected_snapshots:
                raise ValueError(f"Unexpected numerical checkpoint outside producer snapshots: {path.name}")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            manifest = envelope.get("manifest")
            if type(manifest) is not dict or manifest.get("snapshot") != path.stem:
                raise ValueError(f"Checkpoint manifest identity differs from filename: {path.name}")
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
            if envelope.get("checksum") != hashlib.sha256(canonical).hexdigest():
                raise ValueError(f"Checkpoint manifest checksum is invalid: {path.name}")
            payload = manifest.get("payload_sha256")
            if (not isinstance(payload, str) or len(payload) != 64
                    or any(char not in "0123456789abcdef" for char in payload)):
                raise ValueError(f"Malformed payload identity in checkpoint manifest: {path.name}")
            manifests[path.stem] = payload
        else:
            states.append(path)
        rows.append({"relative_path": path.relative_to(store_root).as_posix(),
                     "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if set(manifests) != expected_snapshots:
        raise ValueError("Checkpoint manifest inventory is incomplete")
    expected_names = {snapshot + ".json" for snapshot in expected_snapshots}
    expected_names |= {payload + ".state" for payload in manifests.values()}
    observed_names = {Path(row["relative_path"]).name for row in rows}
    if observed_names != expected_names:
        raise ValueError("Checkpoint manifest/state inventory is incomplete or has extras")
    expected_payloads = set(manifests.values())
    file_digests = {Path(row["relative_path"]).name: row["sha256"] for row in rows}
    for path in states:
        # Payload files are content-addressed by the manifest. Their name
        # alone is insufficient; verify the bytes before authorizing eviction.
        if path.stem not in expected_payloads:
            raise ValueError(f"Checkpoint state has no matching manifest payload identity: {path.name}")
        if file_digests[path.name] != path.stem:
            raise ValueError(f"Checkpoint payload bytes do not match their SHA-256 filename: {path.name}")
    return rows


def evict_checkpoint_files(store_root: Path, expected_snapshots: set[str], inventory: list[dict]) -> dict:
    """Delete exactly the prehashed manifest/state inventory; no recursive removal."""
    store_root = store_root.resolve(strict=True)
    try:
        before = plan_checkpoint_eviction(store_root, expected_snapshots)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Checkpoint inventory changed after it was frozen") from exc
    if before != inventory:
        raise RuntimeError("Checkpoint inventory changed after it was frozen")
    removed = []
    for row in inventory:
        path = store_root / row["relative_path"]
        if path.resolve(strict=True).parent != store_root / "checkpoints":
            raise ValueError("Checkpoint path escaped its direct directory")
        path.unlink()
        removed.append(row)
    directory_fd = os.open(store_root / "checkpoints", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if numerical_checkpoint_files(store_root):
        raise RuntimeError("Numerical checkpoints remain after exact eviction")
    return {"removed": removed, "removed_bytes": sum(row["bytes"] for row in removed)}


def _canonical_store_view(store, snapshots: set[str]) -> dict:
    with store._locked():
        refs = store._read_refs()
        records = {snapshot: store.get_snapshot(snapshot) for snapshot in sorted(snapshots)}
        histories = {snapshot: list(store.lineage(snapshot)[1]) for snapshot in sorted(snapshots)}
    return {"refs": refs, "snapshot_records": records, "histories": histories}


def evict_via_disk_pool(store, disk_pool_type, expected_snapshots: set[str],
                        inventory: list[dict]) -> dict:
    """Use the production DiskCheckpointPool reservation path to force LRU eviction."""
    checked = plan_checkpoint_eviction(store.root, expected_snapshots)
    if checked != inventory:
        raise RuntimeError("Checkpoint inventory changed after it was frozen")
    total_bytes = sum(row["bytes"] for row in inventory)
    if total_bytes <= 0:
        raise ValueError("Checkpoint pressure inventory must have positive size")
    canonical_before = _canonical_store_view(store, expected_snapshots)
    canonical_before_sha256 = canonical_json_sha256(canonical_before)
    receipt_before = canonical_before["refs"].get("receipts", {}).get("op-source")
    pool = disk_pool_type(store, byte_limit=total_bytes)
    before = pool.metrics()
    if before.get("file_bytes") != total_bytes or before.get("reserved_bytes") != 0:
        raise AssertionError("Disk pool did not observe the exact preflight inventory and empty reservation state")
    reservation = None
    try:
        reservation = pool.reserve(total_bytes, timeout=0)
        during = pool.metrics()
        if (during.get("file_bytes") != 0 or during.get("reserved_bytes") != total_bytes
                or during.get("files_evicted") != len(inventory)):
            raise AssertionError("Disk pool pressure did not evict the exact frozen inventory")
        if numerical_checkpoint_files(store.root):
            raise AssertionError("Disk pool left numerical checkpoints after the pressure reservation")
    finally:
        if reservation is not None:
            reservation.release()
    after = pool.metrics()
    if after.get("reserved_bytes") != 0 or after.get("file_bytes") != 0:
        raise AssertionError("Disk pool reservation was not released cleanly after eviction")
    if after.get("files_evicted") != len(inventory):
        raise AssertionError("Disk pool eviction counter differs from the exact inventory")
    canonical_after = _canonical_store_view(store, expected_snapshots)
    canonical_after_sha256 = canonical_json_sha256(canonical_after)
    receipt_after = canonical_after["refs"].get("receipts", {}).get("op-source")
    if canonical_after_sha256 != canonical_before_sha256 or receipt_after != receipt_before:
        raise AssertionError("Disk pool eviction changed canonical references, snapshots, histories, or receipt")
    return {"method": "production-DiskCheckpointPool.reserve-pressure",
            "expected_snapshots": sorted(expected_snapshots),
            "pressure_budget_bytes": total_bytes,
            "inventory_file_count": len(inventory),
            "inventory_bytes": total_bytes,
            "evicted_files": inventory,
            "metrics_before": before, "metrics_during_reservation": during,
            "metrics_after_release": after,
            "canonical_before_sha256": canonical_before_sha256,
            "canonical_after_sha256": canonical_after_sha256,
            "producer_receipt_unchanged": True,
            "canonical_snapshots_refs_controllers_histories_unchanged": True,
            "checkpoint_directory_empty": True}


def verify_rebuild_evidence(evidence: object, *, snapshot: str, history_tokens: int,
                            model_contract_id: str, execution_profile_id: str) -> dict:
    """Validate the coordinator's explicit prefix-reconstruction accounting."""
    if type(evidence) is not dict:
        raise ValueError("Missing source_recomputation evidence")
    expected = {"source_kind": "recomputed-history", "durable": False,
                "prior_computation_preserved": False,
                "prefix_history_tokens": history_tokens,
                "prefix_forward_tokens": history_tokens,
                "prefix_consumed_tokens": history_tokens}
    if (any(evidence.get(key) != value for key, value in expected.items())
            or type(evidence.get("durable")) is not bool
            or type(evidence.get("prior_computation_preserved")) is not bool
            or any(type(evidence.get(key)) is not int for key in
                   ("prefix_history_tokens", "prefix_forward_tokens", "prefix_consumed_tokens"))):
        raise ValueError("Source reconstruction evidence does not match canonical history work")
    if evidence.get("snapshot") not in (None, snapshot):
        raise ValueError("Source reconstruction names a different snapshot")
    # Older compatible coordinator summaries omit identity fields, but when
    # present they must bind exactly to this worker's qualified profile.
    for key, value in (("model_contract_id", model_contract_id),
                       ("execution_profile_id", execution_profile_id)):
        if key in evidence and evidence[key] != value:
            raise ValueError(f"Source reconstruction {key} differs from worker profile")
    return dict(expected, snapshot=snapshot,
                schedule_sha256=evidence.get("schedule_sha256"),
                controller_sha256=evidence.get("controller_sha256"),
                seconds=evidence.get("seconds"))


def verify_observed_rebuild_receipt(receipt: object, evidence: dict, *, snapshot: str,
                                    history_tokens: int, model_contract_id: str,
                                    execution_profile_id: str, controller: dict,
                                    expected_schedule_sha256: str) -> dict:
    if type(receipt) is not dict:
        raise ValueError("Rebuild observer did not capture a receipt")
    required = {"snapshot": snapshot, "source_kind": "recomputed-history", "durable": False,
                "prior_computation_preserved": False, "consumed": history_tokens,
                "forward_tokens": history_tokens, "model_contract_id": model_contract_id,
                "execution_profile_id": execution_profile_id}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Observed rebuild receipt does not bind the exact source/history")
    if (type(receipt.get("durable")) is not bool or type(receipt.get("prior_computation_preserved")) is not bool
            or type(receipt.get("consumed")) is not int or type(receipt.get("forward_tokens")) is not int):
        raise ValueError("Observed rebuild receipt has non-exact field types")
    schedule = receipt.get("schedule_sha256")
    controller_hash = receipt.get("controller_sha256")
    if (not isinstance(schedule, str) or len(schedule) != 64
            or any(char not in "0123456789abcdef" for char in schedule)
            or schedule != expected_schedule_sha256
            or evidence.get("schedule_sha256") != schedule):
        raise ValueError("Observed execution schedule digest differs from rebuilt cache/evidence")
    expected_controller = canonical_json_sha256(controller)
    if (controller_hash != expected_controller
            or evidence.get("controller_sha256") != controller_hash):
        raise ValueError("Observed controller digest differs from canonical source/evidence")
    deltas = receipt.get("counter_deltas")
    if (type(deltas) is not dict or deltas.get("forward_tokens") != history_tokens
            or type(deltas.get("forward_tokens")) is not int):
        raise ValueError("Observed native rebuild counters differ from source history")
    return {**required, "schedule_sha256": schedule, "controller_sha256": controller_hash,
            "counter_deltas": deltas}


def is_missing_checkpoint_error(error: BaseException) -> bool:
    """Accept only an ENOENT/FileNotFound chain, not arbitrary restore failures."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, FileNotFoundError):
            return True
        if isinstance(current, OSError) and current.errno == errno.ENOENT:
            return True
        current = current.__cause__ or current.__context__
    return False


def is_detached_incomplete_checkpoint_error(error: BaseException,
                                           integrity_error_type: type[BaseException]) -> bool:
    """Match only the coordinator's exact detached missing-payload diagnostic."""
    return (type(error) is integrity_error_type
            and str(error) == "Invalid or incomplete checkpoint")


def validate_work_split(*, history_tokens: int, suffix_tokens: int, output_tokens: int,
                        prefix_forward: object, prefix_consumed: object,
                        request_forward: object, total_forward_delta: object) -> dict:
    if (type(prefix_forward) is not int or type(prefix_consumed) is not int
            or type(request_forward) is not int or type(total_forward_delta) is not int
            or min(history_tokens, suffix_tokens, output_tokens) < 0):
        raise ValueError("Work counters must be exact integers")
    if prefix_forward != history_tokens or prefix_consumed != history_tokens:
        raise ValueError("Rebuild work does not equal complete source history")
    expected_request = suffix_tokens + output_tokens
    if request_forward != expected_request:
        raise ValueError("Suffix/output work differs from the submitted request")
    if total_forward_delta != history_tokens + expected_request:
        raise ValueError("Total native work does not equal rebuild plus suffix/output work")
    return {"prefix_history_tokens": history_tokens, "prefix_forward_tokens": prefix_forward,
            "prefix_consumed_tokens": prefix_consumed, "suffix_tokens": suffix_tokens,
            "output_tokens": output_tokens, "suffix_and_output_forward_tokens": request_forward,
            "total_forward_tokens": total_forward_delta}


def wait_child_exit(child) -> int:
    """Return the real terminal status even when the polling loop saw exit first."""
    code = child.wait()
    if type(code) is not int:
        raise ValueError("Child process did not provide an integer exit status")
    return code


def require_no_native_work(before: dict, after: dict, phase: str) -> None:
    if type(before) is not dict or type(after) is not dict or before != after:
        raise ValueError(f"{phase} changed native counters before a request was allowed")


def native_stats_delta(before: dict, after: dict) -> dict:
    """Return exact integer counter deltas while retaining full snapshots separately."""
    if type(before) is not dict or type(after) is not dict:
        raise TypeError("Native statistics must be objects")
    delta = {}
    for key in sorted(set(before) | set(after)):
        left, right = before.get(key), after.get(key)
        if type(left) is int and type(right) is int:
            delta[key] = right - left
    return delta


def bind_oracle_result(value: object, *, pid: int, exit_code: int,
                       requested_frontiers: int, generated_limit: int) -> dict:
    """Bind the native_request_probe return shape to our observed process and files."""
    if type(value) is not dict:
        raise ValueError("Native oracle returned a non-object result")
    if type(pid) is not int or pid <= 0 or type(exit_code) is not int or exit_code != 0:
        raise ValueError("Native oracle process did not exit successfully")
    tokens = value.get("tokens")
    prompts = value.get("prompt_frontiers")
    generated = value.get("generated_frontiers")
    work = value.get("work")
    seconds = value.get("seconds")
    command = value.get("command")
    if (type(tokens) is not list or not 0 < len(tokens) <= generated_limit
            or any(type(token) is not int for token in tokens)
            or type(prompts) is not list or len(prompts) != requested_frontiers
            or any(type(item) is not bytes for item in prompts)
            or type(generated) is not list or len(generated) != len(tokens)
            or any(type(item) is not bytes for item in generated)
            or type(work) is not dict or type(seconds) not in (int, float)
            or type(seconds) is bool or seconds < 0
            or type(command) is not list or not command
            or any(not isinstance(item, str) for item in command)):
        raise ValueError("Native oracle return shape or frontier schedule is incomplete")
    return {"pid": pid, "exit_code": exit_code, "tokens": tokens,
            "prompt_frontiers": prompts, "generated_frontiers": generated,
            "work": work, "seconds": seconds, "command": command}


def require_matching_worker_pins(producer: dict, consumer: dict) -> None:
    for key in ("profile_sha256", "profile_id", "model_contract_id", "model_identity",
                "coordinator_overlay_sha256", "api_src_sha256"):
        left, right = producer.get(key), consumer.get(key)
        if left in (None, "", {}) or right in (None, "", {}) or left != right:
            raise ValueError(f"Producer and consumer runtime pin differs: {key}")


def create_canonical_only_fork(coordinator, branch: str, snapshot: str):
    """Create a branch ref without invoking coordinator fork's checkpoint restore."""
    return coordinator.store.fork(branch, snapshot)


def exercise_evicted_checkpoint_branch_actions(coordinator, model, *, source_branch: str,
                                               source_snapshot: str, root_snapshot: str,
                                               answer_branch: str, source_head,
                                               source_receipt: dict) -> dict:
    """Exercise real coordinator fork/rewind without restoring numerical state."""
    store = coordinator.store
    if source_snapshot == root_snapshot:
        raise ValueError("Branch action probe requires a distinct source and root snapshot")
    if numerical_checkpoint_files(store.root):
        raise ValueError("Branch action probe requires all numerical checkpoints to be evicted")
    if store.head(source_branch) != source_head:
        raise AssertionError("Source branch moved before canonical-only branch actions")
    source_record = store.get_snapshot(source_snapshot)
    root_record = store.get_snapshot(root_snapshot)
    source_history = tuple(store.lineage(source_snapshot)[1])
    root_history = tuple(store.lineage(root_snapshot)[1])
    native_before = dict(model.backend.stats)
    with store._locked():
        receipt_before = store._read_refs()["receipts"].get("op-source")
    if receipt_before != source_receipt:
        raise AssertionError("Producer operation receipt changed before branch actions")

    fork = coordinator.fork(answer_branch, source_snapshot)
    fork_head = store.head(answer_branch)
    if (fork.tip != source_snapshot or fork_head.tip != source_snapshot
            or fork_head.revision != 0):
        raise AssertionError("Coordinator fork did not create the expected source branch head")
    if store.head(source_branch) != source_head:
        raise AssertionError("Coordinator fork moved the source branch")
    if dict(model.backend.stats) != native_before:
        raise AssertionError("Coordinator fork performed native work")

    root_move = coordinator.rewind(answer_branch, fork_head.revision,
                                   root_snapshot, "op-rewind-root")
    root_retry = coordinator.rewind(answer_branch, fork_head.revision,
                                    root_snapshot, "op-rewind-root")
    if root_retry != root_move or store.head(answer_branch) != root_move:
        raise AssertionError("Same-operation root rewind did not replay its original branch result")
    if root_move.tip != root_snapshot or root_move.revision != fork_head.revision + 1:
        raise AssertionError("Coordinator root rewind moved to an unexpected revision or snapshot")

    source_move = coordinator.rewind(answer_branch, root_move.revision,
                                     source_snapshot, "op-rewind-source")
    source_retry = coordinator.rewind(answer_branch, root_move.revision,
                                      source_snapshot, "op-rewind-source")
    if source_retry != source_move or store.head(answer_branch) != source_move:
        raise AssertionError("Same-operation source rewind did not replay its original branch result")
    if source_move.tip != source_snapshot or source_move.revision != root_move.revision + 1:
        raise AssertionError("Coordinator source rewind moved to an unexpected revision or snapshot")

    if store.head(source_branch) != source_head:
        raise AssertionError("Canonical branch action changed the original source head")
    if (store.get_snapshot(source_snapshot) != source_record
            or store.get_snapshot(root_snapshot) != root_record
            or tuple(store.lineage(source_snapshot)[1]) != source_history
            or tuple(store.lineage(root_snapshot)[1]) != root_history):
        raise AssertionError("Canonical branch action mutated source or root history")
    with store._locked():
        receipt_after = store._read_refs()["receipts"].get("op-source")
    if receipt_after != source_receipt:
        raise AssertionError("Canonical branch action changed the producer operation receipt")
    if dict(model.backend.stats) != native_before:
        raise AssertionError("Coordinator rewind performed native work")
    if numerical_checkpoint_files(store.root):
        raise AssertionError("Canonical branch actions recreated numerical checkpoints")
    return {
        "fork": {"branch": answer_branch, "snapshot": fork_head.tip,
                 "revision": fork_head.revision},
        "rewinds": [
            {"operation_id": "op-rewind-root", "expected_revision": fork_head.revision,
             "snapshot": root_move.tip, "revision": root_move.revision,
             "idempotent_retry_equal": True},
            {"operation_id": "op-rewind-source", "expected_revision": root_move.revision,
             "snapshot": source_move.tip, "revision": source_move.revision,
             "idempotent_retry_equal": True},
        ],
        "source_head_unchanged": True,
        "source_and_root_history_unchanged": True,
        "producer_receipt_unchanged": True,
        "native_stats_unchanged": True,
        "numerical_checkpoints_absent": True,
    }


class Progress:
    def __init__(self, output: Path):
        self.path = output / "progress.json"
        self.started = time.monotonic()
        self.phase = "starting"
        self.detail = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self):
        self.write()
        self.thread.start()

    def set(self, phase: str, **detail):
        with self.lock:
            self.phase, self.detail = phase, detail
            self._write_locked()

    def write(self):
        with self.lock:
            self._write_locked()

    def _write_locked(self):
        atomic_json(self.path, {"phase": self.phase, "elapsed_seconds": time.monotonic() - self.started,
                                "updated_unix": time.time(), **self.detail})

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.write()

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


def _install_runtime(api_src: Path, coordinator_overlay: Path):
    api_src = api_src.resolve(strict=True)
    overlay = coordinator_overlay.resolve(strict=True)
    if overlay.name != "coordinator.py":
        raise ValueError("Coordinator overlay must be a coordinator.py file")
    sys.path.insert(0, str(api_src))
    importlib.import_module("context_time_travel")
    original = importlib.import_module("context_time_travel.coordinator")
    spec = importlib.util.spec_from_file_location("context_time_travel.coordinator", overlay)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load candidate coordinator overlay")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["context_time_travel.coordinator"] = candidate
    spec.loader.exec_module(candidate)
    package = sys.modules["context_time_travel"]
    package.coordinator = candidate
    return original, candidate


def _wait_request(request, timeout=2400):
    receipt = request.durable.result(timeout=timeout)
    answer = request.snapshot.result(timeout=timeout)
    events = []
    while not request.events.empty():
        events.append(request.events.get())
    return receipt, answer, events


def wait_workers_idle(coordinator, timeout=120):
    deadline = time.monotonic() + timeout
    with coordinator._condition:
        while any(worker["active"] for worker in coordinator._workers.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Coordinator worker did not return to idle")
            coordinator._condition.wait(remaining)


def _runtime(args):
    _install_runtime(args.api_src, args.coordinator_overlay)
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig, PoolTransitionTrace
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _validate_execution, _validate_model
    from pool_campaign import get_state
    return ContextCoordinator, PoolConfig, PoolTransitionTrace, BranchingModel, StrataBackend, _validate_execution, _validate_model, get_state


def _worker(args, phase: str):
    (ContextCoordinator, PoolConfig, PoolTransitionTrace, BranchingModel, StrataBackend,
     validate_execution, validate_model, get_state) = _runtime(args)
    from context_time_travel.store import IntegrityError
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("execution", {}).get("GPU") != [args.gpu]:
        raise ValueError("Profile is not explicitly pinned to the selected GPU")
    validate_execution(profile)
    model_identity = validate_model(profile)
    env = profile["execution"].get("environment", {})
    for key in list(os.environ):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            os.environ.pop(key, None)
    os.environ.update(env)
    pool = PoolConfig(gpus=(args.gpu,), ram_bytes=args.ram_gib * 1024**3,
                      disk_bytes=args.disk_gib * 1024**3,
                      min_available_bytes=args.min_available_gib * 1024**3)
    trace = PoolTransitionTrace("missing-materialization-" + phase, max_events=4096)
    trace_path = args.output / (phase + "-pool-trace.json")
    coordinator = None
    record = {"phase": phase, "status": "RUNNING", "pid": os.getpid(),
              "probe_script": str(Path(__file__).resolve()),
              "probe_script_sha256": sha256_file(Path(__file__).resolve()),
              "api_src_sha256": None, "coordinator_overlay": str(args.coordinator_overlay.resolve()),
              "coordinator_overlay_sha256": sha256_file(args.coordinator_overlay),
              "profile_sha256": sha256_file(args.profile),
              "model_identity": model_identity, "profile_id": profile["execution_profile_id"],
              "model_contract_id": profile["model_contract_id"], "native_pids": {},
              "native_exit_codes": {}, "failures": []}
    api_files = sorted(args.api_src.rglob("*.py"))
    record["api_src_sha256"] = {path.relative_to(args.api_src).as_posix(): sha256_file(path)
                                 for path in api_files}
    native = {}
    started = time.monotonic()
    try:
        coordinator = ContextCoordinator(args.store, pool,
            lambda gpu: BranchingModel(args.store, StrataBackend(profile, gpu_id=gpu), profile),
            transition_trace=trace)
        coordinator.ready()
        model = next(worker["model"] for worker in coordinator._workers.values())
        native = {str(gpu): worker["model"].backend.process for gpu, worker in coordinator._workers.items()}
        record.update(native_pids={key: proc.pid for key, proc in native.items()},
                      startup_seconds=time.monotonic() - started)
        atomic_json(args.output / (phase + "-worker.json"), record)
        if phase == "producer":
            prefix = tuple(100 + i % 17 for i in range(args.prefix))
            suffix = tuple(117 + i % 13 for i in range(args.suffix))
            root_request = coordinator.create(args.root_branch, prefix, "op-root")
            root_receipt, root_answer, root_events = _wait_request(root_request)
            wait_workers_idle(coordinator)
            if root_answer.get("tokens") != [] or root_receipt.get("snapshot") != root_answer.get("snapshot"):
                raise AssertionError("Producer root was not durably acknowledged")
            if root_answer.get("counter_deltas", {}).get("forward_tokens") != len(prefix):
                raise AssertionError("Producer root did not consume exactly its prefix")
            before = dict(model.backend.stats)
            source_request = coordinator.submit(args.root_branch, suffix, args.generated, "op-source")
            source_receipt, source_answer, source_events = _wait_request(source_request)
            wait_workers_idle(coordinator)
            if not source_answer.get("tokens") or source_receipt.get("snapshot") != source_answer.get("snapshot"):
                raise AssertionError("Producer source request did not produce a durable output")
            if source_answer.get("counter_deltas", {}).get("forward_tokens") != len(suffix) + len(source_answer["tokens"]):
                raise AssertionError("Producer suffix/output work is not exact")
            head = coordinator.store.head(args.root_branch)
            snapshot = head.tip
            state = get_state(coordinator, snapshot)
            source_manifest = (model.checkpoint_directory / (snapshot + ".json")).read_bytes()
            (args.output / "producer-source-manifest.json").write_bytes(source_manifest)
            source_history = list(state["history"])
            if source_history != list(prefix + suffix + tuple(source_answer["tokens"])):
                raise AssertionError("Producer canonical history does not bind prefix, suffix, and output")
            record.update(status="PASS", root_snapshot=root_receipt["snapshot"],
                          root_branch=args.root_branch, source_snapshot=snapshot,
                          source_branch=args.root_branch, source_revision=head.revision,
                          source_history=source_history, source_history_tokens=len(source_history),
                          source_controller=state["controller"], source_frontier_sha256=state["frontier_sha256"],
                          source_frontier_file=state["frontier_file"], source_payload_file=state["payload_file"],
                          source_payload_sha256=state["manifest"]["payload_sha256"],
                          source_manifest_sha256=sha256_file(model.checkpoint_directory/(snapshot+".json")),
                          source_manifest_copy_sha256=hashlib.sha256(source_manifest).hexdigest(),
                          source_operation_receipt=source_receipt,
                          source_request_answer=source_answer,
                          source_request_events=source_events,
                          root_request_events=root_events,
                          native_counter_delta={key: model.backend.stats.get(key, 0)-value
                              for key, value in before.items() if type(value) is int and type(model.backend.stats.get(key)) is int})
        else:
            snapshot = args.source_snapshot
            original_head = coordinator.store.head(args.source_branch)
            if original_head.tip != snapshot:
                raise AssertionError("Consumer source branch head changed before rebuild")
            history = tuple(model.history(snapshot))
            controller_before = model.controller(snapshot)
            with coordinator.store._locked():
                receipt_before = coordinator.store._read_refs()["receipts"].get("op-source")
            if receipt_before is None:
                raise AssertionError("Producer operation receipt is missing")
            before = dict(model.backend.stats)
            source_key = coordinator._key(model, snapshot)
            if snapshot in model.cache or source_key in coordinator.pool:
                raise AssertionError("Fresh consumer unexpectedly has a cached source materialization")
            # Strict restore must reject absent checkpoint metadata before issuing native work.
            try:
                model.restore_checkpoint(snapshot)
            except BaseException as exc:
                if not is_missing_checkpoint_error(exc):
                    raise AssertionError("Strict missing-checkpoint restore failed for another reason") from exc
                record["strict_restore_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not record.get("strict_restore_error"):
                raise AssertionError("Strict restore unexpectedly accepted an evicted checkpoint")
            after_strict = dict(model.backend.stats)
            record["strict_missing_checkpoint_control"] = {
                "native_stats_before": before,
                "native_stats_after": after_strict,
                "native_stats_delta": native_stats_delta(before, after_strict),
                "native_stats_unchanged": before == after_strict,
                "rejected": True,
            }
            require_no_native_work(before, after_strict, "Strict missing-checkpoint rejection")
            # Negative control: restore only the original valid manifest while
            # its numerical payload remains absent. Normal routing must reject
            # this incomplete checkpoint rather than rebuilding it.
            manifest_copy = args.output / "producer-source-manifest.json"
            manifest_bytes = manifest_copy.read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != args.source_manifest_sha256:
                raise AssertionError("Source manifest copy differs from producer receipt")
            manifest_path = model.checkpoint_directory / (snapshot + ".json")
            payload_sha = json.loads(manifest_bytes)["manifest"]["payload_sha256"]
            if manifest_path.exists() or (model.checkpoint_directory / (payload_sha + ".state")).exists():
                raise AssertionError("Expected source manifest and payload to be absent before negative control")
            manifest_path.write_bytes(manifest_bytes)
            payload_path = model.checkpoint_directory / (payload_sha + ".state")
            if (not manifest_path.is_file()
                    or sha256_file(manifest_path) != args.source_manifest_sha256
                    or payload_path.exists()):
                raise AssertionError("Negative control did not restore exact producer manifest with payload absent")
            create_canonical_only_fork(coordinator, "strict-control", snapshot)
            control_head_before = coordinator.store.head("strict-control")
            control_before = dict(model.backend.stats)
            control = coordinator.submit("strict-control",
                tuple(150 + i % 11 for i in range(args.suffix)), args.generated, "op-strict-control")
            try:
                control.durable.result(timeout=120)
            except BaseException as exc:
                if not is_detached_incomplete_checkpoint_error(exc, IntegrityError):
                    raise AssertionError("Incomplete checkpoint failed for an unrelated reason") from exc
                record["present_manifest_missing_payload_error"] = {
                    "type": type(exc).__name__, "message": str(exc)}
            if not record.get("present_manifest_missing_payload_error"):
                raise AssertionError("Present manifest with missing payload was not rejected")
            wait_workers_idle(coordinator)
            control_after = dict(model.backend.stats)
            record["present_manifest_missing_payload_control"] = {
                "manifest_sha256": sha256_file(manifest_path),
                "payload_sha256": payload_sha, "payload_absent": not payload_path.exists(),
                "native_stats_before": control_before,
                "native_stats_after": control_after,
                "native_stats_delta": native_stats_delta(control_before, control_after),
                "native_stats_unchanged": control_before == control_after,
            }
            require_no_native_work(control_before, control_after,
                                   "Present-manifest/missing-payload rejection")
            control_head_after = coordinator.store.head("strict-control")
            source_head_after = coordinator.store.head(args.source_branch)
            if (control_head_after.tip != control_head_before.tip
                    or control_head_after.revision != control_head_before.revision
                    or source_head_after.tip != snapshot):
                raise AssertionError("Incomplete checkpoint rejection moved a canonical branch head")
            if (not manifest_path.is_file()
                    or sha256_file(manifest_path) != args.source_manifest_sha256
                    or payload_path.exists()):
                raise AssertionError("Incomplete checkpoint control changed the source manifest/payload state")
            record["present_manifest_missing_payload_control"].update(
                branch_tip_unchanged=True)
            manifest_path.unlink()
            source_frontier = {}
            original_rebuild = model.rebuild_materialization
            def observe_rebuild(source):
                result = original_rebuild(source)
                if source != snapshot:
                    raise AssertionError("Rebuild observer saw a different source")
                image = model.cache.get(source)
                if image is None:
                    raise AssertionError("Rebuild did not leave the source image for observation")
                source_frontier["values"] = tuple(image.frontier)
                source_frontier["thread_id"] = threading.get_ident()
                source_frontier["receipt"] = result
                source_frontier["schedule_sha256"] = result.get("schedule_sha256")
                source_frontier["cache_schedule_sha256"] = __import__(
                    "context_time_travel.execution_receipt", fromlist=["schedule_digest"]
                    ).schedule_digest(image.execution_schedule)
                return result
            model.rebuild_materialization = observe_rebuild
            # The numerical checkpoint is intentionally absent. Exercise the
            # production coordinator branch actions before submitting work;
            # only the explicit incomplete-checkpoint control above uses the
            # canonical Store directly.
            branch_actions = exercise_evicted_checkpoint_branch_actions(
                coordinator, model, source_branch=args.source_branch,
                source_snapshot=snapshot, root_snapshot=args.root_snapshot,
                answer_branch=args.answer_branch, source_head=original_head,
                source_receipt=receipt_before)
            suffix = tuple(150 + i % 11 for i in range(args.suffix))
            request = coordinator.submit(args.answer_branch, suffix, args.generated, "op-rebuild-answer")
            answer_receipt, answer, events = _wait_request(request)
            wait_workers_idle(coordinator)
            final_snapshot = answer_receipt.get("snapshot")
            if not final_snapshot or answer.get("snapshot") != final_snapshot:
                raise AssertionError("Consumer answer lacks one durable canonical snapshot")
            state = get_state(coordinator, final_snapshot)
            if state["history"] != list(history + suffix + tuple(answer.get("tokens", ()))):
                raise AssertionError("Final canonical history differs from source + suffix + actual outputs")
            reconstruction = verify_rebuild_evidence(answer.get("source_recomputation"),
                snapshot=snapshot, history_tokens=len(history),
                model_contract_id=profile["model_contract_id"],
                execution_profile_id=profile["execution_profile_id"])
            observed_rebuild = verify_observed_rebuild_receipt(source_frontier["receipt"],
                reconstruction, snapshot=snapshot, history_tokens=len(history),
                model_contract_id=profile["model_contract_id"],
                execution_profile_id=profile["execution_profile_id"], controller=coordinator.store.get_snapshot(snapshot)["controller"],
                expected_schedule_sha256=source_frontier["cache_schedule_sha256"])
            prefix_delta = reconstruction["prefix_forward_tokens"]
            answer_delta = answer.get("counter_deltas", {}).get("forward_tokens")
            expected_suffix_work = len(suffix) + len(answer.get("tokens", ()))
            if type(answer_delta) is not int or answer_delta != expected_suffix_work:
                raise AssertionError("Answer suffix/output work is not separately exact")
            after_answer = dict(model.backend.stats)
            total_forward = after_answer.get("forward_tokens", 0) - before.get("forward_tokens", 0)
            work_split = validate_work_split(history_tokens=len(history), suffix_tokens=len(suffix),
                output_tokens=len(answer.get("tokens", ())),
                prefix_forward=reconstruction["prefix_forward_tokens"],
                prefix_consumed=reconstruction["prefix_consumed_tokens"],
                request_forward=answer_delta, total_forward_delta=total_forward)
            if not source_frontier:
                raise AssertionError("No rebuilt source frontier was observed")
            if not any(row.get("event") == "source_recomputed" and row.get("source_snapshot") == snapshot
                       and row.get("prefix_forward_tokens") == len(history) for row in events):
                raise AssertionError("Coordinator omitted the source_recomputed event")
            if coordinator.store.head(args.answer_branch).tip != final_snapshot:
                raise AssertionError("Answer branch did not move to the acknowledged snapshot")
            if coordinator.store.head(args.source_branch) != original_head:
                raise AssertionError("Source branch changed while rebuilding its child")
            with coordinator.store._locked():
                receipt_after = coordinator.store._read_refs()["receipts"].get("op-source")
            if receipt_after != receipt_before:
                raise AssertionError("Historical producer operation receipt changed")
            if model.controller(snapshot) != controller_before or model.history(snapshot) != history:
                raise AssertionError("Canonical source controller/history mutated")
            frontier_file = args.output / "rebuilt-source-frontier.f32"
            frontier_bytes = _frontier_bytes(source_frontier["values"])
            frontier_file.write_bytes(frontier_bytes)
            record.update(status="PASS", source_snapshot=snapshot, source_branch=args.source_branch,
                answer_branch=args.answer_branch, source_history=list(history),
                source_history_tokens=len(history), source_controller=controller_before,
                strict_restore_forward_delta=0, source_recomputation=reconstruction,
                observed_rebuild_receipt=observed_rebuild,
                present_manifest_missing_payload_rejected=True,
                rebuilt_source_frontier_file=str(frontier_file),
                rebuilt_source_frontier_sha256=hashlib.sha256(frontier_bytes).hexdigest(),
                rebuild_observer_thread_id=source_frontier["thread_id"],
                answer_operation_receipt=answer_receipt, answer=answer,
                answer_events=events, answer_snapshot=final_snapshot,
                final_history=state["history"], final_controller=state["controller"],
                final_frontier_sha256=state["frontier_sha256"], final_frontier_file=state["frontier_file"],
                final_payload_sha256=state["manifest"]["payload_sha256"],
                prefix_forward_tokens=prefix_delta, suffix_forward_tokens=expected_suffix_work,
                work_split=work_split,
                source_operation_receipt_unchanged=True, source_snapshot_unchanged=True,
                branch_actions=branch_actions,
                branch_setup_scope="coordinator fork/rewind with all numerical checkpoint files absent")
        record["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / (phase + "-worker.json"), record)
        return 0
    except BaseException as exc:
        record.update(status="FAIL", error=str(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        record["failures"].append(str(exc))
        atomic_json(args.output / (phase + "-worker.json"), record)
        raise
    finally:
        if coordinator is not None:
            close_error = None
            try:
                coordinator.close()
            except BaseException as exc:
                close_error = str(exc)
            record["native_exit_codes"] = {key: proc.poll() for key, proc in native.items()}
            record["native_exit_success"] = bool(native) and all(type(code) is int and code == 0
                                                                  for code in record["native_exit_codes"].values())
            if close_error:
                record["cleanup_error"] = close_error
                record["status"] = "FAIL"
            if record["native_exit_success"] is not True:
                record["status"] = "FAIL"
            try:
                trace_snapshot = trace.snapshot()
                atomic_json(trace_path, trace_snapshot)
                record["pool_trace_file"] = str(trace_path)
                record["pool_trace_overflowed"] = trace_snapshot["overflowed"]
                if trace_snapshot["overflowed"]:
                    record["status"] = "FAIL"
                atomic_json(args.output / (phase + "-worker.json"), record)
            except BaseException as exc:
                record["status"] = "FAIL"
                record["trace_capture_error"] = str(exc)
                try:
                    atomic_json(args.output / (phase + "-worker.json"), record)
                except BaseException:
                    pass


def _frontier_bytes(values):
    import struct
    return struct.pack("<" + "f" * len(values), *values)


def _run_native_reference(profile, output: Path, name: str, ids, requests, generated: int) -> dict:
    """Run the pinned independent native oracle and retain its real process receipt."""
    output.mkdir(parents=True, exist_ok=True)
    tokens_path = output / (name + ".tokens")
    target = output / (name + ".f32")
    tokens_path.write_text(",".join(map(str, ids)), encoding="ascii")
    command = list(profile["execution"]["oracle_command"])
    command[command.index("--max-new") + 1] = str(generated)
    command += ["--tokens-file", str(tokens_path)]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
            env.pop(key, None)
    env.update(profile["execution"].get("environment", {}))
    env.update(STRATA_DUMP_FIRST_LOGITS=str(target),
               STRATA_PREFILL_REQUESTS=",".join(map(str, requests)))
    started = time.monotonic()
    stdout_path, stderr_path = output / (name + ".stdout"), output / (name + ".stderr")
    process_path = output / (name + ".process.json")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        pid = child.pid
        atomic_json(process_path, {"pid": pid, "command": command, "status": "RUNNING"})
        try:
            exit_code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            atomic_json(process_path, {"pid": pid, "command": command,
                                       "status": "TIMEOUT", "exit_code": child.returncode})
            raise TimeoutError(f"Independent native oracle timed out; PID {pid}")
    atomic_json(process_path, {"pid": pid, "command": command, "status": "EXITED",
                               "exit_code": exit_code})
    work = json.loads(stdout_path.read_text(encoding="utf-8"))
    schedule = json.loads(Path(str(target) + ".schedule.json").read_text(encoding="utf-8"))
    outputs = work.get("generated")
    if (exit_code != 0 or type(outputs) is not list or not outputs
            or schedule.get("requests") != requests or work.get("prefill_tokens") != len(ids)
            or work.get("normal_forward_tokens") != len(outputs)
            or work.get("consumed") != len(ids) + len(outputs)
            or len(outputs) > generated):
        raise AssertionError("Independent native reference failed its exact work/schedule checks")
    width = 4 * 248320
    raw = Path(str(target) + ".decode.f32").read_bytes()
    prompt_raw = Path(str(target) + ".requests.f32").read_bytes()
    if len(raw) != width * len(outputs) or len(prompt_raw) != width * len(requests):
        raise AssertionError("Independent native reference has incomplete frontier evidence")
    value = {"tokens": outputs,
             "generated_frontiers": [raw[index:index + width] for index in range(0, len(raw), width)],
             "prompt_frontiers": [prompt_raw[index:index + width]
                                  for index in range(0, len(prompt_raw), width)],
             "work": work, "seconds": time.monotonic() - started, "command": command,
             "schedule": schedule, "process_file": str(process_path),
             "stdout_file": str(stdout_path), "stderr_file": str(stderr_path),
             "schedule_file": str(Path(str(target) + ".schedule.json")),
             "frontier_file": str(Path(str(target) + ".decode.f32")),
             "prompt_frontier_file": str(Path(str(target) + ".requests.f32"))}
    result = bind_oracle_result(value, pid=pid, exit_code=exit_code,
                                requested_frontiers=len(requests), generated_limit=generated)
    result["schedule"] = schedule
    result["process_file"] = value["process_file"]
    result["stdout_file"] = value["stdout_file"]
    result["stderr_file"] = value["stderr_file"]
    result["schedule_file"] = value["schedule_file"]
    result["frontier_file"] = value["frontier_file"]
    result["prompt_frontier_file"] = value["prompt_frontier_file"]
    result["stdout_sha256"] = sha256_file(stdout_path)
    result["stderr_sha256"] = sha256_file(stderr_path)
    result["schedule_sha256"] = sha256_file(Path(str(target) + ".schedule.json"))
    return result


def _stop_child(child: subprocess.Popen, grace_seconds: float = 10):
    """Stop only the owned fresh session, including its native descendants."""
    if os.name == "posix":
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        # A leader may exit on TERM while a child ignores it; always kill the
        # session group after the bounded grace period, not just the leader.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Owned child process group {child.pid} did not terminate")
    elif child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def _run_child(args, phase, phase_args, output, progress):
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
               "--profile", str(args.profile), "--api-src", str(args.api_src),
               "--coordinator-overlay", str(args.coordinator_overlay),
               "--store", str(args.store), "--output", str(args.output),
               "--gpu", str(args.gpu), "--ram-gib", str(args.ram_gib),
               "--disk-gib", str(args.disk_gib), "--min-available-gib", str(args.min_available_gib),
               "--prefix", str(args.prefix), "--suffix", str(args.suffix),
               "--generated", str(args.generated)]
    if phase_args.get("source_manifest_sha256"):
        command += ["--source-manifest-sha256", str(phase_args["source_manifest_sha256"])]
    for key, value in phase_args.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    with (output / f"{phase}.stdout").open("wb") as stdout, (output / f"{phase}.stderr").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            atomic_json(output / f"{phase}.process.json", {"pid": child.pid,
                "process_group_id": child.pid, "new_session": True,
                "command": command, "status": "RUNNING"})
            progress.set(phase + "-running", child_pid=child.pid, last_completed_step="child-started")
            deadline = time.monotonic() + 7200
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{phase} child timed out; PID {child.pid}")
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                observed = {}
                worker_path = output / f"{phase}-worker.json"
                if worker_path.exists():
                    try:
                        worker = json.loads(worker_path.read_text(encoding="utf-8"))
                        observed = {"native_pids": worker.get("native_pids", {}),
                                    "native_exit_codes": worker.get("native_exit_codes", {})}
                    except (OSError, ValueError):
                        observed = {"worker_progress_parse_error": True}
                progress.set(phase + "-running", child_pid=child.pid,
                             last_completed_step="waiting-for-child", **observed)
            code = wait_child_exit(child)
            atomic_json(output / f"{phase}.process.json", {"pid": child.pid,
                "process_group_id": child.pid, "new_session": True, "command": command,
                "status": "EXITED", "exit_code": code})
            if code != 0:
                raise RuntimeError(f"{phase} child failed with exit {code}; see {phase}.stderr")
            result_path = output / f"{phase}-worker.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "PASS" or result.get("native_exit_success") is not True:
                raise RuntimeError(f"{phase} worker did not pass: {result.get('error')}")
            return result
        except BaseException as exc:
            cleanup_error = None
            try:
                _stop_child(child)
            except BaseException as cleanup_exc:
                cleanup_error = str(cleanup_exc)
            try:
                atomic_json(output / f"{phase}.process.json", {"pid": child.pid,
                    "process_group_id": child.pid, "new_session": True, "command": command,
                    "status": "FAILED", "exit_code": child.poll(), "error": str(exc),
                    "cleanup_error": cleanup_error})
            except BaseException:
                pass
            raise


def run(args):
    if args.store.absolute() != (args.output.absolute() / "store"):
        raise ValueError("The Store must be exactly OUTPUT/store so eviction scope is harness-owned")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.store.exists():
        raise FileExistsError(f"Refusing to use an existing Store: {args.store}")
    args.store.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output)
    result = {"schema": "missing-materialization-probe-v1", "status": "RUNNING",
              "scope": "real native GPU probe; canonical Store retained; all numerical checkpoint files evicted",
              "fixture_dimensions": {"prefix_tokens": args.prefix,
                                     "suffix_tokens": args.suffix,
                                     "generated_token_limit": args.generated},
              "profile": str(args.profile), "api_src": str(args.api_src),
              "coordinator_overlay": str(args.coordinator_overlay), "store": str(args.store),
              "probe_script": str(Path(__file__).resolve()),
              "probe_script_sha256": sha256_file(Path(__file__).resolve()),
              "phases": [], "failures": []}
    atomic_json(args.output / "result.json", result)
    progress.start()
    try:
        progress.set("producer", store=str(args.store))
        producer = _run_child(args, "producer", {"root_branch": "source-A"}, args.output, progress)
        result["phases"].append({"phase": "producer", "worker": producer})
        result["source_snapshot"] = producer["source_snapshot"]
        progress.set("producer-complete", source_snapshot=producer["source_snapshot"])
        if producer.get("native_exit_success") is not True:
            raise RuntimeError("Producer did not report successful native shutdown")
        expected = {producer["root_snapshot"], producer["source_snapshot"]}
        inventory = plan_checkpoint_eviction(args.store, expected)
        _install_runtime(args.api_src, args.coordinator_overlay)
        from context_time_travel.pools import DiskCheckpointPool
        from context_time_travel.store import Store
        eviction_store = Store(args.store)
        evicted = evict_via_disk_pool(eviction_store, DiskCheckpointPool, expected, inventory)
        result["eviction"] = evicted
        progress.set("evicted", snapshots=sorted(expected), removed_files=evicted["inventory_file_count"],
                     removed_bytes=evicted["inventory_bytes"],
                     files_evicted=evicted["metrics_during_reservation"]["files_evicted"])
        consumer = _run_child(args, "consumer", {
            "source_branch": producer["source_branch"], "source_snapshot": producer["source_snapshot"],
            "root_snapshot": producer["root_snapshot"],
            "answer_branch": "answer-B",
            "source_manifest_sha256": producer["source_manifest_copy_sha256"]}, args.output, progress)
        require_matching_worker_pins(producer, consumer)
        result["phases"].append({"phase": "consumer", "worker": consumer})
        progress.set("consumer-complete", answer_snapshot=consumer["answer_snapshot"])
        progress.set("independent-native-oracle")
        _install_runtime(args.api_src, args.coordinator_overlay)
        scripts = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts))
        from context_time_travel import controller
        from tools.strata_tokenizer import Tokenizer
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        for key in list(os.environ):
            if key.startswith(("STRATA_", "CUDA_", "NVIDIA_", "GGML_")):
                os.environ.pop(key, None)
        os.environ.update(profile["execution"].get("environment", {}))
        source_history = tuple(producer["source_history"])
        suffix = tuple(150 + i % 11 for i in range(args.suffix))
        oracle = _run_native_reference(profile, args.output / "oracle", "fresh-rebuild-arrangement",
                                       source_history + suffix,
                                       [len(source_history), len(suffix)], args.generated)
        rebuilt_source = Path(consumer["rebuilt_source_frontier_file"]).read_bytes()
        final_frontier = Path(consumer["final_frontier_file"]).read_bytes()
        if rebuilt_source != oracle["prompt_frontiers"][0]:
            raise AssertionError("Rebuilt source frontier differs from independent full-history prefill")
        if final_frontier != oracle["generated_frontiers"][-1]:
            raise AssertionError("Final frontier differs from independent full-history plus suffix oracle")
        if consumer["answer"].get("tokens") != oracle["tokens"]:
            raise AssertionError("Rebuilt request output IDs differ from independent native oracle")
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        eos = profile["execution"]["eos_tokens"]
        expected_controller = controller.start(dict(temperature=0, eos_tokens=eos),
            "op-rebuild-answer", source_history + suffix, len(tokenizer.tokens))
        expected_controller = controller.consume_many(expected_controller,
            ((token, tokenizer.token_bytes(token), token in eos) for token in oracle["tokens"]))
        if consumer["final_controller"] != expected_controller:
            raise AssertionError("Final greedy controller differs from fresh native oracle history")
        result.update(status="PASS", source_history_tokens=producer["source_history_tokens"],
            source_snapshot=producer["source_snapshot"], final_snapshot=consumer["answer_snapshot"],
            prefix_reconstruction=consumer["source_recomputation"],
            suffix_forward_tokens=consumer["suffix_forward_tokens"],
            output_tokens=len(consumer["answer"].get("tokens", [])),
            oracle={"name": "fresh-rebuild-arrangement", "pid": oracle["pid"],
                    "exit_code": oracle["exit_code"], "tokens": oracle["tokens"],
                    "schedule": oracle["schedule"], "work": oracle["work"],
                    "command": oracle["command"], "seconds": oracle["seconds"],
                    "process_file": oracle["process_file"],
                    "stdout_file": oracle["stdout_file"], "stderr_file": oracle["stderr_file"],
                    "stdout_sha256": oracle["stdout_sha256"],
                    "stderr_sha256": oracle["stderr_sha256"],
                    "schedule_file": oracle["schedule_file"],
                    "schedule_sha256": oracle["schedule_sha256"],
                    "frontier_file": oracle["frontier_file"],
                    "prompt_frontier_file": oracle["prompt_frontier_file"],
                    "source_frontier_exact": True, "final_frontier_exact": True,
                    "controller_exact": True},
            native_exit_codes={phase: phase_record["worker"]["native_exit_codes"]
                for phase_record in result["phases"] for phase in [phase_record["phase"]]},
            oracle_exit_code=oracle["exit_code"])
        atomic_json(args.output / "result.json", result)
        progress.set("complete", status="PASS")
        return result
    except BaseException as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        result["failures"].append(str(exc))
        atomic_json(args.output / "result.json", result)
        progress.set("failed", status="FAIL", error=str(exc))
        raise
    finally:
        progress.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--api-src", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--coordinator-overlay", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("producer", "consumer"))
    parser.add_argument("--root-branch")
    parser.add_argument("--source-branch")
    parser.add_argument("--source-snapshot")
    parser.add_argument("--root-snapshot")
    parser.add_argument("--answer-branch")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ram-gib", type=int, default=96)
    parser.add_argument("--disk-gib", type=int, default=128)
    parser.add_argument("--min-available-gib", type=int, default=16)
    parser.add_argument("--prefix", type=positive_int, default=128,
                        help="producer prefix length in tokens (default: 128)")
    parser.add_argument("--suffix", type=positive_int, default=4,
                        help="producer/consumer suffix length in tokens (default: 4)")
    parser.add_argument("--generated", type=positive_int, default=8,
                        help="maximum generated tokens per request (default: 8)")
    args = parser.parse_args(argv)
    try:
        if args.phase:
            return _worker(args, args.phase)
        run(args)
        return 0
    except BaseException as exc:
        print(f"missing materialization probe failed: {exc}", file=sys.stderr, flush=True)
        return 1


def positive_int(value):
    """argparse converter for positive fixture dimensions with no arbitrary cap."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
