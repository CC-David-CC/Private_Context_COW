"""Resident CUDA fixture adapter for the canonical context API.

This backend is a model-free data-movement fixture. It is not a language model
and cannot qualify real-model inference. Its child process uses a private
ASCII command pipe and typed JSON replies; it never opens a network socket.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import queue
from pathlib import Path
import selectors
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import weakref

from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation


VOCAB_SIZE = 64
ROW_BYTES = 32
RECURRENCE_MULTIPLIER = 257
LOGIT_MULTIPLIER = 197
DEFAULT_SEED = 0x4D465F434F575F31
_COUNTER_FIELDS = (
    "forward_tokens", "captures", "restores", "serialized_bytes",
    "loaded_bytes", "allocations", "releases", "live_bytes", "cow_pages",
    "copied_bytes", "mappings", "unmappings",
)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_gpus(gpus):
    if not isinstance(gpus, (list, tuple)) or not gpus:
        raise ValueError("GPU selection must be a nonempty sequence")
    workers, physical = [], []
    for worker in gpus:
        group = (worker,) if type(worker) is int else tuple(worker) if isinstance(worker, (tuple, list)) else ()
        if not group or any(type(device) is not int or device < 0 for device in group):
            raise ValueError("GPU IDs must be nonnegative integers")
        if len(set(group)) != len(group):
            raise ValueError("GPU groups cannot contain duplicates")
        workers.append(group[0] if len(group) == 1 else group)
        physical.extend(group)
    if len(set(physical)) != len(physical):
        raise ValueError("GPU worker selections cannot overlap")
    return tuple(workers)


def make_profile(binary, gpus, capacity, *, seed=DEFAULT_SEED, hardware_identity):
    """Build a pinned synthetic-fixture profile for a compiled native binary."""
    path = Path(binary).resolve(strict=True)
    selection = _normalize_gpus(gpus)
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if type(seed) is not int or not 0 <= seed < (1 << 64):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(hardware_identity, str) or not hardware_identity.strip():
        raise ValueError("hardware_identity must be explicit")
    weights = hashlib.sha256(b"model-free fixture; no model weights").hexdigest()
    model_contract = dict(schema=1, name="model-free-cuda-cow-fixture-v1",
                          architecture="integer-recurrence-fixture", vocab_size=VOCAB_SIZE,
                          tokenizer="synthetic-token-ids-0-through-63", weights_sha256=weights,
                          seed=seed, row_bytes=ROW_BYTES)
    execution = dict(
        GPU=[list(item) if isinstance(item, tuple) else item for item in selection],
        capacity=capacity,
        seed=seed,
        binary=str(path),
        binary_sha256=_sha_file(path),
        hardware_identity=hardware_identity,
        resume_contract="model-free-cuda-fixture-v1",
        checkpoint_cache=dict(byte_limit=2 * 1024**3, entry_limit=256),
    )
    model_id = _digest(model_contract)
    profile_identity = _digest(dict(model_contract_id=model_id, execution=execution))
    return dict(backend="model-free-cuda", model_contract_id=model_id,
                model_weights_sha256=weights, model_contract=model_contract,
                execution_profile_id=profile_identity, execution=execution,
                hardware=hardware_identity)


def recurrence_final(seed, token_ids):
    """Independent CPU recurrence for one prefix, returning the final uint16."""
    state = seed & 0xFFFF
    for token in token_ids:
        if type(token) is not int or not 0 <= token < VOCAB_SIZE:
            raise ValueError("token outside synthetic vocabulary")
        state = (state * RECURRENCE_MULTIPLIER + token + 1) & 0xFFFF
    return state


def cpu_frontier(seed, token_ids):
    """Independent full-vocabulary frontier; values are exact float integers."""
    state = recurrence_final(seed, token_ids)
    return tuple(float((state ^ (token * LOGIT_MULTIPLIER)) & 0xFFFF)
                 for token in range(VOCAB_SIZE))


def cpu_stage_rows(seed, token_ids, stage):
    """Return the expected 32-byte logical row for each consumed input."""
    try:
        from . import fixture_reference
    except ImportError:
        import fixture_reference
    rows = []
    for position, token in enumerate(token_ids):
        rows.append(bytes(fixture_reference.byte_at(
            seed, fixture_reference.REGION_CANONICAL, stage, token, byte % ROW_BYTES,
            position * ROW_BYTES + byte) for byte in range(ROW_BYTES)))
    return tuple(rows)


@dataclass(frozen=True)
class HotBlob:
    """A process-local immutable native capture, not canonical history."""
    image: object
    identity: bytes
    process_identity: bytes
    consumed: int
    estimated_bytes: int

    def __len__(self):
        return self.estimated_bytes


class _Lease:
    def __init__(self, backend):
        self.backend = backend
        self.bound = False
        self.consumed = 0
        self.history = ()
        self.image_identity = None
        self.state = "NEW"
        self.owner_thread = None

    def __enter__(self):
        backend = self.backend
        backend._lease_lock.acquire()
        try:
            if backend._active is not None or backend._closed:
                raise InvalidOperation("Native fixture lease is unavailable")
            backend._active = self
            self.owner_thread = threading.get_ident()
            self.state = "ACTIVE"
            response = backend._request("reset")
            if response.get("consumed") != 0:
                raise IntegrityError("Native fixture reset retained consumed inputs")
            return self
        except BaseException:
            backend._active = None
            backend._lease_lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        self.backend._check(self)
        self.state = "CLOSED"
        self.backend._active = None
        self.backend._lease_lock.release()
        return False


def _finalize_image(backend_ref, image):
    backend = backend_ref()
    if backend is not None:
        backend._drops.put(image)


class ModelFreeBackend:
    """Adapter implementing the numerical backend surface used by BranchingModel."""
    def __init__(self, profile, *, gpu_id):
        self.profile = profile
        self.gpu_id = gpu_id
        self.vocab_size = VOCAB_SIZE
        self._process_identity = uuid.uuid4().bytes
        self._io_lock = threading.RLock()
        self._lease_lock = threading.RLock()
        self._active = None
        self._closed = False
        self._failed = False
        self._drops = queue.SimpleQueue()
        self._known_images = {}
        self._sequence = 0
        self.last_execution_receipt = None
        self._temp = tempfile.TemporaryDirectory(prefix="mf-cow-")
        self._directory = Path(self._temp.name).resolve()
        if " " in str(self._directory):
            self._temp.cleanup()
            raise InvalidOperation("Native serialize/import paths cannot contain spaces")
        selected = (gpu_id,) if type(gpu_id) is int else tuple(gpu_id)
        if not selected or any(type(x) is not int or x < 0 for x in selected):
            self._temp.cleanup()
            raise InvalidOperation("Invalid worker GPU selection")
        allowed = _normalize_gpus(profile["execution"]["GPU"])
        worker_selection = selected[0] if len(selected) == 1 else selected
        if worker_selection not in allowed:
            self._temp.cleanup()
            raise InvalidOperation("Worker GPU is outside the pinned fixture profile")
        command = [profile["execution"]["binary"], "--gpu", ",".join(map(str, selected)),
                   "--capacity", str(profile["execution"]["capacity"]),
                   "--seed", str(profile["execution"]["seed"])]
        if _sha_file(command[0]) != profile["execution"]["binary_sha256"]:
            self._temp.cleanup()
            raise IntegrityError("Native fixture binary changed after profile creation")
        self._log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, encoding="utf-8", bufsize=1)
        self._stats = dict(consumed_tokens=0, checkpoint_export_bytes=0,
            checkpoint_import_bytes=0, contexts_created=1, contexts_freed=0,
            lease_acquires=0, lease_releases=0)
        try:
            response = self._request("ready")
            gpu_reply = response.get("GPU")
            expected = list(selected)
            if (response.get("vocab") != VOCAB_SIZE
                    or response.get("capacity") != profile["execution"]["capacity"]
                    or gpu_reply != expected):
                raise IntegrityError("Native fixture geometry/profile differs from pinned worker")
        except BaseException:
            self.close()
            raise

    def _read_reply(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=18000):
                raise TimeoutError("Native fixture command timed out")
        line = self.process.stdout.readline()
        if not line:
            self._log.seek(0)
            message = self._log.read().decode(errors="replace")[-4000:]
            raise RuntimeError("Native fixture process closed: " + message)
        try:
            response = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise IntegrityError("Native fixture emitted invalid JSON") from error
        if not isinstance(response, dict) or type(response.get("ok")) is not bool:
            raise IntegrityError("Native fixture response lacks typed ok field")
        if not response["ok"]:
            message = response.get("error")
            if not isinstance(message, str) or not message:
                raise IntegrityError("Native fixture error response is malformed")
            raise IntegrityError("Native fixture rejected command: " + message)
        return response

    def _request(self, command):
        with self._io_lock:
            if self._failed:
                raise IntegrityError("Native fixture pipe failed; restart the worker")
            if self._closed or self.process.poll() is not None:
                raise IntegrityError("Native fixture process is unavailable")
            if not isinstance(command, str) or not command or "\n" in command or "\r" in command:
                raise ValueError("Fixture protocol commands must be one ASCII line")
            try:
                command.encode("ascii")
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
                return self._read_reply()
            except BaseException:
                self._failed = True
                raise

    def _check(self, lease):
        if (self._active is not lease or lease.backend is not self or lease.state != "ACTIVE"
                or lease.owner_thread != threading.get_ident()):
            raise InvalidOperation("Inactive or foreign native fixture lease")

    def metadata(self, key):
        values = {
            "general.architecture": "model-free-cuda-cow-fixture-v1",
            "model_free.vocab_size": str(VOCAB_SIZE),
            "model_free.row_bytes": str(ROW_BYTES),
        }
        return values.get(key)

    def contract_fields(self):
        return dict(attention="causal-integer-fixture",
                    state_kind="model-free-kv-rows-plus-u16-recurrence")

    def new_cache(self):
        from context_time_travel.checkpoint_cache import CheckpointCache
        policy = self.profile["execution"]["checkpoint_cache"]
        return CheckpointCache(policy["byte_limit"], policy["entry_limit"])

    def checkpoint_compatible(self,manifest,directory):
        """Inspect only immutable CFB1 header bytes, never a native lease.

        Full payload hashing still occurs before restore. The canonical store
        retains this source throughout queueing, so numerical GC cannot remove
        its file between routing and activation.
        """
        path=Path(directory)/(manifest['payload_sha256']+'.state')
        with path.open('rb') as stream:
            header=stream.read(32)
            if len(header)!=32 or header[:4]!=b'CFB1':
                raise IntegrityError('Missing fixture checkpoint layout')
            seed,capacity,parts,consumed=struct.unpack_from('<QQIQ',header,4)
            allowed=_normalize_gpus(self.profile['execution']['GPU'])
            maximum=sum(1 if type(x) is int else len(x) for x in allowed)
            if (seed!=self.profile['execution']['seed'] or capacity!=self.profile['execution']['capacity']
                    or not 0<parts<=maximum or consumed!=manifest['consumed']):
                raise IntegrityError('Invalid fixture checkpoint layout')
            data=stream.read(4*parts)
        if len(data)!=4*parts:raise IntegrityError('Truncated fixture placement')
        devices=struct.unpack('<'+'i'*parts,data)
        selected=(self.gpu_id,) if type(self.gpu_id) is int else tuple(self.gpu_id)
        return devices==selected

    @contextmanager
    def lease(self):
        lease = _Lease(self)
        try:
            yield lease.__enter__()
        except BaseException:
            if lease.state == "ACTIVE":
                lease.__exit__(None, None, None)
            raise
        else:
            if lease.state == "ACTIVE":
                lease.__exit__(None, None, None)

    def position(self, lease):
        self._check(lease)
        return lease.consumed - 1 if lease.bound else -1

    def _validate_tokens(self, ids, lease):
        ids = tuple(ids)
        if not ids or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
            raise InvalidOperation("Native fixture consume requires token IDs in [0, 64)")
        if lease.consumed + len(ids) > self.profile["execution"]["capacity"]:
            raise InvalidOperation("Native fixture capacity exceeded")
        return ids

    def consume(self, lease, ids):
        return self._consume(lease, ids)

    def consume_generated(self, lease, ids):
        if len(tuple(ids)) != 1:
            raise InvalidOperation("Fixture greedy decode consumes one token at a time")
        return self._consume(lease, ids)

    def _consume(self, lease, ids):
        self._check(lease)
        ids = self._validate_tokens(ids, lease)
        before = lease.consumed
        command = "consume " + str(len(ids)) + " " + " ".join(map(str, ids))
        response = self._request(command)
        if type(response.get("consumed")) is not int or response["consumed"] != before + len(ids):
            raise IntegrityError("Native fixture consumed count differs from request")
        lease.bound = True
        lease.consumed = response["consumed"]
        lease.history += ids
        lease.image_identity = None
        self._stats["consumed_tokens"] += len(ids)

    def logits(self, lease):
        self._check(lease)
        response = self._request("frontier")
        raw = response.get("logits")
        if (response.get("consumed") != lease.consumed or not isinstance(raw, list)
                or len(raw) != VOCAB_SIZE
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in raw)):
            raise IntegrityError("Native fixture returned an invalid full frontier")
        result = tuple(float(value) for value in raw)
        expected = cpu_frontier(self.profile["execution"]["seed"], lease.history)
        if result != expected:
            raise IntegrityError("Native fixture frontier differs from independent CPU recurrence")
        return result

    def is_eog(self, token):
        if type(token) is not int or not 0 <= token < VOCAB_SIZE:
            raise InvalidOperation("Invalid synthetic token")
        return False

    def piece(self, token):
        if type(token) is not int or not 0 <= token < VOCAB_SIZE:
            raise InvalidOperation("Invalid synthetic token")
        return str(token).encode("ascii") + b" "

    def export(self, lease):
        self._check(lease)
        if not lease.bound or lease.consumed <= 0:
            raise InvalidOperation("Cannot capture an empty fixture state")
        response = self._request("capture")
        image = response.get("image")
        identity = response.get("identity")
        consumed, size = response.get("consumed"), response.get("bytes")
        if (type(image) not in (int, str) or type(image) is bool
                or not isinstance(identity, str) or not identity
                or type(consumed) is not int or consumed != lease.consumed
                or type(size) is not int or size <= 0):
            raise IntegrityError("Native fixture capture receipt is incomplete")
        blob = HotBlob(image, identity.encode("utf-8"), self._process_identity,
                       consumed, size)
        self._known_images[str(image)] = weakref.finalize(
            blob, _finalize_image, weakref.ref(self), image)
        lease.image_identity = blob.identity
        self._stats["checkpoint_export_bytes"] += size
        return blob

    def _validate_restored(self, response, history, frontier):
        if (type(response.get("consumed")) is not int
                or response["consumed"] != len(history)
                or response.get("history") != list(history)
                or not isinstance(response.get("logits"), list)
                or len(response["logits"]) != VOCAB_SIZE):
            raise IntegrityError("Native fixture restore lost canonical inputs or position")
        logits = tuple(float(value) for value in response["logits"])
        if not all(math.isfinite(value) for value in logits) or logits != tuple(frontier):
            raise IntegrityError("Native fixture restore frontier differs from checkpoint")

    def restore(self, lease, state, *, history, frontier):
        self._check(lease)
        history = tuple(history)
        if isinstance(state, HotBlob):
            if state.process_identity != self._process_identity or state.consumed != len(history):
                raise IntegrityError("Hot fixture image belongs to another worker or history")
            response = self._request("restore " + str(state.image))
            self._validate_restored(response, history, frontier)
            resident = state
        elif isinstance(state, bytes) and state:
            path = self._directory / "import.state"
            path.write_bytes(state)
            try:
                response = self._request("import " + str(path))
            finally:
                path.unlink(missing_ok=True)
            self._validate_restored(response, history, frontier)
            resident = None
            self._stats["checkpoint_import_bytes"] += len(state)
        else:
            raise IntegrityError("Unsupported or empty fixture checkpoint")
        lease.bound = True
        lease.consumed = len(history)
        lease.history = history
        blob = self.export(lease) if resident is None else resident
        lease.image_identity = blob.identity
        return blob

    def serialize(self, lease, state):
        self._check(lease)
        if (not isinstance(state, HotBlob) or state.process_identity != self._process_identity
                or lease.image_identity != state.identity):
            raise IntegrityError("Fixture serialization requires the checked active image")
        path = self._directory / "image.state"
        response = self._request("serialize " + str(state.image) + " " + str(path))
        payload = path.read_bytes()
        path.unlink(missing_ok=True)
        if type(response.get("bytes")) is not int or response["bytes"] != len(payload) or not payload:
            raise IntegrityError("Native fixture serialized byte extent differs from file")
        self._stats["checkpoint_export_bytes"] += len(payload)
        return payload

    def park(self, lease):
        # Captures are already immutable native images; no second RAM history cache exists.
        self._check(lease)
        return None

    def restore_cached(self, lease, history):
        self._check(lease)
        raise IntegrityError("Fixture backend refuses prefix replay without a checkpoint")

    def clear_cached_histories(self):
        return 0

    def _flush_drops(self):
        import queue
        while True:
            try:
                image = self._drops.get_nowait()
            except queue.Empty:
                return
            if not self._closed and not self._failed and self.process.poll() is None:
                self._request("drop " + str(image))
            self._known_images.pop(str(image), None)

    def release_checkpoints(self):
        self._flush_drops()

    def work_counters(self):
        response = self._request("counters")
        counters = {}
        for key in _COUNTER_FIELDS:
            value = response.get(key)
            if type(value) is not int or value < 0:
                raise IntegrityError("Native fixture counter is missing or invalid: " + key)
            counters[key] = value
        self._stats.update(counters)
        self._stats["consumed_tokens"] = counters["forward_tokens"]
        self._stats["checkpoint_export_bytes"] = counters["serialized_bytes"]
        self._stats["checkpoint_import_bytes"] = counters["loaded_bytes"]
        return dict(self._stats)

    def checkpoint_metrics(self, values):
        images = {id(item.blob): item.blob for item in values}
        return dict(owned_checkpoint_bytes=sum(len(blob) for blob in images.values()),
                    native_counters=self.work_counters())

    def close(self):
        if self._closed:
            return
        try:
            self._flush_drops()
            if self.process.poll() is None:
                if not self._failed:
                    self._request("quit")
                    self.process.wait(timeout=15)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self._closed = True
            self._temp.cleanup()
            self._log.close()


class FixtureModel(BranchingModel):
    """Test-only BranchingModel variant with no prefix-replay escape hatch."""
    @contextmanager
    def _materialize(self, snapshot):
        manifest = self.checkpoint_directory / (snapshot + ".json")
        if snapshot not in self.cache and not manifest.is_file():
            raise IntegrityError("Fixture restore requires a retained numerical checkpoint")
        with super()._materialize(snapshot) as materialized:
            yield materialized

    def run_greedy_request(self, snapshot, ids, max_steps, *, stream, on_token=None):
        """Consume one request with the declared integer fixture, never a model gate."""
        started = time.monotonic()
        self.store.authorize(snapshot, "append")
        ids = tuple(ids)
        if (any(type(token) is not int or not 0 <= token < self.backend.vocab_size for token in ids)
                or type(max_steps) is not int or max_steps <= 0):
            raise InvalidOperation("Invalid fixture request")
        execution = self.profile["execution"]
        history = self.history(snapshot)
        if len(history) + len(ids) + max_steps > execution["capacity"]:
            raise InvalidOperation("Fixture context capacity exceeded")
        prompt_history = history + ids
        decoder = self.controller(snapshot)
        if ids or decoder["status"] == "idle":
            from context_time_travel import controller
            decoder = controller.start(dict(temperature=0, eos_tokens=[]), stream,
                                      prompt_history, self.backend.vocab_size)
        outputs = []
        generation_started = time.monotonic()
        with self.store.read_lease(snapshot):
            with self._materialize(snapshot) as (lease, _):
                if ids:
                    self.backend.consume(lease, ids)
                for _ in range(max_steps):
                    frontier = self.backend.logits(lease)
                    token = max(range(self.backend.vocab_size), key=lambda item: (frontier[item], -item))
                    self.backend.consume_generated(lease, (token,))
                    outputs.append(token)
                    if on_token is not None:
                        on_token(token)
                generation_seconds = time.monotonic() - generation_started
                consumed_history = prompt_history + tuple(outputs)
                from context_time_travel import controller
                decoder = controller.consume_many(decoder,
                    ((token, self.backend.piece(token), False) for token in outputs))
                self._lease_fault("after_token_consume")
                saved = self._save(lease, consumed_history, ())
                if saved.consumed != len(consumed_history):
                    raise IntegrityError("Fixture capture omitted consumed inputs")
        sid = self._publish_feed(snapshot, ids + tuple(outputs), decoder, saved)
        return dict(snapshot=sid, tokens=outputs, generation_seconds=generation_seconds,
                    request_seconds=time.monotonic() - started)
