"""Run and audit the bounded P4 repeated-return latency experiment.

Importing this module does not load a model. The explicit command-line runner
imports selected canonical checkpoints and starts one coordinator when invoked.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import argparse
import queue
import statistics
import shutil
import sys
import tempfile
import threading
import time
import uuid
from typing import Iterable

from context_time_travel.checkpoint_cache import validate_manifest
from context_time_travel.store import IntegrityError, Store, _decode, _hash, _json_bytes


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_file(path: Path) -> None:
    """Flush an already-written file before its containing directory is synced."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on the local POSIX filesystem used by P4."""
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_snapshot(source: Path, identity: str) -> dict:
    if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('invalid snapshot identity')
    path = source / 'objects' / (identity + '.json')
    raw = path.read_bytes()
    if _hash(raw) != identity:
        raise IntegrityError('source snapshot digest mismatch: ' + identity)
    obj = _decode(raw)
    if obj.get('kind') != 'snapshot' or obj.get('schema') != 1:
        raise IntegrityError('source object is not a supported snapshot')
    return obj


def _validated_chain(source: Path, snapshot_id: str, contract: str):
    reverse = []
    seen = set()
    cursor = snapshot_id
    while cursor is not None:
        if cursor in seen:
            raise IntegrityError('cycle in requested source lineage')
        seen.add(cursor)
        row = _read_snapshot(source, cursor)
        if row.get('model_contract_id') != contract:
            raise IntegrityError('source lineage crosses model contracts')
        parent = row.get('parent')
        if parent is not None and (not isinstance(parent, str) or len(parent) != 64):
            raise IntegrityError('invalid source parent identity')
        reverse.append((cursor, row))
        cursor = parent
    return list(reversed(reverse))


def _manifest_expected(root_state: dict, profile: dict, history: list[int]):
    manifest = root_state.get('manifest')
    if not isinstance(manifest, dict):
        raise ValueError('root state has no checkpoint manifest')
    contract = profile.get('model_contract_id')
    profile_id = profile.get('execution_profile_id')
    if not contract or not profile_id:
        raise ValueError('profile must identify model contract and execution profile')
    if manifest.get('model_contract_id') != contract:
        raise ValueError('checkpoint model contract does not match profile')
    if manifest.get('execution_profile_id') != profile_id:
        raise ValueError('checkpoint execution profile does not match profile')
    if manifest.get('snapshot') != root_state.get('snapshot'):
        raise ValueError('checkpoint is not for the requested root snapshot')
    input_identity = hashlib.sha256(_json_bytes(history)).hexdigest()
    expected = dict(schema=manifest.get('schema'), snapshot=root_state['snapshot'],
                    input_identity=input_identity, model_contract_id=contract,
                    execution_profile_id=profile_id, consumed=len(history))
    execution = profile.get('execution', {})
    resume_contract = execution.get('resume_contract')
    if resume_contract in ('identical-native-batches-v1',
                           'identical-native-operators-v1'):
        from context_time_travel.execution_receipt import (
            validate_operator_schedule, validate_schedule,
        )
        schema, validator = ((2, validate_schedule)
                             if resume_contract == 'identical-native-batches-v1'
                             else (3, validate_operator_schedule))
        if manifest.get('schema') != schema:
            raise ValueError('checkpoint schema does not match profile resume contract')
        schedule = manifest.get('execution_schedule')
        digest = validator(schedule, history, execution.get('prefill_chunk'))
        if digest != manifest.get('schedule_sha256'):
            raise ValueError('checkpoint schedule digest does not match schedule')
        expected.update(schema=schema, execution_schedule=schedule,
                        schedule_sha256=digest)
    elif resume_contract in (None, 'legacy', 'native-request-primitive-experiment-v1'):
        if manifest.get('schema') != 1:
            raise ValueError('legacy profile requires schema-1 checkpoint')
        expected['schema'] = 1
    else:
        raise ValueError('unsupported profile resume contract')
    return expected


def _validate_and_copy_checkpoint(source: Path, destination: Path,
                                  root_state: dict, profile: dict,
                                  history: list[int]):
    expected = _manifest_expected(root_state, profile, history)
    snapshot = root_state['snapshot']
    envelope_path = source / 'checkpoints' / (snapshot + '.json')
    envelope = _decode(envelope_path.read_bytes())
    manifest = validate_manifest(envelope, expected)
    if manifest['positions'] != 'absolute-zero-based':
        raise IntegrityError('unsupported checkpoint position convention')
    payload_path = source / 'checkpoints' / (manifest['payload_sha256'] + '.state')
    digest, size = _sha_file(payload_path)
    if digest != manifest['payload_sha256']:
        raise IntegrityError('checkpoint payload digest mismatch')
    if size != manifest['state_bytes'] + 4 * manifest['logits_count']:
        raise IntegrityError('checkpoint payload size does not match manifest')
    # Validate finite frontier logits without loading the model/checkpoint into
    # the numerical backend.  The payload suffix is little-endian float32.
    import math
    import struct
    with payload_path.open('rb') as stream:
        stream.seek(manifest['state_bytes'])
        raw_logits = stream.read(4 * manifest['logits_count'])
    logits = struct.unpack('<' + 'f' * manifest['logits_count'], raw_logits)
    if not all(math.isfinite(value) for value in logits):
        raise IntegrityError('checkpoint frontier contains nonfinite logits')

    destination.mkdir(parents=True, exist_ok=True)
    dest_manifest = destination / envelope_path.name
    dest_payload = destination / payload_path.name
    for src, dst in ((envelope_path, dest_manifest), (payload_path, dest_payload)):
        if dst.exists():
            if _sha_file(dst) != _sha_file(src):
                raise IntegrityError('conflicting checkpoint identity in import')
        else:
            shutil.copyfile(src, dst)
            if _sha_file(dst) != _sha_file(src):
                raise IntegrityError('checkpoint copy verification failed')
        # A successful copy/hash only verifies bytes in the page cache.  Flush
        # the imported checkpoint bytes before the disk arm can acknowledge its
        # newly published Store.
        _fsync_file(dst)
    _fsync_directory(destination)
    return manifest


def import_minimal_store(source_store, destination_store,
                         root_states: Iterable[dict], profile: dict) -> dict:
    """Import requested histories and checkpoints; publish target atomically.

    `root_states` are campaign-captured records with `snapshot`, exact flattened
    `history`, `controller`, and the root checkpoint `manifest`.  The source is
    read-only.  Failure leaves destination absent; only a fully validated
    staging Store is renamed into place.
    """
    source = Path(source_store).resolve(strict=True)
    target = Path(destination_store).absolute()
    states = list(root_states)
    if not states:
        raise ValueError('at least one prepared root is required')
    if target.exists():
        raise FileExistsError(str(target))
    if not target.parent.is_dir():
        raise FileNotFoundError('destination parent must already exist')
    resolved_target = target.parent.resolve() / target.name
    try:
        if os.path.commonpath((str(source), str(resolved_target))) == str(source):
            raise ValueError('destination must not be inside the frozen source Store')
    except ValueError as error:
        if 'destination must not' in str(error):
            raise
    contract = profile.get('model_contract_id')
    profile_id = profile.get('execution_profile_id')
    if not contract or not profile_id:
        raise ValueError('profile must identify model contract and execution profile')

    # Validate all source state before making even a staging Store usable.
    validated = []
    all_objects = {}
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('invalid root state')
        snapshot = state.get('snapshot')
        history = state.get('history')
        controller = state.get('controller')
        if not isinstance(snapshot, str) or not isinstance(history, list):
            raise ValueError('root state must include snapshot and history')
        if not isinstance(controller, dict):
            raise ValueError('root state must include controller mapping')
        chain = _validated_chain(source, snapshot, contract)
        exact_history = [token for _, row in chain for token in row['inputs']]
        if history != exact_history:
            raise IntegrityError('campaign history differs from canonical Store lineage')
        if chain[-1][1].get('controller') != controller:
            raise IntegrityError('campaign controller differs from canonical snapshot')
        manifest = state.get('manifest')
        if not isinstance(manifest, dict):
            raise ValueError('root state has no checkpoint manifest')
        if hashlib.sha256(_json_bytes(history)).hexdigest() != manifest.get('input_identity'):
            raise IntegrityError('checkpoint input identity differs from source history')
        for identity, row in chain:
            prior = all_objects.setdefault(identity, row)
            if prior != row:
                raise IntegrityError('conflicting snapshot data for same identity')
        validated.append((state, chain, history))

    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '.staging-',
                                    dir=str(target.parent)))
    try:
        store_path = staging / 'store'
        store = Store(store_path)
        imported = set()
        for _, chain, _ in validated:
            parent_id = None
            for identity, row in chain:
                if identity in imported:
                    parent_id = identity
                    continue
                if parent_id is None:
                    actual = store.create_root(contract, row['inputs'], row['controller'])
                else:
                    actual = store.append(parent_id, row['inputs'], row['controller'])
                if actual != identity:
                    raise IntegrityError('Store API did not reproduce canonical snapshot ID')
                imported.add(identity)
                parent_id = identity
        checkpoints = store_path / 'checkpoints'
        imported_manifests = []
        for state, _, history in validated:
            imported_manifests.append(_validate_and_copy_checkpoint(
                source, checkpoints, state, profile, history))
        # Read back every lineage/controller after checkpoint validation.
        for state, chain, history in validated:
            root_id, actual_history = store.lineage(state['snapshot'])
            if actual_history != history or root_id != chain[0][0]:
                raise IntegrityError('imported history verification failed')
            actual = store.get_snapshot(state['snapshot'])
            if actual['controller'] != state['controller']:
                raise IntegrityError('imported controller verification failed')
        del store
        # The checkpoint directory itself was created after Store initialized;
        # persist the new child entry before publishing the complete Store.
        _fsync_directory(store_path)
        os.replace(store_path, target)
        # Persist the atomic Store-directory publication in the real output
        # filesystem.  The imported checkpoint files and their directory were
        # already synced above.
        _fsync_directory(target.parent)
        return dict(snapshots=[state['snapshot'] for state, _, _ in validated],
                    objects=len(imported), manifests=imported_manifests,
                    checkpoint_payloads=sorted(p.name for p in (target/'checkpoints').glob('*.state')))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def classify_checkpoint_tier(key, *, pool_contains: bool,
                             model_cache_contains: bool, events,
                             native_counters=None) -> str:
    """Classify only with pool membership and exact-key disk trace evidence."""
    if model_cache_contains:
        raise ValueError('worker model-cache hit bypasses the measured pool tier')
    normalized = list(key)
    exact = [e for e in events if list(e.get('key', ())) == normalized]
    starts = [e for e in exact if e.get('event') == 'disk_preload_started']
    successes = [e for e in exact if e.get('event') == 'disk_preload_succeeded']
    if pool_contains:
        if any(e.get('event','').startswith('disk_preload_') for e in exact):
            raise ValueError('pool-resident key also has exact-key disk preload')
        return 'resident_pool'
    if len(starts) != 1 or len(successes) != 1:
        raise ValueError('disk tier needs one exact-key preload start and success')
    start, success = starts[0], successes[0]
    if (type(start.get('sequence')) is not int or type(success.get('sequence')) is not int
            or start['sequence'] >= success['sequence']):
        raise ValueError('disk preload completion is not after its start')
    if (start.get('load_id') != success.get('load_id')
            or type(start.get('load_id')) is not int):
        raise ValueError('disk preload start/success load IDs differ')
    for field in ('process_id','phase','phase_id'):
        if not start.get(field) or start.get(field) != success.get(field):
            raise ValueError('disk preload start/success '+field+' differs')
    if success.get('success') is not True or type(success.get('bytes')) is not int or success['bytes'] <= 0:
        raise ValueError('disk preload success has no positive byte witness')
    return 'disk_preload'


class EventTimeline:
    """Retain producer and consumer timestamps as separate time domains."""
    def __init__(self):
        self.rows = []

    def accept(self, event: dict, *, consumer_monotonic: float):
        producer = event.get('time')
        if type(producer) not in (int, float) or type(consumer_monotonic) not in (int, float):
            raise ValueError('event needs numeric producer and consumer timestamps')
        self.rows.append(dict(event=event.get('event'), producer=float(producer),
                              consumer=float(consumer_monotonic), token=event.get('token'),
                              eos=event.get('eos', event.get('turn_complete'))))

    def summary(self, *, submit_monotonic: float) -> dict:
        starts = [r for r in self.rows if r['event'] == 'request_started']
        tokens = [r for r in self.rows if r['event'] == 'token']
        snapshots = [r for r in self.rows if r['event'] == 'snapshot']
        durable = [r for r in self.rows if r['event'] == 'durable']
        if not starts or not tokens or not snapshots or not durable:
            raise ValueError('timeline missing request/token/snapshot/durable event')
        producer_gaps = [b['producer']-a['producer'] for a,b in zip(tokens,tokens[1:])]
        consumer_gaps = [b['consumer']-a['consumer'] for a,b in zip(tokens,tokens[1:])]
        if any(g < 0 for g in producer_gaps+consumer_gaps):
            raise ValueError('nonmonotonic token event timestamps')
        return dict(submit_to_request_started_seconds=starts[0]['consumer']-submit_monotonic,
                    submit_to_first_token_seconds=tokens[0]['consumer']-submit_monotonic,
                    producer_token_gap_seconds=producer_gaps,
                    consumer_token_gap_seconds=consumer_gaps,
                    submit_to_snapshot_seconds=snapshots[-1]['consumer']-submit_monotonic,
                    submit_to_durable_seconds=durable[-1]['consumer']-submit_monotonic,
                    producer_snapshot_to_durable_seconds=durable[-1]['producer']-snapshots[-1]['producer'],
                    output_tokens=len(tokens), eos=bool(tokens[-1]['eos']))


def validate_work_delta(question_token_count: int, generated_tokens,
                        counters: dict, *, saved_prefix_tokens: int) -> bool:
    if type(question_token_count) is not int or question_token_count < 0:
        raise ValueError('invalid question token count')
    outputs = list(generated_tokens)
    forward = counters.get('forward_tokens')
    prefill = counters.get('prefill_tokens')
    if type(forward) is not int or type(prefill) is not int:
        raise ValueError('native counters lack forward/prefill token counts')
    if forward != question_token_count + len(outputs):
        raise ValueError('native forward work includes unexpected or missing tokens')
    if prefill != question_token_count:
        raise ValueError('native prefill work is not suffix-only')
    if forward >= saved_prefix_tokens:
        raise ValueError('native forward work approaches/repeats saved prefix')
    return True


def validate_reference_output(actual, reference, eos_token_ids,
                              output_limit: int) -> bool:
    actual = list(actual)
    reference = list(reference)
    eos = set(eos_token_ids)
    if type(output_limit) is not int or output_limit <= 0:
        raise ValueError('output limit must be a positive integer')
    expected = reference[:output_limit]
    for index, token in enumerate(actual):
        if index >= len(expected) or token != expected[index]:
            raise ValueError('generated tokens differ from independent fresh reference')
        if token in eos:
            if index != len(actual)-1:
                raise ValueError('tokens follow actual EOS')
            return True
    if len(actual) < min(output_limit, len(reference)):
        raise ValueError('output ended before EOS or the bounded reference prefix')
    return True


def _campaign_inputs(source_store: Path, source_result_path: Path,
                     prepare_path: Path, profile_path: Path,
                     fixture_path: Path, reference_dir: Path, *,
                     expected_prefix_tokens: int = 65536):
    prepared = json.loads(prepare_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    profile = json.loads(profile_path.read_text())
    fixture = json.loads(fixture_path.read_text())
    if prepared.get('status') != 'PASS' or not isinstance(prepared.get('roots'), list):
        raise ValueError('source prepare receipt is not PASS or lacks roots')
    if profile.get('model_contract_id') is None or profile.get('execution_profile_id') is None:
        raise ValueError('source profile lacks model/profile identity')
    if (source_result.get('status') != 'PASS'
            or source_result.get('profile_id') != profile['execution_profile_id']
            or source_result.get('model_contract_id') != profile['model_contract_id']
            or source_result.get('build_sha256') != profile.get('native_build', {}).get('sha256')):
        raise ValueError('source campaign receipt does not qualify the supplied exact profile/build')
    roots = prepared['roots']
    if len(roots) != 3:
        raise ValueError('P4 return runner requires exactly three prepared roots')
    if type(expected_prefix_tokens) is not int or expected_prefix_tokens <= 0:
        raise ValueError('expected prefix token count must be positive')
    for root_index, root in enumerate(roots):
        history = root.get('history')
        if not isinstance(history, list) or len(history) != expected_prefix_tokens:
            raise ValueError(f'prepared root {root_index} history must contain exactly '
                             f'{expected_prefix_tokens} tokens')
    suffixes = fixture.get('suffixes')
    if not isinstance(suffixes, list) or len(suffixes) != 6:
        raise ValueError('P4 fixture must contain exactly six supported suffix trees')
    if any(not isinstance(ids, list) or not ids or
           any(type(token) is not int or token < 0 for token in ids) for ids in suffixes):
        raise ValueError('fixture suffixes must be nonempty lists of nonnegative token IDs')
    mapping = fixture.get('tree_roots')
    if not isinstance(mapping, list) or len(mapping) != 6:
        raise ValueError('P4 fixture must explicitly map all six trees to prepared roots')
    if any(type(root_index) is not int or root_index not in range(3) for root_index in mapping):
        raise ValueError('fixture tree/root mapping contains an unsupported root index')
    if [mapping[index] for index in (0, 2, 4)] != [0, 1, 2]:
        raise ValueError('P4 fixture selected trees 0, 2, and 4 must map to roots 0, 1, and 2')
    selected = []
    for tree_index in (0, 2, 4):
        path = reference_dir / f'fresh-tree-{tree_index}.stdout'
        reference = json.loads(path.read_text())
        generated = reference.get('generated')
        if not isinstance(generated, list) or not generated:
            raise ValueError(f'independent fresh reference is missing for tree {tree_index}')
        selected.append(dict(root_index=mapping[tree_index], tree_index=tree_index,
                             suffix=suffixes[tree_index], reference=generated))
    if {item['root_index'] for item in selected} != {0,1,2}:
        raise ValueError('fixture suffixes 0, 2, and 4 must cover all three roots')
    selected.sort(key=lambda item:item['root_index'])
    return roots, profile, fixture, selected, source_result


def preflight_arm(source_store, target_parent, root_states, profile, *,
                  arm: str, ram_bytes: int, disk_bytes: int,
                  min_available_bytes: int, free_bytes: int | None = None) -> dict:
    """Fail before model startup unless source and conservative disk envelope fit."""
    source = Path(source_store).resolve(strict=True)
    parent = Path(target_parent).resolve(strict=True)
    if len(root_states) != 3:
        raise ValueError('preflight requires exactly three root states')
    if arm not in ('resident', 'disk'):
        raise ValueError('arm must be resident or disk')
    maximum_payload = 0
    maximum_load_reservation = 0
    roots = []
    for state in root_states:
        expected = _manifest_expected(state, profile, state.get('history', []))
        manifest_path = source/'checkpoints'/(state['snapshot']+'.json')
        envelope = _decode(manifest_path.read_bytes())
        manifest = validate_manifest(envelope, expected)
        payload = source/'checkpoints'/(manifest['payload_sha256']+'.state')
        digest, size = _sha_file(payload)
        if digest != manifest['payload_sha256']:
            raise IntegrityError('source checkpoint payload digest mismatch')
        if size != manifest['state_bytes']+4*manifest['logits_count']:
            raise IntegrityError('source checkpoint payload size mismatch')
        maximum_payload = max(maximum_payload, size)
        maximum_load_reservation = max(maximum_load_reservation,
            2*size, 3*manifest['state_bytes']+8*manifest['logits_count'])
        roots.append(dict(snapshot=state['snapshot'], manifest_sha256=_sha_file(manifest_path)[0],
                          payload_sha256=digest, payload_bytes=size))
    if arm == 'resident' and ram_bytes < 15*maximum_payload:
        raise ValueError('resident pool budget cannot conservatively retain roots and working checkpoints')
    if ram_bytes < maximum_load_reservation:
        raise ValueError('RAM pool budget is below one load/capture reservation')
    if disk_bytes < 15*maximum_payload:
        raise ValueError('disk pool budget cannot conservatively retain roots and working checkpoints')
    required_free = 15*maximum_payload + 10*1024**3
    actual_free = shutil.disk_usage(parent).free if free_bytes is None else free_bytes
    if actual_free < required_free:
        raise OSError(f'insufficient free disk: need {required_free}, have {actual_free}')
    available_memory = None
    if min_available_bytes:
        meminfo = Path('/proc/meminfo')
        if not meminfo.is_file():
            raise OSError('host available-memory witness /proc/meminfo is unavailable')
        fields = dict(line.split(':',1) for line in meminfo.read_text().splitlines() if ':' in line)
        if 'MemAvailable' not in fields:
            raise OSError('host available-memory witness lacks MemAvailable')
        available_memory = int(fields['MemAvailable'].split()[0])*1024
        if available_memory < min_available_bytes:
            raise OSError(f'insufficient host memory headroom: need {min_available_bytes}, have {available_memory}')
    if disk_bytes <= 0 or min_available_bytes < 0 or ram_bytes <= 0:
        raise ValueError('pool budgets must be positive (minimum available may be zero)')
    gpu = profile.get('execution', {}).get('GPU')
    from context_time_travel.pools import normalize_gpu_selection
    if len(normalize_gpu_selection(gpu)) != 1:
        raise ValueError('P4 return runner requires one coordinator worker')
    return dict(arm=arm, model_contract_id=profile['model_contract_id'],
                execution_profile_id=profile['execution_profile_id'],
                roots=roots, max_checkpoint_bytes=maximum_payload,
                worst_case_checkpoint_count=15, required_free_bytes=required_free,
                available_free_bytes=actual_free, available_memory_bytes=available_memory,
                maximum_load_reservation_bytes=maximum_load_reservation, ram_bytes=ram_bytes,
                disk_bytes=disk_bytes, min_available_bytes=min_available_bytes,
                gpu=gpu)


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


class RunHeartbeat:
    """Publish phase context every interval while a long run is active."""
    def __init__(self, path: Path, *, interval: float = 30.0, emit=None):
        if type(interval) not in (int, float) or interval <= 0:
            raise ValueError('heartbeat interval must be positive')
        self.path = Path(path)
        self.interval = float(interval)
        self.emit = emit
        self.started = time.monotonic()
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._thread_error = None
        self._state = dict(phase='starting', root_index=None, repeat=None)

    def _payload(self):
        with self._lock:
            state = dict(self._state)
        state.update(pid=self.pid, elapsed_seconds=time.monotonic()-self.started)
        return state

    def update(self, phase: str, *, root_index=None, repeat=None):
        with self._lock:
            self._state = dict(phase=phase, root_index=root_index, repeat=repeat)
        with self._publish_lock:
            _write_json(self.path, self._payload())

    def _run(self):
        try:
            while not self._stop.wait(self.interval):
                payload = self._payload()
                with self._publish_lock:
                    _write_json(self.path, payload)
                if self.emit is not None:
                    self.emit('P4_HEARTBEAT '+json.dumps(payload, sort_keys=True))
        except BaseException as error:
            with self._lock:
                self._thread_error = error
            self._stop.set()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('heartbeat already started')
        self._thread = threading.Thread(target=self._run, name='p4-return-heartbeat', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval+1.0))
            if self._thread.is_alive():
                raise TimeoutError('heartbeat thread did not stop')
        with self._lock:
            error = self._thread_error
        if error is not None:
            raise RuntimeError('heartbeat worker failed') from error


def run_arm(args, *, coordinator_factory=None):
    """Execute one of the two arms. Native execution is an explicit CLI action."""
    source_store = Path(args.source_store).resolve(strict=True)
    output = Path(args.output).absolute()
    if output.exists():
        raise FileExistsError(str(output))
    if not output.parent.is_dir():
        raise FileNotFoundError('output parent must already exist')
    prefix_override = getattr(args, 'test_prefix_tokens', None)
    if prefix_override is not None and coordinator_factory is None:
        raise ValueError('test prefix override requires an injected CPU fake coordinator')
    prefix_tokens = 65536 if prefix_override is None else prefix_override
    if type(prefix_tokens) is not int or prefix_tokens <= 0:
        raise ValueError('prefix token count must be positive')
    roots, profile, fixture, selected, source_result = _campaign_inputs(
        source_store, Path(args.source_result), Path(args.prepare), Path(args.profile), Path(args.fixture),
        Path(args.reference_dir), expected_prefix_tokens=prefix_tokens)
    preflight = preflight_arm(source_store, output.parent, roots, profile,
        arm=args.arm, ram_bytes=args.ram_bytes, disk_bytes=args.disk_bytes,
        min_available_bytes=args.min_available_bytes,
        free_bytes=getattr(args, 'preflight_free_bytes_for_test', None))
    if getattr(args, 'preflight_free_bytes_for_test', None) is not None and coordinator_factory is None:
        raise ValueError('synthetic free-space override is test-only and requires an injected fake coordinator')
    output.mkdir(mode=0o700)
    report = dict(status='RUNNING', scope='same-process coordinator return timing; not HTTP or cold-device latency',
        arm=args.arm, arm_order=args.arm_order, order_seed=args.order_seed,
        output_limit=128, warmups=3, measured_requests=9,
        saved_prefix_tokens=prefix_tokens, pid=os.getpid(), phase='preparing',
        preflight=preflight,
        profile_identity={name:profile.get(name) for name in
            ('model_contract_id','execution_profile_id','model_weights_sha256','native_build')},
        tokenizer_contract=profile.get('tokenizer'),
        hardware=profile.get('hardware'),
        execution_identity={name:profile.get('execution',{}).get(name) for name in
            ('GPU','capacity','prefill_chunk','resume_contract','generated_operator',
             'native_request_protocol','eos_tokens')},
        roots=[row['snapshot'] for row in roots],
        source_store=str(source_store), samples=[], warmup_samples=[],
        input_sha256={str(path):_sha_file(Path(path))[0] for path in
            (args.source_result,args.prepare,args.profile,args.fixture,
             *(Path(args.reference_dir)/f'fresh-tree-{i}.stdout' for i in (0,2,4)))},
        source_receipt={name:source_result.get(name) for name in
            ('status','profile_id','model_contract_id','build_sha256','executed_test_ids')},
        excluded_claims=['HTTP_TTFT','cold_device_latency','cross_quantization_comparison'])
    _write_json(output/'result.json', report)
    heartbeat_emit = getattr(args, 'heartbeat_emit_for_test', None)
    if heartbeat_emit is None and not hasattr(args, 'heartbeat_emit_for_test'):
        heartbeat_emit = lambda line: print(line, file=sys.stderr, flush=True)
    heartbeat = RunHeartbeat(output/'heartbeat.json',
        interval=getattr(args, 'heartbeat_interval_for_test', 30.0), emit=heartbeat_emit).start()
    run_error = None
    cleanup_errors = []

    def phase(name, *, root_index=None, repeat=None):
        report['phase'] = name
        if root_index is not None:
            report['current_root_index'] = root_index
        else:
            report.pop('current_root_index', None)
        if repeat is not None:
            report['current_repeat'] = repeat
        else:
            report.pop('current_repeat', None)
        heartbeat.update(name, root_index=root_index, repeat=repeat)
        _write_json(output/'result.json', report)

    try:
        phase('checkpoint_import')
        imported = import_minimal_store(source_store, output/'store', roots, profile)
        report['import'] = imported
        phase('environment_setup')
        execution = profile['execution']
        for key in list(os.environ):
            if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
                del os.environ[key]
        os.environ.update(execution.get('environment', {}))
        from context_time_travel.coordinator import ContextCoordinator
        from context_time_travel.pools import PoolConfig, PoolTransitionTrace
        if coordinator_factory is None:
            from context_time_travel.real import BranchingModel
            from context_time_travel.strata_backend import StrataBackend
            def coordinator_factory(directory, config, trace):
                def factory(gpu):
                    backend = StrataBackend(profile, gpu_id=gpu)
                    return BranchingModel(directory, backend, profile)
                return ContextCoordinator(directory, config, factory, transition_trace=trace)

        trace = PoolTransitionTrace('p4-return-'+args.arm, max_events=32768)
        gpu_ids = tuple(execution['GPU'])
        pool_config = PoolConfig(gpus=gpu_ids, ram_bytes=args.ram_bytes,
            disk_bytes=args.disk_bytes, min_available_bytes=args.min_available_bytes)
        phase('coordinator_startup')
        startup = time.monotonic()
        coordinator = coordinator_factory(output/'store', pool_config, trace)
        active = [None]
        lock = threading.Lock()
        coordinator.ready()
        startup_seconds = time.monotonic()-startup
        models = [worker['model'] for worker in coordinator._workers.values()]
        if len(models) != 1 or models[0] is None:
            raise RuntimeError('coordinator did not create exactly one model worker')
        model = models[0]
        report['startup_seconds'] = startup_seconds
        report['native_pid'] = getattr(model.backend.process, 'pid', None)

        def record_stage(name, snapshot, elapsed):
            with lock:
                row = active[0]
                if row is not None and snapshot == row['source_snapshot']:
                    row.setdefault('stages', {}).setdefault(name, []).append(elapsed)

        original_load = model._load_checkpoint
        def timed_load(snapshot, history):
            began = time.monotonic()
            try:
                return original_load(snapshot, history)
            finally:
                record_stage('checkpoint_load_validate_seconds', snapshot, time.monotonic()-began)
        model._load_checkpoint = timed_load

        original_restore = model.backend.restore
        def timed_restore(lease, blob, **kwargs):
            began = time.monotonic()
            try:
                return original_restore(lease, blob, **kwargs)
            finally:
                row = active[0]
                if row is not None:
                    row.setdefault('stages', {}).setdefault('backend_restore_seconds', []).append(time.monotonic()-began)
        model.backend.restore = timed_restore

        original_consume = model.backend.consume
        def timed_consume(lease, ids):
            began = time.monotonic()
            try:
                return original_consume(lease, ids)
            finally:
                row = active[0]
                if row is not None:
                    row.setdefault('stages', {}).setdefault('backend_consume_seconds', []).append(time.monotonic()-began)
        model.backend.consume = timed_consume

        original_run = coordinator._run
        def timed_run(bound_model, job):
            row = active[0]
            if row is not None:
                row['worker_run_start_monotonic'] = time.monotonic()
            return original_run(bound_model, job)
        coordinator._run = timed_run

        original_preload = coordinator._preload
        preload_waiters = {}
        preload_future_ids = {}
        def timed_preload(bound_model, snapshot, **kwargs):
            began = time.monotonic()
            future = original_preload(bound_model, snapshot, **kwargs)
            row = active[0]
            if (row is not None and snapshot == row['source_snapshot'] and future is not None):
                future_identity = preload_future_ids.setdefault(id(future),
                    'preload-future-'+str(len(preload_future_ids)+1))
                invocations = row.setdefault('stages', {}).setdefault(
                    'pool_preload_invocations', [])
                record = dict(future_identity=future_identity,
                    invocation=len(invocations)+1,
                    kind='first_call_for_future' if sum(
                        item['future_identity'] == future_identity for item in invocations) == 0
                        else 'shared_future_wait',
                    started_monotonic=began, completed_monotonic=None)
                invocations.append(record)
                done_event = threading.Event()
                preload_waiters.setdefault(id(row), []).append(done_event)
                def done(_future):
                    completed = time.monotonic()
                    record['completed_monotonic'] = completed
                    record['elapsed_seconds'] = completed-began
                    done_event.set()
                row.setdefault('stages', {})
                future.add_done_callback(done)
            return future
        coordinator._preload = timed_preload

        def pool_key(snapshot):
            return (profile['model_contract_id'], profile['execution_profile_id'], snapshot)

        def wait_request(handle, row, submit_time):
            timeline = EventTimeline()
            events = []
            deadline = time.monotonic()+900
            while not handle.durable.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError('request did not reach durable completion within 900 seconds')
                try:
                    event = handle.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                arrived = time.monotonic()
                events.append(event)
                timeline.accept(event, consumer_monotonic=arrived)
            while True:
                try:
                    event = handle.events.get_nowait()
                except queue.Empty:
                    break
                arrived = time.monotonic()
                events.append(event)
                timeline.accept(event, consumer_monotonic=arrived)
            answer = handle.snapshot.result()
            receipt = handle.durable.result()
            if receipt.get('snapshot') != answer.get('snapshot'):
                raise IntegrityError('durable receipt differs from answer snapshot')
            row['events'] = events
            row['answer'] = answer
            row['receipt'] = receipt
            row['consumer_timeline'] = timeline.summary(submit_monotonic=submit_time)
            if row.get('worker_run_start_monotonic') is not None:
                row['submit_to_worker_start_seconds'] = row['worker_run_start_monotonic']-submit_time
            row['submit_to_request_started_event_seconds'] = row['consumer_timeline']['submit_to_request_started_seconds']
            return answer

        def wait_worker_quiescent(timeout=600):
            deadline = time.monotonic()+timeout
            condition = getattr(coordinator, '_condition', None)
            if condition is None:  # Only the explicit CPU fake omits worker threads.
                return
            with condition:
                while any(worker.get('busy') or worker.get('active')
                          for worker in coordinator._workers.values()):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('coordinator worker did not become quiescent')
                    condition.wait(min(remaining, .1))

        def wait_preload_callbacks(row, timeout=600):
            deadline = time.monotonic()+timeout
            for done_event in preload_waiters.pop(id(row), []):
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not done_event.wait(remaining):
                    raise TimeoutError('checkpoint preload timing callback did not finish')

        def submit_one(row, item, max_steps):
            source_snapshot = row['source_snapshot']
            coordinator.store.fork(row['branch'], source_snapshot)
            active[0] = row
            submit_time = time.monotonic()
            row['submit_monotonic'] = submit_time
            handle = coordinator.submit(row['branch'], item['suffix'], max_steps, row['operation'])
            answer = wait_request(handle, row, submit_time)
            wait_worker_quiescent()
            wait_preload_callbacks(row)
            row['counter_delta'] = answer.get('counter_deltas')
            if not isinstance(row['counter_delta'], dict):
                raise IntegrityError('worker answer omitted serialized counter deltas')
            row['pool_after'] = coordinator.metrics()
            return answer

        def run_warmup(item):
            root_index, tree_index = item['root_index'], item['tree_index']
            source_snapshot = roots[root_index]['snapshot']
            branch = 'b-'+uuid.uuid4().hex[:24]
            operation = 'op-'+uuid.uuid4().hex[:24]
            row = dict(root_index=root_index, tree_index=tree_index,
                       source_snapshot=source_snapshot, branch=branch, operation=operation,
                       repeat=-1, warmup=True,
                       suffix_ids=item['suffix'])
            answer = submit_one(row, item, 1)
            row['pool'] = row['pool_after']
            row['trace'] = trace.snapshot()
            return row

        # Read root payload hashes before native activity and prewarm one token
        # per context.  These three warmups are excluded from measured samples.
        source_fingerprints = [(row['manifest_sha256'], row['payload_sha256'])
                               for row in preflight['roots']]
        if args.arm == 'resident':
            phase('resident_root_preload')
            for root_index, row in enumerate(roots):
                phase('resident_root_preload', root_index=root_index)
                future = coordinator._preload(model, row['snapshot'], timeout=600)
                if future is not None:
                    future.result(timeout=600)
        for item in selected:
            phase('warmup', root_index=item['root_index'], repeat=-1)
            warm = run_warmup(item)
            report['warmup_samples'].append(dict(root_index=warm['root_index'],
                tree_index=warm['tree_index'], outputs=len(warm['answer']['tokens'])))
            _write_json(output/'result.json', report)

        previous_root = None
        for repeat in range(3):
            for item in selected:
                phase('measured_request', root_index=item['root_index'], repeat=repeat)
                if previous_root == item['root_index']:
                    raise AssertionError('fixed root order repeated the immediately prior context')
                source_snapshot = roots[item['root_index']]['snapshot']
                key = pool_key(source_snapshot)
                pool_before_metrics = coordinator.metrics()
                before_pool = key in coordinator.pool
                model_cache_before = source_snapshot in model.cache
                if model_cache_before:
                    raise IntegrityError('target root is present in worker per-model cache')
                removed = False
                if args.arm == 'disk' and before_pool:
                    coordinator.pool.remove(key)
                    removed = True
                after_remove = key in coordinator.pool
                pool_after_remove_metrics = coordinator.metrics()
                trace_start = trace.snapshot()['event_count']
                row = dict(root_index=item['root_index'], tree_index=item['tree_index'],
                           source_snapshot=source_snapshot, repeat=repeat,
                           arm=args.arm, paired_order=args.arm_order,
                           operation='op-'+uuid.uuid4().hex[:24],
                           branch='b-'+uuid.uuid4().hex[:24],
                           suffix_ids=item['suffix'], pool_before={'ram_key_present':before_pool,
                               'exact_remove':removed,'key_present_after_remove':after_remove,
                               'metrics':pool_before_metrics},
                           pool_after_remove_metrics=pool_after_remove_metrics,
                           model_cache_before=model_cache_before, trace_start=trace_start,
                           trace=[], pool={})
                answer = submit_one(row, item, 128)
                delta = row['counter_delta']
                trace_doc = trace.snapshot()
                if trace_doc['overflowed']:
                    raise IntegrityError('pool transition trace overflowed')
                events = [event for event in trace_doc['events'] if event['sequence'] >= trace_start]
                row['trace_events_for_request'] = events
                row['trace_identity'] = {name:trace_doc[name] for name in
                    ('phase','process_id','phase_id','event_count','overflowed')}
                row['pool'] = row['pool_after']
                row['answer'] = answer
                row['consumer_timeline'] = row.get('consumer_timeline')
                row['tier'] = classify_checkpoint_tier(key,
                    pool_contains=before_pool if args.arm == 'resident' else after_remove,
                    model_cache_contains=model_cache_before, events=events,
                    native_counters=delta)
                validate_reference_output(answer['tokens'], item['reference'],
                    execution.get('eos_tokens', ()), 128)
                validation_started = time.monotonic()
                validate_work_delta(len(item['suffix']), answer['tokens'], delta,
                                    saved_prefix_tokens=prefix_tokens)
                row['validation_seconds'] = time.monotonic()-validation_started
                if answer.get('capture_seconds') is None:
                    raise IntegrityError('answer omitted checkpoint capture timing')
                row['source_root_fingerprint'] = preflight['roots'][item['root_index']]
                report['samples'].append(row)
                _write_json(output/'result.json', report)
                active[0] = None
                previous_root = item['root_index']

        if len(report['samples']) != 9:
            raise AssertionError('arm did not produce exactly nine measured returns')
        expected_ids = set((roots[i]['snapshot'], selected[i]['tree_index'])
                           for i in range(3) for _ in range(3))
        observed_ids = set((row['source_snapshot'], row['tree_index']) for row in report['samples'])
        if observed_ids != expected_ids:
            raise AssertionError('measured samples do not cover each root three times')
        for expected, state in zip(source_fingerprints, roots):
            manifest = source_store/'checkpoints'/(state['snapshot']+'.json')
            envelope = _decode(manifest.read_bytes())
            payload = source_store/'checkpoints'/(envelope['manifest']['payload_sha256']+'.state')
            if (_sha_file(manifest)[0], _sha_file(payload)[0]) != expected:
                raise IntegrityError('frozen source root checkpoint changed during run')
        report['source_checkpoints_unchanged'] = True
        report['trace'] = trace.snapshot()
        report['executed_test_ids'] = ['P4-RETURN-IMPORT-IDENTITY',
            'P4-RETURN-ROOT-CHECKPOINT-IMMUTABLE','P4-RETURN-SUFFIX-ONLY-WORK',
            'P4-RETURN-REFERENCE-OUTPUT','P4-RETURN-RESIDENT-POOL' if args.arm=='resident'
            else 'P4-RETURN-CONTROLLED-DISK-PRELOAD']
        report['latency_summary'] = _summarize_samples(report['samples'])
        report['status'] = 'PASS'
        phase('finalizing')
    except BaseException as error:
        run_error = error
        report['failed_phase'] = report.get('phase')
        report['status'] = 'FAIL'
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            phase('cleanup')
        except BaseException as error:
            cleanup_errors.append(f'{type(error).__name__} recording cleanup phase: {error}')
        if 'coordinator' in locals():
            processes = []
            try:
                processes = [getattr(worker['model'].backend, 'process', None)
                             for worker in coordinator._workers.values() if worker['model'] is not None]
            except BaseException as error:
                cleanup_errors.append(f'{type(error).__name__} reading worker process handles: {error}')
            try:
                coordinator.close()
            except BaseException as error:
                cleanup_errors.append(f'{type(error).__name__} closing coordinator: {error}')
            try:
                exit_codes = [process.poll() for process in processes if process is not None]
                report['native_exit_codes'] = exit_codes
                report['native_exit_observed'] = all(code is not None for code in exit_codes)
            except BaseException as error:
                cleanup_errors.append(f'{type(error).__name__} checking native process exit: {error}')
                report['native_exit_observed'] = False
                report['native_exit_codes'] = []
            if report.get('status') == 'PASS' and (
                    not report.get('native_exit_observed', False) or
                    any(code != 0 for code in report.get('native_exit_codes', []))):
                report['status'] = 'FAIL'
                report['error'] = 'native worker process did not exit successfully'
        if cleanup_errors:
            report['status'] = 'FAIL'
            report['cleanup_errors'] = cleanup_errors
            if run_error is None:
                report['error'] = '; '.join(cleanup_errors)
        try:
            heartbeat.close()
        except BaseException as error:
            report['status'] = 'FAIL'
            report.setdefault('cleanup_errors', []).append(
                f'{type(error).__name__} stopping heartbeat: {error}')
            if run_error is None:
                report['error'] = report['cleanup_errors'][-1]
        try:
            _write_json(output/'result.json', report)
        except BaseException as error:
            if run_error is not None:
                run_error.add_note(f'failed to persist result.json: {type(error).__name__}: {error}')
            else:
                raise
    if run_error is not None:
        if cleanup_errors:
            run_error.add_note('cleanup failures: '+'; '.join(cleanup_errors))
        raise run_error
    if cleanup_errors:
        raise RuntimeError('coordinator cleanup failed: '+'; '.join(cleanup_errors))
    return report


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered)-1, int((len(ordered)-1)*fraction + 0.5)))
    return ordered[index]


def _summarize_samples(samples):
    metrics = ('submit_to_first_token_seconds','submit_to_snapshot_seconds',
               'submit_to_durable_seconds','producer_snapshot_to_durable_seconds')
    result = {}
    for name in metrics:
        values = [row['consumer_timeline'][name] for row in samples]
        result[name] = dict(median=statistics.median(values),p95=_percentile(values,.95))
    producer_gaps = [gap for row in samples
                     for gap in row['consumer_timeline']['producer_token_gap_seconds']]
    consumer_gaps = [gap for row in samples
                     for gap in row['consumer_timeline']['consumer_token_gap_seconds']]
    result['producer_token_gap_seconds'] = dict(
        median=statistics.median(producer_gaps) if producer_gaps else None,
        p95=_percentile(producer_gaps,.95))
    result['consumer_token_gap_seconds'] = dict(
        median=statistics.median(consumer_gaps) if consumer_gaps else None,
        p95=_percentile(consumer_gaps,.95))
    token_counts = [len(row['answer']['tokens']) for row in samples]
    result['actual_output_tokens'] = token_counts
    result['eos_completed_requests'] = sum(
        bool(row['consumer_timeline']['eos']) for row in samples)
    result['bounded_non_eos_requests'] = sum(
        len(row['answer']['tokens']) == 128 and not row['consumer_timeline']['eos']
        for row in samples)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-store', required=True, type=Path)
    parser.add_argument('--source-result', required=True, type=Path)
    parser.add_argument('--prepare', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('resident','disk'))
    parser.add_argument('--arm-order', required=True, choices=('resident,disk','disk,resident'))
    parser.add_argument('--order-seed', required=True)
    parser.add_argument('--ram-bytes', required=True, type=int)
    parser.add_argument('--disk-bytes', required=True, type=int)
    parser.add_argument('--min-available-bytes', required=True, type=int)
    return parser


def _cli_exit_code(report):
    return 0 if report.get('status') == 'PASS' else 1


def _cli_summary(report, output):
    summary = dict(status=report.get('status', 'FAIL'), output=str(output),
        measured_requests=report.get('measured_requests', len(report.get('samples', []))))
    for field in ('native_exit_observed', 'native_exit_codes', 'preexisting_report_status',
                  'error', 'cleanup_errors'):
        if field in report:
            summary[field] = report[field]
    return summary


def main(argv=None, *, runner=None, stdout=None):
    parsed = _parser().parse_args(argv)
    execute = run_arm if runner is None else runner
    stream = sys.stdout if stdout is None else stdout
    result_path = parsed.output/'result.json'
    preexisting_report = None
    if result_path.is_file():
        try:
            preexisting_report = json.loads(result_path.read_text())
        except (OSError, ValueError):
            preexisting_report = dict(status='UNREADABLE')
    try:
        report = execute(parsed)
    except BaseException as error:
        if preexisting_report is not None:
            report = dict(preexisting_report)
            report['preexisting_report_status'] = preexisting_report.get('status')
            report['status'] = 'FAIL'
            report['invocation_error'] = f'{type(error).__name__}: {error}'
            if preexisting_report.get('status') == 'PASS':
                report['error'] = report['invocation_error']
            elif 'error' not in report:
                report['error'] = report['invocation_error']
        else:
            try:
                report = json.loads(result_path.read_text())
            except (OSError, ValueError):
                report = dict(samples=[])
            previous_status = report.get('status')
            report['status'] = 'FAIL'
            if previous_status not in (None, 'FAIL'):
                report['last_written_status'] = previous_status
        report['status'] = 'FAIL'
        if preexisting_report is None and not report.get('error'):
            report['error'] = f'{type(error).__name__}: {error}'
        elif preexisting_report is not None and preexisting_report.get('status') == 'PASS':
            report['error'] = f'{type(error).__name__}: {error}'
    summary = _cli_summary(report, parsed.output)
    print(json.dumps(summary, sort_keys=True), file=stream, flush=True)
    return _cli_exit_code(report)


if __name__ == '__main__':
    raise SystemExit(main())
