"""Audit suffix-end logits from retained matching-schedule Q8 references."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API / 'src'))

VOCAB_SIZE = 248320
FRAME_BYTES = VOCAB_SIZE * 4


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return _sha_bytes(raw)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha_file_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _verified_payload(path, expected_sha256):
    actual, size = _sha_file_and_size(path)
    if actual != expected_sha256:
        raise ValueError('Canonical parent payload SHA-256 mismatch')
    return {'sha256': actual, 'size': size}


def _require_finite_f32(raw, label):
    if len(raw) % 4:
        raise ValueError(label + ' has a partial float32 value')
    for (value,) in struct.iter_unpack('<f', raw):
        if not math.isfinite(value):
            raise ValueError(label + ' contains a non-finite float32 value')


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Invalid audit JSON: ' + str(path)) from error


def _expected_chunks(requests, chunk_size):
    chunks = []
    position = 0
    for count in requests:
        remaining = count
        while remaining:
            amount = min(remaining, chunk_size)
            chunks.append([position, amount])
            position += amount
            remaining -= amount
    return chunks


def _fixture_ids(fixture, profile, branch):
    if fixture.get('model_contract_id') != profile.get('model_contract_id'):
        raise ValueError('Fixture/profile model contract mismatch')
    prefix, suffix = fixture.get('prefix'), fixture.get(branch)
    for label, ids in (('prefix', prefix), (branch, suffix)):
        if (not isinstance(ids, list) or not ids
                or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids)):
            raise ValueError('Invalid fixture token IDs: ' + label)
    return prefix, suffix


def load_suffix_reference(case_dir, profile, *, vocab_size=VOCAB_SIZE):
    """Validate retained token/schedule provenance and return both prompt frames.

    The second request frame is the logits frontier after the completed suffix
    prefill. Reading it is CPU-only; this function never starts a backend.
    """
    case = Path(case_dir)
    fixture_path = case / 'fixture.json'
    fixture_raw = fixture_path.read_bytes()
    fixture = json.loads(fixture_raw)
    result = _read_json(case / 'result.json')
    prepared = _read_json(case / 'prepare.json')
    if result.get('status') != 'PASS' or prepared.get('status') != 'PASS':
        raise ValueError('Campaign and producer must both be PASS')
    if result.get('profile_id') != profile.get('execution_profile_id'):
        raise ValueError('Campaign/profile identity mismatch')
    if result.get('fixture_sha256') != _sha_bytes(fixture_raw):
        raise ValueError('Campaign fixture hash mismatch')
    if fixture.get('model_contract_id') != profile.get('model_contract_id'):
        raise ValueError('Fixture/profile model contract mismatch')
    prefix = fixture.get('prefix')
    if (not isinstance(prefix, list) or not prefix
            or any(type(token) is not int or not 0 <= token < vocab_size for token in prefix)):
        raise ValueError('Invalid fixture prefix token IDs')
    root = prepared.get('root')
    if not isinstance(root, dict) or root.get('consumed') != len(prefix):
        raise ValueError('Saved root consumed position differs from fixture prefix')
    if root.get('history') != prefix or root.get('controller', {}).get('status') != 'idle':
        raise ValueError('Saved P history/controller differs from the idle fixture prefix')
    root_path = case / root['frontier']
    root_raw = root_path.read_bytes()
    if len(root_raw) != vocab_size * 4 or _sha_bytes(root_raw) != root.get('frontier_sha256'):
        raise ValueError('Saved root frontier is incomplete or has changed')
    _require_finite_f32(root_raw, 'Saved root frontier')
    root_save = prepared.get('root_save')
    if not isinstance(root_save, dict):
        raise ValueError('Missing durable root_save receipt')
    root_branch = root_save.get('branch')
    if (not isinstance(root_branch, dict)
            or root_branch.get('name') != 'prefix'
            or root_branch.get('revision') != 1
            or root_branch.get('tip') != root.get('snapshot')
            or root_branch.get('root') != root.get('snapshot')
            or root_save.get('snapshot') != root.get('snapshot')
            or root_save.get('payload_sha256') != prepared.get('parent_payload_sha256')
            or root_save.get('model_contract_id') != profile.get('model_contract_id')
            or root_save.get('execution_profile_id') != profile.get('execution_profile_id')):
        raise ValueError('Durable root_save receipt does not bind P to this profile and payload')
    branches = prepared.get('branches')
    if not isinstance(branches, dict):
        raise ValueError('Missing prepared branches')

    chunk_size = profile.get('execution', {}).get('prefill_chunk')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('Invalid pinned prefill chunk size')

    result_branches = {}
    expected_suffix_lengths = {}
    for branch in ('A', 'B'):
        prompt, suffix = _fixture_ids(fixture, profile, branch)
        requests = [len(prompt), len(suffix)]
        stem = case / ('fresh-' + branch + '.f32')
        token_path = case / ('fresh-' + branch + '.tokens')
        try:
            token_ids = [int(value) for value in token_path.read_text().split(',')]
        except (OSError, ValueError) as error:
            raise ValueError('Invalid reference token file for branch ' + branch) from error
        if token_ids != prompt + suffix:
            raise ValueError('Reference token IDs differ from bound fixture: ' + branch)

        schedule = _read_json(Path(str(stem) + '.schedule.json'))
        if (schedule.get('operator') != 'batched-prefill'
                or schedule.get('requests') != requests
                or schedule.get('completed_chunks') != _expected_chunks(requests, chunk_size)):
            raise ValueError('Reference batched-prefill schedule mismatch: ' + branch)
        work = _read_json(case / ('fresh-' + branch + '.stdout'))
        generated = work.get('generated')
        if (work.get('prefill_tokens') != len(prompt) + len(suffix)
                or work.get('prompt_consumed') != len(prompt) + len(suffix)
                or work.get('normal_forward_tokens') != len(generated or [])
                or work.get('consumed') != len(prompt) + len(suffix) + len(generated or [])
                or not generated):
            raise ValueError('Reference work counters do not match token schedule: ' + branch)
        if branches.get(branch, {}).get('result', {}).get('tokens') != generated:
            raise ValueError('Reference output tokens differ from canonical branch: ' + branch)

        request_path = Path(str(stem) + '.requests.f32')
        request_raw = request_path.read_bytes()
        expected_bytes = len(requests) * vocab_size * 4
        if len(request_raw) != expected_bytes:
            raise ValueError('Reference request frontier bytes are truncated or oversized: ' + branch)
        split = vocab_size * 4
        prefix_frontier = request_raw[:split]
        suffix_frontier = request_raw[split:]
        if prefix_frontier != root_raw:
            raise ValueError('Reference prefix frontier differs from saved root: ' + branch)
        result_branches[branch] = {
            'token_ids_sha256': _sha_file(token_path),
            'schedule_sha256': _sha_file(Path(str(stem) + '.schedule.json')),
            'request_frontier_file_sha256': _sha_file(request_path),
            'reference_output_count': len(generated),
            'suffix_frontier': suffix_frontier,
            'suffix_frontier_sha256': _sha_bytes(suffix_frontier),
            'prefix_frontier_sha256': _sha_bytes(prefix_frontier),
            'suffix_ids': suffix,
            'requests': requests,
        }
        expected_suffix_lengths[branch] = len(suffix)

    if expected_suffix_lengths['A'] != expected_suffix_lengths['B']:
        raise ValueError('Branch suffix lengths differ')
    return {
        'fixture': fixture,
        'fixture_sha256': _sha_bytes(fixture_raw),
        'profile_id': profile['execution_profile_id'],
        'model_contract_id': profile['model_contract_id'],
        'root_snapshot': root['snapshot'],
        'root_frontier': root_raw,
        'root_frontier_sha256': _sha_bytes(root_raw),
        'branches': result_branches,
        'requests': [len(prefix), expected_suffix_lengths['A']],
    }


def _small_store_fingerprint(store_path, root_snapshot):
    """Fingerprint all small canonical files, excluding payloads and lock files."""
    store = Path(store_path)
    record = {}
    if not store.is_dir() or store.is_symlink():
        raise ValueError('Canonical Store directory is missing or a symlink')
    directories = []
    for path in sorted(store.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink found in canonical Store: ' + str(path))
        if path.is_dir():
            directories.append(path.relative_to(store).as_posix())
            continue
        if not path.is_file():
            raise ValueError('Non-regular canonical Store entry: ' + str(path))
        relative = path.relative_to(store)
        # Checkpoint .state files are large durable tensor payloads and are
        # validated separately against their prepared receipt. Lock files are
        # coordination state, not canonical metadata, and may be touched by
        # read leases while this audit runs.
        if (relative.parts[0] == 'checkpoints' and path.suffix == '.state'):
            continue
        if path.name.endswith('.lock'):
            continue
        raw = path.read_bytes()
        record[relative.as_posix()] = {
            'size': len(raw), 'sha256': _sha_bytes(raw)}
    required = {'refs.json', 'ref-order.json',
                'objects/' + root_snapshot + '.json',
                'root-index/' + root_snapshot + '.json',
                'checkpoints/' + root_snapshot + '.json'}
    if not required.issubset(record):
        raise ValueError('Canonical Store metadata set is incomplete')
    record['$directories'] = sorted(directories)
    return record


def _payload_stat(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'inode': stat.st_ino,
            'mtime_ns': stat.st_mtime_ns}


def _runtime_import_witness(api_source, profile):
    api_source = Path(api_source).resolve()
    source_text = str(api_source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    expected_sources = profile.get('native_build', {}).get('sha256', {})
    names = (
        'context_time_travel.real',
        'context_time_travel.strata_backend',
        'context_time_travel.store',
        'context_time_travel.controller',
        'context_time_travel.execution_receipt',
        'context_time_travel.digest_attestation',
        'context_time_travel.pools',
    )
    witness = {}
    for name in names:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        try:
            origin.relative_to(api_source)
        except ValueError as error:
            raise ValueError('Runtime module imported outside pinned API source: ' + name) from error
        expected = next((digest for path, digest in expected_sources.items()
                         if Path(path).resolve() == origin), None)
        actual = _sha_file(origin)
        if expected is None or actual != expected:
            raise ValueError('Runtime module hash differs from profile build pin: ' + name)
        witness[name] = {'origin': str(origin), 'sha256': actual}
    return witness


def preflight_suffix_audit(profile_path, case_dir, *, api_source=None):
    """Validate retained CPU evidence and runtime pins without loading a model."""
    profile_path, case = Path(profile_path), Path(case_dir)
    profile = _read_json(profile_path)
    api_source = Path(api_source) if api_source is not None else API / 'src'
    references = load_suffix_reference(case, profile)
    prepared = _read_json(case / 'prepare.json')
    origins = _runtime_import_witness(api_source, profile)
    store_path = case / 'store'
    metadata = _small_store_fingerprint(store_path, references['root_snapshot'])
    parent = Path(prepared['parent_payload'])
    expected_parent = (store_path / 'checkpoints' /
                       (prepared['parent_payload_sha256'] + '.state')).resolve()
    if parent.resolve() != expected_parent:
        raise ValueError('Prepared parent payload path differs from canonical Store object')
    stat = parent.stat()
    if stat.st_size <= 0:
        raise ValueError('Canonical parent payload is empty')
    return {
        'status': 'CPU_PREFLIGHT_PASS',
        'profile_file_sha256': _sha_file(profile_path),
        'profile_id': profile['execution_profile_id'],
        'fixture_sha256': references['fixture_sha256'],
        'root_snapshot': references['root_snapshot'],
        'root_save': prepared['root_save'],
        'request_lengths': references['requests'],
        'runtime_imports': origins,
        'canonical_store_small_metadata': metadata,
        'parent_payload': {
            'path': str(parent),
            'expected_sha256': prepared['parent_payload_sha256'],
            'size': stat.st_size,
            'content_sha256_verified': False,
            'content_sha256_check': 'deferred until archive stream is terminal',
        },
    }


def audit_suffix_frontiers(profile_path, case_dir, output_dir, *, api_source=None):
    """Restore exact P and independently consume each suffix without publishing."""
    profile_path, case, output = Path(profile_path), Path(case_dir), Path(output_dir)
    profile = _read_json(profile_path)
    api_source = Path(api_source) if api_source is not None else API / 'src'
    origins = _runtime_import_witness(api_source, profile)
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import _validate_execution, _validate_model

    references = load_suffix_reference(case, profile)
    prepared = _read_json(case / 'prepare.json')
    root_snapshot = references['root_snapshot']
    store_path = case / 'store'
    before_store = _small_store_fingerprint(store_path, root_snapshot)
    parent_payload = Path(prepared['parent_payload'])
    expected_parent = (case / 'store' / 'checkpoints' /
                       (prepared['parent_payload_sha256'] + '.state')).resolve()
    if parent_payload.resolve() != expected_parent:
        raise ValueError('Prepared parent payload path differs from canonical Store object')
    parent_payload_before = _verified_payload(parent_payload, prepared['parent_payload_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    model_identity = _validate_model(profile)

    execution = profile['execution']
    build = profile.get('native_build', {})
    provenance = {
        'profile_file_sha256': _sha_file(profile_path),
        'profile_id': profile['execution_profile_id'],
        'model_contract_id': profile['model_contract_id'],
        'model_weights_sha256': profile.get('model_weights_sha256'),
        'model_hash_receipt_sha256': profile.get('model_hash_receipt_sha256'),
        'model_identity_validation': model_identity,
        'native_build_revision': build.get('revision'),
        'native_build_input_sha256_map': build.get('sha256'),
        'native_build_map_sha256': _sha_json(build.get('sha256', {})),
        'sut_argv': execution.get('sut_command'),
        'sut_argv_sha256': _sha_json(execution.get('sut_command')),
        'oracle_argv': execution.get('oracle_command'),
        'oracle_argv_sha256': _sha_json(execution.get('oracle_command')),
    }

    record = {
        'status': 'RUNNING',
        'scope': 'suffix frontier check only; existing independent fresh references; no generation',
        'profile_id': profile['execution_profile_id'],
        'provenance': provenance,
        'runtime_imports': origins,
        'fixture_sha256': references['fixture_sha256'],
        'case_dir': str(case),
        'root_snapshot': root_snapshot,
        'root_consumed': prepared['root']['consumed'],
        'parent_payload_sha256': prepared['parent_payload_sha256'],
        'reference_requests': references['requests'],
        'reference_frames': {},
        'branches': {},
        'canonical_store_before': before_store,
    }
    backend = None
    close_code = None
    try:
        backend = BranchingModel.load_backend(profile)
        record['native_pid'] = backend.process.pid
        model = BranchingModel(store_path, backend, profile)
        head = model.store.head('prefix')
        if head.tip != root_snapshot:
            raise ValueError('Canonical prefix head no longer references prepared P')
        if prepared.get('root_save', {}).get('execution_profile_id') != profile['execution_profile_id']:
            raise ValueError('Prepared root profile differs from current profile')

        for branch in ('A', 'B'):
            reference = references['branches'][branch]
            record['reference_frames'][branch] = {
                'prefix_sha256': reference['prefix_frontier_sha256'],
                'suffix_sha256': reference['suffix_frontier_sha256'],
                'request_file_sha256': reference['request_frontier_file_sha256'],
                'schedule_sha256': reference['schedule_sha256'],
                'token_ids_sha256': reference['token_ids_sha256'],
            }
            before_materialize = backend.work_counters()
            with model.materialize(root_snapshot) as (lease, prefix_frontier):
                after_materialize = backend.work_counters()
                restore_forward = after_materialize['forward_tokens'] - before_materialize['forward_tokens']
                restore_prefill = after_materialize['prefill_tokens'] - before_materialize['prefill_tokens']
                restore_consumed = after_materialize['consumed_tokens'] - before_materialize['consumed_tokens']
                if restore_forward != 0 or restore_prefill != 0 or restore_consumed != 0:
                    raise AssertionError('Restoring exact P performed forward/prefill work')
                if backend.position(lease) + 1 != len(references['fixture']['prefix']):
                    raise AssertionError('Restored P consumed position differs from prefix length')
                if any(not math.isfinite(float(value)) for value in prefix_frontier):
                    raise AssertionError('Restored P frontier contains non-finite logits')
                packed_prefix = struct.pack('<' + 'f' * len(prefix_frontier), *prefix_frontier)
                if packed_prefix != references['root_frontier']:
                    raise AssertionError('Restored P frontier differs from independent prefix frontier')

                before_suffix = backend.work_counters()
                suffix = reference['suffix_ids']
                backend.consume(lease, suffix)
                after_suffix = backend.work_counters()
                position = backend.position(lease)
                receipt = getattr(backend, 'last_execution_receipt', None)
                expected_chunks = _expected_chunks([len(suffix)], profile['execution']['prefill_chunk'])
                expected_chunks = [[len(references['fixture']['prefix']) + start, count]
                                   for start, count in expected_chunks]
                if (after_suffix['forward_tokens'] - before_suffix['forward_tokens'] != len(suffix)
                        or after_suffix['prefill_tokens'] - before_suffix['prefill_tokens'] != len(suffix)
                        or after_suffix['consumed_tokens'] - before_suffix['consumed_tokens'] != len(suffix)
                        or after_suffix['decode_timed_tokens'] - before_suffix['decode_timed_tokens'] != 0
                        or position != len(references['fixture']['prefix']) + len(suffix) - 1):
                    raise AssertionError('Suffix work counters/zero-based position differ')
                if (not isinstance(receipt, dict)
                        or receipt.get('operator') != 'batched-prefill'
                        or receipt.get('position') != len(references['fixture']['prefix'])
                        or receipt.get('count') != len(suffix)
                        or receipt.get('chunks') != expected_chunks):
                    raise AssertionError('Suffix native batch receipt differs from exact request')
                observed = struct.pack('<' + 'f' * VOCAB_SIZE, *backend.logits(lease))
                expected = reference['suffix_frontier']
                _require_finite_f32(observed, 'Observed suffix frontier')
                _require_finite_f32(expected, 'Reference suffix frontier')
                if observed != expected:
                    raise AssertionError('Suffix-end frontier differs from independent reference')
            record['branches'][branch] = {
                'status': 'PASS',
                'restore_forward_tokens': restore_forward,
                'restore_prefill_tokens': restore_prefill,
                'restore_consumed_delta': restore_consumed,
                'suffix_tokens': len(suffix),
                'forward_delta': after_suffix['forward_tokens'] - before_suffix['forward_tokens'],
                'prefill_delta': after_suffix['prefill_tokens'] - before_suffix['prefill_tokens'],
                'consumed_delta': after_suffix['consumed_tokens'] - before_suffix['consumed_tokens'],
                'decode_timed_delta': after_suffix['decode_timed_tokens'] - before_suffix['decode_timed_tokens'],
                'position_zero_based': position,
                'native_receipt': receipt,
                'frontier_sha256': _sha_bytes(observed),
                'frontier_match': True,
            }
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAIL', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if backend is not None:
            process = backend.process
            try:
                backend.close()
            except BaseException as error:
                record['status'] = 'FAIL'
                record['cleanup_error'] = type(error).__name__ + ': ' + str(error)
            finally:
                close_code = process.poll()
                record['native_exit_code'] = close_code
                record['native_exit_success'] = type(close_code) is int and close_code == 0
                if not record['native_exit_success']:
                    record['status'] = 'FAIL'
                    record['native_exit_error'] = 'Native backend did not exit with numeric code zero'
        try:
            after_store = _small_store_fingerprint(store_path, root_snapshot)
            record['canonical_store_after'] = after_store
            record['canonical_store_unchanged'] = after_store == before_store
            if after_store != before_store:
                record['status'] = 'FAIL'
                record['canonical_store_error'] = 'Canonical Store metadata changed during suffix audit'
            parent_payload_after = _verified_payload(
                parent_payload, prepared['parent_payload_sha256'])
            record['parent_payload'] = {
                'path': str(parent_payload),
                'expected_sha256': prepared['parent_payload_sha256'],
                'size': parent_payload_after['size'],
                'sha256': parent_payload_after['sha256'],
                'bytes_validated_before_and_after': True,
                'unchanged': parent_payload_after == parent_payload_before,
            }
            if parent_payload_after != parent_payload_before:
                record['status'] = 'FAIL'
                record['parent_payload_error'] = 'Canonical parent payload bytes changed during suffix audit'
        except BaseException as error:
            record['status'] = 'FAIL'
            record['post_audit_integrity_error'] = type(error).__name__ + ': ' + str(error)
        output_file = output / 'suffix-frontier-audit.json'
        temporary = output_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.replace(output_file)
    if record['status'] != 'PASS':
        raise RuntimeError('Suffix frontier audit failed')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-source', type=Path,
                        help='Pinned branching_context/src used by the profile build receipt')
    args = parser.parse_args()
    audit_suffix_frontiers(args.profile, args.case, args.output, api_source=args.api_source)


if __name__ == '__main__':
    main()
