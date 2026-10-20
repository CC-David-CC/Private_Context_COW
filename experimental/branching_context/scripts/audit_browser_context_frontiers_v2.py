#!/usr/bin/env python3
"""Compare durable browser COW answers with independent fresh native references.

The reference path receives only the fixture's rendered token IDs and the
pinned profile. It never opens the canonical Store or checkpoint payloads.
The auditor separately verifies Store lineage/receipts and compares each
durable final frontier with the corresponding fresh-reference final frontier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
import traceback

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API / 'src'))
sys.path.insert(0, str(API / 'scripts'))

VOCAB_SIZE = 248320
FRAME_BYTES = VOCAB_SIZE * 4


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_json(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=False, allow_nan=False).encode('utf-8'))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Invalid or missing JSON: ' + str(path)) from error


def _ids(value, label: str, vocab_size: int = VOCAB_SIZE) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or not 0 <= token < vocab_size for token in value)):
        raise ValueError('Invalid token IDs: ' + label)
    return value


def _finite_frame(raw: bytes, label: str, vocab_size: int = VOCAB_SIZE) -> None:
    if len(raw) != vocab_size * 4:
        raise ValueError(label + ' frontier is truncated or has the wrong vocabulary size')
    for (value,) in struct.iter_unpack('<f', raw):
        if not math.isfinite(value):
            raise ValueError(label + ' frontier contains a non-finite float32 value')


def compare_frontier(actual: bytes, expected: bytes, label: str,
                     vocab_size: int = VOCAB_SIZE) -> dict:
    """Require exact float32 bytes; return hashes and the observed max error."""
    _finite_frame(actual, label + ' durable', vocab_size)
    _finite_frame(expected, label + ' fresh reference', vocab_size)
    if actual == expected:
        maximum = 0.0
    else:
        maximum = max(abs(a - b) for (a,), (b,) in zip(
            struct.iter_unpack('<f', actual), struct.iter_unpack('<f', expected)))
        raise ValueError(f'{label} exact frontier mismatch; max absolute error={maximum!r}')
    return {'actual_sha256': _sha_bytes(actual), 'expected_sha256': _sha_bytes(expected),
            'max_abs_error': maximum, 'exact_bytes_match': True,
            'values': vocab_size}


def _context_report(report: dict, name: str) -> dict:
    rows = report.get('contexts')
    if rows is None:
        rows = report.get('reusable_context_report', {}).get('contexts')
    if not isinstance(rows, list):
        raise ValueError('Browser report has no contexts list')
    matches = [row for row in rows if isinstance(row, dict) and row.get('name') == name]
    if len(matches) != 1:
        raise ValueError('Browser report context name is missing or ambiguous: ' + name)
    row = matches[0]
    nested = row.get('report')
    if not isinstance(nested, dict):
        raise ValueError('Browser context report is missing: ' + name)
    return row, nested, rows.index(row) + 1


def _validate_store_receipt(store, op_id: str, receipt: dict, namespace_id: str,
                            label: str) -> dict:
    """Bind a captured HTTP receipt to the Store's validated durable receipt."""
    stored = store.presentation_receipt(op_id)
    expected_branch = receipt.get('branch')
    result = stored.get('result')
    presentation = stored.get('presentation')
    if (not isinstance(expected_branch, dict) or not isinstance(result, dict)
            or result.get('name') != expected_branch.get('name')
            or result.get('revision') != expected_branch.get('revision')
            or result.get('tip') != receipt.get('snapshot')
            or result.get('tip') != expected_branch.get('tip')
            or result.get('root') != expected_branch.get('root')
            or not isinstance(presentation, dict)
            or presentation.get('namespace_id') != namespace_id):
        raise ValueError('Captured receipt differs from canonical Store presentation: ' + label)
    return {'operation_id': op_id, 'digest': stored.get('digest'),
            'result': result, 'presentation_kind': presentation.get('kind'),
            'namespace_id': presentation.get('namespace_id')}


def _validate_root_store_operation(store, op_id: str, receipt: dict, namespace_id: str,
                                   root_messages: list, renderer_id: str,
                                   root_token_ids: list, label: str) -> dict:
    """Validate create's generic move receipt and its separate root annotation.

    The coordinator deliberately gives Store.fork() a separate ``root-...``
    presentation operation. The public create operation is a subsequent plain
    move, so it must be checked through the namespace catalog rather than
    presentation_receipt(op_id).
    """
    from context_time_travel.store import _json_bytes
    branch = receipt.get('branch')
    if not isinstance(branch, dict) or store.branch_namespace(branch.get('name')) != namespace_id:
        raise ValueError('Root operation branch/namespace differs from canonical Store: ' + label)
    catalog = store.presentation_catalog(namespace_id)
    events = [event for event in catalog.get('events', [])
              if event.get('operation_id') == op_id and event.get('branch') == branch.get('name')]
    if len(events) != 1:
        raise ValueError('Root operation is missing or ambiguous in canonical Store catalog: ' + label)
    event = events[0]
    result = event.get('result')
    if (event.get('type') != 'move' or event.get('digest') is not None
            or not isinstance(result, dict)
            or result.get('name') != branch.get('name')
            or result.get('revision') != branch.get('revision')
            or result.get('tip') != receipt.get('snapshot')
            or result.get('tip') != branch.get('tip')
            or result.get('root') != branch.get('root')):
        raise ValueError('Root create operation differs from canonical Store move receipt: ' + label)
    root_events = [item for item in catalog.get('events', [])
                   if item.get('branch') == branch.get('name')
                   and (item.get('presentation') or {}).get('kind') == 'root'
                   and item.get('result', {}).get('tip') == receipt.get('snapshot')]
    if len(root_events) != 1:
        raise ValueError('Root annotation is missing or ambiguous in canonical Store: ' + label)
    root_event = root_events[0]
    root_receipt = store.presentation_receipt(root_event['operation_id'])
    presentation = root_receipt.get('presentation')
    normalized = ([{'role': message.get('role'), 'content': message.get('content')}
                   for message in presentation.get('messages', [])]
                  if isinstance(presentation, dict) else None)
    if (not isinstance(presentation, dict)
            or root_receipt.get('result', {}).get('name') != branch.get('name')
            or root_receipt.get('result', {}).get('revision') != 0
            or root_receipt.get('result', {}).get('tip') != receipt.get('snapshot')
            or presentation.get('kind') != 'root'
            or presentation.get('namespace_id') != namespace_id
            or presentation.get('renderer_id') != renderer_id
            or normalized != root_messages
            or presentation.get('input_span', {}).get('start') != 0
            or presentation.get('input_span', {}).get('count') != len(root_token_ids)
            or presentation.get('input_span', {}).get('token_ids_sha256') !=
               _sha_bytes(_json_bytes(root_token_ids))):
        raise ValueError('Canonical root presentation does not match captured context: ' + label)
    return {'operation_id': op_id, 'result': result, 'namespace_id': namespace_id,
            'root_annotation_operation_id': root_receipt['operation_id'],
            'root_annotation_digest': root_receipt['digest'],
            'root_annotation_result': root_receipt['result']}


def _fixture_contexts(fixture_set_path: Path, profile: dict):
    fixture_set_path = Path(fixture_set_path).resolve()
    fixture_set = _read_json(fixture_set_path)
    if fixture_set.get('schema') != 'cow-browser-context-set-v1':
        raise ValueError('Unsupported browser fixture-set schema')
    if fixture_set.get('status') not in ('PASS_CPU_TOKENIZER_TEMPLATE_ONLY', 'PASS'):
        raise ValueError('Browser fixture-set is not a passing CPU-built fixture')
    if (fixture_set.get('execution_profile_id') not in (None, profile.get('execution_profile_id'))
            or fixture_set.get('model_contract_id') not in (None, profile.get('model_contract_id'))):
        raise ValueError('Fixture-set profile/model identity mismatch')
    contexts = fixture_set.get('contexts')
    if not isinstance(contexts, list) or len(contexts) != 3:
        raise ValueError('Expected exactly three browser contexts in fixture-set')
    loaded = []
    seen = set()
    for row in contexts:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str):
            raise ValueError('Malformed fixture-set context entry')
        name = row['name']
        if name in seen:
            raise ValueError('Duplicate fixture-set context name: ' + name)
        seen.add(name)
        fixture_path = (fixture_set_path.parent / row['fixture_file']).resolve()
        if fixture_set_path.parent not in fixture_path.parents:
            raise ValueError('Fixture file escapes the fixture-set directory')
        fixture_raw = fixture_path.read_bytes()
        fixture_sha = _sha_bytes(fixture_raw)
        if fixture_sha != row.get('fixture_sha256'):
            raise ValueError('Standalone fixture SHA-256 mismatch: ' + name)
        fixture = json.loads(fixture_raw)
        if (fixture.get('model_contract_id') != profile.get('model_contract_id')
                or fixture.get('execution_profile_id') != profile.get('execution_profile_id')):
            raise ValueError('Context fixture profile/model identity mismatch: ' + name)
        root_ids = _ids(fixture.get('root_token_ids'), name + ' root')
        root_digest = _sha_json(root_ids)
        if (fixture.get('root_token_ids_sha256') != root_digest
                or row.get('root_token_ids_sha256') not in (None, root_digest)):
            raise ValueError('Root token digest mismatch: ' + name)
        branches = fixture.get('branches')
        if not isinstance(branches, dict) or set(('A', 'B')) - set(branches):
            raise ValueError('Fixture A/B branches are missing: ' + name)
        branch_rows = {}
        for label in ('A', 'B'):
            branch = branches[label]
            suffix = _ids(branch.get('suffix_token_ids'), name + ' ' + label + ' suffix')
            digest = _sha_json(suffix)
            if branch.get('suffix_token_ids_sha256') != digest:
                raise ValueError('Question suffix token digest mismatch: ' + name + ' ' + label)
            branch_rows[label] = {'suffix_ids': suffix, 'suffix_sha256': digest}
        loaded.append({'name': name, 'fixture': fixture, 'fixture_path': fixture_path,
                       'fixture_sha256': fixture_sha, 'root_ids': root_ids,
                       'root_sha256': root_digest, 'branches': branch_rows,
                       'source_entry': row})
    return fixture_set, loaded


def _request_record(report_dir: Path, context_dir: Path, nested: dict, key: str,
                    expected_operation_id: str, expected_branch: str | None = None) -> dict:
    ref = nested.get(key)
    if not isinstance(ref, dict):
        raise ValueError('Browser request artifact receipt is missing: ' + key)
    relative = ref.get('path')
    if not isinstance(relative, str):
        raise ValueError('Browser request artifact path is malformed: ' + key)
    base = (report_dir / context_dir).resolve()
    body_path = (base / relative).resolve()
    if base not in body_path.parents:
        raise ValueError('Browser request artifact escapes its context directory')
    raw = body_path.read_bytes()
    if len(raw) != ref.get('bytes') or _sha_bytes(raw) != ref.get('sha256'):
        raise ValueError('Browser request body size/SHA differs from receipt: ' + key)
    body = json.loads(raw)
    if body.get('operation_id') != expected_operation_id:
        raise ValueError('Browser request operation ID differs from durable receipt: ' + key)
    if expected_branch is not None and body.get('branch') != expected_branch:
        raise ValueError('Browser request branch differs from durable receipt: ' + key)
    return {'path': str(body_path), 'bytes': len(raw), 'sha256': _sha_bytes(raw),
            'operation_id': expected_operation_id, 'body': body}


def _validate_browser_context(report: dict, fixture_row: dict, profile: dict,
                              report_dir: Path) -> dict:
    name = fixture_row['name']
    row, nested, context_index = _context_report(report, name)
    if row.get('fixture_sha256') != fixture_row['fixture_sha256']:
        raise ValueError('Browser report fixture digest differs: ' + name)
    if row.get('root_token_ids_sha256') != fixture_row['root_sha256']:
        raise ValueError('Browser report root token digest differs: ' + name)
    question_digests = row.get('question_token_ids_sha256')
    if not isinstance(question_digests, dict):
        raise ValueError('Browser report question token digests are missing: ' + name)
    for label in ('A', 'B'):
        if question_digests.get(label) != fixture_row['branches'][label]['suffix_sha256']:
            raise ValueError('Browser report question token digest differs: ' + name + ' ' + label)
    if row.get('renderer_id') != fixture_row['fixture'].get('renderer_id'):
        raise ValueError('Browser report renderer differs from fixture: ' + name)
    if row.get('execution_profile_id') != profile.get('execution_profile_id'):
        raise ValueError('Browser context execution profile differs from fixture: ' + name)
    if (nested.get('server_execution_profile_id_from_durable_receipts')
            not in (None, profile.get('execution_profile_id'))):
        raise ValueError('Browser context execution profile mismatch: ' + name)
    context_request = nested.get('context_request')
    if not isinstance(context_request, dict) or not isinstance(context_request.get('receipt'), dict):
        raise ValueError('Browser root durable receipt is missing: ' + name)
    root_receipt = context_request['receipt']
    root_snapshot = root_receipt.get('snapshot')
    root_payload = root_receipt.get('payload_sha256')
    root_head = root_receipt.get('branch')
    if (root_receipt.get('model_contract_id') != profile.get('model_contract_id')
            or root_receipt.get('execution_profile_id') != profile.get('execution_profile_id')
            or not isinstance(root_head, dict)
            or root_head.get('tip') != root_snapshot
            or root_head.get('root') != root_snapshot):
        raise ValueError('Browser root receipt profile/model mismatch: ' + name)
    context_artifact_dir = row.get('artifact_dir', 'context-' + str(context_index))
    request_records = {
        'context': _request_record(report_dir, context_artifact_dir, nested,
                                   'context_request', context_request.get('operation_id', '')),
    }
    expected_root_messages = fixture_row['fixture'].get('root_messages')
    if (expected_root_messages is not None
            and request_records['context']['body'].get('messages') != expected_root_messages):
        raise ValueError('Captured context request messages differ from fixture: ' + name)
    namespace_id = row.get('namespace_id')
    if (request_records['context']['body'].get('namespace_id') != namespace_id
            or request_records['context']['body'].get('branch') != root_head.get('name')):
        raise ValueError('Captured context request namespace/branch differs from root receipt: ' + name)
    branches = {}
    for label, key, request_name in (
            ('A', 'ask_a_request', 'actual-ask-a-request.json'),
            ('B', 'ask_b_request', 'actual-ask-b-request.json')):
        request = nested.get(key)
        if not isinstance(request, dict) or not isinstance(request.get('answer'), dict):
            raise ValueError('Browser durable answer receipt missing: ' + name + ' ' + label)
        durable = request.get('receipt')
        answer = request['answer']
        if not isinstance(durable, dict):
            raise ValueError('Browser answer Store receipt missing: ' + name + ' ' + label)
        durable_head = durable.get('branch')
        generated = _ids(answer.get('tokens'), name + ' ' + label + ' answer')
        snapshot = durable.get('snapshot')
        if (answer.get('snapshot') != snapshot
                or durable.get('model_contract_id') != profile.get('model_contract_id')
                or durable.get('execution_profile_id') != profile.get('execution_profile_id')
                or not isinstance(durable_head, dict)
                or durable_head.get('tip') != snapshot
                or durable_head.get('root') != root_snapshot):
            raise ValueError('Browser answer receipt identity mismatch: ' + name + ' ' + label)
        request_records[label] = _request_record(report_dir, context_artifact_dir, nested,
            key, request.get('operation_id', ''), durable_head.get('name'))
        expected_question = fixture_row['fixture']['branches'][label].get('question')
        if (expected_question is not None
                and request_records[label]['body'].get('content') != expected_question):
            raise ValueError('Captured question text differs from fixture: ' + name + ' ' + label)
        request_namespace = request_records[label]['body'].get('namespace_id')
        if request_namespace not in (None, namespace_id):
            raise ValueError('Captured question request namespace differs from fixture: ' + name + ' ' + label)
        branches[label] = {'snapshot': snapshot, 'payload_sha256': durable.get('payload_sha256'),
                           'operation_id': request.get('operation_id'),
                           'generated_ids': generated, 'durable_receipt': durable,
                           'answer': answer, 'request': request}
    return {'name': name, 'row': row, 'nested': nested, 'namespace_id': namespace_id,
            'root_snapshot': root_snapshot, 'root_payload_sha256': root_payload,
            'root_receipt': root_receipt, 'request_artifacts': request_records,
            'branches': branches}


def _canonical_checkpoint_frontier(store, snapshot_id: str, tokens: list[int],
                                   profile: dict, payload_sha256: str,
                                   label: str, vocab_size: int = VOCAB_SIZE):
    from context_time_travel.checkpoint_cache import validate_manifest
    from audit_cow_browser_context_store import _validate_profile_schedule
    from context_time_travel.store import _json_bytes, IntegrityError

    checkpoint_dir = store.root / 'checkpoints'
    envelope_path = checkpoint_dir / (snapshot_id + '.json')
    envelope = json.loads(envelope_path.read_text(encoding='utf-8'))
    manifest = envelope.get('manifest')
    if not isinstance(manifest, dict):
        raise ValueError(label + ' checkpoint manifest is malformed')
    schema = manifest.get('schema')
    if type(schema) is not int or schema not in (1, 2, 3):
        raise ValueError(label + ' checkpoint schema is unsupported')
    expected = {'schema': schema, 'snapshot': snapshot_id,
                'input_identity': _sha_bytes(_json_bytes(tokens)),
                'model_contract_id': profile['model_contract_id'],
                'execution_profile_id': profile['execution_profile_id'],
                'consumed': len(tokens), 'positions': 'absolute-zero-based'}
    try:
        manifest = validate_manifest(envelope, expected)
        _validate_profile_schedule(manifest, profile, tokens)
    except (IntegrityError, KeyError, TypeError, ValueError) as error:
        raise ValueError(label + ' checkpoint manifest validation failed') from error
    if manifest.get('payload_sha256') != payload_sha256:
        raise ValueError(label + ' checkpoint payload does not match the durable receipt')
    payload_path = checkpoint_dir / (payload_sha256 + '.state')
    payload_raw = payload_path.read_bytes()
    if _sha_bytes(payload_raw) != payload_sha256:
        raise ValueError(label + ' checkpoint payload SHA-256 mismatch')
    state_bytes = manifest.get('state_bytes')
    logits_count = manifest.get('logits_count')
    if (type(state_bytes) is not int or state_bytes <= 0 or logits_count != vocab_size
            or len(payload_raw) != state_bytes + vocab_size * 4):
        raise ValueError(label + ' checkpoint payload layout is invalid')
    frontier = payload_raw[state_bytes:]
    _finite_frame(frontier, label, vocab_size)
    return {'manifest': manifest, 'payload_sha256': payload_sha256,
            'payload_bytes': len(payload_raw), 'frontier': frontier,
            'frontier_sha256': _sha_bytes(frontier)}


def _expected_chunks(requests: list[int], chunk_size: int) -> list[list[int]]:
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


def _verify_reference_artifacts(reference_dir: Path, stem: str, ids: list[int],
                                requests: list[int], expected: dict, profile: dict,
                                vocab_size: int, label: str) -> dict:
    stem_path = Path(reference_dir) / (stem + '.f32')
    token_path = Path(reference_dir) / (stem + '.tokens')
    schedule_path = Path(str(stem_path) + '.schedule.json')
    prompt_path = Path(str(stem_path) + '.requests.f32')
    generated_path = Path(str(stem_path) + '.decode.f32')
    stdout_path = Path(reference_dir) / (stem + '.stdout')
    stderr_path = Path(reference_dir) / (stem + '.stderr')
    try:
        token_ids = [int(value) for value in token_path.read_text(encoding='utf-8').split(',')]
        schedule = _read_json(schedule_path)
        work = _read_json(stdout_path)
        prompt_raw = prompt_path.read_bytes()
        generated_raw = generated_path.read_bytes()
        stderr_path.stat()
    except (OSError, ValueError) as error:
        raise ValueError(label + ' independent fresh-reference artifacts are incomplete') from error
    if token_ids != ids:
        raise ValueError(label + ' fresh-reference input IDs differ from fixture')
    chunk_size = profile.get('execution', {}).get('prefill_chunk')
    if (type(chunk_size) is not int or chunk_size <= 0
            or schedule.get('operator') != 'batched-prefill'
            or schedule.get('requests') != requests
            or schedule.get('completed_chunks') != _expected_chunks(requests, chunk_size)):
        raise ValueError(label + ' fresh-reference request schedule differs from fixture/profile')
    generated_ids = expected.get('tokens')
    if (work.get('generated') != generated_ids
            or work.get('prefill_tokens') != len(ids)
            or work.get('prompt_consumed') != len(ids)
            or work.get('normal_forward_tokens') != len(generated_ids or [])
            or work.get('consumed') != len(ids) + len(generated_ids or [])):
        raise ValueError(label + ' fresh-reference work counters do not match output/input spans')
    prompt_count = len(requests)
    if len(prompt_raw) != prompt_count * vocab_size * 4:
        raise ValueError(label + ' fresh-reference request frontier file is truncated')
    if (not isinstance(generated_ids, list) or not generated_ids
            or len(generated_raw) != len(generated_ids) * vocab_size * 4):
        raise ValueError(label + ' fresh-reference generated frontier file is truncated')
    prompt_frames = [prompt_raw[i:i + vocab_size * 4]
                     for i in range(0, len(prompt_raw), vocab_size * 4)]
    generated_frames = [generated_raw[i:i + vocab_size * 4]
                        for i in range(0, len(generated_raw), vocab_size * 4)]
    if prompt_frames != expected.get('prompt_frontiers') or generated_frames != expected.get('generated_frontiers'):
        raise ValueError(label + ' fresh-reference vectors differ from captured reference files')
    for index, frame in enumerate(generated_frames):
        _finite_frame(frame, label + ' fresh reference generated frame ' + str(index), vocab_size)
    stderr_hash = _sha_file(stderr_path)
    command = expected.get('command')
    return {
        'input_token_file_sha256': _sha_file(token_path),
        'schedule_file_sha256': _sha_file(schedule_path),
        'request_frontier_file_sha256': _sha_file(prompt_path),
        'generated_frontier_file_sha256': _sha_file(generated_path),
        'stdout_sha256': _sha_file(stdout_path), 'stderr_sha256': stderr_hash,
        'command_sha256': _sha_json(command) if command is not None else None,
        'schedule': schedule,
        'output_count': len(generated_ids),
        'prefix_frontier_sha256': _sha_bytes(prompt_frames[0]),
        'suffix_frontier_sha256': _sha_bytes(prompt_frames[1]),
        'final_frontier_sha256': _sha_bytes(generated_frames[-1]),
    }


def _validate_profile_runtime(profile: dict) -> dict:
    """Apply the pinned process environment, then verify code/model/hardware pins."""
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    return _validate_model(profile)


def _preflight(profile_path: Path, fixture_set_path: Path, browser_report_path: Path,
               store_dir: Path, browser_artifacts_dir: Path, *, runtime_validator=None,
               store_receipt_validator=None, progress_callback=None,
               vocab_size: int = VOCAB_SIZE) -> dict:
    profile_path, report_path, store_dir = map(Path, (profile_path, browser_report_path, store_dir))
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    if report.get('status') != 'PASS' or report.get('inference_kind') != 'real':
        raise ValueError('Browser report is not a passing real-inference report')
    if (report.get('server_execution_profile_id_from_durable_receipts')
            != profile.get('execution_profile_id')
            or report.get('server_model_contract_id_from_durable_receipts')
            != profile.get('model_contract_id')):
        raise ValueError('Browser report/profile identity mismatch')
    if runtime_validator is None:
        runtime_validator = _validate_profile_runtime
    runtime_identity = runtime_validator(profile)
    if progress_callback is not None:
        progress_callback({'phase': 'validated', 'runtime_validation': runtime_identity,
                           'completed_branches': 0})
    fixture_set, fixtures = _fixture_contexts(Path(fixture_set_path), profile)
    fixture_set_digest = _sha_bytes(Path(fixture_set_path).read_bytes())
    reported_fixture_set_digest = report.get('context_set_manifest_sha256',
                                               report.get('fixture_set_sha256'))
    if (reported_fixture_set_digest is not None
            and reported_fixture_set_digest != fixture_set_digest):
        raise ValueError('Browser report fixture-set digest mismatch')
    browser_artifacts_dir = Path(browser_artifacts_dir).resolve()
    from context_time_travel.store import Store
    store = Store(store_dir)
    from audit_cow_browser_context_store import _authoritative_files_fingerprint
    canonical_before = _authoritative_files_fingerprint(store_dir)
    prepared = []
    for index, fixture_row in enumerate(fixtures, 1):
        browser = _validate_browser_context(report, fixture_row, profile, browser_artifacts_dir)
        name = fixture_row['name']
        root_ids = fixture_row['root_ids']
        root_id = browser['root_snapshot']
        lineage_root, canonical_root_ids = store.lineage(root_id)
        if lineage_root != root_id or canonical_root_ids != root_ids:
            raise ValueError('Canonical browser root history differs from fixture: ' + name)
        root_snapshot = store.get_snapshot(root_id)
        if (root_snapshot.get('parent') is not None
                or root_snapshot.get('model_contract_id') != profile['model_contract_id']
                or root_snapshot.get('controller', {}).get('status') != 'idle'):
            raise ValueError('Canonical root identity/controller mismatch: ' + name)
        root_branch_name = browser['root_receipt']['branch']['name']
        branch_namespace = store.branch_namespace(root_branch_name)
        if branch_namespace != browser['namespace_id']:
            raise ValueError('Captured root namespace differs from canonical Store branch: ' + name)
        if store_receipt_validator is None:
            store_receipt_validator = _validate_store_receipt
        root_store_receipt = _validate_root_store_operation(store,
            browser['nested']['context_request']['operation_id'], browser['root_receipt'],
            browser['namespace_id'], browser['request_artifacts']['context']['body']['messages'],
            fixture_row['fixture']['renderer_id'], root_ids, name + ' root')
        root_checkpoint = _canonical_checkpoint_frontier(store, root_id, root_ids, profile,
            browser['root_payload_sha256'], name + ' root', vocab_size)
        branch_inputs = {}
        for label in ('A', 'B'):
            suffix = fixture_row['branches'][label]['suffix_ids']
            full_prompt = root_ids + suffix
            branch = browser['branches'][label]
            answer_ids = branch['generated_ids']
            expected_history = full_prompt + answer_ids
            branch_root, canonical_history = store.lineage(branch['snapshot'])
            if branch_root != root_id or canonical_history != expected_history:
                raise ValueError('Canonical browser history differs from fixture and answer: ' + name + ' ' + label)
            snapshot = store.get_snapshot(branch['snapshot'])
            controller = snapshot.get('controller')
            if (snapshot.get('model_contract_id') != profile['model_contract_id']
                    or not isinstance(controller, dict)
                    or controller.get('status') not in ('active', 'stopped')
                    or controller.get('accepted_in_scope') != len(answer_ids)):
                raise ValueError('Canonical browser answer controller/count mismatch: ' + name + ' ' + label)
            store_answer_receipt = store_receipt_validator(store, branch['operation_id'],
                branch['durable_receipt'], browser['namespace_id'], name + ' ' + label)
            payload = _canonical_checkpoint_frontier(store, branch['snapshot'], expected_history,
                profile, branch['payload_sha256'], name + ' ' + label, vocab_size)
            branch_inputs[label] = {'suffix': suffix, 'full_prompt': full_prompt,
                'answer_ids': answer_ids, 'expected_history': expected_history,
                'branch': branch, 'payload': payload, 'snapshot': snapshot,
                'store_receipt': store_answer_receipt}
        prepared.append({'fixture': fixture_row, 'browser': browser, 'root_ids': root_ids,
                         'root_snapshot': root_id, 'root_checkpoint': root_checkpoint,
                         'root_store_receipt': root_store_receipt, 'branches': branch_inputs})
    canonical_after = _authoritative_files_fingerprint(store_dir)
    if canonical_before != canonical_after:
        raise ValueError('Auditing changed canonical references, objects, or presentation metadata')
    return {'profile': profile, 'profile_raw': profile_raw, 'report': report,
            'report_raw': report_raw, 'fixture_set': fixture_set,
            'fixture_set_digest': fixture_set_digest, 'runtime_validation': runtime_identity,
            'store': store, 'canonical_fingerprint': canonical_after, 'prepared': prepared}


def _preflight_result(preflight: dict) -> dict:
    return {
        'schema': 1,
        'status': 'PREPARED_NOT_NUMERICALLY_QUALIFIED',
        'scope': 'CPU metadata/store preflight only; no fresh reference process was launched',
        'profile_id': preflight['profile']['execution_profile_id'],
        'model_contract_id': preflight['profile']['model_contract_id'],
        'profile_sha256': _sha_bytes(preflight['profile_raw']),
        'runtime_validation': preflight['runtime_validation'],
        'fixture_set_sha256': preflight['fixture_set_digest'],
        'browser_report_sha256': _sha_bytes(preflight['report_raw']),
        'contexts': len(preflight['prepared']),
        'captured_root_requests_validated': len(preflight['prepared']),
        'captured_answer_requests_validated': len(preflight['prepared']) * 2,
        'captured_requests_validated': len(preflight['prepared']) * 3,
        'canonical_store_receipts_validated': sum(3 for _ in preflight['prepared']),
        'canonical_root_annotations_validated': len(preflight['prepared']),
        'store_manifests_validated': sum(3 for _ in preflight['prepared']),
        'canonical_metadata_unchanged': True,
        'fresh_reference_count': 0,
        'numerical_max_error': None,
    }


def audit(profile_path: Path, fixture_set_path: Path, browser_report_path: Path,
          store_dir: Path, browser_artifacts_dir: Path, fresh_reference_dir: Path,
          *, reference_runner=None, runtime_validator=None, store_receipt_validator=None,
          progress_callback=None,
          preflight_only: bool = False, vocab_size: int = VOCAB_SIZE) -> dict:
    preflight = _preflight(profile_path, fixture_set_path, browser_report_path,
        store_dir, browser_artifacts_dir, runtime_validator=runtime_validator,
        store_receipt_validator=store_receipt_validator,
        progress_callback=progress_callback, vocab_size=vocab_size)
    if preflight_only:
        return _preflight_result(preflight)
    profile, report = preflight['profile'], preflight['report']
    store = preflight['store']
    fresh_reference_dir = Path(fresh_reference_dir).resolve()
    fresh_reference_dir.mkdir(parents=True, exist_ok=False)
    from native_request_probe import reference as run_reference
    if reference_runner is None:
        reference_runner = run_reference
    from audit_cow_browser_context_store import _authoritative_files_fingerprint
    results = []
    total_max_error = 0.0
    reference_prefill_tokens = reference_decode_tokens = reference_consumed_tokens = 0
    completed_references = 0
    root_frontier_comparisons = 0
    for index, item in enumerate(preflight['prepared'], 1):
        fixture_row, root_ids = item['fixture'], item['root_ids']
        root_checkpoint = item['root_checkpoint']
        branch_results = {}
        references_for_context = {}
        for label in ('A', 'B'):
            inputs = item['branches'][label]
            suffix, full_prompt = inputs['suffix'], inputs['full_prompt']
            answer_ids, branch, payload = inputs['answer_ids'], inputs['branch'], inputs['payload']
            if progress_callback is not None:
                progress_callback({'phase': 'fresh-reference', 'context': fixture_row['name'],
                                   'branch': label, 'completed_branches': completed_references})
            reference_dir = Path(fresh_reference_dir) / ('context-' + str(index) + '-' + label)
            reference_dir.mkdir(mode=0o700)
            expected = reference_runner(profile, reference_dir, 'fresh', full_prompt,
                [len(root_ids), len(suffix)], report.get('output_limit', len(answer_ids)))
            ref_artifacts = _verify_reference_artifacts(reference_dir, 'fresh', full_prompt,
                [len(root_ids), len(suffix)], expected, profile, vocab_size,
                fixture_row['name'] + ' ' + label)
            reference_prefill_tokens += expected['work']['prefill_tokens']
            reference_decode_tokens += expected['work']['normal_forward_tokens']
            reference_consumed_tokens += expected['work']['consumed']
            if expected['tokens'] != answer_ids:
                raise ValueError('Browser answer token IDs differ from independent fresh reference: '
                                 + fixture_row['name'] + ' ' + label)
            frontier_result = compare_frontier(payload['frontier'],
                expected['generated_frontiers'][-1], fixture_row['name'] + ' ' + label, vocab_size)
            total_max_error = max(total_max_error, frontier_result['max_abs_error'])
            root_compare = compare_frontier(root_checkpoint['frontier'],
                expected['prompt_frontiers'][0], fixture_row['name'] + ' root ' + label, vocab_size)
            root_frontier_comparisons += 1
            references_for_context[label] = expected
            branch_results[label] = {
                'snapshot': branch['snapshot'], 'payload_sha256': payload['payload_sha256'],
                'payload_bytes': payload['payload_bytes'], 'token_count': len(answer_ids),
                'token_ids_sha256': _sha_json(answer_ids),
                'reference_tokens_sha256': _sha_json(expected['tokens']),
                'request_schedule': [len(root_ids), len(suffix)],
                'reference_work': {key: expected['work'].get(key) for key in
                    ('prefill_tokens', 'normal_forward_tokens', 'consumed', 'generated')},
                'independent_reference_artifacts': ref_artifacts,
                'durable_final_frontier': frontier_result,
                'root_frontier_vs_reference': root_compare,
                'suffix_frontier_coverage': 'not-retained-as-a-canonical-snapshot',
                'intermediate_generated_frontier_coverage': 0,
            }
            completed_references += 1
            if progress_callback is not None:
                progress_callback({'phase': 'complete', 'context': fixture_row['name'],
                                   'branch': label, 'completed_branches': completed_references,
                                   'comparison': {'context': fixture_row['name'], 'branch': label,
                                      'token_ids_sha256': _sha_json(expected['tokens']),
                                      'root_frontier_sha256': root_compare['expected_sha256'],
                                      'final_frontier_sha256': frontier_result['expected_sha256']}})
        # An independent third A run checks exact repeatability for the same prompt/schedule.
        a_input = item['branches']['A']
        repeat_dir = Path(fresh_reference_dir) / ('context-' + str(index) + '-A-repeat')
        repeat_dir.mkdir(mode=0o700)
        if progress_callback is not None:
            progress_callback({'phase': 'fresh-reference-repeat', 'context': fixture_row['name'],
                               'branch': 'A-repeat', 'completed_branches': completed_references})
        repeat = reference_runner(profile, repeat_dir, 'repeat', a_input['full_prompt'],
            [len(root_ids), len(a_input['suffix'])],
            report.get('output_limit', len(a_input['answer_ids'])))
        repeat_artifacts = _verify_reference_artifacts(repeat_dir, 'repeat', a_input['full_prompt'],
            [len(root_ids), len(a_input['suffix'])], repeat, profile, vocab_size,
            fixture_row['name'] + ' A-repeat')
        first_a = references_for_context['A']
        if repeat['tokens'] != first_a['tokens']:
            raise ValueError('Independent A-repeat token IDs are not exact: ' + fixture_row['name'])
        if (repeat['prompt_frontiers'] != first_a['prompt_frontiers']
                or repeat['generated_frontiers'] != first_a['generated_frontiers']):
            raise ValueError('Independent A-repeat frontier bytes are not exact: ' + fixture_row['name'])
        reference_prefill_tokens += repeat['work']['prefill_tokens']
        reference_decode_tokens += repeat['work']['normal_forward_tokens']
        reference_consumed_tokens += repeat['work']['consumed']
        branch_results['A']['independent_repeat'] = {'exact_tokens': True,
            'exact_frontier_frames': True, 'artifacts': repeat_artifacts,
            'work': repeat['work']}
        completed_references += 1
        if progress_callback is not None:
            progress_callback({'phase': 'complete', 'context': fixture_row['name'],
                'branch': 'A-repeat', 'completed_branches': completed_references,
                'comparison': {'context': fixture_row['name'], 'branch': 'A-repeat',
                    'token_ids_sha256': _sha_json(repeat['tokens']),
                    'final_frontier_sha256': repeat_artifacts['final_frontier_sha256']}})
        results.append({'name': fixture_row['name'], 'fixture_sha256': fixture_row['fixture_sha256'],
                        'root_snapshot': item['root_snapshot'], 'root_token_count': len(root_ids),
                        'root_payload_sha256': root_checkpoint['payload_sha256'],
                        'root_frontier_sha256': root_checkpoint['frontier_sha256'],
                        'branches': branch_results})
    canonical_after = _authoritative_files_fingerprint(store_dir)
    if preflight['canonical_fingerprint'] != canonical_after:
        raise ValueError('Auditing changed canonical references, objects, or presentation metadata')
    return {
        'schema': 1,
        'status': 'PASS_EXACT_BROWSER_TOKENS_AND_DURABLE_FINAL_FRONTIERS',
        'scope': 'three-context browser COW results compared with independent fresh native references; canonical token lineage and durable final checkpoint frontiers validated; no history was repacked into reference from Store',
        'profile_id': profile['execution_profile_id'],
        'model_contract_id': profile['model_contract_id'],
        'profile_sha256': _sha_bytes(preflight['profile_raw']),
        'runtime_validation': preflight['runtime_validation'],
        'auditor_sha256': _sha_file(Path(__file__)),
        'native_reference_script_sha256': _sha_file(API / 'scripts' / 'native_request_probe.py'),
        'fixture_set_sha256': preflight['fixture_set_digest'],
        'browser_report_sha256': _sha_bytes(preflight['report_raw']),
        'canonical_metadata_unchanged': True,
        'contexts': results,
        'context_count': len(results),
        'frontier_coverage': {
            'distinct_root_prefix_frontier_states': len(results),
            'root_prefix_frontier_comparisons': root_frontier_comparisons,
            'durable_final_answer_frontiers_compared': len(results) * 2,
            'independent_A_repeat_token_and_frontier_checks': len(results),
            'suffix_frontiers_retained': 0,
            'intermediate_generated_frontiers_retained': 0,
            'limitations': 'Suffix-end and intermediate generated frontiers are not retained as separate canonical snapshots in the browser run; only root and durable final-answer checkpoint frontiers are compared.'},
        'numerical_max_error': total_max_error,
        'sut_forward_evaluations': 0,
        'fresh_reference_work': {
            'prefill_tokens': reference_prefill_tokens,
            'decode_tokens': reference_decode_tokens,
            'forward_evaluations': reference_prefill_tokens + reference_decode_tokens,
            'consumed_tokens': reference_consumed_tokens,
            'reference_count': completed_references,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fixture-set', type=Path, required=True)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--browser-artifacts-dir', type=Path, required=True)
    parser.add_argument('--store-dir', type=Path, required=True)
    parser.add_argument('--fresh-reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true',
                        help='validate profile/model/store/report bindings without launching a fresh reference')
    args = parser.parse_args()
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    recorder = _ProgressRecorder(args.output.with_suffix(args.output.suffix + '.progress.json'),
                                 args.output)
    recorder.start()
    try:
        result = audit(args.profile, args.fixture_set, args.browser_report,
                       args.store_dir, args.browser_artifacts_dir,
                       args.fresh_reference_dir, progress_callback=recorder.update,
                       preflight_only=args.preflight_only)
    except BaseException as error:
        recorder.fail(error)
        print(json.dumps({'status': 'FAIL', 'error': str(error),
                          'completed_branches': recorder.state['completed_branches']}), file=sys.stderr)
        return 1
    else:
        recorder.finish(result)
        summary = {'status': result['status']}
        if 'context_count' in result:
            summary['context_count'] = result['context_count']
        elif 'contexts' in result:
            summary['context_count'] = result['contexts']
        if 'frontier_coverage' in result:
            summary['frontier_coverage'] = result['frontier_coverage']
        summary['numerical_max_error'] = result.get('numerical_max_error')
        print(json.dumps(summary))
        return 0
    finally:
        recorder.stop()


class _ProgressRecorder:
    """Persist bounded progress and a 30-second heartbeat for long references."""
    def __init__(self, progress_path: Path, result_path: Path):
        self.progress_path = Path(progress_path)
        self.result_path = Path(result_path)
        self.started = time.time()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state = {'schema': 1, 'status': 'RUNNING', 'started_unix': self.started,
                      'phase': 'preflight', 'completed_branches': 0, 'progress_count': 0,
                      'completed_comparisons': []}
        self.thread = None

    def _atomic_write(self, path: Path, value: dict):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _save(self):
        self.state['updated_unix'] = time.time()
        self.state['elapsed_seconds'] = self.state['updated_unix'] - self.started
        self._atomic_write(self.progress_path, self.state)

    def start(self):
        with self.lock:
            self._save()
        self.thread = threading.Thread(target=self._heartbeat, name='audit-heartbeat', daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop_event.wait(30):
            with self.lock:
                self.state['heartbeat'] = True
                self._save()

    def update(self, update: dict):
        with self.lock:
            update = dict(update)
            comparison = update.pop('comparison', None)
            if comparison is not None:
                self.state['completed_comparisons'].append(comparison)
            self.state.update(update)
            self.state['progress_count'] += 1
            self.state.pop('heartbeat', None)
            self._save()

    def finish(self, result: dict):
        with self.lock:
            result_bytes = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode('utf-8')
            self.state.update(status=result['status'], completed_branches=
                              result.get('fresh_reference_work', {}).get('reference_count',
                                  result.get('captured_requests_validated', 0)),
                              phase='complete', result_sha256=_sha_bytes(result_bytes))
            self._save()
            self._atomic_write(self.result_path, result)

    def fail(self, error: BaseException):
        with self.lock:
            failed = dict(self.state, status='FAIL', error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc(),
                          phase='failed', partial_result=True)
            self._atomic_write(self.result_path, failed)
            self.state.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                              phase='failed')
            self._save()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


if __name__ == '__main__':
    raise SystemExit(main())
