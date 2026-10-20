#!/usr/bin/env python3
"""Read-only post-service audit of a browser context root and its durable bytes.

Run after the temporary server is stopped. This opens no model backend and does
not issue native work; it checks the canonical token lineage and checkpoint
payload against the browser run's durable root receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authoritative_files_fingerprint(store_dir: Path) -> str:
    """Fingerprint active references, order, canonical objects, and presentations."""
    rows = []
    for relative in ('refs.json', 'ref-order.json', 'objects', 'presentation-namespaces',
                     'presentation-bindings'):
        path = store_dir / relative
        if path.is_file():
            rows.append((relative, sha(path.read_bytes())))
    # Legacy migration publishes references-<sha256>; include every strictly
    # named reference directory so the active format and migrated layouts are
    # both covered even when inspection starts before Store has selected one.
    for path in sorted(store_dir.iterdir()):
        if path.is_dir() and re.fullmatch(r'references(?:-[0-9a-f]{64})?', path.name):
            for child in sorted(path.rglob('*')):
                if child.is_file():
                    rows.append((child.relative_to(store_dir).as_posix(), sha(child.read_bytes())))
    for relative in ('objects', 'presentation-namespaces', 'presentation-bindings'):
        path = store_dir / relative
        if path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file():
                    rows.append((child.relative_to(store_dir).as_posix(), sha(child.read_bytes())))
    return sha(json.dumps(rows, separators=(',', ':')).encode('utf-8'))


def _validate_profile_schedule(manifest, profile, tokens):
    execution = profile.get('execution', {})
    resume_contract = execution.get('resume_contract')
    if resume_contract in (None, ''):
        if manifest.get('schema') != 1 or 'execution_schedule' in manifest:
            raise ValueError('checkpoint schedule does not match the profile resume contract')
        return
    from context_time_travel.execution_receipt import (
        schedule_digest, validate_operator_schedule, validate_schedule,
    )
    if resume_contract == 'identical-native-batches-v1':
        expected_schema, validator = 2, validate_schedule
    elif resume_contract == 'identical-native-operators-v1':
        expected_schema, validator = 3, validate_operator_schedule
    else:
        raise ValueError('profile has an unsupported resume contract for this audit')
    if manifest.get('schema') != expected_schema:
        raise ValueError('checkpoint schema differs from the profile resume contract')
    schedule = manifest.get('execution_schedule')
    digest = validator(schedule, tokens, execution.get('prefill_chunk'))
    if digest != manifest.get('schedule_sha256') or digest != schedule_digest(schedule):
        raise ValueError('checkpoint operator schedule digest/history/positions are invalid')


def _validate_root(store, root_id, model_contract_id, fixture_token_ids):
    lineage_root, tokens = store.lineage(root_id)
    if lineage_root != root_id:
        raise ValueError('canonical snapshot is not its own root lineage')
    snapshot = store.get_snapshot(root_id)
    if snapshot.get('model_contract_id') != model_contract_id:
        raise ValueError('canonical root model contract differs from the audited profile')
    if list(tokens) != fixture_token_ids:
        raise ValueError('canonical root token IDs differ from the browser fixture')
    if snapshot.get('parent') is not None or snapshot.get('controller', {}).get('status') != 'idle':
        raise ValueError('context root is not a root snapshot with an idle controller')
    return snapshot, tokens


def _validate_controller(snapshot, label):
    from context_time_travel.controller import _validated_state
    from context_time_travel.store import IntegrityError

    controller = snapshot.get('controller')
    if not isinstance(controller, dict):
        raise ValueError(f'{label} controller is missing or malformed')
    try:
        validated = _validated_state(controller)
    except (IntegrityError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{label} controller failed structural validation') from error
    status = controller.get('status')
    if status not in {'idle', 'active', 'stopped'} or validated != controller:
        raise ValueError(f'{label} controller failed structural validation')
    from context_time_travel.store import _json_bytes
    return status, sha(_json_bytes(controller))


def _audit_checkpoint(store, validate_manifest, snapshot_id, tokens, model_contract_id,
                      execution_profile_id, profile, expected_payload, label):
    from context_time_travel.store import _json_bytes
    from context_time_travel.store import IntegrityError

    checkpoint_dir = store.root / 'checkpoints'
    envelope = json.loads((checkpoint_dir / (snapshot_id + '.json')).read_text(encoding='utf-8'))
    untrusted = envelope.get('manifest')
    if not isinstance(untrusted, dict):
        raise ValueError(f'{label} checkpoint manifest is missing')
    expected = {
        'schema': untrusted.get('schema'),
        'snapshot': snapshot_id,
        'input_identity': sha(_json_bytes(list(tokens))),
        'model_contract_id': model_contract_id,
        'execution_profile_id': execution_profile_id,
        'consumed': len(tokens),
        'positions': 'absolute-zero-based',
    }
    try:
        manifest = validate_manifest(envelope, expected)
        _validate_profile_schedule(manifest, profile, list(tokens))
    except (IntegrityError, KeyError, TypeError) as error:
        raise ValueError(f'{label} checkpoint manifest or schedule validation failed') from error
    if manifest.get('payload_sha256') != expected_payload:
        raise ValueError(f'{label} checkpoint digest differs from durable receipt')
    payload_path = checkpoint_dir / (manifest['payload_sha256'] + '.state')
    with payload_path.open('rb') as stream:
        payload_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if payload_digest != manifest['payload_sha256']:
        raise ValueError(f'{label} checkpoint file SHA-256 differs from its manifest')
    expected_size = manifest['state_bytes'] + 4 * manifest['logits_count']
    if payload_path.stat().st_size != expected_size:
        raise ValueError(f'{label} checkpoint size differs from manifest layout')
    with payload_path.open('rb') as stream:
        stream.seek(manifest['state_bytes'])
        raw_frontier = stream.read(4 * manifest['logits_count'])
    if len(raw_frontier) != 4 * manifest['logits_count']:
        raise ValueError(f'{label} checkpoint frontier is truncated')
    frontier = struct.unpack('<' + 'f' * manifest['logits_count'], raw_frontier)
    if not all(math.isfinite(value) for value in frontier):
        raise ValueError(f'{label} checkpoint frontier contains non-finite values')
    return manifest, payload_digest, expected_size


def _validate_branch(store, validate_manifest, label, request, branch_fixture, root_id,
                     root_tokens, model_contract_id, execution_profile_id, profile):
    if not isinstance(request, dict) or not isinstance(branch_fixture, dict):
        raise ValueError(f'branch {label} receipt or fixture is missing')
    answer = request.get('answer')
    receipt = request.get('receipt')
    if not isinstance(answer, dict) or not isinstance(receipt, dict):
        raise ValueError(f'branch {label} durable receipt is malformed')
    branch_head = receipt.get('branch')
    branch_name = request.get('branch')
    if branch_name is None and isinstance(branch_head, dict):
        branch_name = branch_head.get('name')
    snapshot_id = receipt.get('snapshot')
    answer_snapshot = answer.get('snapshot')
    suffix = branch_fixture.get('suffix_token_ids')
    generated = answer.get('tokens')
    if (not isinstance(branch_head, dict) or not isinstance(snapshot_id, str)
            or answer_snapshot != snapshot_id
            or not isinstance(branch_name, str)
            or branch_head.get('name') != branch_name
            or branch_head.get('tip') != snapshot_id
            or type(branch_head.get('revision')) is not int
            or branch_head['revision'] < 1
            or not isinstance(suffix, list) or any(type(token) is not int for token in suffix)
            or not isinstance(generated, list) or any(type(token) is not int for token in generated)):
        raise ValueError(f'branch {label} durable receipt or token spans are malformed')
    if branch_head.get('root') != root_id:
        raise ValueError(f'branch {label} canonical root differs from the context root')
    expected_tokens = list(root_tokens) + list(suffix) + list(generated)
    lineage_root, tokens = store.lineage(snapshot_id)
    if lineage_root != root_id:
        raise ValueError(f'branch {label} canonical root differs from the context root')
    if tokens != expected_tokens:
        raise ValueError(f'branch {label} canonical history differs from root, suffix, and answer tokens')
    snapshot = store.get_snapshot(snapshot_id)
    if snapshot.get('model_contract_id') != model_contract_id:
        raise ValueError(f'branch {label} model contract differs from the profile')
    controller_status, controller_sha = _validate_controller(snapshot, f'branch {label}')
    if receipt.get('model_contract_id') != model_contract_id:
        raise ValueError(f'branch {label} receipt model contract differs from the profile')
    if receipt.get('execution_profile_id') != execution_profile_id:
        raise ValueError(f'branch {label} receipt execution profile differs from the profile')
    manifest, payload_sha, payload_bytes = _audit_checkpoint(
        store, validate_manifest, snapshot_id, tokens, model_contract_id,
        execution_profile_id, profile, receipt.get('payload_sha256'), f'branch {label}')
    return {
        'branch': branch_name,
        'snapshot': snapshot_id,
        'canonical_root': lineage_root,
        'revision': branch_head['revision'],
        'token_count': len(tokens),
        'token_ids_sha256': sha(json.dumps(tokens, separators=(',', ':')).encode()),
        'input_identity_sha256': manifest['input_identity'],
        'consumed': manifest['consumed'],
        'model_contract_id': model_contract_id,
        'execution_profile_id': execution_profile_id,
        'suffix_token_count': len(suffix),
        'generated_token_count': len(generated),
        'controller_status': controller_status,
        'controller_sha256': controller_sha,
        'checkpoint_schema': manifest['schema'],
        'checkpoint_file_bytes': payload_bytes,
        'checkpoint_state_bytes': manifest['state_bytes'],
        'frontier_values': manifest['logits_count'],
        'checkpoint_sha256': payload_sha,
        'schedule_sha256': manifest.get('schedule_sha256'),
    }


def audit(repo_root: Path, store_dir: Path, profile_path: Path,
          browser_report_path: Path, fixture_path: Path) -> dict:
    repo_root, store_dir = repo_root.resolve(), store_dir.resolve()
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    report = json.loads(browser_report_path.read_text(encoding='utf-8'))
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    if report.get('status') != 'PASS' or not report.get('reusable_context'):
        raise ValueError('browser report is not a passing reusable-context run')
    reusable = report.get('reusable_context_report')
    nested_identity = (reusable.get('server_receipt_identity')
                       if isinstance(reusable, dict) else None)
    top_identity = report.get('server_receipt_identity')
    server_identity = nested_identity if nested_identity is not None else top_identity
    expected_execution = profile.get('execution_profile_id')
    expected_model = profile.get('model_contract_id')
    if (not isinstance(server_identity, dict)
            or server_identity.get('execution_profile_id') != expected_execution
            or server_identity.get('model_contract_id') != expected_model
            or (nested_identity is not None and top_identity is not None
                and nested_identity != top_identity)
            or report.get('server_execution_profile_id_from_durable_receipts') != expected_execution
            or report.get('server_model_contract_id_from_durable_receipts') != expected_model
            or ('server_execution_profile_id_caller_supplied' in report
                and report['server_execution_profile_id_caller_supplied'] != expected_execution)):
        raise ValueError('browser durable server receipts differ from the audited profile')
    if fixture.get('model_contract_id') != profile.get('model_contract_id'):
        raise ValueError('fixture model contract differs from the audited profile')
    root_id = report['reusable_context_report']['root_snapshot']
    expected_payload = report['reusable_context_report']['root_payload_sha256']
    sys.path.insert(0, str(repo_root / 'experimental' / 'branching_context' / 'src'))
    from context_time_travel.store import Store, _json_bytes
    from context_time_travel.checkpoint_cache import validate_manifest

    if not (store_dir / 'refs.json').is_file():
        raise FileNotFoundError('Expected an existing COW store; refusing to create one')
    before = _authoritative_files_fingerprint(store_dir)
    store = Store(store_dir)
    snapshot, tokens = _validate_root(store, root_id, profile['model_contract_id'],
                                      fixture['root_token_ids'])
    root_controller_status, root_controller_sha = _validate_controller(snapshot, 'context root')
    if root_controller_status != 'idle':
        raise ValueError('context root controller is not idle')
    from context_time_travel.checkpoint_cache import validate_manifest
    root_manifest, payload_digest, expected_size = _audit_checkpoint(
        store, validate_manifest, root_id, tokens, profile['model_contract_id'],
        profile['execution_profile_id'], profile, expected_payload, 'context root')

    branch_fixture = fixture.get('branches')
    if not isinstance(branch_fixture, dict):
        raise ValueError('A/B branch suffix fixture is missing')
    reusable_report = report['reusable_context_report']
    branches = {}
    for label, key in (('A', 'ask_a_request'), ('B', 'ask_b_request')):
        branches[label] = _validate_branch(
            store, validate_manifest, label, reusable_report.get(key),
            branch_fixture.get(label), root_id, tokens, profile['model_contract_id'],
            profile['execution_profile_id'], profile)
    if (branches['A']['branch'] == branches['B']['branch']
            or branches['A']['snapshot'] == branches['B']['snapshot']):
        raise ValueError('A/B branches must have distinct canonical tips')
    if branches['A']['canonical_root'] != branches['B']['canonical_root']:
        raise ValueError('A/B branches do not share the context root')
    after = _authoritative_files_fingerprint(store_dir)
    if before != after:
        raise ValueError('Store initialization or audit changed canonical refs/objects/presentation records')
    return {
        'schema': 1,
        'status': 'PASS_READ_ONLY_DURABLE_CONTEXT_AND_BRANCH_AUDIT',
        'scope': 'root and A/B canonical token histories, branch controller structural validation, complete checkpoint file digests/manifests and profile schedule validation; no model load or GPU work',
        'store_initialization': 'Store initializer may validate/ensure lock plumbing; before/after fingerprint confirms no refs, objects, or presentation records changed',
        'snapshot': root_id,
        'token_count': len(tokens),
        'token_ids_sha256': sha(json.dumps(list(tokens), separators=(',', ':')).encode()),
        'fixture_token_count': fixture['root_token_count'],
        'fixture_token_ids_sha256': fixture['root_token_ids_sha256'],
        'controller_status': root_controller_status,
        'controller_sha256': root_controller_sha,
        'checkpoint_file_bytes': expected_size,
        'checkpoint_state_bytes': root_manifest['state_bytes'],
        'frontier_values': root_manifest['logits_count'],
        'checkpoint_sha256': payload_digest,
        'receipt_sha256_match': True,
        'branches': branches,
        'independent_controller_continuation_equivalence': 'not_claimed',
        'model_loaded': False,
        'native_forward_work': 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--store-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo_root, args.store_dir, args.profile,
                   args.browser_report, args.fixture)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
