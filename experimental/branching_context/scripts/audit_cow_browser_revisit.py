#!/usr/bin/env python3
"""Compare one cold-restored browser resume with a fresh native reference.

This is a separate, read-only audit for a durable A128 -> resume A128 turn.
The reference receives only fixture root/question IDs with the same two
batched-Prefill request boundaries; it never reads the canonical Store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API / 'src'))
sys.path.insert(0, str(API / 'scripts'))

import audit_browser_context_frontiers_v2 as base


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value, raw


def _select_oldest_context(report: dict) -> tuple[dict, dict, dict]:
    """Resolve the report row whose first A answer is the resumed saved tip."""
    container = report.get('reusable_context_report')
    if not isinstance(container, dict):
        raise ValueError('Browser report is missing reusable_context_report')
    revisit = container.get('revisited_oldest_branch')
    rows = container.get('contexts')
    if not isinstance(revisit, dict) or not isinstance(rows, list):
        raise ValueError('Browser report lacks its revisit record or context rows')
    request = revisit.get('request')
    if not isinstance(request, dict):
        raise ValueError('Revisit request record is missing')
    source_tip = revisit.get('source_tip')
    branch_name = request.get('branch')
    matches = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('report'), dict):
            continue
        ask = row['report'].get('ask_a_request')
        durable = ask.get('receipt') if isinstance(ask, dict) else None
        branch = durable.get('branch') if isinstance(durable, dict) else None
        if (isinstance(branch, dict) and branch.get('tip') == source_tip
                and branch.get('name') == branch_name):
            matches.append((row, row['report'], revisit))
    if len(matches) != 1:
        raise ValueError('Revisit source does not identify exactly one first-A browser answer')
    return matches[0]


def _safe_artifact(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Browser evidence path must be a relative path')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('Browser evidence path escapes its artifact directory')
    if not path.is_file() or path.is_symlink():
        raise ValueError('Browser evidence file is missing or not a regular file: ' + relative)
    return path


def _validate_resume_transition(*, root_ids: list[int], suffix_ids: list[int],
                                first_tokens: list[int], source_history: list[int],
                                final_history: list[int], request: dict, ack: dict,
                                source_controller: dict, final_controller: dict,
                                counters: dict, source_branch: str,
                                source_revision: int, source_tip: str,
                                final_tip: str, final_parent: str,
                                report_generated: int, policy: dict) -> list[int]:
    """Validate the canonical A128 -> A256 transition and exact resumed delta."""
    expected_source = root_ids + suffix_ids + first_tokens
    if source_history != expected_source:
        raise ValueError('Saved first-A canonical history differs from fixture/answer tokens')
    if final_history[:len(expected_source)] != expected_source:
        raise ValueError('Resumed canonical history changed its saved first-A prefix')
    resumed = final_history[len(expected_source):]
    if (len(first_tokens) != 128 or report_generated != 128 or len(resumed) != 128):
        raise ValueError('Expected exactly 128 first-A and 128 resumed tokens')
    if final_parent != source_tip:
        raise ValueError('Resume snapshot is not a direct append to the saved first-A tip')
    if (request.get('branch') != source_branch or request.get('revision') != source_revision
            or type(request.get('max_tokens')) is not int or request['max_tokens'] != 128
            or request.get('stream') is not True):
        raise ValueError('Resume request does not target the saved active A branch')
    if (ack.get('operation_id') != request.get('operation_id')
            or ack.get('branch') != source_branch
            or ack.get('response_status') != 200 or ack.get('eof_consumed') is not True):
        raise ValueError('Resume acknowledgement differs from the captured request')
    receipt = ack.get('receipt')
    if (not isinstance(receipt, dict) or receipt.get('branch') != source_branch
            or receipt.get('revision') != source_revision + 1
            or receipt.get('snapshot') != final_tip):
        raise ValueError('Resume durable acknowledgement differs from canonical Store tip')
    if (source_controller.get('status') != 'active'
            or final_controller.get('status') != 'active'
            or source_controller.get('schema') != 1 or final_controller.get('schema') != 1
            or source_controller.get('stream_key') != final_controller.get('stream_key')
            or source_controller.get('policy') != final_controller.get('policy')
            or source_controller.get('next_draw') != 0 or final_controller.get('next_draw') != 0
            or source_controller.get('accepted_in_scope') != len(first_tokens)
            or final_controller.get('accepted_in_scope') != len(first_tokens) + len(resumed)
            or source_controller.get('policy') != policy):
        raise ValueError('Resume changed or lost its active greedy controller scope')
    sampler = source_controller.get('policy')
    if (not isinstance(sampler, dict) or sampler.get('temperature') != 0.0
            or sampler.get('grammar') is not None or sampler.get('stop_sequences') != []):
        raise ValueError('Saved turn is not the supported greedy, no-stop scope')
    integer_fields = ('prefill_tokens', 'forward_tokens', 'consumed_tokens',
                      'decode_timed_tokens', 'loaded_file_bytes',
                      'checkpoint_import_bytes', 'integrity_mismatches')
    if any(type(counters.get(key)) is not int for key in integer_fields):
        raise ValueError('Resume counters are missing exact integer values')
    if (counters['prefill_tokens'] != 0 or counters['forward_tokens'] != len(resumed)
            or counters['consumed_tokens'] != len(resumed)
            or counters['decode_timed_tokens'] != len(resumed)
            or counters['loaded_file_bytes'] <= 0
            or counters['checkpoint_import_bytes'] <= 0
            or counters['integrity_mismatches'] != 0):
        raise ValueError('Resume did not perform the expected disk reload and suffix-only decode work')
    return resumed


def _load_resume_evidence(report: dict, browser_dir: Path) -> dict:
    row, nested, revisit = _select_oldest_context(report)
    request = revisit['request']
    evidence = revisit.get('evidence')
    if not isinstance(evidence, dict):
        raise ValueError('Revisit evidence map is missing')
    request_path = _safe_artifact(browser_dir, evidence.get('request'))
    request_body, request_raw = _json(request_path)
    if request_body != request:
        raise ValueError('Captured revisit request body differs from the browser report')
    client_stream_path = _safe_artifact(browser_dir, evidence.get('client_stream'))
    stream_raw = client_stream_path.read_bytes()
    ack = revisit.get('client_ack')
    evidence_ack = evidence.get('client_ack')
    if (not isinstance(ack, dict) or len(stream_raw) != ack.get('raw_ndjson_bytes')
            or _sha(stream_raw) != ack.get('raw_ndjson_sha256')
            or not isinstance(evidence_ack, dict)
            or evidence_ack.get('sha256') != _sha(stream_raw)
            or evidence_ack.get('bytes') != len(stream_raw)):
        raise ValueError('Revisit client stream differs from its durable acknowledgement')
    return {'row': row, 'nested': nested, 'revisit': revisit,
            'request': request_body, 'request_path': str(request_path),
            'request_sha256': _sha(request_raw), 'client_stream_path': str(client_stream_path),
            'client_stream_sha256': _sha(stream_raw), 'ack': ack}


def _validate_complete_controller(source: dict, final: dict, tokens: list[int],
                                  token_bytes, eos_tokens: list[int]) -> dict:
    """Reconstruct every controller field after the resumed token IDs."""
    from context_time_travel import controller
    expected = controller.consume_many(source, (
        (token, token_bytes(token), token in eos_tokens) for token in tokens))
    if expected != final:
        differing = sorted(key for key in set(expected) | set(final)
                           if expected.get(key) != final.get(key))
        raise ValueError('Complete resumed controller differs in fields: ' + ','.join(differing))
    return expected


def _preflight(profile_path: Path, fixture_set_path: Path, browser_report_path: Path,
               store_dir: Path, browser_artifacts_dir: Path, *, runtime_validator=None) -> dict:
    profile, profile_raw = _json(profile_path)
    report, report_raw = _json(browser_report_path)
    if report.get('status') != 'PASS' or report.get('inference_kind') != 'real':
        raise ValueError('Browser report is not a passing real-inference report')
    if (report.get('server_execution_profile_id_from_durable_receipts')
            != profile.get('execution_profile_id')
            or report.get('server_model_contract_id_from_durable_receipts')
            != profile.get('model_contract_id')):
        raise ValueError('Browser report/profile identity mismatch')
    if runtime_validator is None:
        runtime_validator = base._validate_profile_runtime
    runtime_identity = runtime_validator(profile)
    fixture_set, fixtures = base._fixture_contexts(Path(fixture_set_path), profile)
    fixture_set_raw = Path(fixture_set_path).read_bytes()
    fixture_set_sha = _sha(fixture_set_raw)
    if report.get('context_set_manifest_sha256') != fixture_set_sha:
        raise ValueError('Browser report fixture-set digest mismatch')
    selected, nested, revisit = _select_oldest_context(report)
    fixture = next((item for item in fixtures if item['name'] == selected.get('name')), None)
    if fixture is None:
        raise ValueError('Revisited context is absent from the pinned fixture-set')
    browser = base._validate_browser_context(report, fixture, profile, Path(browser_artifacts_dir))
    evidence = _load_resume_evidence(report, browser_artifacts_dir)
    root_ids = fixture['root_ids']
    suffix = fixture['branches']['A']['suffix_ids']
    first_tokens = browser['branches']['A']['generated_ids']
    source_tip = revisit.get('source_tip')
    branch_name = evidence['request'].get('branch')
    if (browser['branches']['A']['snapshot'] != source_tip
            or browser['branches']['A']['operation_id'] != nested['ask_a_request']['operation_id']
            or browser['namespace_id'] != selected.get('namespace_id')):
        raise ValueError('Revisit source does not match the first A answer or namespace')
    from context_time_travel.store import Store
    from audit_cow_browser_context_store import _authoritative_files_fingerprint
    store_dir = Path(store_dir)
    canonical_before = _authoritative_files_fingerprint(store_dir)
    store = Store(store_dir)
    root_snapshot = browser['root_snapshot']
    root_lineage, root_history = store.lineage(root_snapshot)
    if root_lineage != root_snapshot or root_history != root_ids:
        raise ValueError('Canonical root history differs from the pinned fixture')
    root_record = store.get_snapshot(root_snapshot)
    if (root_record.get('parent') is not None
            or root_record.get('model_contract_id') != profile['model_contract_id']
            or root_record.get('controller', {}).get('status') != 'idle'):
        raise ValueError('Canonical root snapshot/controller is not the expected idle root')
    if store.branch_namespace(branch_name) != browser['namespace_id']:
        raise ValueError('Revisited branch namespace differs from the browser context')
    root_operation = base._validate_root_store_operation(store,
        nested['context_request']['operation_id'], browser['root_receipt'],
        browser['namespace_id'], browser['request_artifacts']['context']['body']['messages'],
        fixture['fixture']['renderer_id'], root_ids, fixture['name'] + ' root')
    first_receipt = base._validate_store_receipt(store,
        browser['branches']['A']['operation_id'], browser['branches']['A']['durable_receipt'],
        browser['namespace_id'], fixture['name'] + ' first A')
    source_snapshot = store.get_snapshot(source_tip)
    source_lineage_root, source_history = store.lineage(source_tip)
    if source_lineage_root != root_snapshot:
        raise ValueError('First A snapshot does not descend from the saved context root')
    resume_operation = evidence['request']['operation_id']
    resume_receipt = store.presentation_receipt(resume_operation)
    final_tip = revisit.get('new_tip')
    ack = evidence['ack']
    if (resume_receipt.get('result', {}).get('tip') != final_tip
            or resume_receipt.get('result', {}).get('name') != branch_name
            or resume_receipt.get('result', {}).get('revision') != evidence['request'].get('revision', -1) + 1
            or resume_receipt.get('presentation', {}).get('kind') != 'resume'
            or resume_receipt.get('presentation', {}).get('source', {}).get('snapshot') != source_tip):
        raise ValueError('Canonical Store resume receipt differs from the captured operation')
    source_message = store.active_presentation(branch_name, snapshot_id=source_tip)
    resume_message_id = resume_receipt['presentation'].get('output_span', {}).get('message_id')
    if (source_message.get('message_id') != resume_message_id
            or source_message.get('namespace_id') != browser['namespace_id']):
        raise ValueError('Resume did not preserve the existing assistant message identity')
    final_snapshot = store.get_snapshot(final_tip)
    final_lineage_root, final_history = store.lineage(final_tip)
    counters = revisit.get('counter_deltas')
    if not isinstance(counters, dict):
        raise ValueError('Resume native counters are missing')
    resumed_tokens = _validate_resume_transition(
        root_ids=root_ids, suffix_ids=suffix, first_tokens=first_tokens,
        source_history=source_history, final_history=final_history,
        request=evidence['request'], ack=ack,
        source_controller=source_snapshot.get('controller', {}),
        final_controller=final_snapshot.get('controller', {}), counters=counters,
        source_branch=branch_name,
        source_revision=evidence['request'].get('revision', -1),
        source_tip=source_tip, final_tip=final_tip,
        final_parent=final_snapshot.get('parent'),
        report_generated=revisit.get('generated_tokens'),
        policy=source_snapshot.get('controller', {}).get('policy'))
    from tools.strata_tokenizer import Tokenizer
    tokenizer = Tokenizer.from_gguf(profile['model_path'])
    if len(tokenizer.tokens) != base.VOCAB_SIZE:
        raise ValueError('Q4 tokenizer vocabulary differs from audited frontier width')
    complete_controller = _validate_complete_controller(
        source_snapshot['controller'], final_snapshot['controller'], resumed_tokens,
        tokenizer.token_bytes, profile['execution']['eos_tokens'])
    if final_lineage_root != root_snapshot:
        raise ValueError('Resumed A snapshot changed canonical root ancestry')
    if ack.get('receipt', {}).get('snapshot') != final_tip:
        raise ValueError('Client durable ACK snapshot differs from canonical resume')
    root_payload = browser['root_payload_sha256']
    first_payload = browser['branches']['A']['payload_sha256']
    root_checkpoint = base._canonical_checkpoint_frontier(store, root_snapshot, root_ids,
        profile, root_payload, fixture['name'] + ' root')
    first_checkpoint = base._canonical_checkpoint_frontier(store, source_tip, source_history,
        profile, first_payload, fixture['name'] + ' first A')
    resume_envelope_path = store.root / 'checkpoints' / (final_tip + '.json')
    resume_envelope, _ = _json(resume_envelope_path)
    resume_manifest = resume_envelope.get('manifest')
    if not isinstance(resume_manifest, dict) or not isinstance(resume_manifest.get('payload_sha256'), str):
        raise ValueError('Resume checkpoint manifest lacks its payload identity')
    resume_checkpoint = base._canonical_checkpoint_frontier(store, final_tip, final_history,
        profile, resume_manifest['payload_sha256'], fixture['name'] + ' resumed A')
    canonical_after = _authoritative_files_fingerprint(store_dir)
    if canonical_before != canonical_after:
        raise ValueError('Read-only resume audit changed canonical Store metadata')
    return {'profile': profile, 'profile_raw': profile_raw, 'report': report,
            'report_raw': report_raw, 'fixture_set': fixture_set,
            'fixture_set_sha256': fixture_set_sha, 'runtime_validation': runtime_identity,
            'store': store, 'canonical_fingerprint': canonical_after,
            'fixture': fixture, 'browser': browser, 'root_ids': root_ids,
            'suffix_ids': suffix, 'first_tokens': first_tokens,
            'resumed_tokens': resumed_tokens, 'source_history': source_history,
            'final_history': final_history, 'source_tip': source_tip,
            'final_tip': final_tip, 'source_branch': branch_name,
            'resume_operation_id': resume_operation, 'request_evidence': evidence,
            'root_operation': root_operation, 'first_receipt': first_receipt,
            'resume_receipt': resume_receipt, 'source_message_id': source_message['message_id'],
            'complete_controller': complete_controller,
            'source_controller': source_snapshot['controller'],
            'final_controller': final_snapshot['controller'],
            'root_checkpoint': root_checkpoint, 'first_checkpoint': first_checkpoint,
            'resume_checkpoint': resume_checkpoint, 'counters': counters}


def audit(profile_path: Path, fixture_set_path: Path, browser_report_path: Path,
          store_dir: Path, browser_artifacts_dir: Path, fresh_reference_dir: Path,
          *, reference_runner=None, runtime_validator=None, preflight_only=False,
          progress_callback=None) -> dict:
    checked = _preflight(profile_path, fixture_set_path, browser_report_path,
        store_dir, browser_artifacts_dir, runtime_validator=runtime_validator)
    if preflight_only:
        return {'schema': 1, 'status': 'PREPARED_NOT_NUMERICALLY_QUALIFIED',
                'scope': 'CPU-only canonical resume preflight; no fresh native inference launched',
                'profile_id': checked['profile']['execution_profile_id'],
                'model_contract_id': checked['profile']['model_contract_id'],
                'profile_sha256': _sha(checked['profile_raw']),
                'fixture_set_sha256': checked['fixture_set_sha256'],
                'browser_report_sha256': _sha(checked['report_raw']),
                'context': checked['fixture']['name'],
                'root_tokens': len(checked['root_ids']),
                'question_suffix_tokens': len(checked['suffix_ids']),
                'first_answer_tokens': len(checked['first_tokens']),
                'resumed_tokens': len(checked['resumed_tokens']),
                'resume_operation_id': checked['resume_operation_id'],
                'resume_counters': {key: checked['counters'].get(key) for key in
                    ('prefill_tokens', 'forward_tokens', 'consumed_tokens',
                     'loaded_file_bytes', 'checkpoint_import_bytes', 'integrity_mismatches')},
                'canonical_metadata_unchanged': True,
                'complete_controller_exact': True,
                'fresh_reference_count': 0, 'numerical_max_error': None}
    if progress_callback:
        progress_callback({'phase': 'preflight_complete', 'context': checked['fixture']['name'],
                           'time_unix': time.time(), 'completed_references': 0})
    fresh_reference_dir = Path(fresh_reference_dir).resolve()
    fresh_reference_dir.mkdir(parents=True, exist_ok=False)
    if reference_runner is None:
        from native_request_probe import reference as reference_runner
    ids = checked['root_ids'] + checked['suffix_ids']
    requests = [len(checked['root_ids']), len(checked['suffix_ids'])]
    if progress_callback:
        progress_callback({'phase': 'reference_start', 'context': checked['fixture']['name'],
                           'time_unix': time.time(), 'completed_references': 0,
                           'request_schedule': requests, 'decode_limit': 256})
    expected = reference_runner(checked['profile'], fresh_reference_dir, 'fresh-resume',
                               ids, requests, 256)
    if progress_callback:
        progress_callback({'phase': 'reference_complete', 'context': checked['fixture']['name'],
                           'time_unix': time.time(), 'completed_references': 1,
                           'reference_tokens': len(expected.get('tokens', [])),
                           'reference_work': expected.get('work')})
    if len(expected.get('tokens', [])) != 256:
        raise ValueError('Fresh native reference stopped before the complete 256-token scope')
    if (len(expected.get('prompt_frontiers', [])) != 2
            or len(expected.get('generated_frontiers', [])) != 256
            or expected.get('work', {}).get('prefill_tokens') != sum(requests)
            or expected.get('work', {}).get('normal_forward_tokens') != 256
            or expected.get('work', {}).get('consumed') != sum(requests) + 256):
        raise ValueError('Fresh native reference work/frontier counts do not match its two Prefill requests and 256 decodes')
    exact = checked['first_tokens'] + checked['resumed_tokens']
    if expected['tokens'] != exact:
        raise ValueError('Canonical first-A plus resumed token IDs differ from fresh native reference')
    root_vs_ref = base.compare_frontier(checked['root_checkpoint']['frontier'],
        expected['prompt_frontiers'][0], checked['fixture']['name'] + ' root')
    first_vs_ref = base.compare_frontier(checked['first_checkpoint']['frontier'],
        expected['generated_frontiers'][127], checked['fixture']['name'] + ' first A128')
    final_vs_ref = base.compare_frontier(checked['resume_checkpoint']['frontier'],
        expected['generated_frontiers'][-1], checked['fixture']['name'] + ' resumed A256')
    from audit_cow_browser_context_store import _authoritative_files_fingerprint
    final_fingerprint = _authoritative_files_fingerprint(store_dir)
    if checked['canonical_fingerprint'] != final_fingerprint:
        raise ValueError('Fresh-reference audit changed canonical Store metadata')
    maximum = max(root_vs_ref['max_abs_error'], first_vs_ref['max_abs_error'],
                  final_vs_ref['max_abs_error'])
    return {'schema': 1,
            'status': 'PASS_EXACT_COLD_RESTORED_RESUME_TOKENS_AND_FRONTIERS',
            'scope': 'one browser-published A128 answer plus cold-restored A128 resume; independent fresh native request uses root and A suffix as separate batched Prefill inputs then 256 normal-verifier decode tokens',
            'generation_request_segmentation': {
                'sut': [128, 128], 'fresh_reference': [256],
                'request_boundaries_match': False,
                'claim': 'Same per-token normal-verifier operator sequence and exact token/frontier states; fresh oracle is one continuous decode request, not two HTTP/generate calls.'},
            'profile_id': checked['profile']['execution_profile_id'],
            'model_contract_id': checked['profile']['model_contract_id'],
            'profile_sha256': _sha(checked['profile_raw']),
            'fixture_set_sha256': checked['fixture_set_sha256'],
            'browser_report_sha256': _sha(checked['report_raw']),
            'auditor_sha256': _sha(Path(__file__).read_bytes()),
            'native_reference_sha256': _sha((API / 'scripts' / 'native_request_probe.py').read_bytes()),
            'context': checked['fixture']['name'],
            'root_snapshot': checked['browser']['root_snapshot'],
            'first_a_snapshot': checked['source_tip'],
            'resumed_snapshot': checked['final_tip'],
            'resume_operation_id': checked['resume_operation_id'],
            'source_message_id_preserved': checked['source_message_id'],
            'source_controller_sha256': _sha(json.dumps(checked['source_controller'],
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')),
            'final_controller_sha256': _sha(json.dumps(checked['final_controller'],
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')),
            'root_tokens': len(checked['root_ids']),
            'question_suffix_tokens': len(checked['suffix_ids']),
            'first_answer_tokens': len(checked['first_tokens']),
            'resumed_tokens': len(checked['resumed_tokens']),
            'reference_tokens': len(expected['tokens']),
            'reference_request_schedule': requests,
            'reference_work': {key: expected['work'].get(key) for key in
                ('prefill_tokens', 'normal_forward_tokens', 'consumed', 'generated')},
            'resume_counters': {key: checked['counters'].get(key) for key in
                ('prefill_tokens', 'forward_tokens', 'consumed_tokens', 'loaded_file_bytes',
                 'checkpoint_import_bytes', 'integrity_mismatches')},
            'controller_full_state_exact': True,
            'frontiers': {'root': root_vs_ref, 'first_a_128': first_vs_ref,
                          'resumed_a_256': final_vs_ref},
            'canonical_metadata_unchanged': True,
            'sut_forward_evaluations_from_browser': 256,
            'additional_sut_forward_evaluations_by_audit': 0,
            'fresh_reference_count': 1,
            'numerical_max_error': maximum}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fixture-set', type=Path, required=True)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--browser-artifacts-dir', type=Path, required=True)
    parser.add_argument('--store-dir', type=Path, required=True)
    parser.add_argument('--fresh-reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    recorder = base._ProgressRecorder(args.output.with_suffix(args.output.suffix + '.progress.json'),
                                      args.output)
    recorder.start()
    try:
        result = audit(args.profile, args.fixture_set, args.browser_report,
            args.store_dir, args.browser_artifacts_dir, args.fresh_reference_dir,
            preflight_only=args.preflight_only, progress_callback=recorder.update)
    except BaseException as error:
        recorder.fail(error)
        print(json.dumps({'status': 'FAIL', 'error': str(error)}), file=sys.stderr)
        return 1
    else:
        recorder.finish(result)
        print(json.dumps({'status': result['status'],
                          'context': result.get('context'),
                          'numerical_max_error': result.get('numerical_max_error')}))
        return 0
    finally:
        recorder.stop()


if __name__ == '__main__':
    raise SystemExit(main())
