"""CPU-only support tests for browser COW frontier audit helpers.

These tests use tiny synthetic vectors/metadata. They do not exercise a model,
native reference process, browser, GPU, or numerical qualification.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[2]
SCRIPT = HERE / 'scripts' / 'audit_browser_context_frontiers_v2.py'
SPEC = importlib.util.spec_from_file_location('audit_browser_context_frontiers_v2', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
sys.path.insert(0, str(HERE / 'src'))
from context_time_travel.store import Store, _json_bytes


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8'))


class BrowserContextFrontierAuditTests(unittest.TestCase):
    def test_exact_tiny_frontier_match_reports_zero_error(self):
        raw = struct.pack('<2f', 0.25, -1.5)
        result = audit.compare_frontier(raw, raw, 'fixture', vocab_size=2)
        self.assertTrue(result['exact_bytes_match'])
        self.assertEqual(result['max_abs_error'], 0.0)
        self.assertEqual(result['values'], 2)

    def test_frontier_mismatch_is_rejected_without_tolerance(self):
        actual = struct.pack('<2f', 0.25, -1.5)
        expected = struct.pack('<2f', 0.25, -1.4999999)
        with self.assertRaisesRegex(ValueError, 'exact frontier mismatch'):
            audit.compare_frontier(actual, expected, 'fixture', vocab_size=2)

    def test_frontier_rejects_truncation_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            audit.compare_frontier(b'\0\0\0', b'\0\0\0\0', 'fixture', vocab_size=1)
        nan = struct.pack('<f', float('nan'))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            audit.compare_frontier(nan, nan, 'fixture', vocab_size=1)

    def test_fixture_set_binds_three_exact_standalone_fixture_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for index in range(3):
                tokens = [10 + index, 20 + index]
                suffix_a = [30 + index]
                suffix_b = [40 + index]
                fixture = {
                    'execution_profile_id': 'b' * 64,
                    'model_contract_id': 'a' * 64,
                    'renderer_id': 'r' * 64,
                    'root_messages': [{'role': 'user', 'content': 'source'}],
                    'root_token_ids': tokens,
                    'root_token_ids_sha256': _json_sha(tokens),
                    'branches': {
                        'A': {'suffix_token_ids': suffix_a,
                              'suffix_token_ids_sha256': _json_sha(suffix_a),
                              'question': 'question A'},
                        'B': {'suffix_token_ids': suffix_b,
                              'suffix_token_ids_sha256': _json_sha(suffix_b),
                              'question': 'question B'},
                    },
                }
                raw = json.dumps(fixture, separators=(',', ':')).encode('utf-8')
                filename = f'fixture-{index}.json'
                (root / filename).write_bytes(raw)
                entries.append({'name': f'context-{index}', 'fixture_file': filename,
                                'fixture_sha256': _sha(raw),
                                'root_token_ids_sha256': _json_sha(tokens)})
            fixture_set = {'schema': 'cow-browser-context-set-v1', 'status': 'PASS',
                           'execution_profile_id': 'b' * 64,
                           'model_contract_id': 'a' * 64, 'contexts': entries}
            path = root / 'fixture-set.json'
            path.write_text(json.dumps(fixture_set), encoding='utf-8')
            profile = {'execution_profile_id': 'b' * 64, 'model_contract_id': 'a' * 64}
            _, loaded = audit._fixture_contexts(path, profile)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[2]['root_ids'], [12, 22])
            (root / 'fixture-1.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                audit._fixture_contexts(path, profile)

    def test_request_digest_and_operation_binding_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = {'operation_id': 'op-A', 'branch': 'branch-A', 'content': 'question'}
            raw = json.dumps(body, separators=(',', ':')).encode('utf-8')
            (root / 'ask.json').write_bytes(raw)
            nested = {'ask_a_request': {'path': 'ask.json', 'bytes': len(raw),
                                        'sha256': _sha(raw)}}
            checked = audit._request_record(root, Path('.'), nested, 'ask_a_request',
                                            'op-A', 'branch-A')
            self.assertEqual(checked['body'], body)
            with self.assertRaisesRegex(ValueError, 'operation ID'):
                audit._request_record(root, Path('.'), nested, 'ask_a_request',
                                      'op-B', 'branch-A')
            nested['ask_a_request']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'size/SHA'):
                audit._request_record(root, Path('.'), nested, 'ask_a_request',
                                      'op-A', 'branch-A')

    def test_durable_checkpoint_frontier_is_extracted_only_after_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            store_dir = Path(temporary) / 'store'
            store = Store(store_dir)
            model, execution = 'a' * 64, 'b' * 64
            tokens = [11, 29, 31]
            snapshot = store.create_root(model, tokens, {'status': 'idle'})
            raw_frontier = struct.pack('<2f', 0.5, -0.75)
            state = b'KV-state-bytes'
            payload = state + raw_frontier
            payload_sha = _sha(payload)
            (store_dir / 'checkpoints').mkdir(exist_ok=True)
            (store_dir / 'checkpoints' / (payload_sha + '.state')).write_bytes(payload)
            manifest = {
                'schema': 1, 'snapshot': snapshot,
                'input_identity': _sha(_json_bytes(tokens)),
                'model_contract_id': model, 'execution_profile_id': execution,
                'consumed': len(tokens), 'positions': 'absolute-zero-based',
                'logits_count': 2, 'state_bytes': len(state),
                'payload_sha256': payload_sha,
            }
            envelope = {'manifest': manifest,
                        'checksum': _sha(_json_bytes(manifest))}
            (store_dir / 'checkpoints' / (snapshot + '.json')).write_bytes(_json_bytes(envelope))
            profile = {'model_contract_id': model, 'execution_profile_id': execution,
                       'execution': {'prefill_chunk': 16}}
            checked = audit._canonical_checkpoint_frontier(store, snapshot, tokens,
                profile, payload_sha, 'root', vocab_size=2)
            self.assertEqual(checked['frontier'], raw_frontier)
            payload_path = store_dir / 'checkpoints' / (payload_sha + '.state')
            payload_path.write_bytes(state + struct.pack('<2f', 9.0, 10.0))
            with self.assertRaisesRegex(ValueError, 'payload SHA-256 mismatch'):
                audit._canonical_checkpoint_frontier(store, snapshot, tokens,
                    profile, payload_sha, 'root', vocab_size=2)

    def test_progress_recorder_persists_partial_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            recorder = audit._ProgressRecorder(base / 'progress.json', base / 'result.json')
            recorder.start()
            recorder.update({'phase': 'fresh-reference', 'context': 'ctx-2',
                             'branch': 'B', 'completed_branches': 3})
            try:
                raise ValueError('synthetic reference mismatch')
            except ValueError as error:
                recorder.fail(error)
            finally:
                recorder.stop()
            result = json.loads((base / 'result.json').read_text(encoding='utf-8'))
            progress = json.loads((base / 'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['completed_branches'], 3)
            self.assertEqual(result['context'], 'ctx-2')
            self.assertIn('synthetic reference mismatch', result['traceback'])
            self.assertEqual(progress['status'], 'FAIL')

    def test_progress_result_hash_covers_exact_published_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            recorder = audit._ProgressRecorder(base / 'progress.json', base / 'result.json')
            recorder.start()
            try:
                result = {'status': 'PASS', 'value': 'exact bytes'}
                recorder.finish(result)
                raw = (base / 'result.json').read_bytes()
                progress = json.loads((base / 'progress.json').read_text(encoding='utf-8'))
                self.assertTrue(raw.endswith(b'\n'))
                self.assertEqual(progress['result_sha256'], _sha(raw))
            finally:
                recorder.stop()

    def test_three_context_end_to_end_mock_references_match_durable_bytes(self):
        """Synthetic CPU-only integration of fixture/report/Store/reference joins."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture_dir = base / 'fixtures'
            fixture_dir.mkdir()
            browser_dir = base / 'browser'
            browser_dir.mkdir()
            store_dir = base / 'store'
            store = Store(store_dir)
            model, execution = 'a' * 64, 'b' * 64
            (store_dir / 'checkpoints').mkdir()
            profile_path = base / 'profile.json'
            profile = {'model_contract_id': model, 'execution_profile_id': execution,
                       'execution': {'prefill_chunk': 16}}
            profile_path.write_text(json.dumps(profile), encoding='utf-8')
            context_fixtures = []
            browser_contexts = []
            reference_calls = []
            expected_root_frontiers = {}

            def checkpoint(snapshot, tokens, frontier_values):
                state = b'KV-state-' + bytes([len(tokens)])
                vector = struct.pack('<2f', *frontier_values)
                payload = state + vector
                digest = _sha(payload)
                manifest = {
                    'schema': 1, 'snapshot': snapshot,
                    'input_identity': _sha(_json_bytes(tokens)),
                    'model_contract_id': model, 'execution_profile_id': execution,
                    'consumed': len(tokens), 'positions': 'absolute-zero-based',
                    'logits_count': 2, 'state_bytes': len(state),
                    'payload_sha256': digest,
                }
                envelope = {'manifest': manifest,
                            'checksum': _sha(_json_bytes(manifest))}
                (store_dir / 'checkpoints' / (snapshot + '.json')).write_bytes(_json_bytes(envelope))
                (store_dir / 'checkpoints' / (digest + '.state')).write_bytes(payload)
                return digest, vector

            def request_file(context_dir, filename, body, operation_id, branch):
                raw = json.dumps(body, separators=(',', ':')).encode('utf-8')
                (context_dir / filename).write_bytes(raw)
                return {'path': filename, 'bytes': len(raw), 'sha256': _sha(raw),
                        'operation_id': operation_id, 'branch': branch}

            for index in range(1, 4):
                name = f'ctx-{index}'
                root_ids = [10 + index, 20 + index]
                root_messages = [{'role': 'user', 'content': f'source {index}'}]
                suffixes = {'A': [30 + index], 'B': [40 + index]}
                questions = {'A': f'question A {index}', 'B': f'question B {index}'}
                fixture = {
                    'execution_profile_id': execution, 'model_contract_id': model,
                    'renderer_id': 'a' * 64, 'root_messages': root_messages,
                    'root_token_ids': root_ids, 'root_token_ids_sha256': _json_sha(root_ids),
                    'branches': {
                        label: {'suffix_token_ids': suffixes[label],
                                'suffix_token_ids_sha256': _json_sha(suffixes[label]),
                                'question': questions[label]}
                        for label in ('A', 'B')},
                }
                fixture_raw = json.dumps(fixture, separators=(',', ':')).encode('utf-8')
                fixture_filename = f'{name}.json'
                (fixture_dir / fixture_filename).write_bytes(fixture_raw)
                context_fixtures.append({'name': name, 'fixture_file': fixture_filename,
                    'fixture_sha256': _sha(fixture_raw),
                    'root_token_ids_sha256': _json_sha(root_ids)})
                root_snapshot = store.create_root(model, root_ids, {'status': 'idle'})
                root_frontier_values = [0.125 * index, -0.25 * index]
                root_payload, root_frontier = checkpoint(root_snapshot, root_ids,
                                                           root_frontier_values)
                expected_root_frontiers[name] = root_frontier
                namespace = store.create_namespace(f'ctx {index}', f'ns-op-{index}').namespace_id
                root_branch = f'root-{index}'
                ctxdir = browser_dir / f'context-{index}'
                ctxdir.mkdir()
                root_op = f'op-root-{index}'
                root_presentation_op = f'root-presentation-{index}'
                root_presentation = {
                    'schema': 1, 'namespace_id': namespace, 'kind': 'root',
                    'messages': [{'id': f'{index:032x}', **message} for message in root_messages],
                    'source': {'snapshot': None, 'revision': 0, 'token_count': 0},
                    'input_span': {'start': 0, 'count': len(root_ids),
                                   'token_ids_sha256': _sha(_json_bytes(root_ids))},
                    'output_span': None, 'renderer_id': fixture['renderer_id'],
                    'turn_complete': False,
                }
                store.fork(root_branch, root_snapshot, namespace_id=namespace,
                           operation_id=root_presentation_op, presentation=root_presentation)
                store.move(root_branch, 0, root_snapshot, root_op)
                root_body = {'operation_id': root_op, 'branch': root_branch,
                             'namespace_id': namespace, 'messages': root_messages}
                ctx_ref = request_file(ctxdir, 'actual-context-request.json', root_body,
                                       root_op, root_branch)
                nested = {'context_request': {**ctx_ref, 'receipt': {
                    'snapshot': root_snapshot, 'payload_sha256': root_payload,
                    'model_contract_id': model, 'execution_profile_id': execution,
                    'branch': {'name': root_branch, 'revision': 1,
                               'tip': root_snapshot, 'root': root_snapshot}}}}
                for label in ('A', 'B'):
                    answer_tokens = [50 + index, 60 + index]
                    branch_name = root_branch
                    snapshot = store.append(root_snapshot,
                        suffixes[label] + answer_tokens,
                        {'status': 'active', 'accepted_in_scope': len(answer_tokens)})
                    frontier_values = [index + (0.01 if label == 'A' else 0.02),
                                       -index - (0.03 if label == 'A' else 0.04)]
                    payload, final_frontier = checkpoint(snapshot,
                        root_ids + suffixes[label] + answer_tokens, frontier_values)
                    operation = f'op-{label}-{index}'
                    body = {'operation_id': operation, 'branch': branch_name,
                            'namespace_id': None, 'revision': 1,
                            'content': questions[label]}
                    request_ref = request_file(ctxdir,
                        f'actual-ask-{label.lower()}-request.json', body,
                        operation, branch_name)
                    nested[f'ask_{label.lower()}_request'] = {
                        **request_ref,
                        'answer': {'snapshot': snapshot, 'tokens': answer_tokens},
                        'receipt': {'snapshot': snapshot, 'payload_sha256': payload,
                            'model_contract_id': model, 'execution_profile_id': execution,
                            'branch': {'name': branch_name, 'revision': 0,
                                       'tip': snapshot, 'root': root_snapshot}},
                    }
                    reference_calls.append((name, label, root_ids + suffixes[label],
                                            root_frontier, final_frontier,
                                            answer_tokens))
                browser_contexts.append({
                    'name': name, 'index': index, 'artifact_dir': f'context-{index}',
                    'fixture_sha256': _sha(fixture_raw),
                    'root_token_ids_sha256': _json_sha(root_ids),
                    'question_token_ids_sha256': {
                        label: _json_sha(suffixes[label]) for label in ('A', 'B')},
                    'renderer_id': fixture['renderer_id'], 'namespace_id': namespace,
                    'execution_profile_id': execution,
                    'report': nested,
                })

            fixture_set = {'schema': 'cow-browser-context-set-v1', 'status': 'PASS',
                'execution_profile_id': execution, 'model_contract_id': model,
                'contexts': context_fixtures}
            fixture_set_path = fixture_dir / 'fixture-set.json'
            fixture_set_path.write_text(json.dumps(fixture_set), encoding='utf-8')
            fixture_set_digest = _sha(fixture_set_path.read_bytes())
            report_path = browser_dir / 'report.json'
            report_path.write_text(json.dumps({
                'status': 'PASS', 'inference_kind': 'real', 'output_limit': 2,
                'server_execution_profile_id_from_durable_receipts': execution,
                'server_model_contract_id_from_durable_receipts': model,
                'context_set_manifest_sha256': fixture_set_digest,
                'reusable_context_report': {'contexts': browser_contexts}}), encoding='utf-8')

            def mock_reference(profile_arg, output, stem, ids, requests, generated):
                # Calls are identified by their exact prompt IDs, avoiding any
                # dependence on the synthetic context's presentation name.
                match = next(row for row in reference_calls if row[2] == ids)
                if requests != [2, 1] or generated != 2:
                    raise AssertionError('Unexpected reference request schedule')
                tokens = match[5]
                prompt_frames = [match[3], struct.pack('<2f', 99.0, -99.0)]
                generated_frames = [match[4], match[4]]
                stem_path = output / (stem + '.f32')
                (output / (stem + '.tokens')).write_text(','.join(map(str, ids)), encoding='utf-8')
                schedule = {'operator': 'batched-prefill', 'requests': requests,
                            'completed_chunks': [[0, 2], [2, 1]]}
                Path(str(stem_path) + '.schedule.json').write_text(
                    json.dumps(schedule), encoding='utf-8')
                work = {'generated': tokens, 'prefill_tokens': sum(requests),
                        'prompt_consumed': sum(requests), 'normal_forward_tokens': len(tokens),
                        'consumed': sum(requests) + len(tokens)}
                (output / (stem + '.stdout')).write_text(json.dumps(work), encoding='utf-8')
                (output / (stem + '.stderr')).write_bytes(b'')
                Path(str(stem_path) + '.requests.f32').write_bytes(b''.join(prompt_frames))
                Path(str(stem_path) + '.decode.f32').write_bytes(b''.join(generated_frames))
                return {'tokens': tokens, 'prompt_frontiers': prompt_frames,
                        'generated_frontiers': generated_frames, 'work': work,
                        'command': ['synthetic-native-reference', '--tokens-file']}

            def must_not_run_reference(*args, **kwargs):
                raise AssertionError('reference launched before runtime validation')

            fresh_dir = base / 'fresh'
            with self.assertRaisesRegex(RuntimeError, 'runtime pin mismatch'):
                audit.audit(profile_path, fixture_set_path, report_path, store_dir,
                    browser_dir, fresh_dir, reference_runner=must_not_run_reference,
                    runtime_validator=lambda _: (_ for _ in ()).throw(
                        RuntimeError('runtime pin mismatch')), vocab_size=2)
            self.assertFalse(fresh_dir.exists())
            synthetic_receipt = lambda _store, op, receipt, ns, _label: {
                'operation_id': op, 'result': receipt.get('branch'), 'namespace_id': ns}
            prepared = audit.audit(profile_path, fixture_set_path, report_path, store_dir,
                browser_dir, fresh_dir, preflight_only=True,
                runtime_validator=lambda _: {'model_bytes_hashed_this_process': 0,
                                             'model_file_hash_receipts_reused': 0},
                store_receipt_validator=synthetic_receipt, vocab_size=2)
            self.assertEqual(prepared['status'], 'PREPARED_NOT_NUMERICALLY_QUALIFIED')
            self.assertEqual(prepared['captured_answer_requests_validated'], 6)
            self.assertEqual(prepared['canonical_store_receipts_validated'], 9)
            self.assertEqual(prepared['canonical_root_annotations_validated'], 3)
            self.assertFalse(fresh_dir.exists())
            output = audit.audit(profile_path, fixture_set_path, report_path, store_dir,
                browser_dir, fresh_dir, reference_runner=mock_reference,
                runtime_validator=lambda _: {'model_bytes_hashed_this_process': 0,
                                             'model_file_hash_receipts_reused': 0},
                store_receipt_validator=synthetic_receipt,
                vocab_size=2)
            self.assertEqual(output['status'], 'PASS_EXACT_BROWSER_TOKENS_AND_DURABLE_FINAL_FRONTIERS')
            self.assertEqual(output['context_count'], 3)
            self.assertEqual(output['frontier_coverage']['durable_final_answer_frontiers_compared'], 6)
            self.assertEqual(output['frontier_coverage']['distinct_root_prefix_frontier_states'], 3)
            self.assertEqual(output['frontier_coverage']['root_prefix_frontier_comparisons'], 6)
            self.assertEqual(output['sut_forward_evaluations'], 0)
            self.assertEqual(output['fresh_reference_work']['reference_count'], 9)


if __name__ == '__main__':
    unittest.main()
