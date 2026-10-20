"""CPU tests for the browser-root receipt auditor; no backend/model required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
AUDIT_PATH = HERE / 'audit_cow_browser_context_store.py'
SPEC = importlib.util.spec_from_file_location('audit_cow_browser_context_store', AUDIT_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)
SRC = HERE.parent / 'src'
sys.path.insert(0, str(SRC))
from context_time_travel.store import Store, _json_bytes
from context_time_travel.execution_receipt import schedule_digest, tokens_digest
from context_time_travel.controller import start as controller_start, consume as controller_consume


class AuditCowBrowserContextStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store_dir = self.root / 'store'
        self.store = Store(self.store_dir)
        self.model = 'a' * 64
        self.execution = 'b' * 64
        self.tokens = [19, 23, 29]
        self.snapshot = self.store.create_root(self.model, self.tokens, {'status': 'idle'})
        self.profile = self.root / 'profile.json'
        self.fixture = self.root / 'fixture.json'
        self.report = self.root / 'report.json'
        self._write_checkpoint()
        self._write_branch_receipts()

    def tearDown(self):
        self.temp.cleanup()

    def _write_inputs(self, *, model=None, resume='identical-native-operators-v1',
                      branch_suffixes=None, branch_answers=None, branch_receipts=None):
        if branch_suffixes is None:
            branch_suffixes = getattr(self, 'suffixes', {})
        if branch_answers is None:
            branch_answers = getattr(self, 'answers', {})
        if branch_receipts is None:
            branch_receipts = getattr(self, 'branch_receipts', {})
        profile = {'model_contract_id': model or self.model,
                   'execution_profile_id': self.execution,
                   'execution': {'resume_contract': resume, 'prefill_chunk': 8192}}
        fixture = {'model_contract_id': model or self.model,
                   'root_token_ids': self.tokens,
                   'root_token_count': len(self.tokens),
                   'root_token_ids_sha256': hashlib.sha256(
                       json.dumps(self.tokens, separators=(',', ':')).encode()).hexdigest(),
                   'branches': {label: {'suffix_token_ids': values}
                                for label, values in (branch_suffixes or {}).items()}}
        report = {'status': 'PASS', 'reusable_context': True,
                  'server_execution_profile_id_caller_supplied': self.execution,
                  'server_execution_profile_id_from_durable_receipts': self.execution,
                  'server_model_contract_id_from_durable_receipts': model or self.model,
                  'server_receipt_identity': {'execution_profile_id': self.execution,
                                              'model_contract_id': model or self.model},
                  'reusable_context_report': {'root_snapshot': self.snapshot,
                                              'root_payload_sha256': self.root_payload_sha,
                                              'ask_a_request': branch_receipts.get('A'),
                                              'ask_b_request': branch_receipts.get('B')}}
        for path, value in ((self.profile, profile), (self.fixture, fixture), (self.report, report)):
            path.write_text(json.dumps(value), encoding='utf-8')

    def _write_checkpoint(self, *, snapshot=None, tokens=None, schedule=None, digest=None,
                          write_inputs=True):
        snapshot = snapshot or self.snapshot
        tokens = list(self.tokens if tokens is None else tokens)
        # Use the exact profile-independent history identity from RealContext.
        input_identity = hashlib.sha256(_json_bytes(tokens)).hexdigest()
        if schedule is None:
            schedule = [{'operator': 'batched-prefill', 'position': 0, 'count': len(tokens),
                         'tokens_sha256': tokens_digest(tokens), 'chunks': [[0, len(tokens)]]}]
        actual_digest = schedule_digest(schedule)
        manifest = {'schema': 3, 'snapshot': snapshot,
                    'input_identity': input_identity, 'model_contract_id': self.model,
                    'execution_profile_id': self.execution, 'consumed': len(tokens),
                    'positions': 'absolute-zero-based', 'logits_count': 2,
                    'state_bytes': 4, 'payload_sha256': '0' * 64,
                    'execution_schedule': schedule,
                    'schedule_sha256': actual_digest if digest is None else digest}
        import struct
        payload = b'KV!!' + struct.pack('<ff', 1.0, -2.0)
        self.payload_sha = hashlib.sha256(payload).hexdigest()
        manifest['payload_sha256'] = self.payload_sha
        checkpoint_dir = self.store_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        envelope = {'manifest': manifest,
                    'checksum': hashlib.sha256(_json_bytes(manifest)).hexdigest()}
        (checkpoint_dir / (snapshot + '.json')).write_bytes(_json_bytes(envelope))
        (checkpoint_dir / (self.payload_sha + '.state')).write_bytes(payload)
        if write_inputs:
            self.root_payload_sha = self.payload_sha
            self._write_inputs()
        return self.payload_sha

    def _write_branch_receipts(self, *, wrong_root=False):
        suffixes = {'A': [31, 32], 'B': [37, 38]}
        answers = {'A': [41, 43], 'B': [47, 53]}
        branch_receipts = {}
        for label in ('A', 'B'):
            branch_root = self.store.create_root(self.model, [101, 103], {'status': 'idle'}) if wrong_root and label == 'B' else self.snapshot
            history = self.tokens + suffixes[label] + answers[label]
            state = controller_start({}, 'stream-' + label, self.tokens + suffixes[label], 256)
            for token in answers[label]:
                state = controller_consume(state, token, str(token).encode(), False)
            tip = self.store.append(branch_root, suffixes[label] + answers[label], state)
            payload_sha = self._write_checkpoint(snapshot=tip, tokens=history, write_inputs=False)
            branch_receipts[label] = {
                'branch': 'branch-' + label,
                'answer': {'snapshot': tip, 'tokens': answers[label]},
                'receipt': {'snapshot': tip, 'payload_sha256': payload_sha,
                            'model_contract_id': self.model,
                            'execution_profile_id': self.execution,
                            'branch': {'name': 'branch-' + label,
                                       'revision': 2, 'tip': tip, 'root': branch_root}},
            }
        self.branch_tips = {label: branch_receipts[label]['receipt']['snapshot']
                            for label in ('A', 'B')}
        self.suffixes = suffixes
        self.answers = answers
        self.branch_receipts = branch_receipts
        self._write_inputs(branch_suffixes=suffixes, branch_answers=answers,
                           branch_receipts=branch_receipts)

    def test_audits_both_branch_histories_and_structural_controllers(self):
        result = audit_module.audit(self.root, self.store_dir, self.profile,
                                    self.report, self.fixture)
        self.assertEqual(result['status'], 'PASS_READ_ONLY_DURABLE_CONTEXT_AND_BRANCH_AUDIT')
        self.assertEqual(result['branches']['A']['token_count'],
                         len(self.tokens + self.suffixes['A'] + self.answers['A']))
        self.assertEqual(result['branches']['B']['token_count'],
                         len(self.tokens + self.suffixes['B'] + self.answers['B']))
        self.assertNotEqual(result['branches']['A']['snapshot'], result['branches']['B']['snapshot'])
        self.assertEqual(result['branches']['A']['controller_status'], 'active')
        self.assertEqual(result['branches']['B']['controller_status'], 'active')

    def test_rejects_swapped_branch_suffix_history(self):
        fixture = json.loads(self.fixture.read_text(encoding='utf-8'))
        fixture['branches']['A']['suffix_token_ids'] = fixture['branches']['B']['suffix_token_ids']
        self.fixture.write_text(json.dumps(fixture), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'branch A canonical history'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_rejects_branch_tip_from_wrong_root(self):
        self._write_branch_receipts(wrong_root=True)
        with self.assertRaisesRegex(ValueError, 'branch B canonical root'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_audits_real_store_root_and_schema3_schedule_without_mutating_authority(self):
        before = audit_module._authoritative_files_fingerprint(self.store_dir)
        result = audit_module.audit(self.root, self.store_dir, self.profile,
                                    self.report, self.fixture)
        after = audit_module._authoritative_files_fingerprint(self.store_dir)
        self.assertEqual(before, after)
        self.assertEqual(result['snapshot'], self.snapshot)
        self.assertEqual(result['token_count'], 3)
        self.assertEqual(result['native_forward_work'], 0)

    def test_accepts_driver_nested_durable_identity_without_rewriting_report(self):
        report = json.loads(self.report.read_text(encoding='utf-8'))
        identity = report.pop('server_receipt_identity')
        report['reusable_context_report']['server_receipt_identity'] = identity
        self.report.write_text(json.dumps(report), encoding='utf-8')
        result = audit_module.audit(self.root, self.store_dir, self.profile,
                                    self.report, self.fixture)
        self.assertEqual(result['snapshot'], self.snapshot)
        saved = json.loads(self.report.read_text(encoding='utf-8'))
        self.assertNotIn('server_receipt_identity', saved)
        self.assertEqual(saved['reusable_context_report']['server_receipt_identity'], identity)

    def test_rejects_conflicting_nested_and_top_level_durable_identity(self):
        report = json.loads(self.report.read_text(encoding='utf-8'))
        report['reusable_context_report']['server_receipt_identity'] = {
            'execution_profile_id': self.execution,
            'model_contract_id': 'd' * 64,
        }
        self.report.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'durable server receipts'):
            audit_module.audit(self.root, self.store_dir, self.profile,
                                self.report, self.fixture)

    def test_rejects_wrong_model_contract(self):
        self._write_inputs(model='c' * 64)
        with self.assertRaisesRegex(ValueError, 'canonical root model contract'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_rejects_schedule_digest_mismatch(self):
        self._write_checkpoint(digest='f' * 64)
        with self.assertRaisesRegex(ValueError, 'schedule digest'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_rejects_schedule_with_wrong_history_or_positions(self):
        schedule = [{'operator': 'batched-prefill', 'position': 1, 'count': 2,
                     'tokens_sha256': tokens_digest(self.tokens[1:]), 'chunks': [[1, 2]]}]
        self._write_checkpoint(schedule=schedule)
        with self.assertRaisesRegex(ValueError, 'schedule|request|execution'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_rejects_profile_resume_contract_and_manifest_schema_mismatch(self):
        self._write_inputs(resume='identical-native-batches-v1')
        with self.assertRaisesRegex(ValueError, 'schema differs'):
            audit_module.audit(self.root, self.store_dir, self.profile, self.report, self.fixture)

    def test_fingerprint_covers_migrated_reference_directory_and_ref_order(self):
        migrated_dir = self.root / 'migrated-store'
        legacy_store = Store(migrated_dir)
        legacy_marker = {'schema': 1, 'branches': {}, 'receipts': {}, 'events': []}
        (migrated_dir / 'refs.json').write_bytes(_json_bytes(legacy_marker))
        migrated_store = Store(migrated_dir)
        marker = json.loads((migrated_dir / 'refs.json').read_text(encoding='utf-8'))
        self.assertRegex(marker['directory'], r'^references-[0-9a-f]{64}$')
        root_id = migrated_store.create_root(self.model, [7], {'status': 'idle'})
        migrated_store.fork('Alpha', root_id)
        before = audit_module._authoritative_files_fingerprint(migrated_dir)
        child = migrated_store.append(root_id, [11], {'status': 'idle'})
        migrated_store.move('Alpha', 0, child, 'MoveOne')
        after = audit_module._authoritative_files_fingerprint(migrated_dir)
        self.assertNotEqual(before, after)
        order_path = migrated_dir / 'ref-order.json'
        order = json.loads(order_path.read_text(encoding='utf-8'))
        order_path.write_bytes(_json_bytes({'schema': order['schema'], 'next': order['next'] + 1}))
        self.assertNotEqual(after, audit_module._authoritative_files_fingerprint(migrated_dir))


if __name__ == '__main__':
    unittest.main()
