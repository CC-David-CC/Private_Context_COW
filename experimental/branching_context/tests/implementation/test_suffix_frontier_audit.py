"""CPU parser tests for independent batched suffix-frontier evidence."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

BRANCHING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRANCHING / 'scripts'))
from audit_suffix_frontiers import (
    _runtime_import_witness,
    _require_finite_f32,
    _small_store_fingerprint,
    _verified_payload,
    load_suffix_reference,
    preflight_suffix_audit,
)


class SuffixFrontierReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.case = Path(self.temp.name)
        self.fixture = {'prefix': [1, 2], 'A': [3], 'B': [4],
                        'model_contract_id': 'contract-test'}
        (self.case / 'fixture.json').write_text(json.dumps(self.fixture))
        (self.case / 'result.json').write_text(json.dumps({
            'status': 'PASS', 'profile_id': 'profile-test',
            'fixture_sha256': self._sha(self.case / 'fixture.json')}))
        self.vocab_size = 5
        self.root = struct.pack('<5f', 1.25, -2.5, 0.0, 3.0, -4.0)
        (self.case / 'prepared-root.f32').write_bytes(self.root)
        self.prepare = {'status': 'PASS', 'root': {
            'snapshot': 'root-snapshot', 'frontier': 'prepared-root.f32',
            'frontier_sha256': self._sha(self.case / 'prepared-root.f32'),
            'consumed': 2, 'history': [1, 2], 'controller': {'status': 'idle'}},
            'root_save': {'branch': {'name': 'prefix', 'revision': 1,
                                     'tip': 'root-snapshot', 'root': 'root-snapshot'},
                          'snapshot': 'root-snapshot',
                          'payload_sha256': 'payload-digest',
                          'model_contract_id': 'contract-test',
                          'execution_profile_id': 'profile-test'},
            'parent_payload': str(self.case / 'store' / 'checkpoints' /
                                  'payload-digest.state'),
            'parent_payload_sha256': 'payload-digest'}
        (self.case / 'prepare.json').write_text(json.dumps(self.prepare))
        self.profile = {'execution_profile_id': 'profile-test',
                        'model_contract_id': 'contract-test',
                        'execution': {'prefill_chunk': 2}}
        self.width = self.vocab_size * 4
        self.frames = {}
        for name, suffix in (('A', [3]), ('B', [4])):
            stem = self.case / ('fresh-' + name + '.f32')
            (self.case / ('fresh-' + name + '.tokens')).write_text('1,2,' + str(suffix[0]))
            schedule = {'operator': 'batched-prefill', 'requests': [2, 1],
                        'completed_chunks': [[0, 2], [2, 1]]}
            Path(str(stem) + '.schedule.json').write_text(json.dumps(schedule))
            first = self.root
            second = struct.pack('<5f', 10.0 + suffix[0], 20.0 + suffix[0],
                                 30.0, 40.0, 50.0)
            self.frames[name] = second
            Path(str(stem) + '.requests.f32').write_bytes(first + second)
            generated = [9] if name == 'A' else [10]
            (self.case / ('fresh-' + name + '.stdout')).write_text(json.dumps({
                'prefill_tokens': 3, 'normal_forward_tokens': 1, 'consumed': 4,
                'prompt_consumed': 3, 'generated': generated}))
        self.prepare['branches'] = {
            name: {'result': {'tokens': [9] if name == 'A' else [10]}}
            for name in ('A', 'B')}
        (self.case / 'prepare.json').write_text(json.dumps(self.prepare))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _sha(path):
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_loads_suffix_frame_after_validating_fixture_schedule_and_tokens(self):
        result = load_suffix_reference(self.case, self.profile, vocab_size=self.vocab_size)
        self.assertEqual(result['root_snapshot'], 'root-snapshot')
        self.assertEqual({name: row['suffix_frontier']
                          for name, row in result['branches'].items()}, self.frames)
        self.assertEqual(result['requests'], [2, 1])

    def test_rejects_truncated_request_frontier_frames(self):
        path = self.case / 'fresh-A.f32.requests.f32'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'frontier'):
            load_suffix_reference(self.case, self.profile, vocab_size=self.vocab_size)

    def test_rejects_branch_swap_or_schedule_mismatch(self):
        path = self.case / 'fresh-B.f32.schedule.json'
        schedule = json.loads(path.read_text())
        schedule['requests'] = [1, 2]
        path.write_text(json.dumps(schedule))
        with self.assertRaisesRegex(ValueError, 'schedule'):
            load_suffix_reference(self.case, self.profile, vocab_size=self.vocab_size)

    def test_rejects_reference_token_file_from_other_branch(self):
        (self.case / 'fresh-A.tokens').write_text('1,2,4')
        with self.assertRaisesRegex(ValueError, 'token IDs'):
            load_suffix_reference(self.case, self.profile, vocab_size=self.vocab_size)

    def test_rejects_root_save_receipt_not_bound_to_profile(self):
        self.prepare['root_save']['execution_profile_id'] = 'other-profile'
        (self.case / 'prepare.json').write_text(json.dumps(self.prepare))
        with self.assertRaisesRegex(ValueError, 'root_save'):
            load_suffix_reference(self.case, self.profile, vocab_size=self.vocab_size)

    def test_rejects_wrong_parent_payload_hash(self):
        payload = self.case / 'parent.state'
        payload.write_bytes(b'canonical bytes')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            _verified_payload(payload, '0' * 64)

    def test_rejects_nonfinite_frontier_values(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            _require_finite_f32(struct.pack('<2f', 1.0, float('nan')), 'fixture')

    def test_store_fingerprint_covers_nested_reference_metadata_not_payloads_or_locks(self):
        store = self.case / 'store'
        (store / 'objects').mkdir(parents=True)
        (store / 'root-index').mkdir()
        (store / 'checkpoints').mkdir()
        (store / 'references' / 'references-abc').mkdir(parents=True)
        for relative, value in (
                ('refs.json', b'{}'), ('ref-order.json', b'[]'),
                ('objects/root-snapshot.json', b'{}'),
                ('root-index/root-snapshot.json', b'{}'),
                ('checkpoints/root-snapshot.json', b'{}'),
                ('checkpoints/payload.state', b'large payload excluded'),
                ('refs.lock', b''), ('references/references-abc/event.json', b'{"event":1}')):
            (store / relative).write_bytes(value)
        fingerprint = _small_store_fingerprint(store, 'root-snapshot')
        self.assertIn('references/references-abc/event.json', fingerprint)
        self.assertNotIn('checkpoints/payload.state', fingerprint)
        self.assertNotIn('refs.lock', fingerprint)

    @unittest.skipUnless(os.environ.get('COW_SUFFIX_CASE')
                         and os.environ.get('COW_SUFFIX_PROFILE')
                         and os.environ.get('COW_SUFFIX_API_SOURCE'),
                         'retained Q8 case paths are not configured')
    def test_retained_q8_case_has_bound_suffix_frontiers(self):
        case = Path(os.environ['COW_SUFFIX_CASE'])
        profile = json.loads(Path(os.environ['COW_SUFFIX_PROFILE']).read_text())
        result = load_suffix_reference(case, profile)
        self.assertIn('root_save', json.loads((case / 'prepare.json').read_text()))
        metadata = _small_store_fingerprint(case / 'store', result['root_snapshot'])
        self.assertIn('references/prefix.json', metadata)
        self.assertIn('references', metadata['$directories'])
        self.assertIn('checkpoints/' + result['root_snapshot'] + '.json', metadata)
        self.assertNotIn('checkpoints/' + json.loads((case / 'prepare.json').read_text())[
            'parent_payload_sha256'] + '.state', metadata)
        runtime = _runtime_import_witness(os.environ['COW_SUFFIX_API_SOURCE'], profile)
        self.assertEqual(len(runtime), 7)
        preflight = preflight_suffix_audit(
            os.environ['COW_SUFFIX_PROFILE'], case,
            api_source=os.environ['COW_SUFFIX_API_SOURCE'])
        self.assertEqual(preflight['status'], 'CPU_PREFLIGHT_PASS')
        self.assertFalse(preflight['parent_payload']['content_sha256_verified'])
        self.assertEqual(result['requests'], [65536, 8335])
        self.assertEqual(result['branches']['A']['suffix_frontier_sha256'],
                         '205c9d07b30f1d86b55236a60cef0724d5ca2f03485737ba5e23cda51c3342c3')
        self.assertEqual(result['branches']['B']['suffix_frontier_sha256'],
                         'c1fb87baf8322da1905a0f53770936775df856e5cc19488204e16cb405b271f0')


if __name__ == '__main__':
    unittest.main()
