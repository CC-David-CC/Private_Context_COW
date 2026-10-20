"""CPU-only structural tests for the separate cold-revisit auditor.

These tests validate report/store-binding helpers with tiny synthetic data.
They launch no browser, model, native reference, or GPU workload.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[2]
SCRIPT = HERE / 'scripts' / 'audit_cow_browser_revisit.py'
SPEC = importlib.util.spec_from_file_location('audit_cow_browser_revisit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ColdBrowserRevisitAuditTests(unittest.TestCase):
    def _report(self):
        row = {'name': 'native_cow_engine', 'namespace_id': 'ns-a', 'report': {
            'ask_a_request': {'operation_id': 'answer-a', 'receipt': {
                'branch': {'name': 'branch-a', 'tip': 'snapshot-a128'}}}}}
        revisit = {'source_tip': 'snapshot-a128', 'request': {
            'operation_id': 'resume-a', 'branch': 'branch-a', 'revision': 2,
            'max_tokens': 128, 'stream': True}}
        return {'reusable_context_report': {
            'contexts': [row, {'name': 'other', 'report': {}}],
            'revisited_oldest_branch': revisit}}

    def test_actual_nested_report_shape_selects_saved_a128_branch(self):
        row, nested, revisit = audit._select_oldest_context(self._report())
        self.assertEqual(row['name'], 'native_cow_engine')
        self.assertEqual(nested['ask_a_request']['operation_id'], 'answer-a')
        self.assertEqual(revisit['request']['operation_id'], 'resume-a')

    def test_wrong_source_branch_or_ambiguous_tip_is_rejected(self):
        report = self._report()
        report['reusable_context_report']['revisited_oldest_branch']['request']['branch'] = 'branch-b'
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit._select_oldest_context(report)
        report = self._report()
        report['reusable_context_report']['contexts'].append(
            report['reusable_context_report']['contexts'][0])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit._select_oldest_context(report)

    def _transition(self):
        root, suffix = [1, 2], [3, 4]
        first = list(range(100, 228))
        resumed = list(range(300, 428))
        source = root + suffix + first
        final = source + resumed
        request = {'operation_id': 'resume-a', 'branch': 'branch-a',
                   'revision': 2, 'max_tokens': 128, 'stream': True}
        ack = {'operation_id': 'resume-a', 'branch': 'branch-a',
               'response_status': 200, 'eof_consumed': True,
               'receipt': {'snapshot': 'snapshot-a256', 'branch': 'branch-a',
                           'revision': 3}}
        policy = {'temperature': 0.0, 'grammar': None, 'stop_sequences': []}
        source_controller = {'schema': 1, 'status': 'active', 'stream_key': 'stream-a',
            'policy': policy, 'next_draw': 0, 'accepted_in_scope': 128}
        final_controller = {**source_controller, 'accepted_in_scope': 256}
        counters = {'prefill_tokens': 0, 'forward_tokens': 128,
            'consumed_tokens': 128, 'decode_timed_tokens': 128,
            'loaded_file_bytes': 323649780, 'checkpoint_import_bytes': 323649780,
            'integrity_mismatches': 0}
        return locals()

    def test_resume_transition_binds_history_controller_and_reload_counters(self):
        v = self._transition()
        actual = audit._validate_resume_transition(root_ids=v['root'],
            suffix_ids=v['suffix'], first_tokens=v['first'], source_history=v['source'],
            final_history=v['final'], request=v['request'], ack=v['ack'],
            source_controller=v['source_controller'], final_controller=v['final_controller'],
            counters=v['counters'], source_branch='branch-a', source_revision=2,
            source_tip='snapshot-a128', final_tip='snapshot-a256',
            final_parent='snapshot-a128', report_generated=128, policy=v['policy'])
        self.assertEqual(actual, v['resumed'])

    def test_resume_rejects_changed_source_history_tokens_and_work_counters(self):
        v = self._transition()
        kwargs = dict(root_ids=v['root'], suffix_ids=v['suffix'],
            first_tokens=v['first'], source_history=v['source'], final_history=v['final'],
            request=v['request'], ack=v['ack'], source_controller=v['source_controller'],
            final_controller=v['final_controller'], counters=v['counters'],
            source_branch='branch-a', source_revision=2, source_tip='snapshot-a128',
            final_tip='snapshot-a256', final_parent='snapshot-a128', report_generated=128,
            policy=v['policy'])
        kwargs['source_history'] = v['source'][:-1] + [999]
        with self.assertRaisesRegex(ValueError, 'canonical history'):
            audit._validate_resume_transition(**kwargs)
        kwargs['source_history'] = v['source']
        kwargs['counters'] = {**v['counters'], 'prefill_tokens': 1}
        with self.assertRaisesRegex(ValueError, 'suffix-only decode work'):
            audit._validate_resume_transition(**kwargs)
        kwargs['counters'] = v['counters']
        kwargs['ack'] = {**v['ack'], 'receipt': {'snapshot': 'changed'}}
        with self.assertRaisesRegex(ValueError, 'durable acknowledgement'):
            audit._validate_resume_transition(**kwargs)

    def test_capture_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {'operation_id': 'resume-a', 'branch': 'branch-a'}
            req_raw = json.dumps(request).encode()
            stream_raw = b'{"event":"token"}\n'
            (root / 'request.json').write_bytes(req_raw)
            (root / 'stream.ndjson').write_bytes(stream_raw)
            report = self._report()
            revisit = report['reusable_context_report']['revisited_oldest_branch']
            revisit['request'] = request
            revisit['client_ack'] = {'raw_ndjson_bytes': len(stream_raw),
                'raw_ndjson_sha256': sha(stream_raw), 'operation_id': 'resume-a',
                'branch': 'branch-a', 'response_status': 200, 'eof_consumed': True,
                'receipt': {'snapshot': 'snapshot-a256', 'branch': 'branch-a', 'revision': 3}}
            revisit['evidence'] = {'request': 'request.json', 'client_stream': 'stream.ndjson',
                'client_ack': {'bytes': len(stream_raw), 'sha256': sha(stream_raw)}}
            report = {'reusable_context_report': {
                'contexts': report['reusable_context_report']['contexts'],
                'revisited_oldest_branch': revisit}}
            # request capture and stream checks use the producer's nested evidence shape.
            evidence = audit._load_resume_evidence(report, root)
            self.assertEqual(evidence['request'], request)
            (root / 'stream.ndjson').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'client stream'):
                audit._load_resume_evidence(report, root)

    def test_complete_controller_continuation_carries_utf8_and_all_state(self):
        sys.path.insert(0, str(HERE / 'src'))
        from context_time_travel import controller
        policy = {'temperature': 0.0, 'eos_tokens': [], 'stop_sequences': []}
        source = controller.start(policy, 'stream', (1, 2), 16)
        source = controller.consume_many(source, [(5, b'\xe2', False)])
        token_data = {6: b'\x82\xac', 7: b'x'}
        events = [(6, token_data[6], False), (7, token_data[7], False)]
        expected = controller.consume_many(source, events)
        actual = audit._validate_complete_controller(source, expected, [6, 7],
            token_data.__getitem__, [])
        self.assertEqual(actual, expected)
        self.assertEqual(actual['output_pending_hex'], '')
        self.assertEqual(actual['accepted_in_scope'], 3)
        changed = dict(expected)
        changed['rng_algorithm'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'rng_algorithm'):
            audit._validate_complete_controller(source, changed, [6, 7],
                token_data.__getitem__, [])


if __name__ == '__main__':
    unittest.main()
