"""CPU evidence-auditor tests; these fixtures perform no inference."""
import copy
import unittest

from audit_cow_client_pressure import audit_revisit


class ClientPressureAuditTests(unittest.TestCase):
    def fixture(self):
        key = ['model', 'profile', 'source']
        rows = []
        def add(event, **fields):
            rows.append(dict(sequence=len(rows), monotonic_ns=100+len(rows),
                event=event, key=key, process_id=101, phase_id='phase', **fields))
        add('ram_clean_eviction', freed_bytes=4096)
        add('disk_preload_started', load_id=1)
        add('disk_preload_succeeded', load_id=1, success=True, bytes=4096)
        add('http_request_received', branch='branch', snapshot='source', revision=2,
            operation_id='op', path='/cow/text/resume')
        add('http_request_accepted', branch='branch', snapshot='source', revision=2,
            operation_id='op', path='/cow/text/resume')
        add('request_source_acquired', branch='branch', source_snapshot='source',
            operation_id='op', source_kind='shared_pool')
        add('http_request_durable', branch='branch', snapshot='source',
            operation_id='op', path='/cow/text/resume', result_snapshot='result')
        revisit = dict(path='/cow/text/resume', request_id='cdp-1',
            request=dict(branch='branch', revision=2, operation_id='op', max_tokens=128),
            source_tip='source', new_tip='result', generated_tokens=16,
            counter_deltas=dict(forward_tokens=16, prefill_tokens=0),
            client_ack=dict(request_id='cdp-1', operation_id='op', branch='branch',
                response_status=200, eof_consumed=True,
                receipt=dict(snapshot='result', branch='branch', revision=3)))
        report = dict(status='PASS', inference_kind='real',
            server_model_contract_id_from_durable_receipts='model',
            server_execution_profile_id_from_durable_receipts='profile',
            reusable_context_report=dict(revisited_oldest_branch=revisit))
        trace = dict(event_count=len(rows), overflowed=False, process_id=101,
            phase_id='phase', events=rows)
        return report, trace

    def test_exact_revisit_requires_actual_same_key_freed_reload(self):
        report, trace = self.fixture()
        result = audit_revisit(report, trace)
        self.assertEqual(result['status'], 'PASS_REQUEST_BOUND_SPILL_RELOAD')
        self.assertEqual(result['prefix_forward_tokens'], 0)
        self.assertEqual(result['causal_chain']['freed_bytes'], 4096)
        self.assertEqual(result['numerical_audit'], 'SEPARATE_REQUIRED')

    def test_no_eviction_cannot_qualify(self):
        report, trace = self.fixture()
        trace['events'][0]['freed_bytes'] = 0
        with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_uncommitted_or_different_complete_cannot_qualify(self):
        for field, value in [('eof_consumed', False), ('operation_id', 'other'),
                             ('request_id', 'other'), ('response_status', 500)]:
            with self.subTest(field=field):
                report, trace = self.fixture()
                report['reusable_context_report']['revisited_oldest_branch']['client_ack'][field] = value
                with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_recomputed_prefix_or_wrong_counter_type_cannot_qualify(self):
        for field, value in [('forward_tokens', 17), ('forward_tokens', True),
                             ('prefill_tokens', 1), ('prefill_tokens', False)]:
            with self.subTest(field=field):
                report, trace = self.fixture()
                report['reusable_context_report']['revisited_oldest_branch']['counter_deltas'][field] = value
                with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_missing_work_and_zero_outputs_cannot_qualify(self):
        for mode in ('missing', 'empty'):
            report, trace = self.fixture()
            revisit = report['reusable_context_report']['revisited_oldest_branch']
            if mode == 'missing': del revisit['counter_deltas']['prefill_tokens']
            else: revisit['generated_tokens'] = 0
            with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_trace_result_must_equal_acknowledged_result(self):
        report, trace = self.fixture()
        trace['events'][-1]['result_snapshot'] = 'different'
        with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_success_after_acquisition_cannot_qualify(self):
        report, trace = self.fixture()
        rows = trace['events']
        trace['events'] = [rows[i] for i in (0, 1, 3, 4, 5, 2, 6)]
        for i, row in enumerate(trace['events']):
            row.update(sequence=i, monotonic_ns=100+i)
        with self.assertRaises(ValueError): audit_revisit(report, trace)

    def test_inputs_are_unchanged_and_nonreal_report_rejected(self):
        report, trace = self.fixture()
        before = copy.deepcopy((report, trace))
        audit_revisit(report, trace)
        self.assertEqual((report, trace), before)
        report['inference_kind'] = 'mock'
        with self.assertRaises(ValueError): audit_revisit(report, trace)


if __name__ == '__main__': unittest.main()
