"""CPU controls for opt-in local pool-trace receipts from cow_http."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys


HTTP = Path(__file__).with_name('cow_http.py')
spec = importlib.util.spec_from_file_location('cow_http_pool_trace_test', HTTP)
cow_http = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cow_http)
API_SRC = Path(__file__).parents[1] / 'experimental' / 'branching_context' / 'src'
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))
from context_time_travel.pools import PoolTransitionTrace
from context_time_travel.pools import SharedCheckpointPool
from context_time_travel.coordinator import ContextCoordinator


class CowHttpPoolTraceTests(unittest.TestCase):
    def test_pool_acquisition_marker_is_opt_in_and_carries_operation_source(self):
        ordinary = SharedCheckpointPool(1024)
        self.assertIsNone(ordinary.record_transition('request_source_acquired', ('m','p','s'),
            operation_id='op', source_snapshot='s'))
        ordinary.close()
        trace = PoolTransitionTrace('pool-test')
        observed = SharedCheckpointPool(1024, transition_trace=trace)
        observed.record_transition('request_source_acquired', ('m','p','s'),
            operation_id='op', source_snapshot='s', source_kind='shared_pool')
        self.assertEqual(trace.snapshot()['events'][0]['operation_id'], 'op')
        observed.close()

    def test_worker_close_trace_records_actual_numeric_exit_and_preserves_errors(self):
        class Process:
            pid = 4321
            returncode = None
        class Backend:
            def __init__(self, code, raises=False):
                self.process = Process()
                self.code, self.raises = code, raises
            def close(self):
                self.process.returncode = self.code
                if self.raises:
                    raise RuntimeError('close failed')
        class Model:
            profile = {'model_contract_id':'model', 'execution_profile_id':'profile'}
            def __init__(self, code, raises=False):
                self.backend = Backend(code, raises)

        trace = PoolTransitionTrace('worker-close')
        coordinator = object.__new__(ContextCoordinator)
        coordinator.pool = SharedCheckpointPool(1024, transition_trace=trace)
        coordinator._close_worker_backend(Model(0), 0)
        with self.assertRaisesRegex(RuntimeError, 'close failed'):
            coordinator._close_worker_backend(Model(7, raises=True), 1)
        rows = trace.snapshot()['events']
        self.assertEqual([row['event'] for row in rows],
                         ['native_worker_closed', 'native_worker_closed'])
        self.assertEqual([(row['gpu_id'], row['pid'], row['returncode'], row['close_succeeded'])
                          for row in rows], [(0, 4321, 0, True), (1, 4321, 7, False)])
        coordinator.pool.close()

    @staticmethod
    def _causal_trace():
        key = ['model', 'profile', 'snapshot-42']
        rows = []
        def add(event, **fields):
            rows.append(dict(sequence=len(rows), monotonic_ns=100+len(rows), event=event,
                             key=list(key), process_id=101, phase_id='phase-1', **fields))
        add('ram_clean_eviction', freed_bytes=4096)
        add('disk_preload_started', load_id=1)
        add('disk_preload_succeeded', load_id=1, success=True, bytes=8192)
        add('http_request_received', branch='A', snapshot='snapshot-42', revision=5,
            operation_id='ask-op', path='/cow/text/ask')
        add('http_request_accepted', branch='A', snapshot='snapshot-42', revision=5,
            operation_id='ask-op', path='/cow/text/ask')
        add('request_source_acquired', branch='A', source_snapshot='snapshot-42',
            operation_id='ask-op', source_kind='shared_pool')
        add('http_request_durable', branch='A', snapshot='snapshot-42', operation_id='ask-op',
            path='/cow/text/ask', result_snapshot='snapshot-43')
        return {'event_count':len(rows),'overflowed':False,'process_id':101,
                'phase_id':'phase-1','events':rows}

    def test_causal_eviction_read_request_audit_accepts_matching_chain(self):
        result = cow_http.validate_pool_spill_reload_trace(self._causal_trace(),
            model_contract_id='model', execution_profile_id='profile', branch='A',
            source_snapshot='snapshot-42', operation_id='ask-op')
        self.assertEqual(result['freed_bytes'], 4096)
        self.assertEqual(result['loaded_bytes'], 8192)
        self.assertEqual(result['result_snapshot'], 'snapshot-43')
        self.assertEqual(result['source_kind'], 'shared_pool')

    def test_causal_audit_accepts_on_demand_disk_load_after_request_received(self):
        trace = self._causal_trace()
        rows = trace['events']
        reordered = [rows[3], rows[0], rows[1], rows[2], rows[4], rows[5], rows[6]]
        for index,row in enumerate(reordered):
            row['sequence']=index;row['monotonic_ns']=200+index
        trace['events']=reordered;trace['event_count']=len(reordered)
        result = cow_http.validate_pool_spill_reload_trace(trace,
            model_contract_id='model', execution_profile_id='profile', branch='A',
            source_snapshot='snapshot-42', operation_id='ask-op')
        self.assertEqual(result['request_received_sequence'], 0)
        self.assertEqual(result['load_succeeded_sequence'], 3)

    def test_causal_audit_rejects_disk_load_that_succeeds_after_source_acquisition(self):
        trace = self._causal_trace()
        rows = trace['events']
        reordered = [rows[0], rows[1], rows[3], rows[4], rows[5], rows[6], rows[2]]
        for index, row in enumerate(reordered):
            row['sequence'] = index
            row['monotonic_ns'] = 300 + index
        trace['events'] = reordered
        trace['event_count'] = len(reordered)
        with self.assertRaisesRegex(ValueError, 'No positive freed-eviction'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')

    def test_causal_audit_rejects_missing_eviction_reload_or_request_success(self):
        trace = self._causal_trace()
        trace['events'][0]['freed_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'No positive freed-eviction'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')
        trace = self._causal_trace()
        trace['events'][2]['success'] = False
        with self.assertRaisesRegex(ValueError, 'No positive freed-eviction'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')
        trace = self._causal_trace()
        trace['events'][5]['source_kind'] = 'model_cache'
        with self.assertRaisesRegex(ValueError, 'unique accepted, acquired, and durable'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')

    def test_causal_audit_rejects_intervening_eviction_wrong_key_and_overflow(self):
        trace = self._causal_trace()
        trace['events'].insert(5, dict(sequence=5, monotonic_ns=105,
            event='ram_clean_eviction', key=['model','profile','snapshot-42'], freed_bytes=16,
            process_id=101,phase_id='phase-1'))
        for index,row in enumerate(trace['events']):
            row['sequence']=index;row['monotonic_ns']=100+index
        trace['event_count']=len(trace['events'])
        with self.assertRaisesRegex(ValueError, 'No positive freed-eviction'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')
        trace = self._causal_trace()
        trace['events'][2]['key'][-1] = 'other-snapshot'
        with self.assertRaisesRegex(ValueError, 'No positive freed-eviction'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')
        trace = self._causal_trace();trace['overflowed']=True
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            cow_http.validate_pool_spill_reload_trace(trace, model_contract_id='model',
                execution_profile_id='profile', branch='A', source_snapshot='snapshot-42',
                operation_id='ask-op')

    def test_http_trace_is_opt_in_and_binds_request_to_canonical_source(self):
        self.assertIsNone(inspect.signature(cow_http.make_server).parameters['transition_trace'].default)
        trace = PoolTransitionTrace('http-session', max_events=8)
        head = {'tip': 'snapshot-42', 'revision': 6}
        adapter = type('Adapter', (), {'head': lambda _self, branch: head})()
        request = cow_http.trace_request_source(trace, ('model', 'profile'), adapter,
            '/cow/text/ask', {'branch': 'branch-A', 'operation_id': 'op-1'})
        self.assertEqual(request, {'key': ('model', 'profile', 'snapshot-42'),
            'branch': 'branch-A', 'snapshot': 'snapshot-42', 'revision': 6,
            'operation_id': 'op-1', 'path': '/cow/text/ask'})
        self.assertIsNone(cow_http.trace_request_source(None, ('m', 'p'), adapter,
            '/cow/text/ask', {'branch': 'branch-A', 'operation_id': 'op-1'}))
        self.assertEqual(trace.snapshot()['events'][0]['event'], 'http_request_received')

    def test_incomplete_or_failed_server_lifecycle_never_publishes_trace(self):
        trace = object()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trace.json'
            self.assertFalse(cow_http.persist_pool_trace(path, trace,
                service_ready=False, coordinator_closed=True))
            self.assertFalse(path.exists())
            self.assertFalse(cow_http.persist_pool_trace(path, trace,
                service_ready=True, coordinator_closed=False))
            self.assertFalse(path.exists())

    def test_graceful_ready_close_publishes_bounded_private_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            os.chmod(temp, 0o700)
            path = Path(temp) / 'trace.json'
            trace = PoolTransitionTrace('http-session', max_events=8)
            trace.record('ram_clean_eviction', ('m', 'p', 's'), freed_bytes=64)
            trace.record('disk_preload_succeeded', ('m', 'p', 's'), success=True, bytes=64)
            request = cow_http.trace_request_source(trace, ('m', 'p'),
                type('Adapter', (), {'head': lambda _self, branch: {'tip': 's', 'revision': 7}})(),
                '/cow/text/ask', {'branch': 'A', 'operation_id': 'ask-op'})
            trace.record('http_request_accepted', request['key'], **{
                key: value for key, value in request.items() if key != 'key'})
            trace.record('request_source_acquired', request['key'], operation_id='ask-op',
                         source_snapshot='s', source_kind='shared_pool')
            trace.record('http_request_durable', request['key'], operation_id='ask-op',
                         source_snapshot='s', result_snapshot='tip-2')
            self.assertTrue(cow_http.persist_pool_trace(path, trace,
                service_ready=True, coordinator_closed=True))
            document = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(document['event_count'], 6)
            self.assertEqual(document['events'][0]['key'], ['m', 'p', 's'])
            self.assertEqual(document['events'][1]['bytes'], 64)
            self.assertLess(document['events'][1]['monotonic_ns'],
                            document['events'][2]['monotonic_ns'])
            self.assertEqual(document['events'][2]['event'], 'http_request_received')
            self.assertEqual(document['events'][2]['operation_id'], 'ask-op')
            self.assertEqual(document['events'][2]['snapshot'], 's')
            self.assertEqual(document['events'][4]['source_kind'], 'shared_pool')
            self.assertEqual(document['events'][5]['event'], 'http_request_durable')
            self.assertEqual(document['events'][5]['result_snapshot'], 'tip-2')
            self.assertTrue(all(type(row['monotonic_ns']) is int for row in document['events']))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                cow_http.persist_pool_trace(path, trace,
                    service_ready=True, coordinator_closed=True)

    def test_overflowed_trace_is_not_published_as_complete_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            os.chmod(temp, 0o700)
            path = Path(temp) / 'trace.json'
            trace = PoolTransitionTrace('http-session', max_events=1)
            trace.record('first', ('m', 'p', 's'))
            trace.record('overflow', ('m', 'p', 's'))
            with self.assertRaisesRegex(ValueError, 'overflow'):
                cow_http.persist_pool_trace(path, trace,
                    service_ready=True, coordinator_closed=True)
            self.assertFalse(path.exists())

    def test_trace_target_preflight_requires_private_directory_and_new_path(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'trace.json'
            os.chmod(temp, 0o755)
            with self.assertRaises(PermissionError):
                cow_http.validate_pool_trace_target(target)
            os.chmod(temp, 0o700)
            self.assertEqual(cow_http.validate_pool_trace_target(target), target.absolute())
            target.write_text('{}', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                cow_http.validate_pool_trace_target(target)


if __name__ == '__main__':
    unittest.main()
