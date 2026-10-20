"""CPU contract tests for the real Q4 CUDA-allocation-failure harness.

These tests validate only fail-closed evidence and orchestration controls. They
do not emulate CUDA or qualify an actual device allocation failure.
"""
from __future__ import annotations

import sys
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from run_q4_cuda_oom import (
    OOMProbeError,
    validate_gate_record,
    validate_helper_ready,
    validate_helper_pressure,
    validate_native_oom,
    validate_owned_process_exit,
    validate_probe_result,
    run_status_exit_code,
    validate_fault_http_events,
    validate_import_witness,
    ServiceStartupFailure,
    _start_service_owned,
    _stop_owned_helper,
    _proc_identity,
    _finish_and_persist,
    build_import_expectations,
    main,
)
import run_q4_cuda_oom as probe


class CudaOomEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            'forward_tokens': 0,
            'prefill_tokens': 0,
            'vmm_driver_failures': 0,
            'vmm_out_of_memory_failures': 0,
            'vmm_last_driver_error': 0,
            'vmm_last_failed_operation': 'none',
        }

    def test_gate_is_bound_to_exact_active_generate_and_authoritative_stats(self):
        gate = {'event': 'before_generate', 'operation_id': 'op-q4-oom-1',
                'branch': 'BranchA', 'command': 'generate 16 2 1 2',
                'native_pid': 4401, 'stats': dict(self.baseline),
                'request_start_stats': dict(self.baseline), 'injected_commands': ['pages', 'stats'],
                'request_active': True, 'pages': [{'pool': 0, 'mapped_bytes': 4096}]}
        self.assertEqual(validate_gate_record(gate, operation_id='op-q4-oom-1',
            branch='BranchA', native_pid=4401, expected_forward=0)['event'], 'before_generate')
        for bad in (
            {**gate, 'operation_id': 'other'},
            {**gate, 'branch': 'BranchB'},
            {**gate, 'native_pid': 0},
            {**gate, 'request_active': False},
            {**gate, 'stats': {**self.baseline, 'forward_tokens': 8000}},
                    {**gate, 'request_start_stats': {**self.baseline, 'prefill_tokens': 1}},
                    {**gate, 'stats': {**self.baseline, 'prefill_tokens': 1}},
            {**gate, 'command': 'advance'},
        ):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_gate_record(bad, operation_id='op-q4-oom-1', branch='BranchA',
                                     native_pid=4401, expected_forward=0)

    def test_helper_ready_requires_real_vmm_identity(self):
        self.assertEqual(validate_helper_ready({'status': 'ready', 'granularity': 65536})['granularity'], 65536)
        for bad in ({'status': 'ready', 'granularity': 0},
                    {'status': 'error', 'granularity': 65536},
                    {'status': 'ready', 'granularity': True}, 'ready'):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_helper_ready(bad)

    def test_pressure_requires_actual_driver_oom_and_released_helper(self):
        pressure = {'status': 'full', 'held_bytes': 1024, 'physical_allocations': 1,
                    'free_before': 4096, 'free_after': 0, 'total_bytes': 8192,
                    'driver_failures': 1, 'out_of_memory_failures': 1,
                    'last_error': 2, 'last_operation': 'cuMemCreate'}
        release = {'status': 'released', 'released_bytes': 1024, 'free_after': 4096}
        self.assertEqual(validate_helper_pressure(pressure, release)['out_of_memory_failures'], 1)
        for bad in ({**pressure, 'last_error': 0},
                    {**pressure, 'last_operation': 'cuMemMap'},
                    {**pressure, 'out_of_memory_failures': 0},
                    {**pressure, 'status': 'error'},
                    {**pressure, 'held_bytes': 0},
                    {**pressure, 'physical_allocations': 0}):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_helper_pressure(bad, release)
        with self.assertRaises(OOMProbeError):
            validate_helper_pressure(pressure, {**release, 'released_bytes': 0})

    def test_native_error_requires_oom_counter_delta_without_forward_work(self):
        after = {**self.baseline, 'vmm_out_of_memory_failures': 1,
                 'vmm_driver_failures': 1, 'vmm_last_driver_error': 2,
                 'vmm_last_failed_operation': 'cuMemCreate'}
        accepted = validate_native_oom({'status': 'error', 'message': 'CUDA error 2'},
            self.baseline, after, expected_forward=0)
        self.assertEqual(accepted['forward_delta'], 0)
        for response, counters in (
            ({'status': 'ok'}, after),
            ({'status': 'error', 'message': 'CUDA error 2'}, self.baseline),
            ({'status': 'error', 'message': 'CUDA error 2'}, {**after, 'forward_tokens': 8000}),
            ({'status': 'error', 'message': 'CUDA error 2'}, {**after, 'vmm_last_failed_operation': 'cuMemMap'}),
            ({'status': 'error', 'message': 'CUDA error 2'}, {**after, 'vmm_last_driver_error': 0}),
        ):
            with self.subTest(response=response, counters=counters), self.assertRaises(OOMProbeError):
                validate_native_oom(response, self.baseline, counters, expected_forward=0)

    def test_owned_process_exit_never_accepts_unknown_or_foreign_identity(self):
        valid = {'pid': 4410, 'start_time': '1234', 'exe': '/selected/helper', 'exit_code': 0,
                 'pid_absent_after_wait': True}
        validate_owned_process_exit(valid, expected_pid=4410, expected_start_time='1234',
                                    expected_exe='/selected/helper')
        for bad in ({**valid, 'pid': 4411}, {**valid, 'start_time': 'other'},
                    {**valid, 'exe': '/other'}, {**valid, 'exit_code': None},
                    {**valid, 'exit_code': 1}):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_owned_process_exit(bad, expected_pid=4410, expected_start_time='1234',
                                            expected_exe='/selected/helper')

    def test_result_requires_original_fault_and_restart_completion_for_pass(self):
        result = {'status': 'PASS', 'native_oom': {'response': {'status': 'error'}, 'oom_failure_delta': 1},
                  'same_service_followup': {'event': 'complete', 'tokens': [7]},
                  'restart_followup': {'event': 'complete', 'tokens': [7]}}
        self.assertEqual(validate_probe_result(result), 'PASS')
        fail_recovery = {**result, 'status': 'FAIL-RECOVERY',
            'same_service_followup': {'event': 'error'},
            'restart_followup': {'event': 'complete', 'tokens': [8]}}
        self.assertEqual(validate_probe_result(fail_recovery), 'FAIL-RECOVERY')
        for bad in ({**result, 'native_oom': {'status': 'ok'}},
                    {**result, 'restart_followup': {'event': 'error'}},
                    {**result, 'status': 'PASS', 'same_service_followup': {'event': 'complete'}}):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_probe_result(bad)

    def test_runner_exit_matches_result_status_including_expected_recovery_failure(self):
        valid = {'status': 'PASS', 'native_oom': {'response': {'status': 'error'}, 'oom_failure_delta': 1},
                 'same_service_followup': {'event': 'complete', 'tokens': [1]},
                 'restart_followup': {'event': 'complete', 'tokens': [2]}}
        self.assertEqual(run_status_exit_code(valid), 0)
        fail_recovery = {**valid, 'status': 'FAIL-RECOVERY', 'same_service_followup': {'event': 'error'}}
        self.assertEqual(run_status_exit_code(fail_recovery), 1)
        for bad in ({}, {'status': 'RUNNING'}, {'status': 'MISMATCH'}):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                run_status_exit_code(bad)

    def test_timeout_is_a_failure_not_fail_recovery(self):
        result = {'status': 'FAIL', 'error': {'type': 'TimeoutError', 'message': 'no terminal event'},
            'native_oom': {'response': {'status': 'error'}, 'oom_failure_delta': 1},
            'same_service_followup': {'event': 'timeout'},
            'restart_followup': {'event': 'complete', 'tokens': [8]}}
        self.assertEqual(validate_probe_result(result), 'FAIL')
        self.assertEqual(run_status_exit_code(result), 1)
        mislabeled = {**result, 'status': 'FAIL-RECOVERY'}
        with self.assertRaises(OOMProbeError):
            validate_probe_result(mislabeled)

    def test_readiness_failure_reaps_only_the_started_owned_service(self):
        class Service:
            process = None

            def start(self):
                self.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                raise RuntimeError('readiness rejected profile')

            def stop(self):
                self.process.terminate()
                code = self.process.wait(timeout=5)
                return {'exit_code': code, 'pid': self.process.pid}

        service = Service()
        with self.assertRaises(ServiceStartupFailure) as caught:
            _start_service_owned(service)
        receipt = caught.exception.cleanup_receipt
        self.assertEqual(receipt['pid'], service.process.pid)
        self.assertEqual(receipt['stop']['pid'], service.process.pid)
        self.assertIsNotNone(receipt['returncode'])
        self.assertTrue(receipt['pid_absent'])
        self.assertIsNotNone(service.process.poll())

    def test_helper_cleanup_releases_and_verifies_owned_child(self):
        code = "import json,sys; sys.stdin.readline(); print(json.dumps({'status':'released','released_bytes':1,'free_after':2}),flush=True)"
        process = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
            start_new_session=True)
        identity = _proc_identity(process.pid)
        receipt = _stop_owned_helper(process, identity, 5)
        self.assertEqual(receipt['terminal']['exit_code'], 0)
        self.assertTrue(receipt['terminal']['pid_absent_after_wait'])
        self.assertEqual(receipt['cleanup_errors'], [])

    def test_terminal_result_is_written_after_progress_thread_closes(self):
        class Progress:
            closed = False

            def finish(self, status, error):
                self.status = status

            def close(self):
                self.closed = True

        progress = Progress()
        result = {'status': 'PASS'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            def write_after_close(destination, value):
                self.assertTrue(progress.closed)
                destination.write_text(json.dumps(value), encoding='utf-8')
            with patch.object(probe, '_atomic_json', side_effect=write_after_close):
                error = _finish_and_persist(result, path, progress, [], None)
            self.assertIsNone(error)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'], 'PASS')

    def test_cleanup_failure_changes_terminal_status_and_exit(self):
        class Progress:
            def finish(self, status, error):
                self.status = status

            def close(self):
                pass

        result = {'status': 'PASS'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            error = _finish_and_persist(result, path, Progress(),
                [{'resource': 'helper', 'type': 'Timeout', 'message': 'still alive'}], None)
            self.assertIsInstance(error, OOMProbeError)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'], 'FAIL')

    def test_fault_stream_requires_one_error_and_no_token_or_durable_ack(self):
        valid = [{'event': 'error', 'error': 'native CUDA OOM'}]
        self.assertEqual(validate_fault_http_events(valid)['event_count'], 1)
        for bad in ([], [{'event': 'complete', 'receipt': {}}],
                    [{'event': 'error', 'error': 'OOM'}, {'event': 'durable', 'receipt': {}}],
                    [{'event': 'error', 'error': 'OOM'}, {'event': 'error', 'error': 'again'}],
                    [{'event': 'error', 'error': ''}],
                    [{'event': 'error', 'error': 'OOM', 'receipt': {'snapshot_id': 'bad'}}],
                    [{'event': 'token', 'token_id': 5}, {'event': 'error', 'error': 'OOM'}]):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_fault_http_events(bad)

    def test_import_witness_requires_exact_pinned_module_origins(self):
        names = ('context_time_travel.real','context_time_travel.coordinator',
                 'context_time_travel.strata_backend','context_time_travel.execution_receipt',
                 'context_time_travel.checkpoint_cache','serve.cow_http')
        modules = {name: {'path': f'/runtime/{name}.py', 'sha256': f'{index:064x}'}
                   for index, name in enumerate(names, 1)}
        expected = {row['path']: row['sha256'] for row in modules.values()}
        valid = {'event': 'runtime_import_witness', 'modules': modules}
        self.assertEqual(set(validate_import_witness(valid, expected)['modules']), set(names))
        wrong_path = {**valid, 'modules': {**modules, names[0]:
            {'path': '/old/runtime/real.py', 'sha256': modules[names[0]]['sha256']}}}
        wrong_hash = {**valid, 'modules': {**modules, names[1]:
            {**modules[names[1]], 'sha256': '0' * 64}}}
        for bad in (wrong_path, wrong_hash,
                    {'event': 'runtime_import_witness', 'modules': {k:v for k,v in modules.items() if k != names[-1]}}):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_import_witness(bad, expected)

    def test_http_source_is_independently_pinned_outside_profile_api_closure(self):
        profile_api = {'/pinned/api/real.py': 'a' * 64,
                       '/pinned/api/coordinator.py': 'b' * 64}
        with tempfile.TemporaryDirectory() as directory:
            http = Path(directory) / 'cow_http.py'
            http.write_text('service source', encoding='utf-8')
            import hashlib
            http_sha = hashlib.sha256(http.read_bytes()).hexdigest()
            combined, service = build_import_expectations(profile_api,
                http_path=http, http_sha256=http_sha, expected_http_sha256=http_sha)
            self.assertEqual(combined['/pinned/api/real.py'], 'a' * 64)
            self.assertEqual(combined[str(http.resolve())], http_sha)
            self.assertEqual(service['binding'], 'independent_explicit_service_source_pin')
            with self.assertRaises(OOMProbeError):
                build_import_expectations(profile_api, http_path=http,
                    http_sha256=http_sha, expected_http_sha256='0' * 64)
            with self.assertRaises(OOMProbeError):
                build_import_expectations({**profile_api, str(http.resolve()): http_sha},
                    http_path=http, http_sha256=http_sha, expected_http_sha256=http_sha)

    def test_preflight_only_reports_prepared_not_run_without_creating_run_or_starting_probe(self):
        fake = {'profile_sha256': 'a' * 64, 'native_sha256': 'b' * 64,
                'service_build_source': {'http': {'sha256': 'c' * 64}},
                'profile_api_imports': {'/api/real.py': 'd' * 64}}
        output = io.StringIO()
        argv = ['--profile', '/unused/profile.json', '--store', '/unused/store',
            '--runtime-root', '/unused/runtime', '--api-src', '/unused/api/src',
            '--api-scripts', '/unused/api/scripts', '--run-dir', '/unused/run',
            '--expected-profile-sha256', 'a' * 64, '--expected-native-sha256', 'b' * 64,
            '--expected-http-sha256', 'c' * 64, '--expected-helper-sha256', 'e' * 64,
            '--pressure-helper', '/unused/helper', '--expected-source-snapshot', 'snapshot',
            '--preflight-only']
        with (
            patch.object(probe, 'preflight_configuration', return_value=fake),
            patch.object(probe, 'run_probe', side_effect=AssertionError('preflight started runtime')),
            redirect_stdout(output),
        ):
            self.assertEqual(main(argv), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['status'], 'PREPARED_NOT_RUN')
        self.assertEqual(value['service_build_source']['http']['sha256'], 'c' * 64)


if __name__ == '__main__':
    unittest.main()
