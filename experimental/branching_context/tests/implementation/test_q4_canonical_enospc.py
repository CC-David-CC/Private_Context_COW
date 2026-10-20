"""CPU-only evidence guards for the canonical-reference ENOSPC harness."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import redirect_stdout
import io

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_q4_canonical_enospc as probe


class CanonicalEnospcControls(unittest.TestCase):
    def setUp(self):
        self.operation = 'op-publication-q4-test'
        self.branch = 'CliA-fault-test'
        self.target = 'a' * 64
        self.ref_path = '/private/store/references/CliA-fault-test.json'
        self.old_sha = 'b' * 64

    def valid_mount(self):
        return dict(mount_namespace_verified=True,
            current_mount_namespace='mnt:[4026533000]', parent_mount_namespace='mnt:[4026531832]',
            image=dict(sha256='c' * 64, uuid='4df1febe-d3f8-401e-908a-81da3824b712',
                       loop_device='/dev/loop77', inode=5254728),
            canonical=dict(mountpoint='/fixture/store-mount', filesystem='ext4', device='/dev/loop77'),
            checkpoint=dict(mountpoint='/fixture/store-mount/checkpoints', filesystem='ext4',
                            device='/dev/loop78', st_dev=1235))

    def valid_mount_with_stat(self):
        receipt = self.valid_mount()
        receipt['canonical']['st_dev'] = 1234
        return receipt

    def valid_writer(self):
        return dict(operation_id=self.operation, branch=self.branch, branch_write=True,
            reference_path=self.ref_path, old_reference_sha256=self.old_sha,
            reference_sha256_after=self.old_sha, target_snapshot=self.target,
            exception_type='OSError', errno=errno.ENOSPC, message='[Errno 28] No space left on device',
            stage_files_after=[], target_checkpoint_valid=True, source_checkpoint_unchanged=True,
            serialized_reference_bytes=2814)

    def test_mount_guard_requires_exact_private_ext4_and_separate_checkpoint_device(self):
        probe.validate_mount_layout(self.valid_mount_with_stat(), image_sha256='c' * 64,
            image_uuid='4df1febe-d3f8-401e-908a-81da3824b712', image_inode=5254728,
            canonical_mount='/fixture/store-mount',
            checkpoint_directory='/fixture/store-mount/checkpoints')
        for key, value in [('mount_namespace_verified', False)]:
            bad = self.valid_mount_with_stat(); bad[key] = value
            with self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_mount_layout(bad, image_sha256='c' * 64,
                    image_uuid='4df1febe-d3f8-401e-908a-81da3824b712', image_inode=5254728,
                    canonical_mount='/fixture/store-mount', checkpoint_directory='/fixture/store-mount/checkpoints')
        bad = self.valid_mount_with_stat(); bad['checkpoint']['device'] = '/dev/loop77'; bad['checkpoint']['st_dev'] = 1234
        with self.assertRaises(probe.CanonicalEnospcError):
            probe.validate_mount_layout(bad, image_sha256='c' * 64,
                image_uuid='4df1febe-d3f8-401e-908a-81da3824b712', image_inode=5254728,
                canonical_mount='/fixture/store-mount', checkpoint_directory='/fixture/store-mount/checkpoints')
        bad = self.valid_mount_with_stat(); bad['current_mount_namespace'] = 'mnt:[not-a-number]'
        with self.assertRaises(probe.CanonicalEnospcError):
            probe.validate_mount_layout(bad, image_sha256='c' * 64,
                image_uuid='4df1febe-d3f8-401e-908a-81da3824b712', image_inode=5254728,
                canonical_mount='/fixture/store-mount', checkpoint_directory='/fixture/store-mount/checkpoints')

    def test_fault_requires_exact_branch_reference_writer_and_preserved_old_ref(self):
        accepted = probe.validate_reference_fault(self.valid_writer(), operation_id=self.operation,
            branch=self.branch, reference_path=self.ref_path, old_reference_sha256=self.old_sha,
            target_snapshot=self.target)
        self.assertEqual(accepted['target_snapshot'], self.target)
        for key, value in [('operation_id', 'op-other'), ('branch', 'other'),
                           ('reference_path', '/wrong/path'), ('branch_write', False),
                           ('errno', errno.EIO), ('exception_type', 'RuntimeError'),
                           ('old_reference_sha256', 'd' * 64), ('stage_files_after', ['.stage-x']),
                           ('reference_sha256_after', None)]:
            bad = self.valid_writer(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_reference_fault(bad, operation_id=self.operation, branch=self.branch,
                    reference_path=self.ref_path, old_reference_sha256=self.old_sha,
                    target_snapshot=self.target)

    def test_stream_requires_one_original_errno_error_and_no_durable_ack(self):
        events = [{'event': 'token', 'token_id': 42},
                  {'event': 'error', 'cancelled': False,
                   'error': '[Errno 28] No space left on device'}]
        probe.validate_error_stream(events)
        for bad in [events + [{'event': 'complete'}],
                    [{'event': 'error', 'cancelled': False, 'error': '[Errno 5] Input/output error'}],
                    [{'event': 'durable'}, events[-1]],
                    [{'event': 'error', 'cancelled': False, 'error': '[Errno 28] No space left on device',
                      'durable': True}],
                    [{'event': 'error', 'cancelled': False, 'error': '[Errno 28] No space left on device',
                      'receipt': {'snapshot': self.target}}],
                    [events[-1], events[-1]], []]:
            with self.subTest(events=bad), self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_error_stream(bad)

    def test_same_operation_retry_and_following_request_guards(self):
        status = dict(state='failed', operation_id=self.operation, branch=self.branch,
                      durable=False, publication_retryable=True)
        old = dict(name=self.branch, revision=4, tip='d' * 64)
        probe.validate_retryable_failure(status, old, dict(old), operation_id=self.operation,
            branch=self.branch, parent_snapshot=old['tip'])
        with self.assertRaises(probe.CanonicalEnospcError):
            probe.validate_retryable_failure(status, old, dict(old, revision=5),
                operation_id=self.operation, branch=self.branch, parent_snapshot=old['tip'])
        receipt = dict(snapshot=self.target, branch=dict(name=self.branch, revision=5, tip=self.target))
        durable_status = dict(operation_id=self.operation, branch=self.branch, state='durable',
                              durable=True, receipt=receipt)
        probe.validate_retry(dict(event='complete', receipt=receipt), operation_id=self.operation,
            target_snapshot=self.target, branch=self.branch, revision_before=4,
            moves_before=['op-old'], moves_after=['op-old', self.operation],
            operation_status=durable_status, work_before=b'native-work', work_after=b'native-work')
        with self.assertRaises(probe.CanonicalEnospcError):
            probe.validate_retry(dict(event='complete', receipt=receipt), operation_id=self.operation,
                target_snapshot=self.target, branch=self.branch, revision_before=4,
                moves_before=[], moves_after=[self.operation, self.operation],
                operation_status=durable_status, work_before=b'native-work', work_after=b'changed')

    def test_target_checkpoint_is_required_even_when_source_branch_stays_put(self):
        probe.validate_target_checkpoint(dict(snapshot=self.target, payload_sha256='c' * 64,
            manifest_sha256='d' * 64, payload_bytes=327985908),
                                         target_snapshot=self.target)
        for bad in [None, dict(snapshot=self.target, payload_sha256='f' * 64),
                    dict(snapshot='e' * 64, payload_sha256='c' * 64,
                         manifest_sha256='d' * 64, payload_bytes=327985908)]:
            with self.subTest(checkpoint=bad), self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_target_checkpoint(bad, target_snapshot=self.target)

    def test_receipt_is_fsynced_and_verified_before_release_and_eio_aborts(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt_path = Path(temp) / 'filler-receipt.json'
            calls = []
            receipt = dict(schema='canonical-enospc-filler-v1', errno=errno.ENOSPC,
                fill_path='/bounded/.filler', filled_bytes=60 * 1024 * 1024,
                allocated_bytes=60 * 1024 * 1024,
                available_bytes_after=0,
                filesystem_write_error='[Errno 28] No space left on device')
            def release():
                saved = json.loads(receipt_path.read_text(encoding='utf-8'))
                self.assertEqual(saved, receipt)
                probe.validate_filler_receipt(saved, canonical_mount='/bounded',
                    max_fill_bytes=64 * 1024 * 1024)
                calls.append('release')
            probe.persist_receipt_before_release(receipt_path, receipt, release, canonical_st_dev=123)
            self.assertEqual(calls, ['release'])
            self.assertEqual(json.loads(receipt_path.read_text(encoding='utf-8')), receipt)

            def persistence_fails(_path, _value):
                raise OSError(errno.EIO, 'receipt device failed')
            old = probe._atomic_json
            try:
                probe._atomic_json = persistence_fails
                calls.clear()
                with self.assertRaises(OSError) as caught:
                    probe.persist_receipt_before_release(receipt_path, receipt, release, canonical_st_dev=123)
                self.assertEqual(caught.exception.errno, errno.EIO)
                self.assertEqual(calls, [])
                with self.assertRaises(probe.CanonicalEnospcError):
                    probe.persist_receipt_before_release(receipt_path, receipt, release,
                        canonical_st_dev=receipt_path.parent.stat().st_dev)
                self.assertEqual(calls, [])
            finally:
                probe._atomic_json = old

    def test_request_completion_immediately_after_release_is_not_misclassified(self):
        class CompletingFuture:
            complete = False
            def done(self):
                return self.complete
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt_path = root / 'filler-receipt.json'
            release_path = root / 'release.json'
            future = CompletingFuture()
            receipt = dict(errno=errno.ENOSPC, fill_path='/bounded/.fill', filled_bytes=4096,
                allocated_bytes=4096, available_bytes_after=0,
                filesystem_write_error='[Errno 28] No space left on device')
            def release():
                probe._require_request_waiting_at_release(future)
                release_path.write_text('released', encoding='utf-8')
                future.complete = True  # Simulate fast kernel/request completion after release.
            probe.persist_receipt_before_release(receipt_path, receipt, release,
                canonical_st_dev=root.stat().st_dev + 1)
            self.assertTrue(future.done())
            self.assertEqual(release_path.read_text(encoding='utf-8'), 'released')
            self.assertEqual(json.loads(receipt_path.read_text(encoding='utf-8')), receipt)

    def test_request_already_finished_before_release_is_rejected(self):
        class DoneFuture:
            def done(self):
                return True
        with self.assertRaises(probe.CanonicalEnospcError):
            probe._require_request_waiting_at_release(DoneFuture())

    def test_run_exit_refuses_nonpass_status(self):
        parser = probe.build_parser()
        args = parser.parse_args(['--runtime', '/r', '--api-src', '/a', '--profile', '/p',
            '--store', '/s', '--original-store', '/o', '--branch', 'CliA', '--run-dir', '/run',
            '--canonical-mount', '/mnt/store', '--checkpoint-directory', '/mnt/store/checkpoints',
            '--loop-image', '/image', '--parent-mount-namespace', '4026531840',
            '--expected-profile-sha256', 'a' * 64, '--expected-native-sha256', 'b' * 64,
            '--expected-source-snapshot', 'c' * 64, '--expected-image-uuid', '4df1febe-d3f8-401e-908a-81da3824b712',
            '--expected-image-inode', '5254728'])
        self.assertEqual(args.max_tokens, 16)
        self.assertEqual(probe.exit_code_for_status('PASS'), 0)
        for status in ('FAIL', 'RUNNING', 'PREPARED_NOT_RUN', None):
            with self.subTest(status=status):
                self.assertNotEqual(probe.exit_code_for_status(status), 0)
        with redirect_stdout(io.StringIO()):
            code = probe.main(['--runtime', '/r', '--api-src', '/a', '--profile', '/p',
                '--store', '/s', '--original-store', '/o', '--branch', 'CliA', '--run-dir', '/run',
                '--canonical-mount', '/mnt/store', '--checkpoint-directory', '/mnt/store/checkpoints',
                '--loop-image', '/image', '--parent-mount-namespace', 'mnt:[4026531832]',
                '--expected-profile-sha256', 'a' * 64, '--expected-native-sha256', 'b' * 64,
                '--expected-source-snapshot', 'c' * 64, '--expected-image-uuid', '4df1febe-d3f8-401e-908a-81da3824b712',
                '--expected-image-inode', '5254728'], probe_runner=lambda _args: {'status': 'FAIL'})
        self.assertEqual(code, 1)

    def test_filler_receipt_must_prove_errno28_and_zero_available_bytes(self):
        valid = dict(errno=errno.ENOSPC, fill_path='/bounded/.fill', filled_bytes=1_000_000,
            allocated_bytes=1_000_000,
            available_bytes_after=0, filesystem_write_error='[Errno 28] No space left on device')
        probe.validate_filler_receipt(valid, canonical_mount='/bounded', max_fill_bytes=2_000_000)
        for key, value in [('errno', errno.EIO), ('filled_bytes', 0),
                           ('filled_bytes', 2_000_001), ('available_bytes_after', 4096),
                           ('allocated_bytes', 0),
                           ('fill_path', '/elsewhere/.fill')]:
            bad = dict(valid); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_filler_receipt(bad, canonical_mount='/bounded', max_fill_bytes=2_000_000)

    def test_run_directory_parent_must_exist_and_result_must_be_off_bounded_volume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = root / 'bounded-store'
            store.mkdir()
            parent = root / 'runs'
            parent.mkdir()
            probe.validate_run_directory(parent / 'campaign', store)
            (parent / 'campaign').mkdir()
            with self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_run_directory(parent / 'campaign', store)
            with self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_run_directory(store / 'result', store)
            with self.assertRaises(probe.CanonicalEnospcError):
                probe.validate_run_directory(root / 'missing' / 'campaign', store)

    def test_partial_writes_continue_until_real_kernel_errno(self):
        writes = [3, 2]
        def short_then_full(fd, view):
            if writes:
                count = min(writes.pop(0), len(view))
                return count
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        with mock.patch.object(probe.os, 'write', side_effect=short_then_full):
            count, error = probe._write_until_error(9, b'abcdefgh')
        self.assertEqual(count, 5)
        self.assertIsInstance(error, OSError)
        self.assertEqual(error.errno, errno.ENOSPC)

    def test_filler_is_removed_when_early_fsync_fails_after_real_write_errno(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real_fsync = os.fsync
            fsync_calls = []
            def fail_file_fsync_then_sync_cleanup(fd):
                fsync_calls.append(fd)
                if len(fsync_calls) == 1:
                    raise OSError(errno.EIO, 'injected filler fsync failure')
                return real_fsync(fd)
            with mock.patch.object(probe, '_write_until_error',
                    return_value=(17, OSError(errno.ENOSPC, 'full'))), \
                 mock.patch.object(probe.os, 'fsync', side_effect=fail_file_fsync_then_sync_cleanup):
                with self.assertRaises(OSError) as caught:
                    probe._fill_to_enospc(root, 1024, 'fsync-eio')
            self.assertEqual(caught.exception.errno, errno.EIO)
            self.assertFalse((root / '.q4-canonical-enospc-fsync-eio.filler').exists())

    def test_first_large_write_enospc_drains_remaining_space_with_smaller_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            available = [4096, 0, 0]
            def staged_statvfs(_path):
                value = available.pop(0)
                return SimpleNamespace(f_bavail=value, f_frsize=1)
            def large_fails_then_small_writes_finish(fd, data):
                calls.append(len(data))
                if len(calls) == 1:
                    return 0, OSError(errno.ENOSPC, 'large allocation did not fit')
                if len(calls) == 2:
                    count = os.write(fd, data)
                    return count, None
                return 0, OSError(errno.ENOSPC, 'bounded filesystem is full')
            with mock.patch.object(probe, '_write_until_error',
                    side_effect=large_fails_then_small_writes_finish), \
                 mock.patch.object(probe.os, 'statvfs', side_effect=staged_statvfs):
                receipt = probe._fill_to_enospc(root, 8192, 'residual-space')
            self.assertEqual(calls, [8192, 4096, 4096])
            self.assertEqual(receipt['errno'], errno.ENOSPC)
            self.assertEqual(receipt['filled_bytes'], 4096)
            self.assertEqual(receipt['available_bytes_after'], 0)
            self.assertGreater(receipt['allocated_bytes'], 0)
            self.assertTrue(Path(receipt['fill_path']).is_file())

    def test_filler_cap_failure_removes_only_the_owned_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(probe, '_write_until_error', return_value=(1024, None)):
                with self.assertRaisesRegex(probe.CanonicalEnospcError, 'byte cap'):
                    probe._fill_to_enospc(root, 1024, 'cap-failure')
            self.assertFalse((root / '.q4-canonical-enospc-cap-failure.filler').exists())
            foreign = root / '.q4-canonical-enospc-cap.filler'
            # The driver must refuse an already-existing path and leave it intact.
            foreign.write_bytes(b'owned by the fixture')
            with self.assertRaises(probe.CanonicalEnospcError):
                probe._fill_to_enospc(root, 1024, 'cap')
            self.assertEqual(foreign.read_bytes(), b'owned by the fixture')

    def test_started_service_is_stopped_when_readiness_start_raises(self):
        class FakeService:
            process = None
            stop_calls = 0
            def start(self):
                self.process = object()  # Models Popen succeeding before readiness fails.
                raise RuntimeError('readiness failed')
            def stop(self):
                self.stop_calls += 1
                return dict(exit_code=0, graceful=True)
        service = FakeService()
        result = {}
        try:
            service.start()
        except RuntimeError:
            attempted = probe._stop_service_if_process_exists(service, result, 'service_stop')
        self.assertTrue(attempted)
        self.assertEqual(service.stop_calls, 1)
        self.assertEqual(result['service_stop'], dict(exit_code=0, graceful=True))

    def test_service_without_started_child_is_not_stopped(self):
        class NotStarted:
            process = None
            def stop(self):
                raise AssertionError('must not stop a process that was never created')
        self.assertFalse(probe._stop_service_if_process_exists(NotStarted(), {}, 'stop'))

    def test_generated_server_wrapper_is_valid_python_and_gates_only_one_exact_operation(self):
        source = probe._reference_wrapper_source(Path('/run/control'), self.branch,
            self.operation, Path('/fixture/store'))
        compile(source, '<canonical-enospc-server-wrapper>', 'exec')
        self.assertIn('not (control/"gate-consumed.json").exists()', source)
        self.assertIn('getattr(local,"operation",None)==target_operation', source)
        self.assertIn('path.resolve()==(self.references/(branch_name+".json")).resolve()', source)
        self.assertIn('reference_sha256_after', source)

    def test_preflight_only_serializes_validated_not_run_receipt(self):
        argv = ['--runtime', '/r', '--api-src', '/a', '--profile', '/p',
            '--store', '/s', '--original-store', '/o', '--branch', 'CliA', '--run-dir', '/run',
            '--canonical-mount', '/mnt/store', '--checkpoint-directory', '/mnt/store/checkpoints',
            '--loop-image', '/image', '--parent-mount-namespace', 'mnt:[4026531832]',
            '--expected-profile-sha256', 'a' * 64, '--expected-native-sha256', 'b' * 64,
            '--expected-source-snapshot', 'c' * 64, '--expected-image-uuid', '4df1febe-d3f8-401e-908a-81da3824b712',
            '--expected-image-inode', '5254728', '--preflight-only']
        output = io.StringIO()
        with mock.patch.object(probe, 'preflight', return_value={'mount': 'checked'}), redirect_stdout(output):
            code = probe.main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()),
            {'status': 'PREPARED_NOT_RUN', 'preflight': {'mount': 'checked'}})

    def test_runner_serializes_failure_after_it_owns_run_control_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / 'run'
            run_dir.mkdir()
            (run_dir / 'control').mkdir()
            argv = ['--runtime', '/r', '--api-src', '/a', '--profile', '/p',
                '--store', '/s', '--original-store', '/o', '--branch', 'CliA',
                '--run-dir', str(run_dir), '--canonical-mount', '/mnt/store',
                '--checkpoint-directory', '/mnt/store/checkpoints', '--loop-image', '/image',
                '--parent-mount-namespace', 'mnt:[4026531832]',
                '--expected-profile-sha256', 'a' * 64, '--expected-native-sha256', 'b' * 64,
                '--expected-source-snapshot', 'c' * 64,
                '--expected-image-uuid', '4df1febe-d3f8-401e-908a-81da3824b712',
                '--expected-image-inode', '5254728']
            with redirect_stdout(io.StringIO()):
                code = probe.main(argv, probe_runner=lambda _args: {'status': 'FAIL', 'error': 'controlled failure'})
            self.assertEqual(code, 1)
            self.assertEqual(json.loads((run_dir / 'result.json').read_text(encoding='utf-8')),
                {'schema': 'q4-canonical-reference-enospc-v1', 'status': 'FAIL',
                 'error': 'controlled failure'})


if __name__ == '__main__':
    unittest.main()
