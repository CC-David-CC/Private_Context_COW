"""CPU-only contracts for the real private-ext4 Q4 ENOSPC harness.

These tests validate evidence classification and mount guards. They do not
simulate a kernel ENOSPC or qualify COW/native/GPU behavior.
"""
from __future__ import annotations

import errno
import ast
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import run_q4_enospc as probe


class EnospcEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.writer = {
            'operation_id': 'op-enospc', 'branch': 'BranchA', 'snapshot': 'snap-target',
            'destination_kind': 'checkpoint-payload-state', 'exception_type': 'OSError',
            'errno': errno.ENOSPC, 'message': '[Errno 28] No space left on device',
            'destination_directory': '/tmp/checkpoints',
            'payload_name': 'abc.state', 'requested_payload_bytes': 331_334_904,
            'published_payload_exists': False, 'published_manifest_exists': False,
            'staging_files_after': [],
        }

    def test_accepts_only_real_state_payload_enospc(self):
        result = probe.validate_enospc_writer_observation(self.writer, 'op-enospc', 'BranchA')
        self.assertEqual(result['errno'], errno.ENOSPC)
        for change in (
            {'errno': errno.EFBIG}, {'destination_kind': 'native-export'},
            {'operation_id': 'other'}, {'payload_name': 'manifest.json'},
            {'published_payload_exists': True}, {'published_manifest_exists': True},
            {'staging_files_after': ['.stage-leftover']},
            {'requested_payload_bytes': 0},
        ):
            with self.subTest(change=change), self.assertRaises(probe.EnospcProbeError):
                probe.validate_enospc_writer_observation({**self.writer, **change}, 'op-enospc', 'BranchA')

    def test_requires_explicit_uncancelled_error_and_never_complete(self):
        valid = [{'event': 'error', 'error': '[Errno 28] No space left on device', 'cancelled': False}]
        probe.validate_enospc_stream(valid)
        for events in (
            [], [{'event': 'error', 'error': 'write failed', 'cancelled': False}],
            [{'event': 'error', 'error': valid[0]['error'], 'cancelled': True}],
            [*valid, {'event': 'complete', 'receipt': {}}],
            [{'event': 'durable', 'receipt': {'snapshot': 'must-not-ack'}}, *valid],
            [{'event': 'error', 'error': valid[0]['error'], 'cancelled': False,
              'durable': True}],
            [{'event': 'error', 'error': valid[0]['error'], 'cancelled': False,
              'receipt': {'snapshot': 'must-not-ack'}}],
            [*valid, {'event': 'error', 'error': valid[0]['error'], 'cancelled': False}],
            ['malformed'],
        ):
            with self.subTest(events=events), self.assertRaises(probe.EnospcProbeError):
                probe.validate_enospc_stream(events)

    def test_filler_must_report_bounded_kernel_enospc_and_zero_free_blocks(self):
        good = {'errno': errno.ENOSPC, 'attempted_bytes': 12_000_000,
                'staged_bytes': 11_999_232, 'available_before': 12_000_000,
                'available_after': 0}
        probe.validate_filler_result(good, capacity_bytes=1024**3, max_filler_bytes=16_000_000)
        for change in ({'errno': errno.EIO}, {'attempted_bytes': 16_000_001},
                       {'available_after': 8 * 1024**2 + 1}, {'staged_bytes': 20_000_000},
                       {'available_before': 1024**3 + 1}):
            with self.subTest(change=change), self.assertRaises(probe.EnospcProbeError):
                probe.validate_filler_result({**good, **change}, capacity_bytes=1024**3,
                                             max_filler_bytes=16_000_000)

    def test_writer_observer_gates_once_before_calling_the_unchanged_writer(self):
        generated = probe._writer_wrapper_source(Path('/tmp/events'), Path('/tmp/gate'), Path('/tmp/release'))
        compile(generated, 'server_wrapper.py', 'exec')
        self.assertIn('original_write(self, destination, parts)', generated)
        self.assertIn('requested_payload_bytes', generated)
        self.assertIn('destination_directory', generated)
        self.assertIn('!= "RELEASE"', generated)
        self.assertIn('original_coordinator_publish(self, model, job, prepared, key, pin, lease)', generated)

    def test_filler_orchestration_persists_result_before_releasing_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, release_path = root / 'writer-gate.json', root / 'writer-release'
            filler_path = root / 'filler-result.json'
            gate_path.write_text(json.dumps({
                'event': 'writer_gate', 'operation_id': 'op-123', 'branch': 'Q4ENOSPCabc',
                'snapshot': 'b' * 64, 'requested_payload_bytes': 331_328_756,
            }), encoding='utf-8')
            fill = {'errno': errno.ENOSPC, 'attempted_bytes': 900_000_000,
                    'staged_bytes': 899_000_000, 'available_before': 900_000_000,
                    'available_after': 0, 'filler_path': str(root / '.filler'),
                    'filler_retained_bytes': 899_000_000}
            holder, errors = {}, queue.Queue()
            release_observations = []
            original_signal = probe._write_signal

            def observe_release(path, payload):
                if payload == b'RELEASE\n':
                    self.assertTrue(filler_path.is_file(), 'filler receipt must exist before RELEASE')
                    persisted = json.loads(filler_path.read_text(encoding='utf-8'))
                    self.assertEqual(persisted, fill)
                    release_observations.append(persisted)
                return original_signal(path, payload)

            with mock.patch.object(probe, '_fill_private_mount', return_value=fill) as fill_fs, \
                 mock.patch.object(probe, '_write_signal', side_effect=observe_release):
                probe._fill_after_writer_gate(gate_path=gate_path, release_path=release_path,
                    filler_path=filler_path, mount=root, operation_id='op-123',
                    branch='Q4ENOSPCabc', source_snapshot='a' * 64, timeout=1,
                    max_fill_bytes=1400 * 1024**2, fill_result_holder=holder, fill_error=errors)
            orchestration_error = errors.get_nowait() if not errors.empty() else None
            self.assertIsNone(orchestration_error,
                f'filler gate orchestration failed before RELEASE: {orchestration_error!r}')
            self.assertEqual(filler_path.read_text(encoding='utf-8'), json.dumps(fill, sort_keys=True,
                indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            self.assertEqual(release_path.read_bytes(), b'RELEASE\n')
            self.assertEqual(holder['gate']['operation_id'], 'op-123')
            self.assertEqual(release_observations, [fill])
            fill_fs.assert_called_once_with(root, max_bytes=1400 * 1024**2)

    def test_filler_error_aborts_writer_without_persisting_success_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, release_path = root / 'writer-gate.json', root / 'writer-release'
            filler_path = root / 'filler-result.json'
            gate_path.write_text(json.dumps({
                'event': 'writer_gate', 'operation_id': 'op-123', 'branch': 'Q4ENOSPCabc',
                'snapshot': 'b' * 64, 'requested_payload_bytes': 331_328_756,
            }), encoding='utf-8')
            holder, errors = {}, queue.Queue()
            with mock.patch.object(probe, '_fill_private_mount',
                                   side_effect=OSError(errno.ENOSPC, 'bounded fill failed')):
                probe._fill_after_writer_gate(gate_path=gate_path, release_path=release_path,
                    filler_path=filler_path, mount=root, operation_id='op-123',
                    branch='Q4ENOSPCabc', source_snapshot='a' * 64, timeout=1,
                    max_fill_bytes=1400 * 1024**2, fill_result_holder=holder, fill_error=errors)
            failure = errors.get_nowait()
            self.assertIsInstance(failure, OSError)
            self.assertEqual(failure.errno, errno.ENOSPC)
            self.assertEqual(release_path.read_bytes(), b'ABORT\n')
            self.assertFalse(filler_path.exists())

    def test_filler_receipt_persistence_error_aborts_before_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, release_path = root / 'writer-gate.json', root / 'writer-release'
            filler_path = root / 'filler-result.json'
            gate_path.write_text(json.dumps({
                'event': 'writer_gate', 'operation_id': 'op-123', 'branch': 'Q4ENOSPCabc',
                'snapshot': 'b' * 64, 'requested_payload_bytes': 331_328_756,
            }), encoding='utf-8')
            fill = {'errno': errno.ENOSPC, 'attempted_bytes': 900_000_000,
                    'staged_bytes': 899_000_000, 'available_before': 900_000_000,
                    'available_after': 0}
            holder, errors = {}, queue.Queue()
            with mock.patch.object(probe, '_fill_private_mount', return_value=fill), \
                 mock.patch.object(probe._base, '_atomic_json',
                                   side_effect=OSError(errno.EIO, 'receipt fsync failed')):
                probe._fill_after_writer_gate(gate_path=gate_path, release_path=release_path,
                    filler_path=filler_path, mount=root, operation_id='op-123',
                    branch='Q4ENOSPCabc', source_snapshot='a' * 64, timeout=1,
                    max_fill_bytes=1400 * 1024**2, fill_result_holder=holder, fill_error=errors)
            failure = errors.get_nowait()
            self.assertIsInstance(failure, OSError)
            self.assertEqual(failure.errno, errno.EIO)
            self.assertEqual(release_path.read_bytes(), b'ABORT\n')
            self.assertFalse(filler_path.exists())

    def test_all_json_persistence_calls_resolve_through_pinned_helper(self):
        tree = ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        unresolved = [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == '_atomic_json']
        self.assertEqual(unresolved, [])

    def test_mount_guard_requires_ext4_loop_backed_by_exact_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount = root / 'mount'
            image = root / 'image.ext4'
            mount.mkdir()
            image.write_bytes(b'x' * (1024**3))
            findmnt = {'filesystems': [{'fstype': 'ext4', 'source': '/dev/loop42'}]}
            loop = {'loopdevices': [{'name': '/dev/loop42', 'back-file': str(image), 'maj:min': '7:42'}]}
            stat = mock.Mock(f_blocks=1024**3 // 4096, f_frsize=4096,
                             f_bavail=1024**3 // 8192)
            with mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=[json.dumps(findmnt), json.dumps(loop)]), \
                 mock.patch.object(probe.os, 'statvfs', return_value=stat), \
                 mock.patch.object(probe.os, 'readlink', side_effect=['mnt:[4026539999]', 'mnt:[4026531840]']):
                info = probe.validate_ext4_mount(mount, image)
            self.assertEqual(info['source'], '/dev/loop42')
            self.assertEqual(info['image_bytes'], 1024**3)
            self.assertEqual(info['loop_major_minor'], '7:42')

    def test_mount_guard_rejects_nonloop_or_wrong_backing_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount, image = root / 'mount', root / 'image.ext4'
            mount.mkdir(); image.write_bytes(b'x' * (1024**3))
            with mock.patch.object(probe.subprocess, 'check_output',
                                   return_value=json.dumps({'filesystems': [
                                       {'fstype': 'ext4', 'source': '/dev/sda1'}]})):
                with self.assertRaises(probe.EnospcProbeError):
                    probe.validate_ext4_mount(mount, image)

    def test_mount_guard_rejects_wrong_loop_image_and_shared_mount_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount, image, other = root / 'mount', root / 'image.ext4', root / 'other.ext4'
            mount.mkdir(); image.write_bytes(b'x' * (1024**3)); other.write_bytes(b'y' * (1024**3))
            findmnt = {'filesystems': [{'fstype': 'ext4', 'source': '/dev/loop42'}]}
            loop = {'loopdevices': [{'name': '/dev/loop42', 'back-file': str(other), 'maj:min': '7:42'}]}
            with mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=[json.dumps(findmnt), json.dumps(loop)]):
                with self.assertRaises(probe.EnospcProbeError):
                    probe.validate_ext4_mount(mount, image)
            loop['loopdevices'][0]['back-file'] = str(image)
            stat = mock.Mock(f_blocks=1024**3 // 4096, f_frsize=4096,
                             f_bavail=1024**3 // 8192)
            with mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=[json.dumps(findmnt), json.dumps(loop)]), \
                 mock.patch.object(probe.os, 'statvfs', return_value=stat), \
                 mock.patch.object(probe.os, 'readlink', return_value='mnt:[same]'):
                with self.assertRaises(probe.EnospcProbeError):
                    probe.validate_ext4_mount(mount, image)

    def test_mount_guard_rejects_unreadable_parent_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount, image = root / 'mount', root / 'image.ext4'
            mount.mkdir(); image.write_bytes(b'x' * (1024**3))
            findmnt = {'filesystems': [{'fstype': 'ext4', 'source': '/dev/loop42'}]}
            loop = {'loopdevices': [{'name': '/dev/loop42', 'back-file': str(image), 'maj:min': '7:42'}]}
            stat = mock.Mock(f_blocks=1024**3 // 4096, f_frsize=4096,
                             f_bavail=1024**3 // 8192)
            with mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=[json.dumps(findmnt), json.dumps(loop)]), \
                 mock.patch.object(probe.os, 'statvfs', return_value=stat), \
                 mock.patch.object(probe.os, 'readlink', side_effect=['mnt:[private]', '']):
                with self.assertRaises(probe.EnospcProbeError):
                    probe.validate_ext4_mount(mount, image)

    def test_mount_guard_compares_captured_parent_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount, image = root / 'mount', root / 'image.ext4'
            mount.mkdir(); image.write_bytes(b'x' * (1024**3))
            findmnt = {'filesystems': [{'fstype': 'ext4', 'source': '/dev/loop42'}]}
            loop = {'loopdevices': [{'name': '/dev/loop42', 'back-file': str(image), 'maj:min': '7:42'}]}
            stat = mock.Mock(f_blocks=1024**3 // 4096, f_frsize=4096,
                             f_bavail=1024**3 // 8192)
            with mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=[json.dumps(findmnt), json.dumps(loop)]), \
                 mock.patch.object(probe.os, 'statvfs', return_value=stat), \
                 mock.patch.object(probe.os, 'readlink', return_value='mnt:[private]'):
                info = probe.validate_ext4_mount(
                    mount, image, parent_mount_namespace='mnt:[captured-parent]')
            self.assertEqual(info['mount_namespace'], 'mnt:[private]')
            self.assertEqual(info['init_mount_namespace'], 'mnt:[captured-parent]')

    def test_mount_guard_rejects_symlinked_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_mount, image = root / 'mount', root / 'image.ext4'
            real_mount.mkdir(); image.write_bytes(b'x' * (1024**3))
            alias = root / 'mount-alias'
            alias.symlink_to(real_mount, target_is_directory=True)
            with self.assertRaises(probe.EnospcProbeError):
                probe.validate_ext4_mount(alias, image)

    def test_main_returns_failure_when_probe_result_is_not_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / 'store'
            original = root / 'original'
            mount = store / 'checkpoints'
            store.mkdir(); original.mkdir(); mount.mkdir()
            (store / 'refs.json').write_text('{}', encoding='utf-8')
            image = root / 'private.ext4'
            image.write_bytes(b'image')
            args = SimpleNamespace(max_tokens=8, ram_gib=96, disk_gib=128,
                max_fill_mib=1400, store=store, original_store=original,
                checkpoint_mount=mount, loop_image=image,
                parent_mount_namespace='mnt:[parent]', preflight_only=False,
                run_dir=root / 'run')
            with mock.patch.object(probe, 'build_parser') as parser, \
                 mock.patch.object(probe, 'validate_preflight_inputs', return_value={}), \
                 mock.patch.object(probe, 'validate_ext4_mount', return_value={}), \
                 mock.patch.object(probe, 'run_probe', return_value={'status': 'FAIL'}), \
                 redirect_stdout(io.StringIO()):
                parser.return_value.parse_args.return_value = args
                self.assertEqual(probe.main([]), 1)

    def test_preflight_validates_pinned_profile_runtime_and_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, original = root / 'store', root / 'original'
            store.mkdir(); original.mkdir()
            (store / 'checkpoints').mkdir()
            (store / 'refs.json').write_text('{}', encoding='utf-8')
            image = root / 'image.ext4'; image.write_bytes(b'image')
            profile = root / 'profile.json'; profile.write_text('{}', encoding='utf-8')
            runtime = root / 'runtime'; runtime.mkdir()
            api_src = runtime / 'src'; api_src.mkdir()
            api_scripts = runtime / 'scripts'; api_scripts.mkdir()
            native = root / 'native'; native.write_bytes(b'native')
            args = SimpleNamespace(profile=profile, store=store, original_store=original,
                runtime_root=runtime, api_src=api_src, api_scripts=api_scripts,
                loop_image=image, checkpoint_mount=store / 'checkpoints',
                expected_profile_sha256='profile-sha', expected_native_sha256='native-sha',
                expected_source_snapshot='a' * 64)
            profile_data = {'execution': {'sut_command': [str(native)]}}
            with mock.patch.object(probe._base, '_ensure_q4_profile') as ensure, \
                 mock.patch.object(probe._base, '_runtime_closure', return_value={'files': {}}), \
                 mock.patch.object(probe._base, '_sha256', side_effect=lambda path: {
                     profile: 'profile-sha', native: 'native-sha'}[Path(path)]), \
                 mock.patch.object(probe._base, '_snapshot_record_fingerprint', return_value='a' * 64) as record, \
                 mock.patch.object(probe._base, '_snapshot_file_fingerprint', return_value={'a.state': {'bytes': 1}}) as files, \
                 mock.patch.object(probe.json, 'loads', return_value=profile_data):
                receipt = probe.validate_preflight_inputs(args)
            ensure.assert_called_once_with(profile_data)
            record.assert_called_once_with(store, 'a' * 64)
            files.assert_called_once_with(store, 'a' * 64)
            self.assertEqual(receipt['native_sha256'], 'native-sha')
            self.assertEqual(receipt['source_snapshot'], 'a' * 64)


if __name__ == '__main__':
    unittest.main()
