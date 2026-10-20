"""POSIX process-level tests for one shared checkpoint-disk budget."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from context_time_travel.pools import DiskCheckpointPool, PoolFull
from context_time_travel.store import Store


_HOLDER = r"""
import sys
from context_time_travel.pools import DiskCheckpointPool
from context_time_travel.store import Store

root, limit, size, digest = sys.argv[1:]
pool = DiskCheckpointPool(Store(root), int(limit))
with pool.reserve(int(size), timeout=0,
                  payload_sha256=None if digest == '-' else digest):
    print('READY', flush=True)
    command = sys.stdin.readline().strip()
    if command != 'release':
        raise RuntimeError('parent did not release reservation')
print('RELEASED', flush=True)
"""


class DiskPoolProcessReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.pool = DiskCheckpointPool(self.store, 64)
        self._children = []
        self.addCleanup(self._stop_children)

    def start_holder(self, *, limit=64, size=1, digest=None):
        child = subprocess.Popen(
            [sys.executable, '-u', '-c', _HOLDER, self.tmp.name,
             str(limit), str(size), digest or '-'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        self._children.append(child)
        ready = child.stdout.readline().strip()
        if ready != 'READY':
            stdout, stderr = child.communicate(timeout=3)
            self.fail(f'reservation child did not become ready: {ready!r} {stdout!r} {stderr!r}')
        return child

    def stop_holder(self, child):
        if child.poll() is None:
            child.stdin.write('release\n')
            child.stdin.flush()
        stdout, stderr = child.communicate(timeout=3)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertIn('RELEASED', stdout)

    def _stop_children(self):
        for child in self._children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()

    def _reuse_descriptor_when_closed(self, descriptor):
        original_close=os.close
        replacement_path=Path(self.tmp.name)/'descriptor-reuse-probe'
        replacement=[]

        def close_then_reuse(fd):
            original_close(fd)
            expected=descriptor() if callable(descriptor) else descriptor
            if fd==expected:
                opened=os.open(replacement_path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
                if opened!=fd:
                    os.dup2(opened,fd)
                    original_close(opened)
                replacement.append(fd)

        return original_close,replacement_path,replacement,close_then_reuse

    @unittest.skipUnless(os.name == 'posix', 'Store uses POSIX flock leases')
    def test_reservations_from_two_processes_share_one_byte_limit(self):
        child = self.start_holder(size=48)
        metrics=self.pool.metrics()
        self.assertEqual(metrics['reserved_bytes'],48)
        self.assertEqual(metrics['local_reserved_bytes'],0)
        with self.assertRaises(PoolFull):
            with self.pool.reserve(32, timeout=0.08):
                self.fail('a second process over-admitted the shared disk budget')
        self.stop_holder(child)
        with self.pool.reserve(64, timeout=0):
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 64)

    @unittest.skipUnless(os.name == 'posix', 'Store uses POSIX flock leases')
    def test_other_process_collection_preserves_inflight_orphan_payload(self):
        payload = b'written-before-manifest'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        child = self.start_holder(size=0, digest=digest)
        self.pool.collect()
        self.assertTrue(state.exists(), 'an active writer digest reservation must protect its orphan')
        self.stop_holder(child)
        self.pool.collect()
        self.assertFalse(state.exists(), 'released orphan payload should become reclaimable')

    @unittest.skipUnless(os.name == 'posix', 'Store uses POSIX flock leases')
    def test_waiting_digest_is_protected_globally_until_timeout_then_released(self):
        child = self.start_holder(size=64)
        payload = b'pending-publication-payload'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        state.write_bytes(payload)
        outcome = []
        started = threading.Event()

        def wait_for_capacity():
            started.set()
            try:
                with self.pool.reserve(1, timeout=0.25, payload_sha256=digest):
                    outcome.append('acquired')
            except PoolFull:
                outcome.append('timeout')

        waiter = threading.Thread(target=wait_for_capacity)
        waiter.start()
        self.assertTrue(started.wait(1))
        marker_dir = getattr(self.pool, '_reservation_directory', None)
        self.assertIsNotNone(marker_dir, 'cross-process reservation metadata is not implemented')
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if any(digest in path.read_text(encoding='utf-8') for path in marker_dir.glob('*.json')):
                break
            time.sleep(0.005)
        else:
            self.fail('waiting publisher never registered a zero-byte digest protection')
        collector = DiskCheckpointPool(Store(self.tmp.name), 64)
        collector.collect()
        self.assertTrue(state.exists(), 'a waiting publisher must protect its digest before admission')
        waiter.join(2)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(outcome, ['timeout'])
        collector.collect()
        self.assertFalse(state.exists(), 'timeout must release the global digest protection')
        self.stop_holder(child)

    @unittest.skipUnless(os.name == 'posix', 'Store uses POSIX flock leases')
    def test_sigkill_releases_stale_reservation_and_next_process_reclaims_marker(self):
        child = self.start_holder(size=64)
        marker_dir = getattr(self.pool, '_reservation_directory', None)
        self.assertIsNotNone(marker_dir, 'cross-process reservation metadata is not implemented')
        markers=list(marker_dir.glob('*.json'))
        self.assertTrue(markers, 'live reservation marker is missing')
        self.assertEqual(stat.S_IMODE(marker_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(markers[0].stat().st_mode), 0o600)
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=3)
        with self.pool.reserve(64, timeout=0):
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 64)
        self.pool.collect()
        self.assertEqual(list(marker_dir.glob('*.json')), [], 'dead owner reservation was not reclaimed')

    @unittest.skipUnless(os.name == 'posix', 'Store uses POSIX flock leases')
    def test_concurrent_processes_with_different_limits_fail_explicitly(self):
        child = self.start_holder(limit=64, size=1)
        incompatible = DiskCheckpointPool(Store(self.tmp.name), 128)
        with self.assertRaises(PoolFull):
            with incompatible.reserve(1, timeout=0):
                self.fail('concurrent pool byte limits were silently mixed')
        self.stop_holder(child)

    def test_release_closes_reservation_descriptor_once(self):
        reservation=self.pool.reserve(1)
        original_close,path,replacement,close_then_reuse=self._reuse_descriptor_when_closed(
            reservation.marker_fd)
        with mock.patch('context_time_travel.pools.os.close',side_effect=close_then_reuse):
            reservation.release()
        self.assertEqual(replacement,[reservation.marker_fd])
        os.fstat(replacement[0])
        original_close(replacement[0])
        path.unlink()

    def test_failed_admission_cleanup_closes_reservation_descriptor_once(self):
        original_writer=self.pool._reservation_bytes
        writer_calls=0
        marker_fd=[]

        def fail_admission_update(fd,record):
            nonlocal writer_calls
            writer_calls+=1
            if writer_calls==1:
                marker_fd.append(fd)
                return original_writer(fd,record)
            raise OSError('injected failure after reservation marker creation')

        original_close,path,replacement,close_then_reuse=self._reuse_descriptor_when_closed(
            lambda: marker_fd[0] if marker_fd else -1)
        with mock.patch.object(self.pool,'_reservation_bytes',side_effect=fail_admission_update):
            # Capture the actual marker fd before arranging a descriptor reuse.
            def failing_close(fd):
                if marker_fd and fd==marker_fd[0]:
                    close_then_reuse(fd)
                else:
                    original_close(fd)
            with mock.patch('context_time_travel.pools.os.close',side_effect=failing_close):
                with self.assertRaisesRegex(OSError,'injected failure'):
                    self.pool.reserve(1)
        self.assertEqual(replacement,[marker_fd[0]])
        os.fstat(replacement[0])
        original_close(replacement[0])
        path.unlink()


if __name__ == '__main__':
    unittest.main()
