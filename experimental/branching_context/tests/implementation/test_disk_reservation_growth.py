"""The same global disk reservation can be extended without losing its digest lease."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from context_time_travel.pools import DiskCheckpointPool, PoolFull
from context_time_travel.store import Store


_HOLDER = r"""
import sys
from context_time_travel.pools import DiskCheckpointPool
from context_time_travel.store import Store
root, limit, size = sys.argv[1:]
pool = DiskCheckpointPool(Store(root), int(limit))
with pool.reserve(int(size), timeout=0):
    print('READY', flush=True)
    if sys.stdin.readline().strip() != 'release':
        raise RuntimeError('missing release command')
print('RELEASED', flush=True)
"""


class DiskReservationGrowthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.pool = DiskCheckpointPool(self.store, 64)
        self.children = []
        self.addCleanup(self.stop_children)

    def start_holder(self, *, limit=64, size=1):
        child = subprocess.Popen(
            [sys.executable, '-u', '-c', _HOLDER, self.tmp.name, str(limit), str(size)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(child)
        ready = child.stdout.readline().strip()
        if ready != 'READY':
            stdout, stderr = child.communicate(timeout=3)
            self.fail(f'holder failed to reserve: {ready!r} {stdout!r} {stderr!r}')
        return child

    def stop_holder(self, child):
        if child.poll() is None:
            child.stdin.write('release\n')
            child.stdin.flush()
        stdout, stderr = child.communicate(timeout=3)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertIn('RELEASED', stdout)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()

    def marker_rows(self):
        return {path: json.loads(path.read_text(encoding='utf-8'))
                for path in self.pool._reservation_directory.glob('*.json')}

    def test_growth_updates_one_global_marker_and_accounting(self):
        digest = hashlib.sha256(b'future payload').hexdigest()
        reservation = self.pool.reserve(7, payload_sha256=digest)
        try:
            before = self.marker_rows()
            self.assertEqual(len(before), 1)
            marker, row = next(iter(before.items()))
            reservation.grow(9, timeout=0)
            after = self.marker_rows()
            self.assertEqual(set(after), {marker}, 'growth must keep the original lease marker')
            self.assertEqual(after[marker]['reserved_bytes'], 16)
            self.assertEqual(after[marker]['payload_sha256'], digest)
            self.assertEqual(reservation.size, 16)
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 16)
            self.assertEqual(self.pool.metrics()['local_reserved_bytes'], 16)
            self.assertEqual(reservation.grow(0, timeout=0), 16)
            self.assertEqual(reservation.size, 16)
        finally:
            reservation.release()
        self.assertEqual(self.marker_rows(), {})
        self.assertEqual(self.pool.metrics()['reserved_bytes'], 0)

    def test_other_process_capacity_blocks_growth_then_allows_retry(self):
        child = self.start_holder(size=48)
        digest = hashlib.sha256(b'protected while growing').hexdigest()
        reservation = self.pool.reserve(16, payload_sha256=digest)
        rows=self.marker_rows()
        marker=next(path for path,row in rows.items() if row['payload_sha256']==digest)
        try:
            with self.assertRaises(PoolFull):
                reservation.grow(1, timeout=0.08)
            self.assertEqual(reservation.size, 16)
            self.assertEqual(self.marker_rows()[marker]['reserved_bytes'], 16)
            self.assertEqual(self.marker_rows()[marker]['payload_sha256'], digest)
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 64)
            self.assertEqual(self.pool.metrics()['local_reserved_bytes'], 16)
            self.stop_holder(child)
            reservation.grow(1, timeout=0.2)
            self.assertEqual(reservation.size, 17)
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 17)
        finally:
            reservation.release()

    def test_growth_and_collection_keep_digest_until_reservation_release(self):
        payload = b'written before its publication manifest'
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + '.state')
        reservation = self.pool.reserve(4, payload_sha256=digest)
        try:
            state.write_bytes(payload)
            self.pool.collect()
            self.assertTrue(state.exists())
            reservation.grow(12, timeout=0)
            self.pool.collect()
            self.assertTrue(state.exists())
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 16)
        finally:
            reservation.release()
        self.pool.collect()
        self.assertFalse(state.exists())

    def test_timeout_preserves_original_reservation_and_digest_lease(self):
        child = self.start_holder(size=64)
        digest = hashlib.sha256(b'waiter digest').hexdigest()
        reservation = self.pool.reserve(0, payload_sha256=digest)
        rows=self.marker_rows()
        marker=next(path for path,row in rows.items() if row['payload_sha256']==digest)
        try:
            with self.assertRaises(PoolFull):
                reservation.grow(1, timeout=0.08)
            self.assertEqual(reservation.size, 0)
            self.assertEqual(self.marker_rows()[marker]['reserved_bytes'], 0)
            self.assertEqual(self.marker_rows()[marker]['payload_sha256'], digest)
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 64)
            self.assertEqual(self.pool.metrics()['local_reserved_bytes'], 0)
            self.stop_holder(child)
            reservation.grow(1, timeout=0.2)
            self.assertEqual(reservation.size, 1)
        finally:
            reservation.release()

    def test_invalid_and_closed_growth_reject_without_changing_reservation(self):
        reservation = self.pool.reserve(4)
        try:
            for value in (True, -1, 1.5):
                with self.subTest(extra_bytes=value), self.assertRaises(ValueError):
                    reservation.grow(value, timeout=0)
            for timeout in (True, -1, float('inf'), float('nan'), 'slow'):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    reservation.grow(1, timeout=timeout)
            with self.assertRaises(PoolFull):
                reservation.grow(61, timeout=0)
            self.assertEqual(reservation.size, 4)
            self.assertEqual(self.pool.metrics()['reserved_bytes'], 4)
        finally:
            reservation.release()
        with self.assertRaises(RuntimeError):
            reservation.grow(1, timeout=0)

    def test_active_conflicting_byte_limit_blocks_growth(self):
        digest = hashlib.sha256(b'live conflict').hexdigest()
        reservation = self.pool.reserve(8, payload_sha256=digest)
        conflict_path = self.pool._reservation_directory / 'incompatible-live.json'
        conflict_fd = os.open(conflict_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        fcntl.flock(conflict_fd, fcntl.LOCK_SH)
        record = dict(schema=1, pid=os.getpid(), byte_limit=128,
                      reserved_bytes=0, payload_sha256=None)
        data = json.dumps(record, sort_keys=True, separators=(',', ':')).encode('utf-8')
        os.write(conflict_fd, data)
        os.fsync(conflict_fd)
        try:
            with self.assertRaises(PoolFull):
                reservation.grow(1, timeout=0)
            self.assertEqual(reservation.size, 8)
            self.assertEqual(self.marker_rows()[next(
                path for path in self.marker_rows() if path != conflict_path)]['reserved_bytes'], 8)
        finally:
            reservation.release()
            os.close(conflict_fd)
            conflict_path.unlink(missing_ok=True)

    def test_marker_storage_failure_aborts_corrupted_reservation(self):
        digest = hashlib.sha256(b'failed metadata growth').hexdigest()
        reservation = self.pool.reserve(4, payload_sha256=digest)
        original_write = DiskCheckpointPool._reservation_bytes

        def write_then_fail(fd, record):
            original_write(fd, record)
            if record['reserved_bytes'] > 4:
                raise OSError(28, 'injected marker fsync failure')

        with mock.patch.object(DiskCheckpointPool, '_reservation_bytes',
                               side_effect=write_then_fail):
            with self.assertRaisesRegex(OSError, 'injected marker fsync failure'):
                reservation.grow(8, timeout=0)

        self.assertTrue(reservation.closed)
        self.assertEqual(self.pool.metrics()['local_reserved_bytes'], 0)
        self.assertEqual(self.pool.metrics()['reserved_bytes'], 0)
        self.assertEqual(self.marker_rows(), {})


if __name__ == '__main__':
    unittest.main()
