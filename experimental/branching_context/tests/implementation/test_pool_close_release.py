"""Closing drops unpinned derived RAM while preserving active borrowers."""
import errno
import threading
import time
import unittest
import weakref

from context_time_travel.pools import SharedCheckpointPool
from context_time_travel.real import Checkpoint


def image(blob=b'abcd', snapshot='a'):
    return Checkpoint('m', 'e', 1, blob, (1.,), Checkpoint.digest(blob, (1.,)), snapshot)


class SharedPoolCloseTests(unittest.TestCase):
    def test_close_drops_clean_failed_and_dirty_entries_and_releases_objects(self):
        pool = SharedCheckpointPool(32)
        checkpoints = [image(b'clean'), image(b'failed'), image(b'dirty')]
        refs = [weakref.ref(cp) for cp in checkpoints]
        pool.admit('clean', checkpoints[0], durable=True)
        pool.admit('failed', checkpoints[1], durable=False)
        pool.mark_failed('failed')
        pool.admit('dirty', checkpoints[2], durable=False)
        del checkpoints

        pool.close()

        self.assertEqual(pool.metrics()['entries'], 0)
        self.assertEqual(pool.metrics()['unique_payloads'], 0)
        self.assertEqual(pool.metrics()['resident_bytes'], 0)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_active_borrower_stays_accounted_until_release_then_is_dropped(self):
        pool = SharedCheckpointPool(16)
        checkpoint = image()
        ref = weakref.ref(checkpoint)
        pool.admit('borrowed', checkpoint, durable=True)
        del checkpoint

        with pool.pin('borrowed'):
            pool.close()
            metrics = pool.metrics()
            self.assertEqual((metrics['entries'], metrics['pins'], metrics['resident_bytes']), (1, 1, 8))
        metrics = pool.metrics()
        self.assertEqual((metrics['entries'], metrics['pins'], metrics['resident_bytes']), (0, 0, 0))
        self.assertIsNone(ref())
        pool.close()
        self.assertEqual(pool.metrics()['entries'], 0)

    def test_close_racing_disk_preload_prevents_late_admission_and_releases_reservation(self):
        pool = SharedCheckpointPool(16)
        started, release = threading.Event(), threading.Event()
        loaded = image(b'loaded')
        loaded_ref = weakref.ref(loaded)

        def loader():
            started.set()
            if not release.wait(5):
                raise TimeoutError('test loader release was not signaled')
            return loaded

        future = pool.preload('disk', 10, loader)
        self.assertTrue(started.wait(2))
        closer = threading.Thread(target=pool.close)
        closer.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with pool._condition:
                if pool._closed:
                    break
            time.sleep(.005)
        self.assertTrue(pool._closed, 'close did not mark pool closed')
        release.set()
        closer.join(5)
        self.assertFalse(closer.is_alive(), 'close did not finish after loader was released')
        self.assertTrue(future.done())
        failure = future.exception(timeout=1)
        self.assertIsInstance(failure, RuntimeError)
        self.assertIn('closed', str(failure))
        del failure
        del loaded
        self.assertEqual(pool.metrics()['entries'], 0)
        self.assertEqual(pool.metrics()['reserved_bytes'], 0)
        self.assertEqual(pool.metrics()['resident_bytes'], 0)
        self.assertIsNone(loaded_ref())

    def test_new_pins_are_rejected_after_close_while_existing_borrower_remains(self):
        pool = SharedCheckpointPool(16)
        pool.admit('borrowed', image(), durable=True)
        with pool.pin('borrowed'):
            pool.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                with pool.pin('borrowed'):
                    self.fail('a new borrower entered after close')
            metrics = pool.metrics()
            self.assertEqual((metrics['entries'], metrics['pins'], metrics['resident_bytes']), (1, 1, 8))
        self.assertEqual(pool.metrics()['entries'], 0)

    def test_failed_preload_preserves_oserror_errno_and_message(self):
        pool = SharedCheckpointPool(16)
        error = OSError(errno.ENOSPC, 'simulated checkpoint read failure')
        try:
            future = pool.preload('failed-read', 8, lambda: (_ for _ in ()).throw(error))
            failure = future.exception(timeout=2)
            self.assertIsInstance(failure, OSError)
            self.assertEqual(failure.errno, errno.ENOSPC)
            self.assertIn('simulated checkpoint read failure', str(failure))
        finally:
            pool.close()

    def test_new_reservations_admissions_and_preloads_fail_after_idempotent_close(self):
        pool = SharedCheckpointPool(16)
        pool.close()
        pool.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.reserve(0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.admit('new', image())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.preload('new', 8, image)
        self.assertEqual(pool.metrics()['entries'], 0)

    def test_reservation_created_before_close_cannot_admit_after_close(self):
        pool = SharedCheckpointPool(16)
        reservation = pool.reserve(8)
        try:
            pool.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                reservation.admit('late', image())
        finally:
            reservation.release()
        self.assertEqual(pool.metrics()['reserved_bytes'], 0)
        self.assertEqual(pool.metrics()['entries'], 0)
        self.assertEqual(pool.metrics()['resident_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
