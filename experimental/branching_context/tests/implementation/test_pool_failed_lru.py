"""Failed RAM checkpoints are retryable derived data, never durable state."""
import gc
import weakref
import unittest
from dataclasses import replace

from context_time_travel.pools import PoolFull, PoolTransitionTrace, SharedCheckpointPool
from context_time_travel.real import Checkpoint


def image(blob, identity=None):
    frontier=(1.0, 2.0)
    return Checkpoint('model','execution',1,blob,frontier,
                      Checkpoint.digest(blob,frontier),identity or blob.hex())


class FailedCheckpointLRUTests(unittest.TestCase):
    def test_failed_and_durable_evictions_have_distinct_metrics_and_trace_events(self):
        trace=PoolTransitionTrace('failed-ram-test')
        pool=SharedCheckpointPool(9,transition_trace=trace)
        pool.admit('failed',image(b'x'),durable=False)
        pool.mark_failed('failed')
        with pool.reserve(9):
            self.assertNotIn('failed',pool)
        metrics=pool.metrics()
        self.assertEqual(metrics['clean_evictions'],0)
        self.assertEqual(metrics['failed_evictions'],1)
        self.assertEqual([row['event'] for row in trace.snapshot()['events']],
                         ['ram_failed_eviction'])

        trace=PoolTransitionTrace('durable-ram-test')
        pool=SharedCheckpointPool(9,transition_trace=trace)
        pool.admit('durable',image(b'd'),durable=True)
        with pool.reserve(9):
            self.assertNotIn('durable',pool)
        metrics=pool.metrics()
        self.assertEqual(metrics['clean_evictions'],1)
        self.assertEqual(metrics['failed_evictions'],0)
        self.assertEqual([row['event'] for row in trace.snapshot()['events']],
                         ['ram_clean_eviction'])

    def test_mark_failed_does_not_reclassify_already_durable_entry(self):
        pool=SharedCheckpointPool(32)
        pool.admit('durable',image(b'published'),durable=True)
        pool.mark_failed('durable')
        self.assertEqual(pool.metrics()['failed_entries'],0)
        self.assertEqual(pool.metrics()['failed_bytes'],0)
        with self.assertRaises(PoolFull):
            with pool.pin('durable'):
                pool.reserve(32)

    def test_failed_unpinned_entry_becomes_lru_victim_under_pressure(self):
        pool=SharedCheckpointPool(18)
        pool.admit('failed',image(b'x'),durable=False)
        pool.mark_failed('failed')
        pool.admit('clean',image(b'y'),durable=True)
        with pool.reserve(9):
            self.assertNotIn('failed',pool)
            self.assertIn('clean',pool)
        metrics=pool.metrics()
        self.assertEqual(metrics['failed_entries'],0)
        self.assertEqual(metrics['failed_bytes'],0)
        self.assertEqual(metrics['clean_evictions'],0)
        self.assertEqual(metrics['failed_evictions'],1)

    def test_pinned_failure_is_not_an_automatic_victim(self):
        pool=SharedCheckpointPool(16)
        pool.admit('failed',image(b'fail'),durable=False)
        pool.mark_failed('failed')
        with pool.pin('failed'):
            with self.assertRaises(PoolFull): pool.reserve(16)
            self.assertIn('failed',pool)
            self.assertEqual(pool.metrics()['failed_entries'],1)

    def test_dirty_unmarked_entry_remains_protected(self):
        pool=SharedCheckpointPool(16)
        pool.admit('dirty',image(b'dirty'),durable=False)
        with self.assertRaises(PoolFull): pool.reserve(16)
        self.assertIn('dirty',pool)
        self.assertEqual(pool.metrics()['failed_entries'],0)

    def test_failed_entry_participates_in_lru_mru_order(self):
        pool=SharedCheckpointPool(33)
        pool.admit('failed-old',image(b'old'),durable=False)
        pool.mark_failed('failed-old')
        pool.admit('clean',image(b'mid'),durable=True)
        pool.admit('failed-recent',image(b'new'),durable=False)
        pool.mark_failed('failed-recent')
        pool.get('failed-recent')
        with pool.reserve(11):
            self.assertNotIn('failed-old',pool)
            self.assertIn('clean',pool)
            self.assertIn('failed-recent',pool)

    def test_mark_durable_clears_failed_lifecycle_after_successful_retry(self):
        pool=SharedCheckpointPool(16)
        pool.admit('retry',image(b'retry'),durable=False)
        pool.mark_failed('retry')
        self.assertEqual(pool.metrics()['failed_bytes'],13)
        pool.mark_durable('retry')
        self.assertEqual(pool.metrics()['failed_entries'],0)
        self.assertEqual(pool.metrics()['failed_bytes'],0)
        with pool.pin('retry'):
            with self.assertRaises(PoolFull): pool.admit('other',image(b'other'),durable=True)
        self.assertIn('retry',pool)

    def test_failed_alias_accounting_tracks_unique_payload_until_last_reference(self):
        pool=SharedCheckpointPool(16)
        checkpoint=image(b'shared')
        pool.admit('bad-alias',checkpoint,durable=False)
        pool.admit('clean-alias',replace(checkpoint,input_identity='other'),durable=True)
        pool.mark_failed('bad-alias')
        self.assertEqual(pool.metrics()['failed_entries'],1)
        self.assertEqual(pool.metrics()['failed_bytes'],14)
        pool.remove('bad-alias',discard=True)
        self.assertEqual(pool.metrics()['resident_bytes'],14)
        pool.mark_failed('clean-alias')
        pool.remove('clean-alias',discard=True)
        self.assertEqual(pool.metrics()['failed_entries'],0)
        self.assertEqual(pool.metrics()['failed_bytes'],0)
        self.assertEqual(pool.metrics()['resident_bytes'],0)

    def test_evicted_failed_entry_leaves_no_pool_owned_checkpoint_reference(self):
        pool=SharedCheckpointPool(16)
        checkpoint=image(b'weakref')
        observed=weakref.ref(checkpoint)
        pool.admit('failed',checkpoint,durable=False)
        pool.mark_failed('failed')
        del checkpoint
        with pool.reserve(16):
            self.assertNotIn('failed',pool)
        gc.collect()
        self.assertIsNone(observed(), 'eviction must release the pool’s last checkpoint reference')

    def test_unpinned_failed_value_can_be_replaced_but_pinned_failed_value_cannot(self):
        pool=SharedCheckpointPool(16)
        pool.admit('key',image(b'old'),durable=False)
        pool.mark_failed('key')
        new=image(b'new')
        pool.admit('key',new,durable=False)
        self.assertEqual(pool.get('key').blob,b'new')
        pool.mark_failed('key')
        with pool.pin('key'):
            with self.assertRaises(PoolFull): pool.admit('key',image(b'next'),durable=False)
        self.assertEqual(pool.get('key').blob,b'new')


if __name__=='__main__': unittest.main()
