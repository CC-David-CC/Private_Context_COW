"""Cancellation boundaries over actual coordinator/store with a CPU fixture.

No GPU/model arithmetic is qualified here. Late cancellation cannot reverse an
already-started durable publication or turn the same operation ID into new work.
"""
from concurrent.futures import CancelledError
import threading
import unittest

import serve.test_cow_vision_preparer as fixture


class RootCancelPublicationTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionPreparerTests()
        self.addCleanup(self.flow.doCleanups)
        self.flow.setUp()

    def test_prelaunch_cancel_does_not_encode_or_forward(self):
        f=self.flow
        body=f.body();presentation=f.preparer.presentation(body,'already-cancelled')
        before=f.models[-1].backend.stats['forward_tokens']
        handle=f.manager.prepare_root('never-started',body,'already-cancelled',
            presentation=presentation,cancelled=lambda:True)
        with self.assertRaises(CancelledError):handle.durable.result(5)
        self.assertEqual(f.encoder.calls,[])
        self.assertEqual(f.models[-1].backend.stats['forward_tokens'],before)
        self.assertNotIn('never-started',[head.name for head in f.store.branches()])
        self.assertEqual(f.manager.pool.metrics()['reserved_bytes'],0)
        self.assertEqual(f.manager.disk.metrics()['reserved_bytes'],0)

    def test_cancellation_after_publication_starts_keeps_durable_result_and_retry(self):
        f=self.flow;cancelled=threading.Event();model=f.models[-1]
        original=model.save_prepared_branch
        def publish_then_complete(*args,**kwargs):
            cancelled.set()
            return original(*args,**kwargs)
        model.save_prepared_branch=publish_then_complete
        body=f.body();presentation=f.preparer.presentation(body,'publication-wins')
        handle=f.manager.prepare_root('retained',body,'publication-wins',
            presentation=presentation,cancelled=cancelled.is_set)
        receipt=handle.durable.result(5)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(f.store.head('retained').tip,receipt['snapshot'])
        self.assertEqual(f.manager.operation_status('publication-wins')['state'],'durable')
        work=model.backend.stats['forward_tokens'];calls=list(f.encoder.calls)
        again=f.manager.prepare_root('retained',body,'publication-wins',
            presentation=presentation,cancelled=cancelled.is_set)
        self.assertIs(again,handle)
        self.assertEqual(again.durable.result(5),receipt)
        self.assertEqual(model.backend.stats['forward_tokens'],work)
        self.assertEqual(f.encoder.calls,calls)


if __name__=='__main__':unittest.main()
