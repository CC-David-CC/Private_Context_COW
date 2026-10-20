"""Behavioral baseline for removing repeated RAM digest work in one handoff."""
import hashlib
import unittest
from unittest import mock

import test_durable_restore as durable_fixtures
from context_time_travel.real import Checkpoint
from context_time_travel.pools import SharedCheckpointPool


class CheckpointDigestAttestationBaselineTests(unittest.TestCase):
    def test_prepared_pool_writer_handoff_does_not_rehash_same_immutable_owner(self):
        durable_fixtures.DurableRestoreControlTests.setUp(self)
        self.addCleanup(durable_fixtures.DurableRestoreControlTests.tearDown, self)
        self.model.cache[self.sid] = Checkpoint('model','execution',2,b'native-state',
            (1.,2.),Checkpoint.digest(b'native-state',(1.,2.)),
            self.model._input_identity(self.model.history(self.sid)))
        self.backend.serialize = lambda *_: b'native-state'
        prepared = self.model.prepare_checkpoint(self.sid)
        pool = SharedCheckpointPool(1024)

        original = hashlib.sha256
        hashed_payloads = []
        owner_blob = prepared.checkpoint.blob
        def count_payload_digest(data=b'', *args, **kwargs):
            if data is owner_blob:
                hashed_payloads.append(len(data))
            return original(data, *args, **kwargs)

        # File digests use sha256() followed by streamed updates and remain
        # required; this counts only RAM digests seeded with immutable bytes.
        with mock.patch('hashlib.sha256', side_effect=count_payload_digest):
            pool.admit('prepared', prepared.checkpoint, durable=True)
            self.model.publish_prepared_checkpoint(prepared)

        self.assertEqual(hashed_payloads, [])
        self.assertEqual(pool.get('prepared'), prepared.checkpoint)
        pool.close()


if __name__ == '__main__':
    unittest.main()
