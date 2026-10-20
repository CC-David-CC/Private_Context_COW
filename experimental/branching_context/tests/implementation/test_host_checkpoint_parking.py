"""Host-cache control-plane tests; no mock is GPU qualification."""
import hashlib
import unittest
import test_durable_restore as fixtures
from context_time_travel.real import Checkpoint
from context_time_travel.checkpoint_cache import CheckpointCache
from context_time_travel.store import IntegrityError


class HostCheckpointParkingTests(unittest.TestCase):
    setUp = fixtures.DurableRestoreControlTests.setUp

    def hot(self):
        self.backend.stats['forward_tokens'] = 0
        self.backend.serialize = lambda lease,state: b'parked-native-bytes'
        self.model.cache = CheckpointCache(1024,3)
        class Image:
            identity = b'hot-native'
            def __len__(self): return 16
        state = Image()
        checkpoint = Checkpoint('model','execution',2,state,(1.,2.),
            Checkpoint.digest(state,(1.,2.)),self.model._input_identity(self.model.history(self.sid)))
        self.model.cache[self.sid] = checkpoint
        self.model.store.fork('retained',self.sid)
        return checkpoint

    def test_host_parking_preserves_history_controller_reference_and_frontier(self):
        original = self.hot()
        before = self.model.store.head('retained')
        canonical = self.model.snapshot_bytes(self.sid)
        receipt = self.model.park_checkpoint(self.sid)
        parked = self.model.cache[self.sid]
        self.assertEqual(parked.blob,b'parked-native-bytes')
        self.assertEqual(parked.frontier,original.frontier)
        self.assertEqual(parked.execution_schedule,original.execution_schedule)
        self.assertEqual(receipt['forward_tokens'],0)
        self.assertEqual(receipt['state_bytes'],19)
        self.assertEqual(receipt['state_sha256'],hashlib.sha256(parked.blob).hexdigest())
        self.assertEqual(self.model.snapshot_bytes(self.sid),canonical)
        self.assertEqual(self.model.store.head('retained'),before)

    def test_failed_serialization_or_admission_leaves_hot_image_and_reference(self):
        original = self.hot()
        def fail(*args): raise OSError('injected serialization failure')
        self.backend.serialize = fail
        with self.assertRaises(OSError): self.model.park_checkpoint(self.sid)
        self.assertIs(self.model.cache[self.sid],original)
        self.backend.serialize = lambda *args:b'x'*1025
        with self.assertRaises(MemoryError): self.model.park_checkpoint(self.sid)
        self.assertIs(self.model.cache[self.sid],original)
        self.assertEqual(self.model.store.head('retained').tip,self.sid)

    def test_missing_work_witness_and_replay_are_rejected(self):
        original = self.hot()
        del self.backend.stats['forward_tokens']
        with self.assertRaises(IntegrityError): self.model.park_checkpoint(self.sid)
        self.backend.stats['forward_tokens'] = 0
        def replay(*args):
            self.backend.stats['forward_tokens'] += 2
            return b'native'
        self.backend.serialize = replay
        with self.assertRaises(IntegrityError): self.model.park_checkpoint(self.sid)
        self.assertIs(self.model.cache[self.sid],original)


if __name__ == '__main__': unittest.main()
