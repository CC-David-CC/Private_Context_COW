"""CPU adapter protocol checks; these do not simulate GPU numerical success."""
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.strata_backend import StrataImage
from context_time_travel.store import IntegrityError
import test_vision_backend_binding as binding_tests


class VisionColdProtocolTests(unittest.TestCase):
    def setUp(self):
        binding_tests.VisionBackendBindingTests.setUp(self)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.backend._directory = Path(self.tmp.name)
        self.backend.stats = dict(forward_tokens=0)
        self.backend._refresh_stats = lambda: dict(self.backend.stats)
        self.backend._protocol_failed = False
        self.state = StrataImage(b'identity', b'worker', 3, 9, 1, 64, (), self.history.digest)
        self.lease.image_identity = self.state.identity
        self.backend.export = lambda lease: self.state
        self.requests = []
        def request(command):
            self.calls.append(command)
            if command.startswith('load_image '):
                decoder = json.JSONDecoder(); rest = command[len('load_image '):]; paths=[]
                while rest:
                    value, end = decoder.raw_decode(rest); paths.append(Path(value)); rest=rest[end:].lstrip()
                self.assertEqual(paths[0].read_bytes(), b'native-cold-state')
                self.requests.append(paths[1].read_text())
                return dict(status='ok', consumed=len(self.history))
            if command.startswith('write '):
                Path(json.loads(command[len('write '):])).write_bytes(b'native-cold-state')
                return dict(status='ok', bytes=len(b'native-cold-state'))
            return dict(self.info)
        self.backend._request = request

    def test_new_lease_cold_restore_transports_metadata_without_embedding_path(self):
        self.lease.bound=False; self.lease.consumed=0; self.lease.inputs=None
        restored=self.backend.restore(self.lease,b'native-cold-state',history=self.history,frontier=(0.,)*16)
        self.assertEqual(restored,self.state)
        self.assertEqual(self.lease.inputs,self.history)
        self.assertEqual(self.backend.stats['forward_tokens'],0)
        self.assertEqual(len(self.requests),1)
        metadata=self.requests[0]
        self.assertIn('COW_VISION_RESTORE_V1',metadata)
        self.assertIn('d'*64,metadata)
        self.assertNotIn('.sve',metadata)
        self.assertFalse(list(self.backend._directory.iterdir()))

    def test_serialization_rechecks_active_input_binding_before_writing(self):
        self.info['native_input_sha256']='0'*64
        with self.assertRaises(IntegrityError): self.backend.serialize(self.lease,self.state)
        self.assertFalse(any(command.startswith('write ') for command in self.calls))

    def test_cold_restore_bad_echo_poisoned_and_scratch_removed(self):
        self.lease.bound=False; self.lease.consumed=0; self.lease.inputs=None
        self.info['position_sha256']='0'*64
        with self.assertRaises(IntegrityError):
            self.backend.restore(self.lease,b'native-cold-state',history=self.history,frontier=(0.,)*16)
        self.assertTrue(self.backend._protocol_failed)
        self.assertEqual(len(self.requests),1)
        self.assertFalse(list(self.backend._directory.iterdir()))

    def test_completed_load_with_wrong_consumed_receipt_cannot_leave_usable_lease(self):
        self.lease.bound=False; self.lease.consumed=0; self.lease.inputs=None
        original=self.backend._request
        def malformed(command):
            response=original(command)
            return dict(response,consumed=10) if command.startswith('load_image ') else response
        self.backend._request=malformed
        with self.assertRaises(IntegrityError):
            self.backend.restore(self.lease,b'native-cold-state',history=self.history,frontier=(0.,)*16)
        self.assertEqual(len(self.requests),1)
        self.assertTrue(self.backend._protocol_failed)
        self.assertFalse(list(self.backend._directory.iterdir()))


if __name__=='__main__': unittest.main()
