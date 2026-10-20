"""CPU-only manifest/receipt controls; no real-backend qualification claim."""
import hashlib
import json
import unittest
import test_durable_restore as fixtures
from context_time_travel.execution_receipt import tokens_digest, schedule_digest
from context_time_travel.store import IntegrityError, _json_bytes


class OperatorBoundCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DurableRestoreControlTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model, self.backend, self.sid = self.fixture.model, self.fixture.backend, self.fixture.sid
        self.fixture.persist()
        self.model.profile['execution'] = dict(resume_contract='identical-native-operators-v1',prefill_chunk=8)
        self.path = self.model.checkpoint_directory/(self.sid+'.json')
        self.schedule = [dict(operator='batched-prefill',position=0,count=1,
                              tokens_sha256=tokens_digest([0]),chunks=[[0,1]]),
                         dict(operator='normal-verifier-decode',position=1,count=1,
                              tokens_sha256=tokens_digest([1]),chunks=[[1,1]])]

    def publish(self, schema=3):
        envelope = json.loads(self.path.read_bytes())
        envelope['manifest'].update(schema=schema,execution_schedule=self.schedule,
                                    schedule_sha256=schedule_digest(self.schedule))
        envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
        self.path.write_bytes(_json_bytes(envelope))

    def test_missing_receipts_rejected_before_native_import(self):
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)

    def test_exact_mixed_schedule_restores_and_roundtrips_version_three(self):
        self.publish()
        receipt = self.model.restore_checkpoint(self.sid,expected_schedule_sha256=schedule_digest(self.schedule))
        self.assertEqual(receipt['forward_tokens'],0)
        self.assertEqual(receipt['schedule_sha256'],schedule_digest(self.schedule))
        self.backend.serialize = lambda lease, blob: blob
        self.model.checkpoint(self.sid)
        self.assertEqual(json.loads(self.path.read_bytes())['manifest']['schema'],3)
        self.assertEqual(list(self.model._load_checkpoint(self.sid,(0,1)).execution_schedule),self.schedule)

    def test_batch_only_version_is_not_interchangeable(self):
        self.publish(schema=2)
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)

    def test_altered_tokens_and_native_boundaries_rejected_before_import(self):
        for field, value in [('tokens_sha256',tokens_digest([0])),('chunks',[[2,1]])]:
            with self.subTest(field=field):
                self.schedule[1][field] = value
                self.publish()
                with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
                self.assertEqual(self.backend.stats['loaded_file_bytes'],0)


if __name__ == '__main__': unittest.main()
