"""Supplemental control-plane tests; no GPU/numerical certification claims."""
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel, Checkpoint
from context_time_travel.store import Store, IntegrityError


class ControlBackend:
    vocab_size = 2
    def __init__(self):
        self.stats = {'consumed_tokens': 0, 'loaded_file_bytes': 0}
        self.profile = {'model_contract_id': 'model', 'execution_profile_id': 'execution'}
        self.replay = False
        self.frontier = (1., 2.)
        self.count = 0
    @contextmanager
    def lease(self):
        yield self
    def work_counters(self):
        return dict(self.stats)
    def restore(self, lease, blob, *, history, frontier):
        self.count = len(history)
        self.frontier = frontier
        self.stats['loaded_file_bytes'] += len(blob)
        if self.replay:
            self.stats['consumed_tokens'] += len(history)
        return blob
    def position(self, lease):
        return self.count - 1
    def logits(self, lease):
        return self.frontier
    def close(self):
        self.closed = True


class DurableRestoreControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = ControlBackend()
        self.model = BranchingModel.__new__(BranchingModel)
        self.model.backend = self.backend
        self.model.profile = self.backend.profile
        self.model.store = Store(self.tmp.name)
        self.model.cache = {}
        self.model.checkpoint_directory = Path(self.tmp.name)/'checkpoints'
        self.model.checkpoint_directory.mkdir()
        self.model._checkpoint_fault = lambda point: None
        self.sid = self.model.store.create_root('model', (0, 1), {'status': 'idle'})

    def persist(self):
        import struct
        payload = b'native-state' + struct.pack('<2f', 1., 2.)
        sha = hashlib.sha256(payload).hexdigest()
        manifest = dict(schema=1, snapshot=self.sid,
            input_identity=self.model._input_identity(self.model.history(self.sid)), model_contract_id='model',
            execution_profile_id='execution', consumed=len(self.model.history(self.sid)), positions='absolute-zero-based',
            state_bytes=12, logits_count=2, payload_sha256=sha)
        from context_time_travel.store import _json_bytes
        envelope = dict(manifest=manifest, checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        self.model._publish_checkpoint_file(self.model.checkpoint_directory/(sha+'.state'),payload)
        self.model._publish_checkpoint_file(self.model.checkpoint_directory/(self.sid+'.json'),_json_bytes(envelope))
        return sha

    def test_dr_001_missing_disk_even_with_hot_image(self):
        self.model.cache[self.sid] = object()
        with self.assertRaises(IntegrityError):
            self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['consumed_tokens'], 0)

    def test_dr_002_valid_receipt_and_frontier(self):
        sha = self.persist()
        receipt = self.model.restore_checkpoint(self.sid)
        self.assertEqual(receipt['payload_sha256'], sha)
        self.assertEqual(receipt['forward_tokens'], 0)
        self.assertEqual(receipt['consumed'], 2)
        self.assertEqual(self.model.cache[self.sid].frontier, (1., 2.))

    def test_dr_schedule_missing_rejected_before_native_load(self):
        self.persist()
        self.model.profile['execution'] = dict(resume_contract='identical-native-batches-v1',prefill_chunk=2)
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)

    def test_dr_schedule_bound_restore_and_wrong_requested_schedule(self):
        self.persist()
        from context_time_travel.execution_receipt import tokens_digest,schedule_digest
        from context_time_travel.store import _json_bytes
        schedule = [dict(operator='batched-prefill',position=0,count=2,
            tokens_sha256=tokens_digest((0,1)),chunks=[[0,2]])]
        path = self.model.checkpoint_directory/(self.sid+'.json')
        envelope = json.loads(path.read_text())
        manifest = envelope['manifest']
        manifest.update(schema=2,execution_schedule=schedule,schedule_sha256=schedule_digest(schedule))
        envelope['checksum'] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
        path.write_bytes(_json_bytes(envelope))
        self.model.profile['execution'] = dict(resume_contract='identical-native-batches-v1',prefill_chunk=2)
        with self.assertRaises(IntegrityError):
            self.model.restore_checkpoint(self.sid,expected_schedule_sha256='0'*64)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)
        receipt = self.model.restore_checkpoint(self.sid,expected_schedule_sha256=schedule_digest(schedule))
        self.assertEqual(receipt['schedule_sha256'],schedule_digest(schedule))
        self.assertEqual(receipt['forward_tokens'],0)
        manifest['execution_schedule'][0]['chunks'] = [[1,2]]
        envelope['checksum'] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
        path.write_bytes(_json_bytes(envelope))
        loaded = self.backend.stats['loaded_file_bytes']
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],loaded)

    def test_dr_007_nonfinite_frontier_rejected(self):
        import struct
        from context_time_travel.store import _json_bytes
        sha = self.persist()
        path = self.model.checkpoint_directory/(sha+'.state')
        payload = path.read_bytes()[:-8] + struct.pack('<2f', float('nan'), 2.)
        new_sha = hashlib.sha256(payload).hexdigest()
        (self.model.checkpoint_directory/(new_sha+'.state')).write_bytes(payload)
        manifest_path = self.model.checkpoint_directory/(self.sid+'.json')
        envelope = json.loads(manifest_path.read_bytes())
        envelope['manifest']['payload_sha256'] = new_sha
        envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
        manifest_path.write_bytes(_json_bytes(envelope))
        with self.assertRaises(IntegrityError):
            self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'], 0)

    def test_dr_008_replay_is_failure(self):
        self.persist()
        self.backend.replay = True
        with self.assertRaises(IntegrityError):
            self.model.restore_checkpoint(self.sid)
        self.assertNotIn(self.sid, self.model.cache)
        self.assertTrue(self.backend.closed)

    def test_dr_006_durable_move_retry_and_stale_revision(self):
        from context_time_travel.store import Conflict, InvalidOperation
        self.model.store.fork('main', self.sid)
        self.model._checkpoint_fault = lambda point: None
        self.model.checkpoint = lambda snapshot: self.persist()
        result = self.model.save_branch('main', 0, self.sid, 'save-one')
        self.assertEqual(result['branch']['revision'], 1)
        self.assertEqual(self.model.save_branch('main', 0, self.sid, 'save-one'), result)
        with self.assertRaises(Conflict):
            self.model.save_branch('main', 0, self.sid, 'save-two')
        with self.assertRaises(InvalidOperation):
            self.model.save_branch('main', 1, self.sid, 'save-one')

    def test_dr_011_failed_write_keeps_old_head(self):
        self.model.store.fork('main', self.sid)
        self.model._checkpoint_fault = lambda point: None
        def fail(snapshot):
            raise OSError(28, 'No space left on device')
        self.model.checkpoint = fail
        with self.assertRaises(OSError):
            self.model.save_branch('main', 0, self.sid, 'save-one')
        self.assertEqual(self.model.store.head('main').revision, 0)

    def test_dr_005_process_kills_at_publication_boundaries(self):
        # Actual process deaths and real filesystem writes, synthetic native bytes.
        # This verifies publication only, never numerical/GPU completeness.
        points = ('payload_during_write','payload_before_publish','payload_durable',
            'manifest_during_write','manifest_before_publish','manifest_durable',
            'before_branch_publish','after_refs_stage_fsync','after_refs_rename_before_dir_fsync',
            'branch_committed_before_ack')
        for point in points:
            with self.subTest(point=point), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, __file__, '--crash', directory, point],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 73, result.stderr)
                store = Store(directory)
                head = store.head('main')
                committed = point in ('after_refs_rename_before_dir_fsync','branch_committed_before_ack')
                self.assertEqual(head.revision, 1 if committed else 0)
                self.assertEqual(store.lineage(head.tip)[1],[0,1,1] if committed else [0,1])
                manifest = Path(directory)/'checkpoints'/(head.tip+'.json')
                self.assertTrue(manifest.exists())
                record = json.loads(manifest.read_bytes())['manifest']
                payload = manifest.parent/(record['payload_sha256']+'.state')
                self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(),record['payload_sha256'])
                if head.revision:
                    self.assertTrue(manifest.exists())
                    self.assertEqual(store.move('main',0,head.tip,'save-crash'),head)

    def test_dr_007_corruption_and_compatibility_matrix(self):
        from context_time_travel.store import _json_bytes
        sha = self.persist()
        manifest_path = self.model.checkpoint_directory/(self.sid+'.json')
        original = manifest_path.read_bytes()
        for field, value in [('model_contract_id','wrong'),('execution_profile_id','wrong'),
            ('positions','offset'),('consumed',3),('input_identity','wrong'),('schema',2)]:
            with self.subTest(field=field):
                envelope = json.loads(original)
                envelope['manifest'][field] = value
                envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
                manifest_path.write_bytes(_json_bytes(envelope))
                with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
                self.assertEqual(self.backend.stats['loaded_file_bytes'],0)
        manifest_path.write_bytes(original)
        payload = self.model.checkpoint_directory/(sha+'.state')
        payload.write_bytes(payload.read_bytes()[:-1])
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)

    def test_dr_006_reader_pins_released_candidate_during_gc(self):
        candidate = self.model.store.append(self.sid,(1,),{'status':'idle'})
        self.model.store.release_history(candidate)
        with self.model.store.read_lease(candidate):
            result = self.model.store.collect_history(self.sid)
            self.assertNotIn(candidate,result['deleted'])
        result = self.model.store.collect_history(self.sid)
        self.assertIn(candidate,result['deleted'])

    def test_dr_011_retry_must_flush_visible_manifest_before_ack(self):
        from unittest.mock import patch
        self.persist()
        self.model.store.fork('main',self.sid)
        # A previous rename may have succeeded while directory fsync failed.
        # Merely seeing the file is insufficient to promise durable completion.
        with patch('context_time_travel.real._sync_dir',side_effect=OSError(5,'I/O error')):
            with self.assertRaises(OSError):
                self.model.save_branch('main',0,self.sid,'save-one')
        self.assertEqual(self.model.store.head('main').revision,0)

    def test_dr_007_unknown_controller_rejected_before_import(self):
        from context_time_travel import controller
        state = controller.start({},'controller-fixture',(0,1),2)
        state['schema'] = 99
        self.sid = self.model.store.append(self.sid,(),state)
        self.persist()
        with self.assertRaises(IntegrityError): self.model.restore_checkpoint(self.sid)
        self.assertEqual(self.backend.stats['loaded_file_bytes'],0)


def crash_worker(directory, point):
    case = DurableRestoreControlTests()
    case.setUp()
    case.model.store = Store(directory)
    case.model.checkpoint_directory = Path(directory)/'checkpoints'
    case.model.checkpoint_directory.mkdir()
    case.sid = case.model.store.create_root('model',(0,1),{'status':'idle'})
    case.persist()  # The old reference already has a complete durable save point.
    case.model.store.fork('main',case.sid)
    case.sid = case.model.store.append(case.sid,(1,),{'status':'idle'})
    case.model.checkpoint = lambda snapshot: case.persist()
    case.model._checkpoint_fault = lambda where: os._exit(73) if where == point else None
    case.model.store.fault_hook = case.model._checkpoint_fault
    case.model.save_branch('main',0,case.sid,'save-crash')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--crash':
        crash_worker(sys.argv[2],sys.argv[3])
    else:
        unittest.main(verbosity=2)
