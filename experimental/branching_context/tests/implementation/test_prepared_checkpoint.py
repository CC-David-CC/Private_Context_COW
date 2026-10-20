"""Publication tests with synthetic native bytes, not model qualification."""
import threading
import unittest
import test_durable_restore as fixtures
from context_time_travel.real import Checkpoint
from context_time_travel.store import Conflict, IntegrityError


class PreparedCheckpointTests(unittest.TestCase):
    setUp = fixtures.DurableRestoreControlTests.setUp

    def prepared(self):
        self.backend.serialize = lambda *_: b'native-state'
        self.model.cache[self.sid] = Checkpoint('model','execution',2,b'native-state',
            (1.,2.),Checkpoint.digest(b'native-state',(1.,2.)),
            self.model._input_identity(self.model.history(self.sid)))
        return self.model.prepare_checkpoint(self.sid)

    def test_publish_in_foreign_thread_never_enters_native_lease(self):
        ready = self.prepared()
        self.model.store.fork('main',self.sid)
        def forbidden(): raise AssertionError('writer touched native execution')
        self.backend.lease = forbidden
        result = []
        def writer(): result.append(self.model.save_prepared_branch('main',0,ready,'publish'))
        t = threading.Thread(target=writer);t.start();t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(result[0]['branch']['revision'],1)
        self.assertEqual(self.model._load_checkpoint(self.sid,self.model.history(self.sid)).blob,b'native-state')

    def test_wrong_profile_or_modified_payload_rejected_before_publish(self):
        from dataclasses import replace
        ready = self.prepared()
        for bad in (replace(ready,checkpoint=replace(ready.checkpoint,blob=b'corrupt')),
                    replace(ready,checkpoint=replace(ready.checkpoint,execution_profile_id='other'))):
            with self.assertRaises(IntegrityError): self.model.publish_prepared_checkpoint(bad)
        self.assertEqual(list(self.model.checkpoint_directory.iterdir()),[])

    def test_failure_before_manifest_keeps_previous_branch(self):
        ready = self.prepared();self.model.store.fork('main',self.sid)
        def fail(point):
            if point == 'before_manifest_publish': raise OSError('disk full')
        self.model._checkpoint_fault = fail
        with self.assertRaises(OSError): self.model.save_prepared_branch('main',0,ready,'save')
        self.assertEqual(self.model.store.head('main').revision,0)
        self.assertFalse((self.model.checkpoint_directory/(self.sid+'.json')).exists())

    def test_retry_validates_files_and_stale_move_does_not_overwrite(self):
        ready = self.prepared();self.model.store.fork('main',self.sid)
        receipt = self.model.save_prepared_branch('main',0,ready,'save')
        self.assertEqual(self.model.save_prepared_branch('main',0,ready,'save'),receipt)
        with self.assertRaises(Conflict): self.model.save_prepared_branch('main',0,ready,'stale')
        payload = self.model.checkpoint_directory/(ready.checkpoint.checksum+'.state')
        payload.write_bytes(b'corrupt')
        with self.assertRaises(IntegrityError): self.model.save_prepared_branch('main',0,ready,'save')

    def test_new_payload_corrupted_during_publication_cannot_move_branch(self):
        self.prepared();self.model.checkpoint(self.sid)
        self.model.store.fork('main',self.sid)
        child=self.model.store.append(self.sid,(0,),{'status':'idle'})
        cp=Checkpoint('model','execution',3,b'child-state',(1.,2.),
            Checkpoint.digest(b'child-state',(1.,2.)),self.model._input_identity(self.model.history(child)))
        prepared=self.model._prepared_checkpoint(child,cp,cp.blob)
        def corrupt(point):
            if point=='payload_durable':
                path=self.model.checkpoint_directory/(cp.checksum+'.state')
                raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
        self.model._checkpoint_fault=corrupt
        with self.assertRaises(IntegrityError): self.model.save_prepared_branch('main',0,prepared,'bad-write')
        self.assertEqual(self.model.store.head('main').tip,self.sid)
        self.assertEqual(self.model.store.head('main').revision,0)
        self.model.validate_checkpoint(self.sid)


if __name__ == '__main__': unittest.main()
