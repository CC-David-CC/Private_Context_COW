"""Publication adapter plumbing; synthetic bytes, not GPU qualification."""
import unittest

import test_prepared_checkpoint as fixtures
from context_time_travel.store import InvalidOperation, IntegrityError
from context_time_travel.store import Store, _hash, _json_bytes


class PreparedPresentationTests(unittest.TestCase):
    setUp = fixtures.PreparedCheckpointTests.setUp
    prepared = fixtures.PreparedCheckpointTests.prepared

    def test_checkpoint_metadata_and_presentation_reach_atomic_move_after_saved_bytes_valid(self):
        ready=self.prepared()
        namespace=self.model.store.create_namespace('Synthetic chat','namespace').namespace_id
        self.model.store.fork('main',self.sid,namespace_id=namespace,operation_id='fork')
        descriptor=dict(schema=1,namespace_id=namespace,kind='root',
            messages=[dict(id='a'*32,role='user',content='synthetic prompt')],
            source=dict(snapshot=None,revision=0,token_count=0),
            input_span=dict(start=0,count=2,token_ids_sha256=_hash(_json_bytes(list(self.model.history(self.sid))))),
            output_span=None,renderer_id='b'*64,turn_complete=False)
        original=self.model.store.move;calls=[]
        def publish(name,revision,snapshot,operation,**kwargs):
            self.model.validate_checkpoint(snapshot)
            calls.append(kwargs)
            return original(name,revision,snapshot,operation,**kwargs)
        self.model.store.move=publish
        result=self.model.save_prepared_branch('main',0,ready,'save',presentation=descriptor)
        self.assertEqual(calls,[{'presentation':descriptor,'checkpoint_receipt':{
            'schema':1,'snapshot':ready.snapshot,
            'payload_sha256':ready.checkpoint.checksum,
            'model_contract_id':ready.checkpoint.model_contract_id,
            'execution_profile_id':ready.checkpoint.execution_profile_id}}])
        self.assertEqual(result['branch']['revision'],1)
        receipt=self.model.store.checkpoint_receipt('save')
        self.assertEqual(receipt['checkpoint_receipt'],calls[0]['checkpoint_receipt'])

    def test_default_prepared_save_binds_checkpoint_metadata_to_atomic_move(self):
        ready=self.prepared();self.model.store.fork('main',self.sid)
        original=self.model.store.move;calls=[]
        def publish(name,revision,snapshot,operation,**kwargs):
            calls.append(kwargs);return original(name,revision,snapshot,operation,**kwargs)
        self.model.store.move=publish
        result=self.model.save_prepared_branch('main',0,ready,'save')
        expected={'schema':1,'snapshot':ready.snapshot,
            'payload_sha256':ready.checkpoint.checksum,
            'model_contract_id':ready.checkpoint.model_contract_id,
            'execution_profile_id':ready.checkpoint.execution_profile_id}
        self.assertEqual(calls,[{'checkpoint_receipt':expected}])
        receipt=self.model.store.checkpoint_receipt('save')
        self.assertEqual(receipt['checkpoint_receipt'],expected)
        self.assertEqual(receipt['result']['tip'],result['snapshot'])

    def test_rejected_presentation_cannot_acknowledge_or_move_branch(self):
        ready=self.prepared();before=self.model.store.fork('main',self.sid)
        def reject(*args,**kwargs):
            if 'presentation' not in kwargs: self.fail('descriptor bypassed validation')
            raise InvalidOperation('invalid canonical presentation')
        self.model.store.move=reject
        with self.assertRaises(InvalidOperation):
            self.model.save_prepared_branch('main',0,ready,'save',presentation={'schema':1})
        self.assertEqual(self.model.store.head('main'),before)

    def test_corrupt_checkpoint_never_enters_presentation_commit(self):
        ready=self.prepared();self.model.store.fork('main',self.sid)
        self.model.publish_prepared_checkpoint(ready)
        payload=self.model.checkpoint_directory/(ready.checkpoint.checksum+'.state')
        payload.write_bytes(b'corrupt')
        def forbidden(*args,**kwargs): self.fail('corrupt numerical bytes reached branch commit')
        self.model.store.move=forbidden
        with self.assertRaises(IntegrityError):
            self.model.save_prepared_branch('main',0,ready,'save',presentation={'schema':1})

    def test_real_store_root_receipt_survives_numeric_publication_and_restart(self):
        ready=self.prepared()
        namespace=self.model.store.create_namespace('Synthetic chat','namespace').namespace_id
        self.model.store.fork('main',self.sid,namespace_id=namespace,operation_id='fork')
        binding=dict(schema=1,namespace_id=namespace,kind='root',
            messages=[dict(id='a'*32,role='user',content='synthetic prompt')],
            source=dict(snapshot=None,revision=0,token_count=0),
            input_span=dict(start=0,count=2,token_ids_sha256=_hash(_json_bytes(list(self.model.history(self.sid))))),
            output_span=None,renderer_id='b'*64,turn_complete=False)
        result=self.model.save_prepared_branch('main',0,ready,'save',presentation=binding)
        restarted=Store(self.model.store.root)
        self.assertEqual(restarted.presentation_receipt('save')['presentation'],binding)
        self.assertEqual(restarted.head('main').revision,result['branch']['revision'])
        self.model.validate_checkpoint(result['snapshot'])


if __name__=='__main__':unittest.main()
