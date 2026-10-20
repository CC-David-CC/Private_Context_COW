"""Publication failures release volatile ownership and retry only retained bytes."""
import gc
import fcntl
import os
import queue
import tempfile
import threading
import time
import unittest
import weakref

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import Conflict, InvalidOperation
from test_pool_coordinator import Model


class FailedPublicationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gate=threading.Event();self.gate.set()
        self.models={};self.trace=[]
        config=PoolConfig(gpus=(0,),ram_bytes=4096,disk_bytes=16384,min_available_bytes=0)
        def factory(gpu):
            model=Model(self.temp.name,gpu,(0,),self.trace,self.gate)
            self.models[gpu]=model
            return model
        self.coordinator=ContextCoordinator(self.temp.name,config,factory,admission_timeout=1)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(timeout=5)
        self.model=self.models[0]

    def create_durable_branch(self,name='main'):
        return self.coordinator.create(name,(1,2),'create-'+name).durable.result(5)['snapshot']

    def fail_next_publication(self):
        original=self.model.save_prepared_branch
        attempts=[]
        def fail(*args,**kwargs):
            attempts.append(args[2].snapshot)
            raise OSError('simulated publication failure')
        self.model.save_prepared_branch=fail
        return original,attempts

    def failed_request(self,operation='failed-op',branch='main'):
        request=self.coordinator.submit(branch,(7,),1,operation)
        answer=request.snapshot.result(5)
        with self.assertRaises(OSError): request.durable.result(5)
        return request,answer

    def test_failed_publication_releases_pin_lease_and_prepared_object(self):
        self.create_durable_branch()
        original,_=self.fail_next_publication()
        prepared_refs=[]
        prepare=self.model.prepare_checkpoint
        def capture(snapshot):
            prepared=prepare(snapshot)
            prepared_refs.append(weakref.ref(prepared))
            return prepared
        self.model.prepare_checkpoint=capture
        request,answer=self.failed_request()
        deadline=time.monotonic()+2
        while prepared_refs[0]() is not None and time.monotonic()<deadline:
            gc.collect();time.sleep(.005)
        self.assertIsNone(prepared_refs[0]())
        self.assertEqual(self.coordinator.pool.metrics()['pins'],0)
        self.assertEqual(self.coordinator.pool.metrics()['failed_entries'],1)
        retained_error=request.durable.exception()
        self.assertIsNone(retained_error.__traceback__)
        self.assertEqual(retained_error.args,('simulated publication failure',))
        self.assertNotIn(answer['snapshot'],self.model.cache)
        lock_path=self.model.store.root/'read-leases'/(answer['snapshot']+'.lock')
        descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600)
        try:
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(descriptor,fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        self.assertTrue(self.coordinator.operation_status('failed-op')['publication_retryable'])
        pending=self.coordinator.pending_request('main')
        self.assertEqual(pending['operation_id'],'failed-op')
        self.assertTrue(pending['publication_retryable'])
        self.assertNotIn('main',self.coordinator._branches)
        self.model.save_prepared_branch=original

    def test_failed_bytes_are_evictable_and_retry_after_eviction_is_explicit(self):
        self.create_durable_branch()
        original,_=self.fail_next_publication()
        request,answer=self.failed_request()
        forward=self.model.stats['forward_tokens']
        key=self.coordinator._key(self.model,answer['snapshot'])
        self.assertIn(key,self.coordinator.pool)
        self.assertEqual(self.coordinator.pool.metrics()['failed_entries'],1)
        before=self.coordinator.store.head('main')
        with self.coordinator.pool.reserve(self.coordinator.pool.byte_limit):
            self.assertNotIn(key,self.coordinator.pool)
        self.assertEqual(self.coordinator.pool.metrics()['failed_entries'],0)
        self.assertEqual(self.coordinator.store.head('main'),before)
        self.assertFalse(self.coordinator.operation_status('failed-op')['publication_retryable'])
        self.assertIsNone(self.coordinator.pending_request('main'))
        self.model.save_prepared_branch=original
        with self.assertRaisesRegex(InvalidOperation,'evicted|available|retained'):
            self.coordinator.retry_publication('failed-op')
        self.assertEqual(self.model.stats['forward_tokens'],forward)

    def test_retry_before_eviction_reuses_bytes_without_forward_work(self):
        self.create_durable_branch()
        original,attempts=self.fail_next_publication()
        request,answer=self.failed_request()
        forward=self.model.stats['forward_tokens']
        old=self.coordinator.store.head('main')
        self.assertNotEqual(old.tip,answer['snapshot'])
        self.assertEqual(self.coordinator.store.head('main'),old)
        self.model.save_prepared_branch=original
        receipt=self.coordinator.retry_publication('failed-op').result(5)
        self.assertEqual(receipt['snapshot'],answer['snapshot'])
        self.assertEqual(self.model.stats['forward_tokens'],forward)
        self.assertEqual(self.coordinator.store.head('main').tip,answer['snapshot'])
        self.assertEqual(len(attempts),1)
        self.assertEqual(self.coordinator.metrics()['failed_publications'],0)

    def test_new_request_can_advance_branch_after_failure(self):
        self.create_durable_branch()
        original,_=self.fail_next_publication()
        _,failed=self.failed_request()
        self.model.save_prepared_branch=original
        next_request=self.coordinator.submit('main',(8,),1,'next-op')
        result=next_request.durable.result(5)
        self.assertNotEqual(result['snapshot'],failed['snapshot'])
        self.assertEqual(self.coordinator.store.head('main').tip,result['snapshot'])

    def test_stale_retry_cannot_overwrite_new_branch_head(self):
        self.create_durable_branch()
        original,_=self.fail_next_publication()
        self.failed_request()
        self.model.save_prepared_branch=original
        advanced=self.coordinator.submit('main',(8,),1,'advance-op').durable.result(5)
        with self.assertRaises(Conflict): self.coordinator.retry_publication('failed-op')
        self.assertEqual(self.coordinator.store.head('main').tip,advanced['snapshot'])
        self.assertFalse(self.coordinator.operation_status('failed-op')['publication_retryable'])
        self.assertIsNone(self.coordinator.pending_request('main'))
        self.assertEqual(self.model.stats['forward_tokens'],6)

    def test_newer_active_request_supersedes_failed_publication_status(self):
        self.create_durable_branch()
        original,_=self.fail_next_publication()
        self.failed_request()
        self.model.save_prepared_branch=original
        self.gate.clear()
        try:
            newer=self.coordinator.submit('main',(8,),1,'newer-op')
            pending=self.coordinator.pending_request('main')
            self.assertEqual(pending['operation_id'],'newer-op')
            self.assertNotEqual(pending['operation_id'],'failed-op')
        finally:
            self.gate.set()
        newer.durable.result(5)
        self.assertIsNone(self.coordinator.pending_request('main'))

    def test_collected_failed_history_is_not_retryable_and_never_recomputes(self):
        root=self.create_durable_branch()
        original,_=self.fail_next_publication()
        request,answer=self.failed_request()
        key=self.coordinator._key(self.model,answer['snapshot'])
        self.assertIn(key,self.coordinator.pool)
        self.model.store.release_history(answer['snapshot'])
        removed=self.model.store.collect_history(root)
        self.assertIn(answer['snapshot'],removed['deleted'])
        self.assertFalse((self.model.store.objects/(answer['snapshot']+'.json')).exists())
        forward=self.model.stats['forward_tokens']
        self.assertFalse(self.coordinator.operation_status('failed-op')['publication_retryable'])
        self.assertIsNone(self.coordinator.pending_request('main'))
        self.model.save_prepared_branch=original
        with self.assertRaisesRegex(InvalidOperation,'history|collected|available'):
            self.coordinator.retry_publication('failed-op')
        self.assertEqual(self.model.stats['forward_tokens'],forward)
        self.assertEqual(self.coordinator.pool.metrics()['pins'],0)
        self.assertNotIn('main',self.coordinator._branches)

    def test_failure_for_existing_durable_alias_does_not_mark_or_drop_it(self):
        root=self.create_durable_branch('first')
        key=self.coordinator._key(self.model,root)
        self.assertIn(key,self.coordinator.pool)
        self.assertIn(root,self.model.cache)
        old_head=self.coordinator.store.head('first')
        original,_=self.fail_next_publication()
        request=self.coordinator.create('second',(1,2),'create-second')
        self.assertEqual(request.snapshot.result(5)['snapshot'],root)
        with self.assertRaises(OSError): request.durable.result(5)
        self.assertEqual(self.coordinator.store.head('first'),old_head)
        self.assertIn(key,self.coordinator.pool)
        metrics=self.coordinator.pool.metrics()
        self.assertEqual(metrics['failed_entries'],0)
        self.assertTrue((self.model.checkpoint_directory/(root+'.json')).is_file())
        self.model.save_prepared_branch=original

    def test_visible_manifest_after_failed_save_does_not_prove_durability(self):
        self.create_durable_branch()
        original=self.model.save_prepared_branch
        def fail_after_visible_manifest(branch,revision,prepared,operation,*args,**kwargs):
            (self.model.checkpoint_directory/(prepared.snapshot+'.json')).write_text('{"partial":true}')
            raise OSError(28,'No space left on device')
        self.model.save_prepared_branch=fail_after_visible_manifest
        request,answer=self.failed_request()
        key=self.coordinator._key(self.model,answer['snapshot'])
        self.assertTrue((self.model.checkpoint_directory/(answer['snapshot']+'.json')).is_file())
        self.assertFalse(self.coordinator.pool.is_durable(key))
        self.assertEqual(self.coordinator.pool.metrics()['failed_entries'],1)
        error=request.durable.exception()
        self.assertEqual(error.errno,28)
        self.assertEqual(error.strerror,'No space left on device')
        self.model.save_prepared_branch=original

    def test_durable_checkpoint_stays_durable_when_re_admitted(self):
        root=self.create_durable_branch()
        key=self.coordinator._key(self.model,root)
        checkpoint=self.coordinator.pool.get(key)
        self.assertTrue(self.coordinator.pool.is_durable(key))
        with self.coordinator.pool.reserve(0) as reservation:
            reservation.admit(key,checkpoint,durable=False)
        self.assertTrue(self.coordinator.pool.is_durable(key))

    def test_detached_oserror_preserves_errno_27_and_28(self):
        for code,message in ((27,'File too large'),(28,'No space left on device')):
            with self.subTest(errno=code):
                source=OSError(code,message,'checkpoint.state')
                detached=self.coordinator._detached_error(source)
                self.assertIs(type(detached),OSError)
                self.assertEqual(detached.errno,code)
                self.assertEqual(detached.strerror,message)
                self.assertEqual(detached.filename,'checkpoint.state')
                self.assertIsNone(detached.__traceback__)

    def test_errno_27_and_28_survive_request_future_and_failure_event(self):
        original=self.model.save_prepared_branch
        for code,message in ((27,'File too large'),(28,'No space left on device')):
            with self.subTest(errno=code):
                branch='errno-'+str(code)
                self.create_durable_branch(branch)
                def fail(*args,**kwargs): raise OSError(code,message,'checkpoint.state')
                self.model.save_prepared_branch=fail
                operation='op-'+str(code)
                request=self.coordinator.submit(branch,(7,),1,operation)
                request.snapshot.result(5)
                with self.assertRaises(OSError) as raised:
                    request.durable.result(5)
                self.assertEqual(raised.exception.errno,code)
                self.assertEqual(raised.exception.strerror,message)
                events=[]
                while True:
                    try: events.append(request.events.get_nowait())
                    except queue.Empty: break
                failure=next(event for event in events if event['event']=='failure')
                self.assertEqual(failure['errno'],code)
                self.assertIn('[Errno '+str(code)+']',failure['error'])
                self.assertIn(message,failure['error'])
                self.model.save_prepared_branch=original


if __name__=='__main__': unittest.main()
