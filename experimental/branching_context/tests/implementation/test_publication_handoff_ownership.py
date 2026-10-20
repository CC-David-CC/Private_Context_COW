"""CPU ownership controls for bounded cache/publication handoff.

Synthetic native-image fixture only; no GPU or numerical qualification.
"""
from pathlib import Path
import copy
import fcntl
import json
import sys
import tempfile
import threading
import time
import unittest

TESTS=Path(__file__).resolve().parent
sys.path.insert(0,str(TESTS))
from test_bounded_tree_retention import NativeImageModel
from context_time_travel.checkpoint_cache import CheckpointCache
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.store import Store, _id


class PublicationHandoffOwnershipTests(unittest.TestCase):
    def start(self,entries):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        models={};gate=threading.Event();gate.set()
        def factory(gpu):
            models[gpu]=NativeImageModel(temp.name,gpu,(0,),[],gate,entries=entries)
            return models[gpu]
        c=ContextCoordinator(temp.name,PoolConfig(gpus=(0,),ram_bytes=4096,
            disk_bytes=16384,min_available_bytes=0),factory)
        self.addCleanup(c.close);c.ready(5)
        return c,models[0]

    @staticmethod
    def wait(predicate,message,timeout=5):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            value=predicate()
            if value:return value
            time.sleep(.005)
        raise AssertionError(message)

    @classmethod
    def create_roots(cls,c,model,names):
        roots=[]
        for index,name in enumerate(names):
            sid=c.create(name,(10+index,20+index),'create-'+name).durable.result(5)['snapshot']
            roots.append(sid)
        cls.wait(lambda:not any(c.metrics()['active_per_gpu'].values()),'root workers did not idle')
        # Give the retention test exactly the roots it intends to exercise.
        model.cache.clear()
        for sid in roots:model.image(sid)
        for index,name in enumerate(names):c.fork(name+'-child',roots[index])
        return tuple(roots)

    @staticmethod
    def lease_is_held(store,snapshot):
        path=store.root/'read-leases'/(_id(snapshot)+'.lock')
        if not path.exists():return False
        with path.open('r+b') as descriptor:
            try:
                fcntl.flock(descriptor.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
        return False

    def test_entry_limit_one_and_two_evictions_do_not_steal_blocked_writer_ownership(self):
        for entry_limit in (1,2):
            with self.subTest(entry_limit=entry_limit):
                c,model=self.start(entry_limit)
                roots=self.create_roots(c,model,('a','b','c'))
                original_save=model.save_prepared_branch
                entered=threading.Event();release=threading.Event()
                self.addCleanup(release.set)
                def blocked_save(name,revision,prepared,operation):
                    if operation=='op-a':
                        entered.set()
                        if not release.wait(5):raise TimeoutError('writer hold did not release')
                    return original_save(name,revision,prepared,operation)
                model.save_prepared_branch=blocked_save
                first=c.submit('a-child',(101,),1,'op-a')
                first_answer=first.snapshot.result(5)
                self.assertTrue(entered.wait(2))
                siblings=[c.submit(name+'-child',(token,),1,operation)
                    for name,token,operation in (('b',102,'op-b'),('c',103,'op-c'))]
                sibling_answers=[item.snapshot.result(5) for item in siblings]

                target=first_answer['snapshot'];pool_key=c._key(model,target)
                self.assertNotIn(target,model.cache,'small LRU did not exercise target eviction')
                self.assertIn(pool_key,c.pool)
                self.assertGreaterEqual(c.pool.metrics()['pins'],1)
                self.assertTrue(self.lease_is_held(c.store,target),'writer lost its canonical read lease')
                for branch in ('a-child','b-child','c-child'):
                    self.assertEqual(c.store.head(branch).tip,roots[('a-child','b-child','c-child').index(branch)])
                for item in (first,*siblings):
                    self.assertFalse(item.durable.done())
                    self.assertFalse(any(row.get('event')=='durable' for row in self.drain(item)))

                release.set()
                receipts=[first.durable.result(5),*(item.durable.result(5) for item in siblings)]
                expected=((101,0),(102,0),(103,0))
                for branch,answer,receipt,tail in zip(('a-child','b-child','c-child'),
                        (first_answer,*sibling_answers),receipts,expected):
                    self.assertEqual(receipt['snapshot'],answer['snapshot'])
                    self.assertEqual(c.store.head(branch).tip,answer['snapshot'])
                    self.assertEqual(model.history(answer['snapshot']),
                        model.history(roots[('a-child','b-child','c-child').index(branch)])+tail)
                self.assertEqual(c.pool.metrics()['pins'],0)
                self.assertEqual(c.pool.metrics()['reserved_bytes'],0)

    @staticmethod
    def drain(request):
        rows=[]
        while True:
            try:rows.append(request.events.get_nowait())
            except Exception:return rows

    def test_capture_failure_after_target_cache_admission_discards_only_new_image(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        source_image=model.cache[root].blob
        original_prepare=model.prepare_checkpoint
        def fail_capture(snapshot):
            if snapshot!=root:raise OSError(5,'synthetic capture failure after image admission')
            return original_prepare(snapshot)
        model.prepare_checkpoint=fail_capture
        before_head=c.store.head('child')
        before_keys=set(model.cache)
        request=c.submit('child',(77,),1,'capture-fails')
        with self.assertRaises(OSError) as snapshot_error:request.snapshot.result(5)
        self.assertEqual(snapshot_error.exception.errno,5)
        with self.assertRaises(OSError) as error:request.durable.result(5)
        self.assertEqual(error.exception.errno,5)
        self.assertEqual(c.store.head('child'),before_head)
        self.assertEqual(set(model.cache),before_keys)
        self.assertIn(root,model.cache)
        self.assertIs(model.cache[root].blob,source_image)
        self.assertFalse(any(row.get('event')=='durable' for row in self.drain(request)))
        self.assertEqual(c.pool.metrics()['pins'],0)

    def test_submit_failure_after_pool_pin_releases_pin_lease_and_nondurable_entry(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        before_work=model.stats['forward_tokens']
        original_submit=c._submit_publication
        def fail_submit(*_args):raise OSError(28,'synthetic writer handoff failure')
        c._submit_publication=fail_submit
        self.addCleanup(setattr,c,'_submit_publication',original_submit)
        before_head=c.store.head('child')
        request=c.submit('child',(88,),1,'submit-fails')
        target=request.snapshot.result(5)['snapshot']
        with self.assertRaises(OSError) as error:request.durable.result(5)
        self.assertEqual(error.exception.errno,28)
        self.assertEqual(c.store.head('child'),before_head)
        self.assertFalse(any(row.get('event')=='durable' for row in self.drain(request)))
        self.assertEqual(c.pool.metrics()['pins'],0,'pre-submit failure leaked the pool pin')
        self.assertEqual(c.pool.metrics()['reserved_bytes'],0)
        self.assertIn(c._key(model,target),c.pool,'prepared bytes should remain retryable when eligible')
        self.assertFalse(self.lease_is_held(c.store,target),'pre-submit failure leaked the canonical read lease')
        self.assertNotIn(target,model.cache,'pre-submit failure retained failed native image')
        self.assertEqual(model.history(target),model.history(root)+(88,0))
        self.assertTrue(c.operation_status('submit-fails')['publication_retryable'])
        self.assertEqual(c.store.head('child'),before_head)
        c._submit_publication=original_submit
        receipt=c.retry_failed('submit-fails').result(5)
        self.assertEqual(receipt['snapshot'],target)
        self.assertEqual(c.store.head('child').tip,target)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)
        self.assertEqual(c.pool.metrics()['pins'],0)

    def test_backend_cleanup_failure_before_writer_handoff_cannot_publish_half_owned_job(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        before_work=model.stats['forward_tokens']
        original_release=model.release_checkpoints
        calls=0
        def fail_release():
            nonlocal calls
            calls+=1
            if calls==1:raise OSError(5,'synthetic native release failure')
            return original_release()
        model.release_checkpoints=fail_release
        self.addCleanup(setattr,model,'release_checkpoints',original_release)
        before_head=c.store.head('child')
        request=c.submit('child',(99,),1,'release-fails')
        target=request.snapshot.result(5)['snapshot']
        with self.assertRaises(OSError) as error:request.durable.result(5)
        self.assertEqual(error.exception.errno,5)
        self.assertEqual(c.store.head('child'),before_head)
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertFalse(self.lease_is_held(c.store,target))
        self.assertIn(c._key(model,target),c.pool)
        self.assertTrue(c.operation_status('release-fails')['publication_retryable'])
        calls_before_retry=calls
        receipt=c.retry_failed('release-fails').result(5)
        self.assertEqual(receipt['snapshot'],target)
        self.assertEqual(calls,calls_before_retry)
        self.assertEqual(c.store.head('child').tip,target)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)
        self.assertEqual(c.pool.metrics()['pins'],0)

    def test_retry_submit_rejection_preserves_failed_payload_and_can_retry_again(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        before_work=model.stats['forward_tokens']
        original_save=model.save_prepared_branch
        attempts=[]
        def fail_first_publish(name,revision,prepared,operation):
            attempts.append(prepared.snapshot)
            if len(attempts)==1:raise OSError(28,'initial disk publication failure')
            return original_save(name,revision,prepared,operation)
        model.save_prepared_branch=fail_first_publish
        first=c.submit('child',(111,),1,'retry-handoff')
        target=first.snapshot.result(5)['snapshot']
        with self.assertRaises(OSError):first.durable.result(5)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)
        self.assertTrue(c.operation_status('retry-handoff')['publication_retryable'])

        original_submit=c._submit_publication
        def reject_retry(*_args):raise OSError(28,'retry writer submission rejected')
        c._submit_publication=reject_retry
        self.addCleanup(setattr,c,'_submit_publication',original_submit)
        try:
            retry_future=c.retry_failed('retry-handoff')
        except OSError as submit_error:
            # Baseline retry_failed raises before terminalizing the replacement
            # future. Complete it only to keep this intentional RED bounded;
            # the assertion below still records the contract failure.
            pending=c._requests['retry-handoff'].durable
            if not pending.done():pending.set_exception(submit_error)
            self.fail('retry submission rejection must return a terminal failed future')
        with self.assertRaises(OSError) as error:retry_future.result(5)
        self.assertEqual(error.exception.errno,28)
        self.assertEqual(c.store.head('child').tip,root)
        self.assertFalse(c._branches.intersection({'child'}))
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertEqual(c.pool.metrics()['reserved_bytes'],0)
        self.assertFalse(self.lease_is_held(c.store,target))
        self.assertIn(c._key(model,target),c.pool)
        self.assertTrue(c.operation_status('retry-handoff')['publication_retryable'])
        self.assertEqual(model.stats['forward_tokens'],before_work+2)

        c._submit_publication=original_submit
        final=c.retry_failed('retry-handoff').result(5)
        self.assertEqual(final['snapshot'],target)
        self.assertEqual(c.store.head('child').tip,target)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)
        self.assertEqual(attempts,[target,target])
        self.assertEqual(c.pool.metrics()['pins'],0)

    def test_enqueue_then_raise_cannot_publish_after_worker_reports_handoff_failure(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        before_work=model.stats['forward_tokens']
        before_head=c.store.head('child')
        original_writer=c._writer
        self.addCleanup(setattr,c,'_writer',original_writer)

        class EnqueueThenRaise:
            def __init__(self):self.jobs=[]
            def submit(self,function,*args,**kwargs):
                self.jobs.append((function,args,kwargs))
                raise OSError(11,'synthetic thread-start failure after enqueue')

        rejecting=EnqueueThenRaise();c._writer=rejecting
        request=c.submit('child',(121,),1,'enqueue-then-raise')
        target=request.snapshot.result(5)['snapshot']
        with self.assertRaises(OSError) as error:request.durable.result(5)
        self.assertEqual(error.exception.errno,11)
        self.assertEqual(c.store.head('child'),before_head)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)
        self.assertTrue(c.operation_status('enqueue-then-raise')['publication_retryable'])
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertFalse(self.lease_is_held(c.store,target))
        self.assertEqual(len(rejecting.jobs),1)

        # A submit implementation may raise after placing work on its queue.
        # Execute that orphan explicitly to prove its handoff gate rejects it.
        c._writer=original_writer
        function,args,kwargs=rejecting.jobs[0]
        orphan_error=None
        try:function(*args,**kwargs)
        except BaseException as error:orphan_error=error
        self.assertIsNone(orphan_error,'rejected queued work should become a harmless no-op')
        self.assertEqual(c.store.head('child'),before_head)
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertFalse(self.lease_is_held(c.store,target))
        self.assertEqual(model.stats['forward_tokens'],before_work+2)

        receipt=c.retry_failed('enqueue-then-raise').result(5)
        self.assertEqual(receipt['snapshot'],target)
        self.assertEqual(c.store.head('child').tip,target)
        self.assertEqual(model.stats['forward_tokens'],before_work+2)

    def test_pre_submit_failure_preserves_identical_durable_alias_and_retries_branch(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        original_release=model.release_checkpoints
        calls=0
        def fail_release_once():
            nonlocal calls
            calls+=1
            if calls==1:raise OSError(5,'release failed while admitting durable alias')
            return original_release()
        model.release_checkpoints=fail_release_once
        self.addCleanup(setattr,model,'release_checkpoints',original_release)
        request=c.create('alias',(10,20),'alias-pre-submit-failure')
        self.assertEqual(request.snapshot.result(5)['snapshot'],root)
        with self.assertRaises(OSError) as error:request.durable.result(5)
        self.assertEqual(error.exception.errno,5)
        key=c._key(model,root)
        self.assertIn(key,c.pool)
        self.assertTrue(c.pool.is_durable(key))
        self.assertEqual(c.pool.metrics()['failed_entries'],0)
        self.assertIn(root,model.cache,'failure cleanup discarded another durable image owner')
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertTrue(c.operation_status('alias-pre-submit-failure')['publication_retryable'])
        model.release_checkpoints=original_release
        receipt=c.retry_failed('alias-pre-submit-failure').result(5)
        self.assertEqual(receipt['snapshot'],root)
        self.assertEqual(c.store.head('alias').tip,root)
        self.assertIn(key,c.pool)
        self.assertTrue(c.pool.is_durable(key))

    def test_pre_submit_retry_keeps_principal_bound_model(self):
        c,model=self.start(2)
        (root,)=self.create_roots(c,model,('source',))
        c.fork('child',root)
        c.store.grant(root,'alice',('read','append','move','checkpoint'))

        owner_save=model.save_prepared_branch
        bound=copy.copy(model)
        bound.store=c.store.for_principal('alice')
        def authorized_save(name,revision,prepared,operation,**kwargs):
            bound.store.authorize(prepared.snapshot,'move')
            return owner_save(name,revision,prepared,operation,**kwargs)
        bound.save_prepared_branch=authorized_save
        model.for_principal=lambda principal: bound
        self.addCleanup(delattr,model,'for_principal')
        def owner_save_is_wrong(*_args,**_kwargs):
            raise AssertionError('retry discarded the principal-bound model')
        model.save_prepared_branch=owner_save_is_wrong
        self.addCleanup(setattr,model,'save_prepared_branch',owner_save)

        original_submit=c._submit_publication
        c._submit_publication=lambda *_args: (_ for _ in ()).throw(
            OSError(28,'injected pre-submit failure'))
        self.addCleanup(setattr,c,'_submit_publication',original_submit)
        request=c.submit('child',(151,),1,'principal-retry',principal='alice')
        target=request.snapshot.result(5)['snapshot']
        with self.assertRaises(OSError):request.durable.result(5)
        c._submit_publication=original_submit

        receipt=c.retry_failed('principal-retry').result(5)
        self.assertEqual(receipt['snapshot'],target)
        self.assertEqual(c.store.head('child').tip,target)
        self.assertEqual(c.pool.metrics()['pins'],0)


if __name__=='__main__':unittest.main()
