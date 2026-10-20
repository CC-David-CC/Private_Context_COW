"""Synthetic scheduler tests; no GPU performance/numerical claims."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import Checkpoint, PreparedCheckpoint
from context_time_travel.store import Store, Conflict


class Model:
    def __init__(self,directory,gpu,ids,trace,gate):
        self.store = Store(directory);self.cache = {};self.gpu = gpu
        self.profile = dict(model_contract_id='m',execution_profile_id='e',execution={'GPU':list(ids)})
        self.trace,self.gate = trace,gate
        self.checkpoint_directory = self.store.root/'checkpoints';self.checkpoint_directory.mkdir(exist_ok=True)
        self.backend = self
        self.stats = {'consumed_tokens':0,'forward_tokens':0,'loaded_file_bytes':0}
    def work_counters(self): return dict(self.stats)
    def release_checkpoints(self): return 0
    def close(self): pass
    def create(self,ids):
        self.stats['forward_tokens'] += len(ids)
        self.stats['consumed_tokens'] += len(ids)
        sid = self.store.create_root('m',ids,{'status':'idle'});self.image(sid);return sid
    def image(self,sid):
        blob = b'state:'+sid.encode()
        self.cache[sid] = Checkpoint('m','e',len(self.store.lineage(sid)[1]),blob,(1.,),
            Checkpoint.digest(blob,(1.,)),sid)
    def run_greedy_request(self,sid,ids,count,*,stream,on_token):
        self.trace.append(('start',self.gpu,stream));self.gate.wait(5)
        self.stats['forward_tokens'] += len(ids)+count
        for token in range(count): on_token(token)
        target = self.store.append(sid,tuple(ids)+tuple(range(count)),{'status':'idle'})
        self.image(target);self.trace.append(('end',self.gpu,stream))
        return dict(snapshot=target,tokens=list(range(count)),request_seconds=.001,generation_seconds=.001)
    def prepare_checkpoint(self,sid):
        cp = self.cache[sid]
        env = dict(manifest=dict(snapshot=sid,state_bytes=len(cp.blob),logits_count=1,payload_sha256=cp.checksum))
        return PreparedCheckpoint(sid,cp,json.dumps(env).encode())
    def _load_checkpoint(self,sid,history):
        envelope = json.loads((self.checkpoint_directory/(sid+'.json')).read_bytes())
        sha = envelope['manifest']['payload_sha256']
        blob = (self.checkpoint_directory/(sha+'.state')).read_bytes()[:-4]
        return Checkpoint('m','e',len(history),blob,(1.,),sha,sid)
    def history(self,sid): return tuple(self.store.lineage(sid)[1])
    def checkpoint_manifest(self,sid):
        return json.loads((self.checkpoint_directory/(sid+'.json')).read_bytes())['manifest']
    def validate_checkpoint(self,sid):
        self._load_checkpoint(sid,self.history(sid));return self.checkpoint_manifest(sid)
    def save_prepared_branch(self,name,revision,prepared,operation):
        blob = prepared.checkpoint.blob+b'\0\0\x80?'
        (self.checkpoint_directory/(prepared.checkpoint.checksum+'.state')).write_bytes(blob)
        (self.checkpoint_directory/(prepared.snapshot+'.json')).write_bytes(prepared.envelope)
        head = self.store.move(name,revision,prepared.snapshot,operation)
        return dict(snapshot=prepared.snapshot,branch=asdict(head),payload_sha256=prepared.checkpoint.checksum,
            model_contract_id='m',execution_profile_id='e')


class CoordinatorTests(unittest.TestCase):
    def start(self,ids=(0,)):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        trace,models = [],{}
        gate = threading.Event();gate.set()
        config = PoolConfig(gpus=ids,ram_bytes=4096,disk_bytes=16384,min_available_bytes=0)
        def factory(gpu):
            models[gpu] = Model(temp.name,gpu,ids,trace,gate);return models[gpu]
        coordinator = ContextCoordinator(temp.name,config,factory)
        self.addCleanup(coordinator.close)
        coordinator.ready(timeout=5)
        return coordinator,models,trace,gate

    def test_only_selected_gpus_created_and_each_has_one_request(self):
        c,models,trace,gate = self.start((1,5,6))
        root = c.create('base',(1,2),'create').durable.result(5)['snapshot']
        for n in range(6): c.fork('branch-'+str(n),root)
        gate.clear()
        jobs = [c.submit('branch-'+str(n),(3,),2,'job-'+str(n)) for n in range(6)]
        deadline = time.monotonic()+3
        while len([r for r in trace if r[0]=='start']) < 3 and time.monotonic()<deadline: time.sleep(.01)
        self.assertEqual({gpu for kind,gpu,_ in trace if kind=='start'},{1,5,6})
        self.assertEqual(len([r for r in trace if r[0]=='start']),3)
        gate.set()
        for job in jobs: job.durable.result(5)
        self.assertEqual(set(models),{1,5,6})
        self.assertEqual(c.metrics()['max_active_per_gpu'],{1:1,5:1,6:1})

    def test_snapshot_precedes_durable_and_writer_does_not_take_gpu_slot(self):
        c,models,trace,gate = self.start()
        root = c.create('base',(1,2),'create').durable.result(5)['snapshot']
        c.fork('a',root);c.fork('b',root)
        entered,release = threading.Event(),threading.Event()
        original = models[0].save_prepared_branch
        def slow(*args): entered.set();release.wait(5);return original(*args)
        models[0].save_prepared_branch = slow
        a = c.submit('a',(3,),1,'a-op')
        a.snapshot.result(5);self.assertTrue(entered.wait(2));self.assertFalse(a.durable.done())
        b = c.submit('b',(4,),1,'b-op')
        b.snapshot.result(5)
        self.assertEqual(c.store.head('a').tip,root)
        release.set();a.durable.result(5);b.durable.result(5)

    def test_same_branch_blocked_and_stale_revision_is_not_rebased(self):
        c,models,trace,gate = self.start()
        root = c.create('base',(1,),'create').durable.result(5)['snapshot']
        c.fork('a',root);gate.clear()
        a = c.submit('a',(2,),1,'a-op')
        stale = c.submit('a',(3,),1,'stale-op')
        gate.set();a.durable.result(5)
        with self.assertRaises(Conflict): stale.durable.result(5)
        self.assertNotIn(('start',0,'stale-op'),trace)

    def test_duplicate_operation_returns_same_handle_and_altered_request_fails(self):
        c,_,_,_ = self.start()
        c.create('base',(1,),'create').durable.result(5)
        one = c.submit('base',(2,),1,'op')
        self.assertIs(c.submit('base',(2,),1,'op'),one)
        with self.assertRaises(ValueError): c.submit('base',(3,),1,'op')
        one.durable.result(5)

    def test_restart_retry_uses_canonical_receipt_without_executing_again(self):
        c,models,trace,_ = self.start()
        c.create('base',(1,),'create').durable.result(5)
        original = c.submit('base',(2,),1,'op',expected_revision=1).durable.result(5)
        directory,factory,config = c.store.root,c._factory,c.config
        c.close()
        resumed = ContextCoordinator(directory,config,factory);self.addCleanup(resumed.close)
        resumed.ready(5)
        count = len(trace)
        retry = resumed.submit('base',(2,),1,'op',expected_revision=1)
        self.assertEqual(retry.durable.result(5),original)
        self.assertEqual(len(trace),count)

    def test_failed_publication_pins_bytes_and_retry_needs_no_gpu(self):
        c,models,trace,_ = self.start()
        root = c.create('base',(1,),'create').durable.result(5)['snapshot']
        old = c.store.head('base')
        original = models[0].save_prepared_branch
        def fail(*args): raise OSError('disk full')
        models[0].save_prepared_branch = fail
        request = c.submit('base',(2,),1,'op')
        request.snapshot.result(5)
        with self.assertRaises(OSError): request.durable.result(5)
        self.assertEqual(c.store.head('base'),old)
        self.assertGreater(c.metrics()['ram']['dirty_bytes'],0)
        count = len(trace)
        models[0].save_prepared_branch = original
        c.retry_failed('op').result(5)
        self.assertEqual(len(trace),count)
        self.assertEqual(c.metrics()['failed_publications'],0)

    def test_pending_cancel_does_not_steal_active_owner(self):
        c,_,trace,gate = self.start()
        root = c.create('base',(1,),'create').durable.result(5)['snapshot']
        c.fork('a',root);c.fork('b',root);gate.clear()
        active = c.submit('a',(2,),1,'active')
        deadline = time.monotonic()+3
        while not trace and time.monotonic()<deadline: time.sleep(.01)
        pending = c.submit('b',(3,),1,'pending')
        self.assertFalse(c.cancel(active))
        self.assertTrue(c.cancel(pending));self.assertTrue(pending.durable.cancelled())
        gate.set();active.durable.result(5)

    def test_queued_source_is_leased_until_cancel_or_completion(self):
        c,_,trace,gate = self.start()
        root = c.create('base',(1,),'create').durable.result(5)['snapshot']
        c.fork('a',root);c.fork('b',root)
        first=c.submit('b',(2,),1,'first');first.durable.result(5)
        old=c.store.head('b').tip
        gate.clear();active=c.submit('a',(2,),1,'active')
        deadline=time.monotonic()+3
        while len(trace)<3 and time.monotonic()<deadline: time.sleep(.01)
        pending=c.submit('b',(3,),1,'pending')
        # Moving the reference behind the queue must not let GC delete its source.
        c.store.move('b',c.store.head('b').revision,root,'external-move')
        self.assertIn(old,c.store.checkpoint_roots())
        self.assertTrue(c.cancel(pending))
        self.assertNotIn(old,c.store.checkpoint_roots())
        gate.set();active.durable.result(5)

    def test_mixed_groups_use_one_scheduler_slot_each(self):
        c,models,_,_ = self.start(((0,1,2),5,6))
        self.assertEqual(set(models),{(0,1,2),5,6})
        c.create('base',(1,),'create').durable.result(5)
        self.assertEqual(c.metrics()['physical_gpu_count'],5)
        self.assertEqual(c.metrics()['worker_count'],3)

    def test_upload_staging_reserved_before_native_use_and_released_after_first_token(self):
        c,models,_,_ = self.start()
        c.create('base',(1,),'create').durable.result(5)
        models[0].cache.clear()
        original=models[0].run_greedy_request
        before,after=[],[]
        def observed(*args,**kwargs):
            cp=models[0].cache[args[0]]
            before.append((c.pool.metrics()['reserved_bytes'],3*(len(cp.blob)+4*len(cp.frontier))))
            callback=kwargs['on_token']
            def token(t):
                callback(t);after.append(c.pool.metrics()['reserved_bytes'])
            kwargs['on_token']=token
            return original(*args,**kwargs)
        models[0].run_greedy_request=observed
        c.submit('base',(2,),2,'op').durable.result(5)
        self.assertGreaterEqual(before[0][0],before[0][1])
        self.assertEqual(after,[0,0])


if __name__ == '__main__': unittest.main()
