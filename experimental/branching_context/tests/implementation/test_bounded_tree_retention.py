"""CPU-only cache policy controls; synthetic images are not GPU evidence."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_pool_coordinator import Model
from context_time_travel.checkpoint_cache import CheckpointCache
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig,PoolTransitionTrace
from context_time_travel.real import Checkpoint,PreparedCheckpoint
from context_time_travel.strata_backend import StrataImage


class NativeImageModel(Model):
    """Exercise actual coordinator routes with explicitly synthetic handles."""
    def __init__(self,*args,entries=6,**kwargs):
        super().__init__(*args,**kwargs)
        self.cache=CheckpointCache(4096,entries)
        self.sequence=0

    def image(self,sid):
        self.sequence+=1
        count=len(self.history(sid))
        image=StrataImage(b'synthetic-'+str(self.sequence).encode(),b'synthetic-process',
                          self.sequence,count,64,4,((self.sequence,0,0,200,'k',0),))
        self.cache[sid]=Checkpoint.create('m','e',count,image,(1.,),sid)

    def prepare_checkpoint(self,sid):
        # Match the production ownership boundary: writer receives immutable
        # serialized bytes, never the synthetic native image itself.
        cp=self.cache[sid]
        blob=b'state:'+sid.encode()
        cold=Checkpoint.create('m','e',cp.consumed,blob,cp.frontier,sid)
        env=dict(manifest=dict(snapshot=sid,state_bytes=len(blob),logits_count=1,
                               payload_sha256=cold.checksum))
        return PreparedCheckpoint(sid,cold,json.dumps(env).encode())


class BoundedTreeRetentionTests(unittest.TestCase):
    def start(self,entries=6):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        gate=threading.Event();gate.set()
        models={};trace=PoolTransitionTrace('cpu-retention',process_id=701,phase_id='trees')
        def factory(gpu):
            models[gpu]=NativeImageModel(directory.name,gpu,(0,),[],gate,entries=entries)
            return models[gpu]
        c=ContextCoordinator(directory.name,
            PoolConfig(gpus=(0,),ram_bytes=4096,disk_bytes=16384,min_available_bytes=0),
            factory,transition_trace=trace)
        self.addCleanup(c.close);c.ready(5)
        return c,models[0],trace

    def idle(self,c):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if not any(c.metrics()['active_per_gpu'].values()):return
            time.sleep(.005)
        self.fail('worker did not finish cache cleanup')

    def roots(self,c,model):
        result=[]
        for index in range(3):
            name='tree'+str(index)
            sid=c.create(name,(10+index,20+index),'create-'+name).durable.result(5)['snapshot']
            self.idle(c);result.append(sid)
        # Establish a known cache independently of create-time cleanup. No
        # inference claim: these are explicitly synthetic process-local images.
        model.cache.clear()
        baseline=model.cache.entry_evictions
        for sid in result:model.image(sid)
        return result,baseline

    def test_hot_roots_survive_tree_switch_and_post_request_pruning_within_budget(self):
        c,model,trace=self.start()
        roots,_=self.roots(c,model)
        images={sid:model.cache[sid].blob for sid in roots}
        for index in range(3):c.fork('ask'+str(index),roots[index])
        for index in range(3):
            c.submit('ask'+str(index),(100+index,),2,'ask-'+str(index)).durable.result(5)
            self.idle(c)
            for sid,image in images.items():
                self.assertIn(sid,model.cache,'a bounded resident root was discarded despite spare budget')
                self.assertIs(model.cache[sid].blob,image)
            self.assertLessEqual(len(model.cache),model.cache.entry_limit)
            self.assertLessEqual(model.cache._referenced_bytes(),model.cache.byte_limit)
        acquired=[row for row in trace.snapshot()['events'] if row['event']=='request_source_acquired']
        self.assertEqual([row['source_kind'] for row in acquired],['model_cache']*3)
        for index,sid in enumerate(roots):
            self.assertEqual(c.store.head('tree'+str(index)).tip,sid)
            self.assertEqual(model.history(sid),(10+index,20+index))
        self.assertEqual(c.metrics()['max_active_per_gpu'],{0:1})

    def test_small_budget_evicts_lru_and_evicted_source_uses_pinned_pool(self):
        c,model,trace=self.start(entries=2)
        roots,before=self.roots(c,model)
        self.assertNotIn(roots[0],model.cache)
        self.assertEqual(model.cache.entry_evictions,before+1)
        c.fork('cold-question',roots[0])
        c.submit('cold-question',(100,),2,'cold-question').durable.result(5)
        self.idle(c)
        acquired=[row for row in trace.snapshot()['events'] if row['event']=='request_source_acquired']
        self.assertEqual(acquired[-1]['source_kind'],'shared_pool')
        self.assertLessEqual(len(model.cache),2)
        self.assertLessEqual(model.cache._referenced_bytes(),4096)
        self.assertNotIn(roots[0],model.cache,'temporary serialized alias outlived its pool pin')
        self.assertEqual(c.pool.metrics()['pins'],0)
        self.assertEqual(c.pool.metrics()['reserved_bytes'],0)
        for index,sid in enumerate(roots):
            self.assertEqual(c.store.head('tree'+str(index)).tip,sid)
            self.assertEqual(model.history(sid),(10+index,20+index))

    def test_cache_still_enforces_byte_budget_and_preserves_existing_images_on_oversize(self):
        c,model,_=self.start()
        roots,_=self.roots(c,model)
        before={sid:model.cache[sid] for sid in roots}
        oversized=replace(before[roots[0]],blob=replace(before[roots[0]].blob,running_bytes=4097))
        with self.assertRaises(MemoryError):model.cache['oversized']=oversized
        self.assertEqual(set(model.cache),set(roots))
        for sid,value in before.items():self.assertIs(model.cache[sid],value)


if __name__=='__main__':unittest.main()
