"""CPU placement controls; native handoff is qualified independently."""
from dataclasses import replace
import time
import unittest

import test_pool_coordinator as fixture
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig


class ExecutionRoleTests(unittest.TestCase):
    def start(self,*,prefill=(1,),decode=(0,),gpus=(0,1),scheduling='fifo'):
        c,models,trace,gate=fixture.CoordinatorTests.start(self,gpus)
        directory,factory=c.store.root,c._factory
        config=replace(c.config,prefill_gpus=prefill,decode_gpus=decode,scheduling=scheduling)
        c.close();c=ContextCoordinator(directory,config,factory).ready(5)
        self.addCleanup(c.close)
        return c,models,trace,gate

    def test_root_prefill_and_generation_obey_explicit_roles(self):
        for policy in ('fifo','weighted-prefix','transfer-aware'):
            with self.subTest(policy=policy):
                c,models,trace,gate=self.start(scheduling=policy)
                root=c.create('base',(1,2,3),'role-root').durable.result(5)
                self.assertEqual(models[1].stats['forward_tokens'],3)
                self.assertEqual(models[0].stats['forward_tokens'],0)
                c.submit('base',(4,),2,'role-answer').durable.result(5)
                self.assertEqual([gpu for event,gpu,_ in trace if event=='start'],[0])
                self.assertTrue(c.capabilities()['features']['prefill_decode_placement']['enabled'])
                self.assertEqual(c.metrics()['workers'][1]['role'],'prefill')
                self.assertEqual(c.metrics()['workers'][0]['role'],'decode')

    def test_prefill_can_run_while_decode_worker_is_busy(self):
        c,models,trace,gate=self.start()
        c.create('base',(1,2),'role-initial').durable.result(5)
        gate.clear()
        try:
            answer=c.submit('base',(3,),2,'role-blocked-answer')
            deadline=time.monotonic()+3
            while not trace:
                if time.monotonic()>deadline:self.fail('decode did not start')
                time.sleep(.005)
            root=c.create('another',(5,6,7),'role-concurrent-prefill').durable.result(5)
            self.assertFalse(answer.durable.done())
            self.assertEqual(models[1].stats['forward_tokens'],5)
            self.assertEqual(c.store.head('another').tip,root['snapshot'])
        finally:gate.set()
        answer.durable.result(5)

    def test_one_gpu_can_explicitly_serve_both_roles(self):
        c,models,trace,gate=self.start(prefill=(0,),decode=(0,),gpus=(0,))
        c.create('base',(1,2),'single-root').durable.result(5)
        c.submit('base',(3,),1,'single-answer').durable.result(5)
        self.assertEqual(c.metrics()['workers'][0]['role'],'combined')

    def test_roles_cover_selected_workers_and_require_both_classes(self):
        for options in (dict(prefill_gpus=(0,)),dict(decode_gpus=(0,)),
                dict(prefill_gpus=(0,),decode_gpus=(0,)),
                dict(prefill_gpus=(0,),decode_gpus=(2,)),
                dict(prefill_gpus=(True,),decode_gpus=(1,)),
                dict(prefill_gpus=(0,0),decode_gpus=(1,))):
            with self.subTest(options=options),self.assertRaises(ValueError):
                PoolConfig(gpus=(0,1),**options)
        config=PoolConfig(gpus=(0,1))
        self.assertIsNone(config.prefill_gpus);self.assertIsNone(config.decode_gpus)


if __name__=='__main__':unittest.main()
