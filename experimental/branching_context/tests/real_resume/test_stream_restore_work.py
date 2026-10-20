"""Real GPU stream regressions against an executed independent batch campaign.

Requires CTT_MATCHING_RESUME_RESULTS; absent evidence is an error, never a skip.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import struct
import unittest

from context_time_travel.real import BranchingModel
from context_time_travel.store import InvalidOperation


class StreamRestoreWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = Path(os.environ['CTT_MATCHING_RESUME_RESULTS'])
        read = lambda n: json.loads((cls.campaign/n).read_text())
        if read('result.json')['status'] != 'PASS' or read('compare.json')['status'] != 'PASS':
            raise AssertionError('Require independently validated real-GPU evidence')
        cls.profile,cls.fixture,cls.reference = [read(n+'.json') for n in ('profile','fixture','resume')]
        if os.environ.get('CTT_PROFILE'):
            actual = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
            for key in ('execution','model_contract_id','hardware'):
                if actual[key] != cls.profile[key]:
                    raise AssertionError('Retained reference does not match tested native math configuration')
            cls.profile = actual
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(cls.profile['execution']['environment'])
        cls.backend = BranchingModel.load_backend(cls.profile)
        cls.addClassCleanup(cls.backend.close)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = BranchingModel(self.temp.name,self.backend,self.profile)
        self.root = self.model.create(self.fixture['prefix'])
        self.original = self.model.snapshot_bytes(self.root)

    def branch(self,name):
        tip = self.model.feed(self.root,self.fixture[name])
        return self.model.start_generation(tip,{'temperature':0},'durable-greedy')

    def assert_frontier(self,tip,name,count):
        consumed = len(self.fixture['prefix'])+len(self.fixture[name])+count
        states = [s for s in self.reference['branches'][name]['states'] if s['consumed'] == consumed]
        self.assertEqual(len(states),1,'Require a reference at this exact consumed boundary')
        values = self.model.next_logits(tip)
        digest = hashlib.sha256(struct.pack('<'+'f'*len(values),*values)).hexdigest()
        self.assertEqual(digest,states[0]['frontier_sha256'])
        self.assertEqual(self.model.controller(tip),states[0]['controller'])

    def test_sr_001_one_restore_at_most_for_one_consumed_output(self):
        tip = self.branch('A')
        before = self.backend.work_counters()
        [(tip,token)] = list(self.model.stream(tip,1))
        after = self.backend.work_counters()
        self.assertEqual(token,self.reference['branches']['A']['generated'][0])
        self.assertEqual(after['consumed_tokens']-before['consumed_tokens'],1)
        self.assertEqual(after['prefill_tokens']-before['prefill_tokens'],1)
        self.assertEqual(after['decode_timed_tokens']-before['decode_timed_tokens'],0)
        self.assert_frontier(tip,'A',1)
        self.assertLessEqual(after['restores']-before['restores'],1,
            'Selection and consumption must share one materialization')

    def test_sr_002_yield_releases_lease_and_interleaving_preserves_branches(self):
        # This restricted profile requires retained numerical checkpoints;
        # its bounded hot cache may evict a tip while another branch runs.
        a = self.branch('A'); self.model.checkpoint(a)
        b = self.branch('B'); self.model.checkpoint(b)
        iterator = self.model.stream(a,2)
        first,token = next(iterator)
        self.model.checkpoint(first)
        self.assertEqual(token,self.reference['branches']['A']['generated'][0])
        self.assertIsNone(self.backend._active)
        other = list(self.model.stream(b,2))
        self.model.checkpoint(other[-1][0])
        second,token = next(iterator)
        iterator.close()
        self.assertEqual(token,self.reference['branches']['A']['generated'][1])
        self.assertEqual([t for _,t in other],self.reference['branches']['B']['generated'][:2])
        self.assert_frontier(second,'A',2)
        self.assert_frontier(other[-1][0],'B',2)
        self.assertEqual(self.model.history(second),tuple(self.fixture['prefix']+self.fixture['A']+
            self.reference['branches']['A']['generated'][:2]))
        self.assertEqual(self.model.snapshot_bytes(self.root),self.original)
        self.assertIsNone(self.backend._active)

    def test_sr_003_consumption_failure_does_not_publish_and_parent_recovers(self):
        tip = self.branch('A')
        before = set(self.model.store.objects.iterdir())
        def fail(point):
            if point == 'after_token_consume': raise RuntimeError('injected consume failure')
        self.model._lease_fault = fail
        with self.assertRaisesRegex(RuntimeError,'injected consume failure'):
            list(self.model.stream(tip,1))
        self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertIsNone(self.backend._active)
        self.model._lease_fault = lambda point: None
        [(new,token)] = list(self.model.stream(tip,1))
        self.assertEqual(token,self.reference['branches']['A']['generated'][0])
        self.assert_frontier(new,'A',1)
        self.assertEqual(self.model.snapshot_bytes(self.root),self.original)

    def test_sr_004_foreign_and_closed_lease_fail_before_mutation(self):
        with self.model.materialize(self.root) as (lease,_):
            before = self.backend.work_counters()['consumed_tokens']
            with ThreadPoolExecutor(max_workers=1) as worker:
                for action in (lambda:self.backend.logits(lease),
                               lambda:self.backend.consume(lease,(1,)),
                               lambda:lease.__exit__(None,None,None)):
                    with self.assertRaises(InvalidOperation): worker.submit(action).result(timeout=10)
            self.assertEqual(self.backend.work_counters()['consumed_tokens'],before)
        with self.assertRaises(InvalidOperation): self.backend.logits(lease)


if __name__ == '__main__': unittest.main()
