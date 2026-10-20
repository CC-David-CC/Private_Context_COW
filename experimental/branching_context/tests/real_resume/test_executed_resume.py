"""Independent audit of real, process-separated campaign artifacts.

Run only against an executed campaign via CTT_RESUME_RUN. Missing evidence fails.
These five cases do not cover RNG draws, all crash points or the complete G2 gate.
"""
import hashlib
import json
import os
from pathlib import Path
import unittest
import array
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from context_time_travel.numerics import assert_logits_close


class ExecutedResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CTT_RESUME_RUN'])
        cls.prepared = json.loads((cls.root/'prepare.json').read_text())
        cls.resumed = json.loads((cls.root/'resume.json').read_text())
        cls.profile = json.loads((cls.root/'profile.json').read_text())

    def test_dr_001_new_native_process_and_zero_prefix_work(self):
        a,b = self.prepared,self.resumed
        self.assertTrue(a['native_exit_observed'])
        self.assertTrue(b['native_exit_observed'])
        self.assertEqual(a['native_exit_code'],0)
        self.assertEqual(b['native_exit_code'],0)
        self.assertNotEqual(a['python_pid'],b['python_pid'])
        self.assertNotEqual(a['native_pid'],b['native_pid'])
        self.assertEqual(len(b['restore_receipts']),2)
        for receipt in b['restore_receipts']:
            self.assertEqual(receipt['forward_tokens'],0)
            self.assertEqual(receipt['consumed'],len(a['root']['history']))
            self.assertEqual(receipt['native_pid'],b['native_pid'])
            self.assertGreater(receipt['counter_deltas']['loaded_file_bytes'],0)
            self.assertGreater(receipt['counter_deltas']['cold_restore_copy_bytes'],0)
            self.assertGreater(receipt['counter_deltas']['cold_restore_copy_calls'],0)

    def test_dr_002_full_frontier_and_real_payload_bytes(self):
        sid = self.prepared['root']['snapshot']
        checkpoint = self.root/'store/checkpoints'
        manifest = json.loads((checkpoint/(sid+'.json')).read_text())['manifest']
        payload = (checkpoint/(manifest['payload_sha256']+'.state')).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),manifest['payload_sha256'])
        self.assertGreater(manifest['state_bytes'],0)
        self.assertEqual(len(payload),manifest['state_bytes']+4*manifest['logits_count'])
        frontier = payload[manifest['state_bytes']:]
        self.assertEqual(hashlib.sha256(frontier).hexdigest(),self.prepared['root']['frontier_sha256'])
        for name in ('A','B'):
            self.assertEqual((self.root/('restored-'+name+'-root.f32')).read_bytes(),frontier)
        self.assertEqual(manifest['model_contract_id'],self.profile['model_contract_id'])
        self.assertEqual(manifest['execution_profile_id'],self.profile['execution_profile_id'])
        self.assertEqual(manifest['positions'],'absolute-zero-based')

    def test_dr_003_independent_branches_and_fresh_oracle(self):
        fixture = json.loads((self.root/'fixture.json').read_text())
        self.assertNotEqual(fixture['A'],fixture['B'])
        for name in ('A','B'):
            a,b = self.prepared['branches'][name],self.resumed['branches'][name]
            self.assertEqual(a['generated'],b['generated'])
            for first,second in zip(a['states'],b['states']):
                self.assertEqual(first['history'],second['history'])
                self.assertEqual(first['controller'],second['controller'])
                self.assertEqual(first['frontier_sha256'],second['frontier_sha256'])
            final = b['states'][-1]['history']
            self.assertEqual(final,fixture['prefix']+fixture[name]+b['generated'])
        comparisons = json.loads((self.root/'compare.json').read_text())
        diagnosis = self.root/'frontier-diagnosis.json'
        if diagnosis.exists():
            detail = json.loads(diagnosis.read_text())
            for name,digest in detail['files'].items():
                self.assertEqual(hashlib.sha256((self.root/name).read_bytes()).hexdigest(),digest)
            actual = array.array('f'); actual.frombytes((self.root/'resumed-A-generated-1.f32').read_bytes())
            expected = array.array('f'); expected.frombytes((self.root/('compare-oracle-'+str(len(comparisons['comparisons']))+'.f32')).read_bytes())
            if sys.byteorder!='little': actual.byteswap(); expected.byteswap()
            assert_logits_close(tuple(actual),tuple(expected),atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.assertEqual(comparisons['status'],'PASS')
        self.assertEqual(len(comparisons['comparisons']),1+sum(len(row['states']) for row in self.resumed['branches'].values()))
        for row in comparisons['comparisons']:
            self.assertTrue(row['numerics']['pass_elementwise'])
            self.assertEqual(row['oracle_work']['work']['prefill_tokens'],row['consumed'])

    def test_dr_008_actual_batched_work_and_exact_suffix_work(self):
        before,after = self.prepared['prefill_counters_before'],self.prepared['prefill_counters_after']
        prefix = len(self.prepared['root']['history'])
        self.assertEqual(after['prefill_tokens']-before['prefill_tokens'],prefix)
        self.assertGreater(prefix,after['prefill_chunks']-before['prefill_chunks'])
        for result in (self.prepared,self.resumed):
            for name in ('A','B'):
                branch = result['branches'][name]
                self.assertEqual(branch['suffix_work'],branch['suffix_tokens'])
                self.assertGreater(branch['suffix_prefill_chunks'],0)

    def test_dr_009_generated_token_boundary_restart(self):
        boundary = json.loads((self.root/'boundary.json').read_text())
        self.assertEqual(boundary['status'],'PASS')
        self.assertTrue(boundary['native_exit_observed'])
        self.assertNotEqual(boundary['native_pid'],self.resumed['native_pid'])
        self.assertEqual(boundary['receipt']['forward_tokens'],0)
        final = self.resumed['branches']['A']['states'][-1]
        self.assertEqual(boundary['final']['history'],final['history'])
        self.assertEqual(boundary['final']['controller'],final['controller'])
        self.assertEqual(boundary['final']['frontier_sha256'],final['frontier_sha256'])


if __name__=='__main__': unittest.main(verbosity=2)
