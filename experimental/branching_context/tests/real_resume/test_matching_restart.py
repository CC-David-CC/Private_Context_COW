"""Audit executed matching-schedule GPU evidence; missing results fail, never skip."""
import hashlib
import json
import os
from pathlib import Path
import unittest


class MatchingRestartEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CTT_MATCHING_RESUME_RESULTS'])
        cls.read = staticmethod(lambda name: json.loads((cls.root/name).read_text()))
        cls.result = cls.read('result.json')
        cls.prepared,cls.resumed = cls.read('prepare.json'),cls.read('resume.json')
        cls.compared,cls.boundary = cls.read('compare.json'),cls.read('boundary.json')

    def test_mr_001_actual_process_separation(self):
        self.assertEqual(self.result['status'],'PASS')
        records = [self.prepared,self.resumed,self.boundary]
        self.assertEqual(len({r['python_pid'] for r in records}),3)
        self.assertEqual(len({self.prepared['native_pid'],self.resumed['native_pid'],
                              self.boundary['receipt']['native_pid']}),3)
        self.assertTrue(all(r['native_exit_observed'] for r in records))
        if self.result['prefix'] >= 8192:
            repeats = [self.read('restore-repeat-'+str(n)+'.json') for n in (2,3)]
            self.assertEqual(len({self.resumed['native_pid'],*[r['native_pid'] for r in repeats]}),3)
            for repeat in repeats:
                self.assertEqual(repeat['status'],'PASS')
                self.assertTrue(repeat['native_exit_observed'])
                self.assertEqual(repeat['receipt']['forward_tokens'],0)
                self.assertEqual(repeat['frontier_sha256'],self.prepared['root']['frontier_sha256'])

    def test_mr_002_disk_restore_zero_forward_work(self):
        receipts = self.resumed['restore_receipts']+[self.boundary['receipt']]
        for receipt in receipts:
            self.assertEqual(receipt['forward_tokens'],0)
            self.assertGreater(receipt['counter_deltas']['loaded_file_bytes'],0)
            self.assertGreater(receipt['counter_deltas']['cold_restore_copy_bytes'],0)
            self.assertEqual(len(receipt['schedule_sha256']),64)

    def test_mr_003_exact_independent_matching_reference(self):
        self.assertEqual(self.compared['status'],'PASS')
        self.assertEqual(self.compared['contract'],'identical-native-batches-v1')
        self.assertTrue(self.compared['comparisons'])
        for item in self.compared['comparisons']:
            self.assertEqual(item['numerics']['max_absolute_error'],0)
            self.assertTrue(item['numerics']['pass_elementwise'])
            self.assertEqual(item['oracle_work']['work']['continuation_operator'],'batched-prefill')
        self.assertEqual(self.read('calibration.json')['atol'],0)
        self.assertEqual(self.read('calibration.json')['rtol'],0)

    def test_mr_004_both_branches_and_consumed_boundary(self):
        fixture = self.read('fixture.json')
        for name in ('A','B'):
            a,b = self.prepared['branches'][name],self.resumed['branches'][name]
            self.assertEqual(a['generated'],b['generated'])
            self.assertEqual([s['controller'] for s in a['states']],[s['controller'] for s in b['states']])
            self.assertEqual([s['frontier_sha256'] for s in a['states']],[s['frontier_sha256'] for s in b['states']])
            self.assertEqual(b['states'][0]['history'],fixture['prefix']+fixture[name])
            self.assertEqual(b['suffix_work'],len(fixture[name]))
            before,after = b['generated_work_before'],b['generated_work_after']
            self.assertEqual(after['prefill_tokens']-before['prefill_tokens'],len(b['generated']))
            self.assertEqual(after['decode_timed_tokens']-before['decode_timed_tokens'],0)
        final = self.resumed['branches']['A']['states'][-1]
        self.assertEqual(self.boundary['final']['frontier_sha256'],final['frontier_sha256'])
        self.assertEqual(self.boundary['final']['controller'],final['controller'])
        sid = self.prepared['root']['snapshot']
        manifest = self.root/'store/checkpoints'/(sid+'.json')
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),fixture['manifest_sha256'])
        envelope = json.loads(manifest.read_text())
        self.assertEqual(envelope['manifest']['schema'],2)
        payload = self.root/'store/checkpoints'/(envelope['manifest']['payload_sha256']+'.state')
        self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(),envelope['manifest']['payload_sha256'])


if __name__ == '__main__': unittest.main()
