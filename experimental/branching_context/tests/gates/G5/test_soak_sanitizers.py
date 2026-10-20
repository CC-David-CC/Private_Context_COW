"""Registered long soak; deferred until exhaustive certification is authorized."""
import json
import shutil
import os
from pathlib import Path
import sys
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from lifecycle_soak import run_soak,qualifies_long_soak,LONG_CYCLES,LONG_SECONDS

class SoakTests(unittest.TestCase):
    def test_g5_005_sanitizer_diagnostics(self):
        from sanitizer_suite import run_suite,REQUIRED
        profile_path=Path(os.environ['CTT_PROFILE'])
        profile=json.loads(profile_path.read_bytes())
        native_root=Path(profile['execution']['sut_command'][0]).resolve().parent.parent
        tool=shutil.which('compute-sanitizer') or '/usr/local/cuda/bin/compute-sanitizer'
        directory=ROOT/'evidence/runs'/('G5-005-'+uuid.uuid4().hex)
        report=run_suite(profile_path,native_root,directory,tool)
        self.assertEqual(report['status'],'PASS',report)
        self.assertEqual(report['execution_profile_id'],profile['execution_profile_id'])
        for name in REQUIRED:
            self.assertEqual(report['tools'][name]['status'],'PASS',name)
            self.assertTrue(report['tools'][name]['instrumented'],name)
        self.assertEqual(report['cuda_target']['oracle_requests'],3)
        self.assertTrue(all(r['max_absolute_error']<=profile['atol'] for r in report['cuda_target']['checks']))
        self.assertTrue((directory/'receipt.json').is_file())

    def test_g5_004_long_soak(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_bytes())
        directory=ROOT/'evidence/runs'/('G5-004-'+uuid.uuid4().hex)
        report=run_soak(profile,directory,minimum_cycles=LONG_CYCLES,minimum_seconds=LONG_SECONDS)
        self.assertEqual(report['status'],'MEASURED')
        self.assertTrue(qualifies_long_soak(report))
        self.assertTrue(report['long_soak_qualified'])
        self.assertGreaterEqual(report['cycles'],64)
        self.assertGreaterEqual(report['seconds'],86400)
        self.assertGreaterEqual(report['numerical_comparisons'],report['cycles']*4)
        self.assertLessEqual(report['max_absolute_error'],profile['atol'])
        self.assertEqual(report['hardware'],profile['hardware'])
        self.assertEqual(report['execution_profile_id'],profile['execution_profile_id'])
        for field in ('latency_histogram','rss_histogram','gpu_histogram'):
            self.assertEqual(sum(b['count'] for b in report['summary'][field]),report['cycles'])
        self.assertTrue((directory/'report.json').is_file())

if __name__=='__main__':unittest.main()
