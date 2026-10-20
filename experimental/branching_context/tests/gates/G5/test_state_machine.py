"""G5-002 exhaustive campaign; deferred until full certification is authorized."""
import datetime
import json
import os
from pathlib import Path
import sys
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from state_machine_campaign import run_campaign


class RandomizedStateMachineTests(unittest.TestCase):
    def test_g5_002_randomized_state_machine(self):
        profile=json.loads(Path(os.environ['CTT_PROFILE']).read_bytes())
        self.assertEqual(profile.get('calibration_status'),'FROZEN')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory=ROOT/'evidence/runs'/('G5-002-'+stamp+'-'+uuid.uuid4().hex)
        report=run_campaign(profile,directory,range(1000),12)
        self.assertEqual(report['executor'],'real')
        self.assertEqual(report['status'],'PASS',report.get('failure',report.get('error')))
        self.assertEqual(report['completed_programs'],1000)
        self.assertEqual(len(report['programs']),1000)
        self.assertEqual(report['seeds'],list(range(1000)))
        self.assertTrue(all(p['status']=='PASS' and len(p['program'])==12 for p in report['programs']))
        self.assertGreaterEqual(report['oracle_requests'],1000)
        self.assertLessEqual(report['max_absolute_error'],profile['atol'])
        self.assertTrue((directory/'campaign.json').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
