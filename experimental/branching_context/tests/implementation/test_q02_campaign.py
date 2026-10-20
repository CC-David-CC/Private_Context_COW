"""Campaign bookkeeping tests; no inference or gate qualification."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import state_machine as actions

class CampaignTests(unittest.TestCase):
    def load(self):
        spec=importlib.util.spec_from_file_location('campaign',ROOT/'scripts/state_machine_campaign.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def profile(self):
        return dict(calibration_status='FROZEN',fixtures=dict(suffix_a=[71],suffix_b=[81]))

    def result(self,program):
        return dict(status='PASS',completed_operations=len(program),oracle_requests=1,
                    boundaries=[dict(max_absolute_error=0)],max_absolute_error=0,executed_operations=[r['op'] for r in program])

    def test_complete_and_incomplete_reports(self):
        module=self.load()
        with tempfile.TemporaryDirectory() as tmp:
            report=module.run_campaign(self.profile(),Path(tmp)/'good',[17,18],6,_execute=lambda p,d,ops,seed:self.result(ops))
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['completed_programs'],2)
            self.assertEqual(report['seeds'],[17,18])
            self.assertEqual(report['attempts'],2)
            with self.assertRaises(FileExistsError):module.run_campaign(self.profile(),Path(tmp)/'good',[17],6)
            def incomplete(p,d,ops,seed):return dict(self.result(ops),completed_operations=0)
            with self.assertRaises(RuntimeError):module.run_campaign(self.profile(),Path(tmp)/'bad',[17],6,_execute=incomplete)

    def test_replays_and_shrinks_same_failure_preserving_dependencies(self):
        module=self.load()
        seed=next(s for s in range(1000) if {'append','generate'}<=set(r['op'] for r in actions.make_program(s,12,[71,81])))
        seen=[]
        def execute(p,d,ops,seed):
            known={'root'}
            for r in ops:
                self.assertIn(r['source'],known);known.add(r['out'])
            seen.append(list(ops))
            if {'append','generate'}<=set(r['op'] for r in ops):
                return dict(status='FAIL',failure=dict(type='ProgramFailure',signature='next-logits'))
            return self.result(ops)
        with tempfile.TemporaryDirectory() as tmp:
            report=module.run_campaign(self.profile(),Path(tmp)/'failure',[seed],12,_execute=execute)
            self.assertEqual(report['status'],'FAIL')
            self.assertTrue(report['failure']['reproduced'])
            self.assertEqual(report['failure']['signature'],['ProgramFailure','next-logits'])
            reduced=report['failure']['minimal_program']
            self.assertLess(len(reduced),12)
            self.assertTrue({'append','generate'}<=set(r['op'] for r in reduced))
            self.assertEqual(seen[0],seen[1]);self.assertEqual(seen[1],seen[2])
            self.assertEqual(seen[-1],reduced);self.assertEqual(seen[-2],reduced)

    def test_unstable_failure_is_retained_without_shrinking(self):
        module=self.load();calls=[]
        def execute(p,d,ops,seed):
            calls.append(ops)
            if len(calls)==1:return dict(status='FAIL',failure=dict(type='ProgramFailure',signature='next-logits'))
            return self.result(ops)
        with tempfile.TemporaryDirectory() as tmp:
            report=module.run_campaign(self.profile(),Path(tmp)/'unstable',[17],6,_execute=execute)
            self.assertEqual(report['status'],'FAIL')
            self.assertFalse(report['failure']['reproduced'])
            self.assertNotIn('minimal_program',report['failure'])
            self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
