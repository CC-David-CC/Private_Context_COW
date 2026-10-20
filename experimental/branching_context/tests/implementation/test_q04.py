"""Mutation qualification policy; no GPU or model is simulated here."""
import importlib.util
from pathlib import Path
import unittest

class CampaignTests(unittest.TestCase):
    def test_q04_invariant(self):
        path=Path(__file__).resolve().parents[2]/'scripts/mutation_campaign.py'
        self.assertTrue(path.is_file(),'Missing strict twelve-mutant campaign runner')
        spec=importlib.util.spec_from_file_location('mutations',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(len(module.REQUIRED),12)
        self.assertIn('M03',module.IMPLEMENTED,'Chunked draw-state mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M03'))
        self.assertIn('M04',module.IMPLEMENTED,'Same-count prefix mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M04'))
        self.assertIn('M06',module.IMPLEMENTED,'ABA stale-revision mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M06'))
        self.assertIn('M07',module.IMPLEMENTED,'Double-retry publication mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M07'))
        self.assertIn('M08',module.IMPLEMENTED,'Corrupt-checkpoint mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M08'))
        self.assertIn('M09',module.IMPLEMENTED,'History-deleting eviction mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M09'))
        self.assertIn('M10',module.IMPLEMENTED,'Evaluator-mutation mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M10'))
        self.assertIn('M11',module.IMPLEMENTED,'Incompatible-state-reuse mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M11'))
        self.assertIn('M12',module.IMPLEMENTED,'Half-consumed error-publication mutant missing')
        self.assertTrue(hasattr(module.RealProbe,'test_M12'))
        control=dict(tests=1,failures=[],errors=[],skips=[],xfail=0,oracle_requests=1)
        mutant=dict(control,failures=['next-logits'],mutation_invocations=1)
        self.assertTrue(module.detected(control,mutant,('next-logits',)))
        for bad in [dict(mutant,failures=[],errors=['NameError']),dict(mutant,skips=['no GPU']),dict(mutant,xfail=1),dict(mutant,mutation_invocations=0),dict(mutant,failures=['unrelated assertion']),dict(mutant,oracle_requests=0),dict(mutant,errors=None),dict(mutant,tests=True),dict(mutant,mutation_invocations=True)]:
            self.assertFalse(module.detected(control,bad,('next-logits',)))
        missing=dict(mutant);del missing['skips']
        self.assertFalse(module.detected(control,missing,('next-logits',)))
        self.assertFalse(module.detected(dict(control,failures=['baseline broken']),mutant,('next-logits',)))
        self.assertFalse(module.campaign_complete({'M01':{'detected':True}}))
        all_rows={name:{'detected':True} for name in module.REQUIRED}
        self.assertTrue(module.campaign_complete(all_rows))
        all_rows['M05']['detected']=False
        self.assertFalse(module.campaign_complete(all_rows))

if __name__=='__main__':unittest.main()
