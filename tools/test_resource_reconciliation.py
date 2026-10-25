from pathlib import Path
import json,tempfile,unittest
from tools.pcc_training_workspace import ResourceBudget,sha,write

class ReconciliationTests(unittest.TestCase):
    def test_audit_preserves_original_charge_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            write(root/'resource-budget.json',dict(ceiling_gpu_seconds=100,prior_upper_bound_gpu_seconds=0,charged_gpu_seconds=0,active={},settled={}))
            b=ResourceBudget(root)
            kw=dict(maximum_gpu_seconds=40,ram_bytes=0,free_ram_bytes=100,reserve_ram_bytes=0,owner={'pid':123})
            b.reserve('op',**kw)
            evidence=root/'audit.json';write(evidence,{'original_process_retired':True,'allocation':0});evidence.chmod(0o600)
            args=dict(allocated_gpu_seconds=0,evidence=evidence,evidence_sha256=sha(evidence))
            with self.assertRaises(ValueError):b.reconcile('op',**args)
            b.settle('op')
            original=json.loads((root/'resource-budget.json').read_text())['settled']['op']
            receipt=b.reconcile('op',**args)
            self.assertEqual(b.reconcile('op',**args),receipt)
            result=json.loads((root/'resource-budget.json').read_text())
            self.assertEqual(result['charged_gpu_seconds'],0)
            self.assertEqual(result['settled']['op'],original)
            self.assertEqual(len(result['reconciliations']),1)
            with self.assertRaises(ValueError):b.reconcile('op',**dict(args,allocated_gpu_seconds=1))
            evidence.write_text('changed')
            with self.assertRaises(ValueError):b.reconcile('op',**args)

    def test_cannot_audit_measured_charge_or_invalid_allocation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            write(root/'resource-budget.json',dict(ceiling_gpu_seconds=100,prior_upper_bound_gpu_seconds=0,charged_gpu_seconds=0,active={},settled={}))
            b=ResourceBudget(root)
            b.reserve('op',maximum_gpu_seconds=40,ram_bytes=0,free_ram_bytes=100,reserve_ram_bytes=0,owner={})
            b.settle('op',2)
            f=root/'audit';f.write_text('owned evidence');f.chmod(0o600)
            for value in [0,-1,True,float('nan'),41]:
                with self.assertRaises(ValueError):b.reconcile('op',allocated_gpu_seconds=value,evidence=f,evidence_sha256=sha(f))

if __name__=='__main__':unittest.main()
