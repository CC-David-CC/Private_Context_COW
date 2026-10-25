from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
from tools.pcc_training_workspace import ResourceBudget,ResourceUnavailable,copy_checkpoint,preserve_evidence,retire_cached_candidate,sha,write


class TrainingWorkspaceTests(unittest.TestCase):
    def test_resource_reservations_prevent_overcommit_and_unknown_work_stays_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'resource-budget.json',dict(ceiling_gpu_seconds=100,prior_upper_bound_gpu_seconds=20,charged_gpu_seconds=0,active={},settled={}))
            budget=ResourceBudget(root)
            kw=dict(maximum_gpu_seconds=40,ram_bytes=30,free_ram_bytes=100,reserve_ram_bytes=20,owner=dict(pid=123))
            budget.reserve('first',**kw)
            with self.assertRaises(ValueError):budget.reserve('first',**kw)
            with self.assertRaises(ResourceUnavailable):budget.reserve('no-ram',**dict(kw,free_ram_bytes=70))
            with self.assertRaises(ResourceUnavailable):budget.reserve('no-time',**dict(kw,maximum_gpu_seconds=41))
            self.assertEqual(budget.settle('first',10)['charged_gpu_seconds'],10)
            self.assertEqual(budget.settle('first',10)['charged_gpu_seconds'],10)
            budget.reserve('second',**kw)
            self.assertEqual(budget.settle('second')['charged_gpu_seconds'],40)
            self.assertEqual(json.loads((root/'resource-budget.json').read_text())['charged_gpu_seconds'],50)

    def candidate(self,root):
        root.mkdir(mode=0o700)
        model=root/'candidate.gguf';model.write_bytes(b'owned model fixture')
        state=root/'optimizer-state.bin';state.write_bytes(b'owned optimizer fixture')
        (root/'result.json').write_text('{"scope":"owned test"}\n')
        return dict(candidate_path=str(model),candidate_sha256=sha(model),optimizer_state_path=str(state),optimizer_state_sha256=sha(state),optimizer_completed_steps=1)

    def test_verified_copy_and_evidence_survive_idempotent_cache_retirement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);point=root/'point';candidate=self.candidate(point)
            saved=root/'milestone';copy_checkpoint(saved,candidate)
            evidence=root/'evidence';preserve_evidence(point,evidence)
            manifest=json.loads((evidence/'evidence-verification.json').read_text())
            self.assertEqual(set(manifest['files']),{'result.json'})
            successor=dict(candidate,optimizer_completed_steps=2,candidate_sha256='a'*64)
            result=retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence,durable_checkpoint=saved)
            self.assertEqual(result['status'],'CACHE_BINARIES_RETIRED')
            self.assertFalse(Path(candidate['candidate_path']).exists())
            self.assertTrue((point/'result.json').exists())
            self.assertTrue((saved/'candidate.gguf').exists())
            self.assertEqual(retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence,durable_checkpoint=saved),result)

    def test_corruption_links_and_unqualified_successors_cannot_delete_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);point=root/'point';candidate=self.candidate(point);evidence=root/'evidence';preserve_evidence(point,evidence)
            with self.assertRaises(ValueError):retire_cached_candidate(root,candidate,successor=candidate,evidence_directory=evidence)
            successor=dict(candidate,optimizer_completed_steps=2,candidate_sha256='a'*64)
            model=Path(candidate['candidate_path']);original=model.read_bytes();model.write_bytes(b'changed')
            with self.assertRaises(ValueError):retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence)
            self.assertTrue(Path(candidate['optimizer_state_path']).exists())
            model.write_bytes(original);model.unlink();model.symlink_to(point/'result.json')
            with self.assertRaises(ValueError):retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence)
            self.assertTrue((point/'result.json').exists())

    def test_interrupted_retirement_reconciles_original_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);point=root/'point';candidate=self.candidate(point);evidence=root/'evidence';preserve_evidence(point,evidence)
            successor=dict(candidate,optimizer_completed_steps=2,candidate_sha256='a'*64)
            original_unlink=Path.unlink
            def interrupt(path,*args,**kwargs):
                if path.name=='optimizer-state.bin':raise RuntimeError('Injected interruption after first unlink')
                return original_unlink(path,*args,**kwargs)
            with patch.object(Path,'unlink',autospec=True,side_effect=interrupt):
                with self.assertRaises(RuntimeError):retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence)
            self.assertFalse(Path(candidate['candidate_path']).exists())
            self.assertTrue(Path(candidate['optimizer_state_path']).exists())
            self.assertTrue((evidence/'cache-retirement-intent.json').exists())
            result=retire_cached_candidate(root,candidate,successor=successor,evidence_directory=evidence)
            self.assertEqual(result['status'],'CACHE_BINARIES_RETIRED')
            self.assertTrue((evidence/'point-evidence.tar.gz').exists())


if __name__=='__main__':unittest.main()
