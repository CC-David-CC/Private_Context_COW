"""Actual owned-file retirement boundaries; fixtures are not model findings."""
from pathlib import Path
import hashlib,json,os,tempfile,unittest
from tools.pcc_curve_retention import retire_point

class CurveRetention(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'named').mkdir()
        self.policy=dict(milestone_updates=[1,2,4,8,16,32,42,64,68],report_complete=False)
    def point(self,update):
        point=self.root/'owned'/('update-'+str(update).zfill(3));(point/'training').mkdir(parents=True)
        model=point/'training/candidate.gguf';model.write_bytes(b'opaque owned model fixture '+bytes([update]))
        state=point/'training/optimizer-state.bin';state.write_bytes(b'opaque owned state fixture '+bytes([update]))
        model_hash=hashlib.sha256(model.read_bytes()).hexdigest()
        (point/'curve-point.json').write_text(json.dumps(dict(arm='owned',update=update,protocol_id='f'*64,model_sha256=model_hash)))
        (point/'response-evidence.json').write_text('Owned synthetic response evidence, not a model score')
        (point/'adapter-logits.bin').write_bytes(b'Numerical evidence to retain')
        name=self.root/'named'/('Owned-U'+str(update)+'.gguf');os.link(model,name)
        files=[dict(path=str(p.relative_to(point)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in point.rglob('*') if p.is_file()]
        (point/'archive-verification.json').write_text(json.dumps(dict(status='VERIFIED',directory=str(point),files=files,named_model=str(name))))
        return point,model,state,name,model_hash
    def test_superseded_binaries_retire_once_while_evidence_remains(self):
        point,model,state,name,_=self.point(3);self.point(4)
        result=retire_point(self.root,'owned',3,self.policy,successor_update=4)
        self.assertFalse(model.exists());self.assertFalse(state.exists());self.assertFalse(name.exists())
        self.assertTrue((point/'curve-point.json').exists());self.assertTrue((point/'response-evidence.json').exists());self.assertTrue((point/'adapter-logits.bin').exists())
        self.assertEqual(retire_point(self.root,'owned',3,self.policy,successor_update=4),result)
    def test_active_or_milestone_points_are_protected(self):
        _,model,state,_,_=self.point(3)
        with self.assertRaises(ValueError):retire_point(self.root,'owned',3,self.policy)
        self.assertTrue(model.exists() and state.exists())
        point,model,state,_,_=self.point(4)
        self.assertEqual(retire_point(self.root,'owned',4,self.policy,successor_update=5)['status'],'PROTECTED')
        self.assertTrue(model.exists() and state.exists())
    def test_changed_binary_refuses_before_any_unlink(self):
        _,model,state,name,_=self.point(5);self.point(6);state.write_bytes(b'Changed after archival')
        with self.assertRaises(ValueError):retire_point(self.root,'owned',5,self.policy,successor_update=6)
        self.assertTrue(model.exists() and state.exists() and name.exists())
    def test_after_report_keeps_selected_model_but_retires_old_state(self):
        _,model,state,name,sha=self.point(4)
        policy=dict(self.policy,report_complete=True)
        with self.assertRaises(ValueError):retire_point(self.root,'owned',4,policy)
        policy.update(report_sha256='a'*64,report_commit='b'*40,selected_current_model_sha256s=[sha])
        retire_point(self.root,'owned',4,policy)
        self.assertTrue(model.exists() and name.exists());self.assertFalse(state.exists())
    def test_report_cleanup_handles_previously_retired_points(self):
        self.point(3);self.point(4);retire_point(self.root,'owned',3,self.policy,successor_update=4)
        policy=dict(self.policy,report_complete=True,report_sha256='a'*64,report_commit='b'*40)
        result=retire_point(self.root,'owned',3,policy)
        self.assertEqual(result['removed'],[])
    def test_escape_and_binary_links_are_rejected(self):
        with self.assertRaises(ValueError):retire_point(self.root,'../outside',3,self.policy)
        _,model,_,_,_=self.point(5);self.point(6);model.unlink();model.symlink_to(self.root/'owned/update-006/training/candidate.gguf')
        with self.assertRaises(ValueError):retire_point(self.root,'owned',5,self.policy,successor_update=6)

if __name__=='__main__':unittest.main()
