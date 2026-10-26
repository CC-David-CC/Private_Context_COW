"""Real owned filesystem/process/archive controls; no model, GPU or inference."""
from pathlib import Path
import io,json,os,subprocess,sys,tarfile,tempfile,unittest
from unittest.mock import patch
from tools.pcc_training_workspace import preserve_evidence,sha
from tools.pcc_closed_store_retention import retire_store,run_pass

class ClosedStoreRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.work=self.root/'work';self.work.mkdir(mode=0o700)
        self.points=self.work/'points';self.points.mkdir(mode=0o700)
        self.arm=self.points/'adamw-full';self.arm.mkdir(mode=0o700)
        self.evidence=self.root/'evidence';self.evidence.mkdir(mode=0o700)
        (self.evidence/'adamw-full').mkdir(mode=0o700)
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
        self.owner=dict(pid=child.pid,start_ticks=int(fields[19]));child.terminate();child.wait(timeout=5)

    def fixture(self,number=1,live=False):
        point=self.arm/('update-%03d'%number);point.mkdir(mode=0o700)
        dev=point/'development';dev.mkdir(mode=0o700);store=dev/'cow-store';store.mkdir(mode=0o700)
        (store/'history').mkdir(mode=0o700);(store/'history/a.json').write_bytes(b'{"tokens":[1,2,3]}')
        (store/'writer.lock').write_bytes(b'');(dev/'0000-private.json').write_bytes(b'{"answer":"preserved"}')
        owner=self.owner
        if live:
            fields=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()
            owner=dict(pid=os.getpid(),start_ticks=int(fields[19]))
        (point/'development-process.json').write_text(json.dumps(owner))
        result=dict(status='COMPLETE',cases=1,passed=1,finished=1,runtime_id='a'*64,weights_updated=False)
        (dev/'result.json').write_text(json.dumps(result));(dev/'settlement.json').write_text(json.dumps(dict(status='COMPLETE',completed_cases=1)))
        (point/'absolute-point.json').write_text(json.dumps(dict(update=number,passed=1,finished=1,runtime_id='a'*64)))
        (point/'candidate.gguf').write_bytes(b'keep-model');(point/'optimizer-state.bin').write_bytes(b'keep-state')
        evidence=self.evidence/'adamw-full'/point.name;preserve_evidence(point,evidence)
        return point,evidence,store

    def test_retire_only_verified_store_keeps_models_raw_responses_and_scores(self):
        point,evidence,store=self.fixture();kept={str(p.relative_to(point)):p.read_bytes()for p in point.rglob('*')if p.is_file()and not p.is_relative_to(store)}
        value=retire_store(self.work,point,evidence)
        self.assertEqual(value['status'],'CLOSED_STORE_RETIRED');self.assertFalse(store.exists())
        self.assertTrue(value['archive_member_bytes_verified']);self.assertEqual(value['models_deleted'],0)
        self.assertEqual(kept,{str(p.relative_to(point)):p.read_bytes()for p in point.rglob('*')if p.is_file()})
        self.assertEqual(retire_store(self.work,point,evidence),value)

    def test_live_original_evaluator_is_not_retired(self):
        point,evidence,store=self.fixture(live=True)
        value=retire_store(self.work,point,evidence)
        self.assertEqual(value['status'],'SKIP');self.assertIn('live',value['reason']);self.assertTrue(store.exists())

    def test_matching_outer_archive_hash_cannot_hide_wrong_inner_bytes(self):
        point,evidence,store=self.fixture();archive=evidence/'point-evidence.tar.gz'
        with tarfile.open(archive,'w:gz')as stream:
            for p in sorted(point.rglob('*')):
                if not p.is_file()or p.suffix in('.gguf','.bin'):continue
                name=p.relative_to(point).as_posix()
                if p==store/'history/a.json':
                    body=b'{"different":true}';info=tarfile.TarInfo(name);info.size=len(body);stream.addfile(info,io.BytesIO(body))
                else:stream.add(p,arcname=name,recursive=False)
        manifest=evidence/'evidence-verification.json';value=json.loads(manifest.read_bytes());value['archive_sha256']=sha(archive);manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Archived store bytes'):retire_store(self.work,point,evidence)
        self.assertTrue((store/'history/a.json').exists());self.assertFalse((evidence/'closed-store-retirement-intent.json').exists())

    def test_changed_store_refuses_before_any_delete(self):
        point,evidence,store=self.fixture();(store/'history/a.json').write_bytes(b'{"changed":true}')
        with self.assertRaisesRegex(ValueError,'store changed'):retire_store(self.work,point,evidence)
        self.assertTrue((store/'writer.lock').exists())

    def test_partial_retirement_reconciles_exact_original_intent(self):
        point,evidence,store=self.fixture();original=Path.unlink;calls=[]
        def interrupted(path,*a,**k):
            if path.is_relative_to(store):
                calls.append(path)
                if len(calls)==2:raise OSError('injected maintenance interruption')
            return original(path,*a,**k)
        with patch.object(Path,'unlink',interrupted):
            with self.assertRaisesRegex(OSError,'injected'):retire_store(self.work,point,evidence)
        self.assertTrue((evidence/'closed-store-retirement-intent.json').exists())
        self.assertFalse((evidence/'closed-store-retirement-complete.json').exists())
        self.assertEqual(retire_store(self.work,point,evidence)['status'],'CLOSED_STORE_RETIRED')
        self.assertFalse(store.exists())

    def test_policy_retains_three_latest_and_bounds_one_pass(self):
        points=[self.fixture(i)for i in range(1,5)]
        policy=dict(schema=1,workspace=str(self.work),groups=[dict(points=str(self.points),evidence=str(self.evidence),arms=['adamw-full'])],
            keep_latest=3,maximum_stores_per_pass=1,maintenance_lock=str(self.root/'maintenance.lock'))
        self.assertEqual(run_pass(policy)['stores'],1)
        self.assertFalse(points[0][2].exists());self.assertTrue(all(p[2].exists()for p in points[1:]))
        policy['groups']=[]
        with self.assertRaises(ValueError):run_pass(policy)

    def test_owned_group_ancestors_can_be_readable_but_point_remains_private(self):
        points=[self.fixture(i)for i in range(1,5)]
        self.points.chmod(0o755);self.evidence.chmod(0o755)
        policy=dict(schema=1,workspace=str(self.work),groups=[dict(points=str(self.points),evidence=str(self.evidence),arms=['adamw-full'])],
            keep_latest=3,maximum_stores_per_pass=1,maintenance_lock=str(self.root/'maintenance.lock'))
        self.assertEqual(run_pass(policy)['stores'],1)
        points[1][0].chmod(0o755)
        with self.assertRaisesRegex(ValueError,'Private owned'):retire_store(self.work,points[1][0],points[1][1])
        self.assertTrue(points[1][2].exists())

    def test_missing_archive_and_linked_paths_never_delete(self):
        point,evidence,store=self.fixture();(evidence/'point-evidence.tar.gz').unlink()
        with self.assertRaises(FileNotFoundError):retire_store(self.work,point,evidence)
        self.assertTrue(store.exists());alias=self.work/'alias';alias.symlink_to(point,target_is_directory=True)
        with self.assertRaises(ValueError):retire_store(self.work,alias,evidence)

    def test_finite_owner_cli_runs_without_model_or_workflow_dispatch(self):
        points=[self.fixture(i)for i in range(1,5)]
        policy=dict(schema=1,workspace=str(self.work),groups=[dict(points=str(self.points),evidence=str(self.evidence),arms=['adamw-full'])],
            keep_latest=3,maximum_stores_per_pass=1,maintenance_lock=str(self.root/'maintenance.lock'),completion_points=[str(points[-1][0])])
        path=self.root/'policy.json';path.write_text(json.dumps(policy));out=self.root/'maintenance'
        result=subprocess.run([sys.executable,'-m','tools.pcc_closed_store_retention','--policy',str(path),
            '--output',str(out),'--maximum-passes','2','--interval-seconds','1'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((out/'exit.json').read_bytes())['passes'],2)
        self.assertFalse(points[0][2].exists());self.assertTrue(points[-1][2].exists())

if __name__=='__main__':unittest.main()
