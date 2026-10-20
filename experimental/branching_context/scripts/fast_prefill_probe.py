"""Hard-coded real CUDA batched-prefill/COW/disk demonstration; zero generated tokens."""
import argparse
import array
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(path):
    result=array.array('f')
    result.frombytes(Path(path).read_bytes())
    if len(result)!=248320 or any(not math.isfinite(x) for x in result):
        raise ValueError('Incomplete/nonfinite full-vocabulary frontier')
    return tuple(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('api',type=Path)
    parser.add_argument('profile',type=Path)
    parser.add_argument('binaries',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--red',action='store_true',help='Require batched instrumentation from the original T=1 driver')
    parser.add_argument('--only-ram',action='store_true',help='Run the focused host-RAM restore check only')
    parser.add_argument('--only-two-restore',action='store_true',help='Run the smallest real prefix-restore/append check')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.api/'src'))
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _digest, _validate_execution, _validate_model
    from context_time_travel import controller
    original=json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')):
            del os.environ[key]
    os.environ.update(original['execution']['environment'])
    _validate_execution(original)
    model_verification=_validate_model(original)
    profile=copy.deepcopy(original)
    if not args.red:
        profile['execution']['sut_command'][0]=str(args.binaries/'strata-batched-branch')
        profile['execution']['oracle_command'][0]=str(args.binaries/'strata-batched-fresh')
        profile['execution']['operator']='batched-prefill-complete-frontier-experimental'
        profile['native_build']['revision']='isolated-fast-prefill-probe-20261013'
        for path in [args.binaries/'strata-batched-branch',args.binaries/'strata-batched-fresh']:
            profile['native_build']['sha256'][str(path)]=sha(path)
        profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=profile['execution'],hardware=profile['hardware']))
        profile['numerical_gate_qualified']=False
    (args.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    report=dict(schema=1,status='RUNNING',scope='hard-coded prefill-only experiment; no G1/G2/G3 qualification',
                timed_generated_tokens=0,cross_operator_control_max_new=1,
                model_verification=model_verification,model_contract_id=profile['model_contract_id'],
                execution_profile_id=profile['execution_profile_id'],hardware=profile['hardware'],
                profile_sha256=sha(args.output/'profile.json'),comparisons=[],forwards=[],timings=[],tests=[])

    def save():
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')

    def fresh(ids,label,binary=None):
        path=args.output/(label+'.f32')
        command=list(profile['execution']['oracle_command'])
        if binary is not None: command[0]=str(binary)
        token_path=args.output/(label+'.tokens')
        token_path.write_text(','.join(map(str,ids)))
        if '--tokens' in command:
            del command[command.index('--tokens'):command.index('--tokens')+2]
        if '--tokens-file' in command:
            command[command.index('--tokens-file')+1]=str(token_path)
        else:
            command+=['--tokens-file',str(token_path)]
        start=time.monotonic()
        result=subprocess.run(command,env=dict(os.environ,STRATA_DUMP_FIRST_LOGITS=str(path)),capture_output=True,timeout=600)
        (args.output/(label+'.stdout')).write_bytes(result.stdout)
        (args.output/(label+'.stderr')).write_bytes(result.stderr)
        row=dict(label=label,command=command,exit_code=result.returncode,seconds=time.monotonic()-start,history=list(ids))
        report['forwards'].append(row);save()
        if result.returncode: raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
        row.update(logits_sha256=sha(path),bytes=path.stat().st_size)
        try: row['counters']=json.loads(result.stdout)
        except ValueError: row['counters']=None
        save()
        return vector(path)

    def compare(a,b,label,enforce=True):
        error=max(abs(x-y) for x,y in zip(a,b))
        row=dict(label=label,compared_logits=len(a),max_absolute_error=error,
                 top1_actual=max(range(len(a)),key=a.__getitem__),top1_expected=max(range(len(b)),key=b.__getitem__),
                 requirement='exact experimental baseline' if enforce else 'cross-operator diagnosis only')
        report['comparisons'].append(row);save()
        if enforce and error!=0: raise AssertionError(label+': max absolute error '+str(error))
        return row

    class Probe(unittest.TestCase):
        def test_00_two_token_operator_diagnosis(self):
            if args.red: return
            normal=Path(original['execution']['sut_command'][0]).parent/'strata'
            expected=fresh((100,117),'two-token-T1',Path(original['execution']['oracle_command'][0]))
            ordinary=fresh((100,117),'two-token-ordinary',normal)
            compare(ordinary,expected,'ordinary prefill versus T1',enforce=False)
            fast=fresh((100,117),'two-token-all-batched')
            repeated=fresh((100,117),'two-token-all-batched-repeat')
            compare(fast,repeated,'independent batched repeat')
            compare(fast,expected,'all-batched versus T1',enforce=False)

        def test_01_two_token_branch_uses_real_prefill(self):
            backend=StrataBackend(profile)
            self.addCleanup(backend.close)
            sut=BranchingModel(args.output/'two-token-store',backend,profile)
            root=sut.create((100,117))
            raw=backend._request('stats')
            self.assertIn('prefill_tokens',raw,'Original T=1 driver does not implement batched prefill')
            self.assertEqual(raw['prefill_tokens'],2)
            self.assertEqual(raw['windows'],0,'No token-level verifier windows in the batched prompt path')
            compare(sut.next_logits(root),fresh((100,117),'two-token-COW-oracle'),'two-token batched COW versus fresh')
            report['two_token_native_counters']=raw
            save()

        def test_02_eight_prefills_with_ram_and_disk_cow(self):
            if args.red: return
            prefix=tuple((tuple(profile['fixtures']['prefix'])*128)[:128])
            endings=[(71+i,81+i,91+i,101+i) for i in range(8)]
            report['prefix']=list(prefix);report['endings']=[list(x) for x in endings]
            expected=fresh(prefix,'prefix-fresh')
            compare(expected,fresh(prefix,'prefix-fresh-repeat'),'128-token independent fresh repeat')
            backend=StrataBackend(profile)
            self.addCleanup(backend.close)
            sut=BranchingModel(args.output/'store',backend,profile)
            started=time.monotonic();root=sut.create(prefix)
            report['prepare_seconds']=time.monotonic()-started
            compare(sut.next_logits(root),expected,'128-token initial batched COW versus fresh')
            root_bytes=sut.snapshot_bytes(root)
            checkpoint=sut.checkpoint(root)
            retained_prefix=sut.cache[root].blob
            parent_audit=backend.audit(retained_prefix)
            report['parent_audit_before']=parent_audit
            payload_sha=sha(checkpoint['payload_path']);manifest_sha=sha(checkpoint['manifest_path'])
            report['checkpoint']=dict(payload_bytes=checkpoint['payload_path'].stat().st_size,
                                      payload_sha256=payload_sha,manifest_sha256=manifest_sha)
            for lane in ('recompute','resident','disk'):
                for index,ending in enumerate(endings):
                    if lane=='disk':
                        sut.cache.clear();backend.release_checkpoints();backend.clear_cached_histories()
                    before=backend._request('stats')
                    started=time.monotonic()
                    if lane=='recompute':
                        with backend.lease() as lease:
                            backend.consume(lease,prefix)
                            prefix_frontier=backend.logits(lease)
                            backend.consume(lease,ending)
                            actual=backend.logits(lease)
                        sid=None
                    else:
                        with sut.materialize(root) as (lease,prefix_frontier):
                            backend.consume(lease,ending) # one actual batched call, no Generate/decode
                            actual=backend.logits(lease)
                            image=sut._save(lease,prefix+ending)
                        decoder=sut.controller(root)
                        for token in ending:
                            decoder=controller.consume(decoder,token,backend.piece(token),backend.is_eog(token))
                        sid=sut.store.append(root,ending,decoder)
                        sut.cache[sid]=image
                        self.assertEqual(sut.history(sid),prefix+ending)
                    seconds=time.monotonic()-started
                    after=backend._request('stats')
                    delta={k:after[k]-before[k] for k in ('forward_tokens','prefill_tokens','prefill_chunks','loaded_file_bytes','cow_pages','copied_bytes')}
                    report['timings'].append(dict(lane=lane,index=index,seconds=seconds,counters=delta,snapshot=sid))
                    save()
                    self.assertEqual(delta['prefill_tokens'],len(prefix)+len(ending) if lane=='recompute' else len(ending))
                    self.assertEqual(delta['forward_tokens'],delta['prefill_tokens'])
                    if lane=='disk': self.assertGreater(delta['loaded_file_bytes'],0)
                    else: self.assertEqual(delta['loaded_file_bytes'],0)
                    compare(prefix_frontier,expected,lane+' prefix '+str(index))
                    compare(actual,fresh(prefix+ending,lane+'-oracle-'+str(index)),lane+' suffix '+str(index))
                    self.assertEqual(root_bytes,sut.snapshot_bytes(root))
                    self.assertEqual(payload_sha,sha(checkpoint['payload_path']))
                    self.assertEqual(manifest_sha,sha(checkpoint['manifest_path']))
                    audit=backend.audit(retained_prefix)
                    self.assertEqual(parent_audit['sha256'],audit['sha256'])
                    report.setdefault('parent_audits',[]).append(dict(lane=lane,index=index,sha256=audit['sha256'],bytes=audit['bytes']))
                    save()
            report['final_native_counters']=backend._request('stats')
            save()

        def test_03_host_ram_backed_prefill(self):
            prefix=tuple((tuple(profile['fixtures']['prefix'])*128)[:128])
            backend=StrataBackend(profile)
            self.addCleanup(backend.close)
            sut=BranchingModel(args.output/'ram-store',backend,profile)
            root=sut.create(prefix)
            retained=sut.cache[root].blob
            parent=backend.audit(retained)
            root_bytes=sut.snapshot_bytes(root)
            expected=fresh(prefix,'ram-prefix-fresh')
            for index,ending in enumerate(((71,81,91,101),(72,82,92,102))):
                sut.cache.clear() # no disk checkpoint exists: force the actual native host-RAM tier
                self.assertFalse((sut.checkpoint_directory/(root+'.json')).exists())
                ram_before=backend._request('ram_stats')
                before=backend._request('stats')
                started=time.monotonic()
                with sut.materialize(root) as (lease,frontier):
                    backend.consume(lease,ending)
                    actual=backend.logits(lease)
                    saved=sut._save(lease,prefix+ending)
                decoder=sut.controller(root)
                for token in ending:
                    decoder=controller.consume(decoder,token,backend.piece(token),backend.is_eog(token))
                sid=sut.store.append(root,ending,decoder)
                sut.cache[sid]=saved
                seconds=time.monotonic()-started
                after=backend._request('stats');ram_after=backend._request('ram_stats')
                delta={k:after[k]-before[k] for k in ('forward_tokens','prefill_tokens','prefill_chunks','loaded_file_bytes','cow_pages','copied_bytes')}
                delta['native_ram_hits']=ram_after['hits']-ram_before['hits']
                report['timings'].append(dict(lane='host-ram',index=index,seconds=seconds,counters=delta,snapshot=sid))
                self.assertEqual(delta['native_ram_hits'],1)
                self.assertEqual(delta['prefill_tokens'],len(ending))
                self.assertEqual(delta['forward_tokens'],len(ending))
                self.assertEqual(delta['loaded_file_bytes'],0)
                compare(frontier,expected,'host-ram prefix '+str(index))
                compare(actual,fresh(prefix+ending,'ram-suffix-fresh-'+str(index)),'host-ram suffix '+str(index))
                self.assertEqual(root_bytes,sut.snapshot_bytes(root))
                audit=backend.audit(retained)
                self.assertEqual(audit['sha256'],parent['sha256'])
                report.setdefault('parent_audits',[]).append(dict(lane='host-ram',index=index,sha256=audit['sha256'],bytes=audit['bytes']))
                save()
            report['final_native_counters']=backend._request('stats')
            save()

        def test_04_two_token_restore_then_prefill(self):
            expected=fresh((100,117),'two-restored-fresh')
            backend=StrataBackend(profile)
            self.addCleanup(backend.close)
            sut=BranchingModel(args.output/'two-restored-store',backend,profile)
            root=sut.create((100,))
            retained=sut.cache[root].blob
            before_bytes=sut.snapshot_bytes(root)
            audit_before=backend.audit(retained)
            sut.checkpoint(root)
            for lane in ('resident','disk','host-ram'):
                with self.subTest(lane=lane):
                    if lane=='disk':
                        sut.cache.clear();backend.release_checkpoints()
                    before=backend._request('stats')
                    if lane=='host-ram':
                        with backend.lease() as lease:
                            self.assertTrue(backend.restore_cached(lease,(100,)))
                            backend.consume(lease,(117,))
                            actual=backend.logits(lease)
                    else:
                        with sut.materialize(root) as (lease,_):
                            backend.consume(lease,(117,))
                            actual=backend.logits(lease)
                    after=backend._request('stats')
                    report.setdefault('two_token_restore_counters',[]).append(dict(lane=lane,forward_tokens=after['forward_tokens']-before['forward_tokens'],prefill_tokens=after['prefill_tokens']-before['prefill_tokens'],native_load_bytes=after['loaded_file_bytes']-before['loaded_file_bytes']))
                    self.assertEqual(after['prefill_tokens']-before['prefill_tokens'],1)
                    self.assertEqual(after['windows'],0)
                    self.assertEqual(before_bytes,sut.snapshot_bytes(root))
                    audit=backend.audit(retained)
                    self.assertEqual(audit['sha256'],audit_before['sha256'])
                    report.setdefault('parent_audits',[]).append(dict(lane=lane,index=0,sha256=audit['sha256'],bytes=audit['bytes']))
                    compare(actual,expected,'restore 100 then batched 117: '+lane)
                    save()

    class Result(unittest.TextTestResult):
        def stopTest(self,test):
            super().stopTest(test)
            report['tests'].append(dict(id=test.id(),failures=len(self.failures),errors=len(self.errors),skips=len(self.skipped)))
            save()

    with (args.output/'unittest.log').open('w') as stream:
        if args.red:
            suite=unittest.TestSuite([Probe('test_01_two_token_branch_uses_real_prefill')])
        elif args.only_ram:
            suite=unittest.TestSuite([Probe('test_03_host_ram_backed_prefill')])
        elif args.only_two_restore:
            suite=unittest.TestSuite([Probe('test_04_two_token_restore_then_prefill')])
        else:
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(Probe)
        result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
    report.update(status='PASS' if result.wasSuccessful() else 'FAIL',test_count=result.testsRun,
                  failures=[dict(id=t.id(),trace=trace) for t,trace in result.failures],
                  errors=[dict(id=t.id(),trace=trace) for t,trace in result.errors],skips=result.skipped)
    save()
    print(json.dumps({key:report[key] for key in ('status','test_count','failures','errors','skips')}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__': main()
