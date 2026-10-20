"""Real branching under a separately bound, unqualified sanitizer profile."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


def run(base,tool,directory):
    from context_time_travel.real import BranchingModel,FreshOracle
    from context_time_travel.numerics import assert_logits_close
    from context_time_travel.strata_backend import _environment,_digest
    if base.get('calibration_status')!='FROZEN':raise ValueError('Frozen base profile required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    p=json.loads(json.dumps(base));p['calibration_status']='SANITIZER_DIAGNOSTIC_UNQUALIFIED';p['numerical_gate_qualified']=False
    p['execution']['environment']=_environment()
    tool=Path(tool).resolve();tool_hash=hashlib.sha256(tool.read_bytes()).hexdigest()
    p['native_build']['sha256'][str(tool)]=tool_hash
    for key,value in p['execution']['environment'].items():
        if key.startswith('CUDA_INJECTION') and Path(value).is_file():
            p['native_build']['sha256'][value]=hashlib.sha256(Path(value).read_bytes()).hexdigest()
    p['execution']['instrumentation']=dict(tool='Compute Sanitizer memcheck',tool_sha256=tool_hash)
    p['execution_profile_id']=_digest(dict(build=p['native_build'],execution=p['execution'],hardware=p['hardware']))
    (directory/'diagnostic-profile.json').write_text(json.dumps(p,indent=2)+'\n')
    report=dict(status='RUNNING',scope='Unqualified instrumented diagnostic, not numerical gate admission',
        execution_profile_id=p['execution_profile_id'],base_execution_profile_id=base['execution_profile_id'],checks=[])
    backend=oracle=sut=None
    try:
        backend=BranchingModel.load_backend(p);oracle=FreshOracle(p);sut=BranchingModel(directory/'store',backend,p)
        prefix=(p['fixtures']['prefix'][0],);root=sut.create(prefix);original=sut.snapshot_bytes(root)
        def check(tip,history):
            actual=sut.next_logits(tip);expected=oracle.forward(history)
            try:result=assert_logits_close(actual,expected,atol=base['atol'],rtol=base['rtol'])
            except BaseException:
                for name,values in [('actual',actual),('expected',expected)]:
                    (directory/(name+'.f32')).write_bytes(struct.pack('<'+str(len(values))+'f',*values))
                raise
            assert sut.history(tip)==history and sut.consumed(tip)==len(history)
            report['checks'].append(dict(tokens=history,max_absolute_error=result.max_absolute_error,
                                        computed_tokens=oracle.last['computed_tokens']))
        check(root,prefix)
        for name in ('suffix_a','suffix_b'):
            token=p['fixtures'][name][0];tip=sut.feed(root,[token]);check(tip,prefix+(token,))
        assert sut.snapshot_bytes(root)==original
        report.update(status='DIAGNOSTIC_COMPLETE',oracle_requests=oracle.requests,
                      backend_counters=dict(backend._refresh_stats()))
        return report
    except BaseException as error:report.update(status='FAIL',error=repr(error));raise
    finally:
        active_error=sys.exc_info()[0] is not None;errors=[]
        actions=([sut.evict_all] if sut is not None else [])+([oracle.close] if oracle is not None else [])+([backend.close] if backend is not None else [])
        for action in actions:
            try:action()
            except BaseException as error:errors.append(repr(error))
        if errors:report.update(status='FAIL',cleanup_errors=errors)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if errors and not active_error:raise RuntimeError('Sanitizer target teardown failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--tool',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    print(json.dumps(run(json.loads(args.profile.read_bytes()),args.tool,args.output),indent=2))

if __name__=='__main__':main()
