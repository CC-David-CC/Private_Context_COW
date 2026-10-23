"""Compare bounded dense sequence training to the unchanged ordinary short path.

Uses a generated tiny CPU model, real native gradients, save and reload.
This does not establish Wen/GPU memory capacity or language quality.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess
from .make_cpu_fixture import make_fixture


def run(executable, output):
    os.umask(0o077)
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(output/'fixture.gguf',context_length=2048)
    original=hashlib.sha256(model.read_bytes()).hexdigest()
    def execute(label, request, should_fail=False):
        path=output/(label+'.json');path.write_text(json.dumps(request)+'\n')
        result=subprocess.run([str(executable),'--train',str(path)],capture_output=True,text=True,timeout=120)
        (output/(label+'.stdout')).write_text(result.stdout)
        (output/(label+'.stderr')).write_text(result.stderr)
        if should_fail:
            assert result.returncode!=0 and not Path(request['candidate']).exists()
            return result.stderr
        assert result.returncode==0,result.stderr[-3000:]
        return json.loads(result.stdout)
    checks=[]
    for mode in ('lora','full'):
        for length in (64,375):
            tokens=[i%128 for i in range(length+1)]
            example=dict(id='owned-sequence',tokens=tokens[:-1],targets=tokens[1:],
                         loss_mask=[i>=length-4 for i in range(length)],verified=True,provenance_sha256='a'*64)
            base=dict(model=str(model),base_sha256=original,mode=mode,rank=2,devices=['CPU'],
                device_budget_bytes={'CPU':8<<30},reserve_bytes=128<<20,examples=[example],
                batch_size=1,steps=1,learning_rate=.0001)
            results={}
            for profile in ('short-v1','dense-short-v1'):
                label=f'{mode}-{length}-{profile}'
                request=dict(base,sequence_profile=profile,candidate=str(output/(label+'.gguf')))
                if profile=='dense-short-v1':request['block_devices']=[0,0,0,0]
                report=execute(label,request);results[profile]=report
                candidate=Path(request['candidate'])
                reload=dict(base,sequence_profile=profile,evaluate_only=True,candidate=str(output/(label+'-unused.gguf')))
                if mode=='full':reload.update(model=str(candidate),base_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
                else:reload.update(initial_adapter=str(candidate),initial_adapter_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
                loaded=execute(label+'-reload',reload)
                assert abs(loaded['losses'][0]-report['steps'][0]['after'])<1e-5
            for field in ('before','after'):
                assert abs(results['short-v1']['steps'][0][field]-results['dense-short-v1']['steps'][0][field])<1e-5
            assert hashlib.sha256(model.read_bytes()).hexdigest()==original
            checks.append(dict(mode=mode,length=length,loss_agreement=True,reload_agreement=True))
    request=dict(base,sequence_profile='dense-short-v1',candidate=str(output/'refused.gguf'))
    request['examples']=[dict(example,tokens=[0]*513,targets=[0]*513,loss_mask=[False]*512+[True])]
    assert 'Sequence admission refused' in execute('overlong',request,True)
    request['examples']=[dict(example,tokens=[0]*512,targets=[0]*512,loss_mask=[True]*512)]
    assert '256 selected target' in execute('too-many-targets',request,True)
    request['examples']=[example]
    request['block_devices']=[0,0,0]
    assert 'Explicit placement must cover' in execute('missing-block',request,True)
    request['block_devices']=[0,0,1,0]
    assert 'unavailable device' in execute('unknown-block-device',request,True)
    result=dict(status='PASS',checks=checks,negative_cases=4,base_unchanged=True,gpu_used=False,
                scope='tiny CPU complete-graph profile equivalence, update and reload; not Wen qualification')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.executable,a.output)))
