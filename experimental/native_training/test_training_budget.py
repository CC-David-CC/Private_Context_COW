"""Actual completed-update budgeting, state continuation and refusal checks."""
from pathlib import Path
import argparse,copy,json,subprocess,hashlib
import numpy as np
from .make_llama_fixture import make_fixture
from .llama_reference import read_gguf
from .test_optimizer_training import state_arrays
from tools.pcc_native_training import execute_native,validate_training_budget_request,validate_training_budget_receipt,require_training_budget_capability


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(executable,output,devices,old_executable=None):
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    capability=require_training_budget_capability(executable)
    model=make_fixture(output/'base.gguf');base=sha(model)
    e=dict(id='a',tokens=[0,1,2,3],targets=[1,2,3,4],loss_mask=[False,True,True,True],verified=True,provenance_sha256='a'*64)
    examples=[e,dict(e,id='b',tokens=[3,2,1,0],targets=[2,1,0,6]),dict(e,id='c',tokens=[1,3,5,2],targets=[3,5,2,0])]
    common=dict(model=str(model),base_sha256=base,mode='full',devices=devices,device_budget_bytes={d:4<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20,sequence_profile='dense-llama-short-v1',steps=5,batch_size=2,learning_rate=.001,examples=examples,optimizer_state_mode='fresh',training_budget_seconds=1e-12,deadline_seconds=60)
    cases=[];negative=0;first_receipt=None;first_request=None
    for label,kind,settings in [('adamw','adamw',{}),('muon-refine','muon',dict(projection='fast-refine-v1',refine_start_step=2,refine_steps=5))]:
        folder=output/label;folder.mkdir()
        request=dict(common,optimizer=kind,optimizer_settings=dict(schema=1,**settings))
        first=execute_native(executable,request,folder/'first',timeout=120)
        assert len(first['receipt']['steps'])==1 and first['receipt']['training_budget']['stop_reason']=='training_budget'
        assert first['receipt']['optimizer_checkpoint']['completed_steps']==1 and first['receipt']['example_cursor']==2
        checkpoint=first['receipt']['optimizer_checkpoint']
        resume=dict(request,model=first['candidate_path'],base_sha256=first['candidate_sha256'],optimizer_state_mode='resume',initial_optimizer_state=checkpoint['path'],initial_optimizer_state_sha256=checkpoint['sha256'],example_cursor=2)
        resumed=execute_native(executable,resume,folder/'resumed',timeout=120)
        assert resumed['receipt']['optimizer_checkpoint']['completed_steps']==2 and resumed['receipt']['example_cursor']==3
        straight_request={k:v for k,v in request.items() if k!='training_budget_seconds'};straight_request['steps']=2
        straight=execute_native(executable,straight_request,folder/'straight',timeout=120)
        actual,_=read_gguf(resumed['candidate_path']);expected,_=read_gguf(straight['candidate_path'])
        assert actual.keys()==expected.keys() and all(np.array_equal(actual[k],expected[k]) for k in actual)
        am,aa=state_arrays(resumed['optimizer_state_path']);em,ea=state_arrays(straight['optimizer_state_path'])
        assert aa.keys()==ea.keys() and all(np.array_equal(aa[k],ea[k]) for k in aa)
        assert 'training_budget' not in straight['receipt'] and all('timing' not in row for row in straight['receipt']['steps'])
        cases.append(dict(arm=label,status='PASS',completed_budget_updates=2,global_step=am['completed_steps'],example_cursor=am['example_cursor'],partial_final_batch=True,weights_and_moments_bit_exact=True,refinement_active=resumed['receipt']['steps'][-1]['optimizer_step']['projection_refinement_active'],first_budget=first['receipt']['training_budget'],resumed_budget=resumed['receipt']['training_budget']))
        if first_receipt is None:first_receipt=first['receipt'];first_request=request
    # A large budget still honors a smaller explicit step cap.
    cap_request=dict(first_request,steps=1,training_budget_seconds=30.)
    capped=execute_native(executable,cap_request,output/'step-cap',timeout=120)
    assert capped['receipt']['training_budget']['stop_reason']=='step_cap' and capped['receipt']['training_budget']['remaining_seconds']>0
    # Untimed SGD historically defaults to one update when steps is omitted.
    sgd={k:v for k,v in common.items() if k not in ('training_budget_seconds','optimizer_state_mode','steps')}
    default=execute_native(executable,sgd,output/'default-sgd',timeout=120)
    assert len(default['receipt']['steps'])==1 and 'training_budget' not in default['receipt']
    # Native entry points enforce invalid budgets even without Python admission.
    for index,value in enumerate([0,True,60.]):
        candidate=output/('refused-native-'+str(index)+'.gguf')
        request=dict(first_request,training_budget_seconds=value,candidate=str(candidate),optimizer_state_output=str(candidate)+'.state')
        path=output/('refused-native-'+str(index)+'.json');path.write_text(json.dumps(request)+'\n')
        proc=subprocess.run([str(executable),'--train',str(path)],capture_output=True,text=True,timeout=15)
        assert proc.returncode and not candidate.exists() and not Path(str(candidate)+'.state').exists()
        assert 'budget' in proc.stderr.lower();negative+=1
    for value in [0,-1,True,float('nan'),float('inf'),60.]:
        try:validate_training_budget_request(dict(first_request,training_budget_seconds=value))
        except ValueError:negative+=1
        else:raise AssertionError('Invalid budget admitted')
    for change in [dict(mode='lora'),dict(evaluate_only=True),dict(gradient_probe_only=True),dict(sequence_profile='short-v1'),dict(steps=0)]:
        try:validate_training_budget_request(dict(first_request,**change))
        except ValueError:negative+=1
        else:raise AssertionError('Unsupported budget route admitted')
    for label,change in [('missing',lambda x:x.pop('training_budget')),('elapsed',lambda x:x['training_budget'].update(elapsed_seconds=99.)),('counter',lambda x:x['training_budget'].update(completed_updates=2)),('boundary',lambda x:x['training_budget'].update(boundary='partial'))]:
        receipt=copy.deepcopy(first_receipt);change(receipt)
        try:validate_training_budget_receipt(first_request,receipt)
        except ValueError:negative+=1
        else:raise AssertionError('Malformed budget receipt admitted: '+label)
    if old_executable:
        try:execute_native(old_executable,first_request,output/'old-binary-refused',timeout=120)
        except ValueError as error:assert 'advertise' in str(error);negative+=1
        else:raise AssertionError('Old binary silently ignored a training budget')
        assert not (output/'old-binary-refused').exists()
    assert sha(model)==base
    result=dict(status='PASS',scope='Tiny ordinary-Llama whole-update timing and completed state; not model learning',devices=devices,capability=capability,cases=cases,negative_cases=negative,step_cap_passed=True,default_unbudgeted_receipt_preserved=True,default_sgd_step_passed=True,promoted=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',default=['CPU']);p.add_argument('--old-executable',type=Path);a=p.parse_args();print(json.dumps(run(a.executable,a.output,a.devices,a.old_executable)))
