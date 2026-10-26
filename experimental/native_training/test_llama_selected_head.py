"""Independent selected-output-head qualification on generated tiny weights.

The full context stays present. Old/dense and new/selected artifacts are compared
on scattered and early labels; FP64 finite differences separately verify the
loss derivatives. This does not train or qualify a real language model.
"""
from pathlib import Path
import argparse,json
import numpy as np
from .test_llama_action_profile import run as action_qualification,example,sha,tensor_arrays
from .llama_reference import read_gguf,model_value
from tools.pcc_native_training import execute_native

PROFILE='dense-llama-action-selected-head-8k-v1'
DENSE='dense-llama-action-8k-v1'

def run(executable,output,devices):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    executable=Path(executable).resolve(strict=True)
    qualification=action_qualification(executable,root/'existing-action-controls',devices,profile=PROFILE)
    assert qualification['status']=='PASS'
    model=root/'existing-action-controls/tiny-f32.gguf';weights,geometry=read_gguf(model)
    common=dict(model=str(model),base_sha256=sha(model),mode='full',rank=1,devices=list(devices),
        memory_profile='allocator-v1',steps=1,batch_size=1,learning_rate=.01,
        device_budget_bytes={d:12<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20)
    checks=[]
    for n,positions in [(7,[0,2,5]),(513,[0,3,241,512])]:
        e=example(n);e['loss_mask']=[i in positions for i in range(n)]
        for mode,steps in [('full',1),('lora',2)]:
            pair=[]
            for profile,label in [(DENSE,'dense'),(PROFILE,'selected')]:
                candidate=execute_native(executable,dict(common,mode=mode,steps=steps,sequence_profile=profile,examples=[e]),
                    root/(str(n)+'-'+mode+'-'+label),timeout=600)
                if label=='selected':assert candidate['receipt']['loss_projection']=='selected-hidden-columns-v1'
                pair.append(candidate)
            left=tensor_arrays(pair[0]['candidate_path']);right=tensor_arrays(pair[1]['candidate_path'])
            assert left.keys()==right.keys()
            maximum=max(float(np.max(np.abs(left[k]-right[k])))for k in left)
            assert maximum<2e-6,('dense/selected saved weights',n,mode,maximum)
            losses=[abs(a[k]-b[k])for a,b in zip(pair[0]['receipt']['steps'],pair[1]['receipt']['steps'])for k in ['before','after']]
            assert max(losses)<1e-5,(n,mode,losses)
            checks.append(dict(name='dense-selected-saved-update-agreement',tokens=n,selected_positions=positions,
                mode=mode,updates=steps,maximum_weight_difference=maximum,maximum_loss_difference=max(losses)))
            if mode!='full':continue
            updated,_=read_gguf(pair[1]['candidate_path']);rate=float(np.float32(common['learning_rate']))
            errors=[];sensitivity=[]
            for name,values in weights.items():
                indices=np.ndindex(values.shape)if n==7 else [(0,0)]
                for index in indices:
                    saved=values[index];estimates=[]
                    for epsilon in (1e-5,2e-5):
                        values[index]=saved+epsilon;plus=model_value(e,weights,geometry)[0]
                        values[index]=saved-epsilon;minus=model_value(e,weights,geometry)[0]
                        values[index]=saved;estimates.append((plus-minus)/(2*epsilon))
                    observed=(saved-updated[name][index])/rate
                    assert np.isclose(observed,estimates[0],rtol=.005,atol=5e-5),(n,name,index,observed,estimates)
                    assert np.isclose(estimates[0],estimates[1],rtol=1e-4,atol=1e-7)
                    errors.append(abs(observed-estimates[0]));sensitivity.append(abs(estimates[0]-estimates[1]))
            checks.append(dict(name='selected-head-vs-independent-fp64',tokens=n,selected_positions=positions,
                scalar_directions=len(errors),all_scalars=n==7,maximum_absolute_gradient_error=max(errors),
                maximum_step_sensitivity=max(sensitivity)))
    result=dict(schema=1,status='PASS',scope='Generated tiny selected-head native mechanics; no real model learning',
        profile=PROFILE,devices=list(devices),existing_action_qualification=qualification,additional_checks=checks,
        native_executable_sha256=sha(executable),test_source_sha256=sha(__file__),real_model_updates=0,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--devices',nargs='+',default=['CPU'])
    args=parser.parse_args();print(json.dumps(run(args.executable,args.output,args.devices)))
