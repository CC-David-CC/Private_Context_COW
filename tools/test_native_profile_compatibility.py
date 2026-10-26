"""Compare old-profile native outputs with the prior qualified executable."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from tools.pcc_native_training import execute_native,sha256
from experimental.native_training.make_llama_fixture import make_fixture
from experimental.native_training.test_llama_action_profile import example,tensor_arrays

def run(executable,reference_executable,output,device,*,include_selected_head=False):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf',dtype='f32',layers=1,context_length=8192)
    checks=[]
    cases=[(4,'dense-llama-short-v1'),(513,'dense-llama-action-8k-v1')]
    if include_selected_head:cases.append((513,'dense-llama-action-selected-head-8k-v1'))
    for n,profile in cases:
        label_prefix=str(n)+'-'+profile
        common=dict(model=str(model),base_sha256=sha256(model),rank=1,devices=[device],sequence_profile=profile,
            memory_profile='allocator-v1',examples=[example(n)],steps=1,batch_size=1,learning_rate=.01,
            device_budget_bytes={device:4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20)
        logits=[]
        for label,binary in [('reference',reference_executable),('current',executable)]:
            path=root/(label_prefix+'-'+label+'-forward.json');artifact=root/(label_prefix+'-'+label+'-logits.f32')
            request=dict(common,mode='full',evaluate_only=True,logits_output=str(artifact),candidate=str(root/'never.gguf'))
            path.write_text(json.dumps(request));p=subprocess.run([str(binary),'--train',str(path)],capture_output=True,text=True,timeout=120)
            assert p.returncode==0,p.stderr[-2400:]
            logits.append(artifact.read_bytes())
        assert logits[0]==logits[1],('Original profile logits changed',profile)
        for mode in ['full','lora']:
            saved=[]
            for label,binary in [('reference',reference_executable),('current',executable)]:
                result=execute_native(binary,dict(common,mode=mode),root/(label_prefix+'-'+mode+'-'+label),timeout=120)
                saved.append(result['candidate_path'])
            left=tensor_arrays(saved[0]);right=tensor_arrays(saved[1]);assert left.keys()==right.keys()
            assert all(np.array_equal(left[k],right[k])for k in left),('Original profile update changed',profile,mode)
            checks.append(dict(profile=profile,input_tokens=n,mode=mode,all_saved_tensors_bit_value_identical=True,
                forward_logit_bytes_identical=True,parameter_tensors=len(left)))
    result=dict(schema=1,status='PASS',scope='Generated tiny old-profile compatibility with prior executable; not real-model learning',
        device=device,checks=checks,current_executable_sha256=sha256(executable),reference_executable_sha256=sha256(reference_executable),
        test_source_sha256=sha256(__file__),real_model_updates=0)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--reference-executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='CPU');p.add_argument('--include-selected-head',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.executable,a.reference_executable,a.output,a.device,include_selected_head=a.include_selected_head)))
