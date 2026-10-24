"""Native LoRA AdamW continuity and untimed full-update compatibility."""
from pathlib import Path
import argparse,copy,json
import numpy as np
from tools.pcc_native_training import execute_native,sha256
from experimental.native_training.make_llama_fixture import make_fixture
from tools.gguf_reader import GGUFFile
from experimental.native_training.test_optimizer_training import state_arrays

def compare(a,b):
    def weights(path):
        info=GGUFFile(path);assert all(t.type_name=='F32' for t in info.tensors)
        return {t.name:np.memmap(path,dtype='<f4',mode='r',offset=info.data_start+t.offset,shape=(t.elements,)) for t in info.tensors}
    x=weights(a['candidate_path']);y=weights(b['candidate_path'])
    assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
    _,x=state_arrays(a['optimizer_state_path']);_,y=state_arrays(b['optimizer_state_path'])
    assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)

def run(executable,old_executable,output,device='CPU'):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf');base=sha256(model)
    examples=[dict(id='one',tokens=[0,1,2,3],targets=[1,2,3,4],loss_mask=[False,True,True,True],verified=True,provenance_sha256='b'*64),
              dict(id='two',tokens=[3,2,1,0],targets=[2,1,0,6],loss_mask=[True,False,True,True],verified=True,provenance_sha256='c'*64)]
    request=dict(model=str(model),base_sha256=base,mode='lora',rank=2,
        devices=[device],sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',
        device_budget_bytes={device:4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
        examples=examples,optimizer='adamw',optimizer_settings=dict(schema=1),
        optimizer_state_mode='fresh',learning_rate=.001,batch_size=1,steps=3,
        record_timing=True,deadline_seconds=120)
    direct=execute_native(executable,request,root/'lora-direct',timeout=130)
    first=execute_native(executable,dict(request,steps=1),root/'lora-first',timeout=130)
    resumed=dict(request,steps=2,optimizer_state_mode='resume',initial_adapter=first['candidate_path'],
        initial_adapter_sha256=first['candidate_sha256'],initial_optimizer_state=first['optimizer_state_path'],
        initial_optimizer_state_sha256=first['optimizer_state_sha256'],example_cursor=1)
    second=execute_native(executable,resumed,root/'lora-resumed',timeout=130)
    compare(direct,second)
    assert second['receipt']['optimizer_checkpoint']['completed_steps']==3
    assert second['receipt']['example_cursor']==3
    assert 'training_budget' not in direct['receipt']
    assert all(s['timing']['seconds']>=s['timing']['optimizer_seconds']>=0 for s in direct['receipt']['steps'])
    # LoRA's zero B initialization correctly gives zero A gradients at update1.
    first_grads=first['receipt']['steps'][0]['gradients']
    assert all(g['l2']==0 for g in first_grads if g['name'].endswith('.lora_a'))
    assert any(g['l2']>0 for g in first_grads if g['name'].endswith('.lora_b'))
    negatives=0
    for change in [dict(initial_adapter_sha256='e'*64),dict(initial_optimizer_state_sha256='e'*64),dict(example_cursor=0),dict(examples=list(reversed(examples)))]:
        try:execute_native(executable,dict(resumed,**change),root/('refused-'+str(negatives)),timeout=130)
        except ValueError:negatives+=1
        else:raise AssertionError('LoRA state mismatch admitted')
    for optimizer,settings in [('adamw',dict(schema=1)),('muon',dict(schema=1,aux_learning_rate=.001))]:
        full=dict(request,mode='full',steps=3,optimizer=optimizer,optimizer_settings=settings)
        current=execute_native(executable,full,root/('new-'+optimizer),timeout=130)
        legacy=execute_native(old_executable,{k:v for k,v in full.items() if k!='record_timing'},root/('old-'+optimizer),timeout=130)
        compare(current,legacy)
        assert 'training_budget' not in current['receipt']
        assert all('timing' in s for s in current['receipt']['steps'])
        assert all('timing' not in s for s in legacy['receipt']['steps'])
    assert sha256(model)==base
    result=dict(schema=1,status='PASS',device=device,native_adamw_lora=True,
        exact_three_update_uninterrupted_vs_resumed_weights_and_moments=True,
        first_update_zero_A_gradient_expected=True,unchanged_full_adamw_muon_weights_and_moments=True,
        timing_without_training_phase_budget=True,refusal_cases=negatives,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--old-executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='CPU');a=p.parse_args()
    print(json.dumps(run(a.executable,a.old_executable,a.output,a.device)))
