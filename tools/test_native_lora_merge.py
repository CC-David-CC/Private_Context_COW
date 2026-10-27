"""Independently check native F32 merged weights and effective LoRA behavior."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from tools.pcc_native_training import execute_native,sha256
from tools.gguf_reader import GGUFFile
from experimental.native_training.make_llama_fixture import make_fixture

def run(executable,output,device='CPU',optimizer='adamw',factorwise=False):
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf');base_hash=sha256(model)
    example=dict(id='owned',tokens=[0,1,2,3],targets=[1,2,3,4],
        loss_mask=[False,True,True,True],verified=True,provenance_sha256='b'*64)
    common=dict(model=str(model),base_sha256=base_hash,mode='lora',rank=2,
        devices=[device],sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',
        device_budget_bytes={device:4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
        examples=[example],deadline_seconds=120)
    training=dict(common,optimizer=optimizer,optimizer_settings=dict(schema=1),
        optimizer_state_mode='fresh',learning_rate=.001,batch_size=1,steps=3)
    if factorwise:training['lora_update_space']='factorwise-v1'
    trained=execute_native(executable,training,
        root/'training',timeout=130)
    merged=root/'merged-F32.gguf'
    request=dict(common,initial_adapter=trained['candidate_path'],
        initial_adapter_sha256=trained['candidate_sha256'],candidate=str(merged))
    def native(name,config,flag='--train',success=True):
        path=root/(name+'-request-private.json');path.write_text(json.dumps(config))
        with (root/(name+'-stdout.json')).open('x') as out,(root/(name+'-stderr.log')).open('x') as err:
            result=subprocess.run([str(executable),flag,str(path)],stdout=out,stderr=err,timeout=130)
        assert (result.returncode==0)==success,(name,(root/(name+'-stderr.log')).read_text())
        return json.loads((root/(name+'-stdout.json')).read_text()) if success else None
    receipt=native('merge',request,'--merge-adapter')
    assert receipt['candidate_sha256']==sha256(merged) and receipt['merged_matrices']==15
    assert all(t.type_name=='F32' for t in GGUFFile(merged).tensors)
    def arrays(path):
        info=GGUFFile(path)
        return {t.name:np.memmap(path,dtype={'F16':'<f2','F32':'<f4'}[t.type_name],mode='r',
            offset=info.data_start+t.offset,shape=(t.elements,)).astype(np.float64).reshape(tuple(reversed(t.shape))) for t in info.tensors}
    base=arrays(model);adapter=arrays(trained['candidate_path']);actual=arrays(merged);worst=0
    assert base.keys()==actual.keys()
    for name,w in base.items():
        a=adapter.get(name+'.lora_a')
        expected=w if a is None else w+adapter[name+'.lora_b']@a
        error=float(np.max(np.abs(expected-actual[name])));worst=max(worst,error)
        assert error<2e-7,(name,error)
    losses=[];logits=[]
    for kind in ['adapter','merged']:
        config=dict(request,evaluate_only=True,logits_output=str(root/(kind+'-logits.bin')))
        if kind=='merged':
            config.update(model=str(merged),base_sha256=sha256(merged),mode='full')
            config.pop('initial_adapter');config.pop('initial_adapter_sha256')
        result=native(kind,config);losses.append(result['losses'][0])
        logits.append(np.fromfile(root/(kind+'-logits.bin'),dtype='<f4'))
    error=float(np.max(np.abs(logits[0]-logits[1])))
    assert abs(losses[0]-losses[1])<1e-6 and error<2e-6
    for index,change in enumerate([dict(base_sha256='e'*64),dict(initial_adapter_sha256='e'*64),{}]):
        native('refused-'+str(index),dict(request,**change),'--merge-adapter',success=False)
    assert sha256(model)==base_hash and sha256(trained['candidate_path'])==trained['candidate_sha256']
    result=dict(schema=1,status='PASS',scope='Tiny native merge mechanics, not real-model LoRA qualification',
        device=device,optimizer=optimizer,factorwise=factorwise,native_f32_merge=True,merged_tensors=len(actual),
        independent_FP64_weight_max_error=worst,adapter_vs_merged_loss_error=abs(losses[0]-losses[1]),
        adapter_vs_merged_logit_max_error=error,actual_tensor_storage='F32',refusal_cases=3,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='CPU')
    p.add_argument('--optimizer',choices=['adamw','muon'],default='adamw');p.add_argument('--factorwise',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.executable,a.output,a.device,a.optimizer,a.factorwise)))
