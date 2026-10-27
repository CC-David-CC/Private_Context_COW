"""Actual generated-model LoRA backprop and completed-state recovery."""
from pathlib import Path
import argparse,json
from tools.pcc_native_training import execute_native,sha256
from tools.test_lora_adamw_native import compare
from experimental.native_training.make_llama_fixture import make_fixture


def run(executable,output,device='CPU',rank=2,devices=None):
    devices=[device] if devices is None else devices
    root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    model=make_fixture(root/'base.gguf');base=sha256(model)
    example=dict(id='owned',tokens=[0,1,2,3],targets=[1,2,3,4],
                 loss_mask=[False,True,True,True],verified=True,provenance_sha256='b'*64)
    common=dict(model=str(model),base_sha256=base,mode='lora',rank=rank,devices=devices,
                sequence_profile='dense-llama-short-v1',memory_profile='allocator-v1',
                device_budget_bytes={d:4<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
                examples=[example],optimizer_state_mode='fresh',learning_rate=.001,
                batch_size=1,steps=3,lora_update_space='factorwise-v1')
    if rank>256:common['lora_rank_profile']='extended-4096-v1'
    rows=[]
    for kind in ('adamw','muon'):
        folder=root/kind;folder.mkdir()
        request=dict(common,optimizer=kind,optimizer_settings=dict(schema=1))
        direct=execute_native(executable,request,folder/'direct',timeout=120)
        first=execute_native(executable,dict(request,steps=1),folder/'first',timeout=120)
        resume=dict(request,steps=2,optimizer_state_mode='resume',
                    initial_adapter=first['candidate_path'],initial_adapter_sha256=first['candidate_sha256'],
                    initial_optimizer_state=first['optimizer_state_path'],
                    initial_optimizer_state_sha256=first['optimizer_state_sha256'],example_cursor=1)
        second=execute_native(executable,resume,folder/'resumed',timeout=120)
        compare(direct,second)
        groups=second['receipt']['optimizer_checkpoint']['groups']
        assert any(g['role']=='hidden_matrix_lora_factor' for g in groups)
        assert all(g['optimizer']==('muon' if kind=='muon' and g['role']=='hidden_matrix_lora_factor' else 'adamw') for g in groups)
        assert second['receipt']['optimizer_checkpoint']['config']['lora_update_space']=='factorwise-v1'
        # A completed state must not cross update spaces, even when the model,
        # adapter, data and optimizer name otherwise match.
        if kind=='adamw':
            incompatible=dict(resume);del incompatible['lora_update_space']
            try:execute_native(executable,incompatible,folder/'wrong-space',timeout=120)
            except RuntimeError:
                assert 'configuration mismatch' in (folder/'wrong-space/stderr.log').read_text(encoding='utf8')
            else:raise AssertionError('Checkpoint crossed update spaces')
            assert not (folder/'wrong-space/candidate.gguf').exists()
        assert sha256(model)==base
        rows.append(dict(optimizer=kind,status='PASS',weights_and_moments_exactly_equal=True))
    result=dict(status='PASS',devices=devices,rank=rank,scope='Generated tiny-model LoRA continuation, not MiniCPM learning',cases=rows)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='CPU')
    p.add_argument('--rank',type=int,default=2)
    p.add_argument('--devices',nargs='+')
    a=p.parse_args();print(json.dumps(run(a.executable,a.output,a.device,a.rank,a.devices)))
