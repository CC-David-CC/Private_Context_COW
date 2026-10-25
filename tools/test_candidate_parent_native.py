"""Actual CPU native full/LoRA parents; exact resume, relocation and refusals.

Generated tiny GGUFs prove mechanics only. Identity templates never load a GPU.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, os, subprocess, sys
import numpy as np
from tools.pcc_native_training import execute_native, sha256
from tools.pcc_candidate_parent import resolve_parent
from tools.pcc_cow_profile import bind_dense_profile, bind_dense_adapter
from tools.pcc_training_workspace import copy_checkpoint
from experimental.native_training.make_llama_fixture import make_fixture
from experimental.native_training.test_optimizer_training import state_arrays
from tools.gguf_reader import GGUFFile


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def saved_tensors(path):
    info=GGUFFile(path)
    return {t.name:np.fromfile(path,dtype={'F16':'<f2','F32':'<f4'}[t.type_name],count=t.elements,
                               offset=info.data_start+t.offset) for t in info.tensors}


def run(executable,output):
    root=Path(output).absolute();root.mkdir(mode=0o700,parents=True,exist_ok=False)
    original=make_fixture(root/'original.gguf',dtype='f32',layers=1)
    base=sha256(original)
    examples=[dict(id='one',tokens=[0,1,2],targets=[1,2,3],loss_mask=[False,True,True],verified=True,provenance_sha256='a'*64),
              dict(id='two',tokens=[2,1,0],targets=[1,0,4],loss_mask=[True,False,True],verified=True,provenance_sha256='b'*64)]
    template=dict(backend='llama-dense-llama.cpp',native_build=dict(sha256={},lora_binding='llama-adapter-per-lease-v1'),
        hardware=dict(gpu_uuid='GPU-identity-fixture'),execution=dict(state_format='llama-dense-frontier-v1',kv_dtype='FP32',
        device_binding='cuda-single-device-v1',environment=dict(CUDA_VISIBLE_DEVICES='0'),GPU=[0]))
    profile=root/'profile-identity-only.json';write(profile,template)
    examples_path=root/'examples.json';write(examples_path,examples)
    base_request=dict(model=str(original),base_sha256=base,devices=['CPU'],sequence_profile='dense-llama-short-v1',
        memory_profile='allocator-v1',device_budget_bytes={'CPU':4<<30},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
        examples=examples,optimizer='adamw',optimizer_settings=dict(schema=1),optimizer_state_mode='fresh',
        learning_rate=.001,batch_size=1,steps=1,rank=2,deadline_seconds=60)
    checks=[];artifacts={}
    def refused(label,call):
        try:call()
        except (ValueError,OSError):checks.append(dict(check=label,passed=True))
        else:raise AssertionError('Accepted invalid parent: '+label)
    for mode in ('full','lora'):
        request=dict(base_request,mode=mode)
        first=execute_native(executable,request,root/(mode+'-first'),timeout=75)
        version=Path(first['candidate_path']).parent/'versions'/(first['candidate_sha256']+'.json')
        bound=bind_dense_profile(template,model_path=Path(first['candidate_path']) if mode=='full' else original,
            model_sha256=first['candidate_sha256'] if mode=='full' else base,gpu_index=0,gpu_uuid='GPU-identity-fixture')
        adapters=[]
        if mode=='lora':
            bound=bind_dense_adapter(bound,adapter_path=first['candidate_path'],adapter_sha256=first['candidate_sha256'])
            adapters=[dict(sha256=first['candidate_sha256'],scale=1.)]
        reference=dict(binding_kind='measured-composite',base_model_sha256=first['candidate_sha256'] if mode=='full' else base,
                       candidate_model_contract_id=bound['model_contract_id'],candidate_adapters=adapters)
        args=dict(artifact_roots=[root],state_policy='resume',base_model=original if mode=='lora' else None,examples=examples)
        resolution=resolve_parent(reference,version,profile,**args)
        resumed_request=dict(request,**resolution['native_parent_fields'],**resolution['optimizer_state_fields'])
        resumed=execute_native(executable,resumed_request,root/(mode+'-resumed'),timeout=75)
        straight=execute_native(executable,dict(request,steps=2),root/(mode+'-straight'),timeout=75)
        x=saved_tensors(resumed['candidate_path']);y=saved_tensors(straight['candidate_path'])
        assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
        _,x=state_arrays(resumed['optimizer_state_path']);_,y=state_arrays(straight['optimizer_state_path'])
        assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
        checks.append(dict(check=mode+'-resolved-resume-matches-uninterrupted-weights-and-moments',passed=True))
        # A changed block must deliberately choose fresh moments on learned weights.
        changed=[dict(examples[1],id='new-owned-block')]
        refused(mode+'-changed-practice-cannot-resume',lambda:resolve_parent(reference,version,profile,**dict(args,examples=changed)))
        fresh=resolve_parent(reference,version,profile,**dict(args,state_policy='fresh',examples=None))
        assert fresh['optimizer_state_fields']=={} and fresh['requires_explicit_fresh_optimizer']
        new=execute_native(executable,dict(request,**fresh['native_parent_fields'],examples=changed),root/(mode+'-new-block'),timeout=75)
        assert new['optimizer_completed_steps']==1 and new['receipt']['example_cursor']==1
        if mode=='full':assert new['base_sha256']==first['candidate_sha256']
        else:assert new['receipt']['parent_adapter_sha256']==first['candidate_sha256']
        assert sha256(first['candidate_path'])==first['candidate_sha256']
        checks.append(dict(check=mode+'-new-block-fresh-moments-retains-learned-parent',passed=True))
        wrong=dict(reference,candidate_model_contract_id='0'*64)
        refused(mode+'-foreign-measured-contract',lambda:resolve_parent(wrong,version,profile,**args))
        forged=deepcopy(first);forged['receipt']['example_cursor']+=1
        bad_version=root/(mode+'-wrong-cursor.json');write(bad_version,forged)
        refused(mode+'-state-and-receipt-cursor-must-match',lambda:resolve_parent(reference,bad_version,profile,**args))
        if mode=='lora':
            wrong=dict(reference,candidate_adapters=[dict(sha256=first['candidate_sha256'],scale=2.)])
            refused('scaled-hat-cannot-silently-warm-start-at-one',lambda:resolve_parent(wrong,version,profile,**args))
            wrong=dict(reference,candidate_adapters=[dict(sha256='0'*64,scale=1.)])
            refused('same-base-does-not-identify-another-hat',lambda:resolve_parent(wrong,version,profile,**args))
            refused('missing-base-is-not-inferred-from-a-hat',lambda:resolve_parent(reference,version,profile,**dict(args,base_model=None)))
        checkpoint=root/(mode+'-durable');copy_checkpoint(checkpoint,first)
        verified=checkpoint/'checkpoint-verification.json'
        relocated=resolve_parent(reference,version,profile,**args,checkpoint_file=verified)
        assert relocated['native_parent_fields']['initial_adapter' if mode=='lora' else 'model']==str(checkpoint/'candidate.gguf')
        # Only generated fixture cache files are retired here, never user models.
        Path(first['candidate_path']).unlink();Path(first['optimizer_state_path']).unlink()
        refused(mode+'-retired-original-does-not-guess-another-path',lambda:resolve_parent(reference,version,profile,**args))
        relocated=resolve_parent(reference,version,profile,**args,checkpoint_file=verified)
        assert relocated['optimizer_state_fields']['initial_optimizer_state']==str(checkpoint/'optimizer-state.bin')
        checks.append(dict(check=mode+'-verified-durable-copy-resolves-after-cache-retirement',passed=True))
        # A missing optimizer is permissible only for explicitly fresh moments.
        (checkpoint/'optimizer-state.bin').unlink()
        refused(mode+'-missing-moments-never-silently-reset',lambda:resolve_parent(reference,version,profile,**args,checkpoint_file=verified))
        fresh=resolve_parent(reference,version,profile,**dict(args,state_policy='fresh',examples=None),checkpoint_file=verified)
        assert fresh['optimizer_state_fields']=={}
        checks.append(dict(check=mode+'-explicit-fresh-can-use-retained-weights-without-old-moments',passed=True))
        artifacts[mode]=dict(version=version,reference=reference,checkpoint=verified)
    for label,settings in [('standard',dict(schema=1,aux_learning_rate=.001)),
                           ('rms',dict(schema=1,aux_learning_rate=.001,scaling='rms',weight_decay=.1,aux_weight_decay=.1)),
                           ('refine',dict(schema=1,aux_learning_rate=.001,scaling='rms',weight_decay=.1,aux_weight_decay=.1,
                                          projection='fast-refine-v1',refine_start_step=2,refine_steps=2))]:
        request=dict(base_request,mode='full',optimizer='muon',optimizer_settings=settings)
        first=execute_native(executable,request,root/('muon-'+label+'-first'),timeout=75)
        version=Path(first['candidate_path']).parent/'versions'/(first['candidate_sha256']+'.json')
        bound=bind_dense_profile(template,model_path=Path(first['candidate_path']),model_sha256=first['candidate_sha256'],gpu_index=0,gpu_uuid='GPU-identity-fixture')
        reference=dict(binding_kind='measured-composite',base_model_sha256=first['candidate_sha256'],
                       candidate_model_contract_id=bound['model_contract_id'],candidate_adapters=[])
        resolution=resolve_parent(reference,version,profile,artifact_roots=[root],state_policy='resume',examples=examples)
        resumed=execute_native(executable,dict(request,**resolution['native_parent_fields'],**resolution['optimizer_state_fields']),root/('muon-'+label+'-resumed'),timeout=75)
        straight=execute_native(executable,dict(request,steps=2),root/('muon-'+label+'-straight'),timeout=75)
        x=saved_tensors(resumed['candidate_path']);y=saved_tensors(straight['candidate_path'])
        assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
        _,x=state_arrays(resumed['optimizer_state_path']);_,y=state_arrays(straight['optimizer_state_path'])
        assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
        checks.append(dict(check='muon-'+label+'-resolved-resume-matches-uninterrupted-weights-and-moments',passed=True))
    chosen=artifacts['full'];reference_file=root/'reference.json';write(reference_file,chosen['reference'])
    output_file=root/'cli-resolution-private.json'
    command=[sys.executable,'-m','tools.pcc_candidate_parent','--reference',str(reference_file),'--version',str(chosen['version']),
        '--profile',str(profile),'--artifact-root',str(root),'--checkpoint',str(chosen['checkpoint']),
        '--state-policy','fresh','--output',str(output_file)]
    done=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    assert 'native_parent_fields' not in done.stdout and json.loads(output_file.read_text())['private_paths']
    before=output_file.read_bytes();again=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert again.returncode==2 and output_file.read_bytes()==before
    checks.append(dict(check='actual-cli-writes-private-new-receipt-and-refuses-overwrite',passed=True))
    # An explicit roots capability cannot be expanded using a symlink.
    alias=root/'linked-candidate.gguf';alias.symlink_to(root/'full-durable/candidate.gguf')
    bad=deepcopy(json.loads(chosen['version'].read_text()));bad['candidate_path']=str(alias)
    bad_file=root/'linked-version.json';write(bad_file,bad)
    refused('symlink-candidate-cannot-expand-owner-roots',lambda:resolve_parent(chosen['reference'],bad_file,profile,artifact_roots=[root],state_policy='fresh'))
    assert sha256(original)==base
    result=dict(schema=1,status='PASS',scope='Generated tiny CPU native parent identity, continuation and refusal mechanics; no model quality or GPU qualification',
        checks=checks,passed=len(checks),executable_sha256=sha256(executable),base_unchanged=True,gpu_used=False,promoted=False)
    write(root/'result.json',result)
    print(json.dumps(dict(status='PASS',passed=len(checks),gpu_used=False)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.executable,args.output)
