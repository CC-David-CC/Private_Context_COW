"""Fingerprint a bounded MiniCPM dense profile over the shared native runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parents[1])]
from context_time_travel.llama_dense_backend import validate_geometry,terminal_tokens
from tools.gguf_reader import GGUFFile

def make_profile(model,build_dir,output,capacity,batch,threads,gpus=None,lora=None,lora_scale=1.,math_profile='native'):
    if (type(capacity) is not int or not 128<=capacity<=131072 or type(batch) is not int
            or not 1<=batch<=capacity or type(threads) is not int or threads<1):
        raise ValueError('Explicit bounded context, prefill and CPU thread count required')
    model=Path(model).resolve(strict=True);build_dir=Path(build_dir).resolve(strict=True)
    md=GGUFFile(model).metadata;geometry=validate_geometry(md)
    build=json.loads((build_dir/'build.json').read_text())
    if build.get('architecture_contract')!='llama-dense-v1':raise ValueError('Explicit dense Llama native build required')
    for path,expected in build['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Build changed: '+path)
    cpu=build.get('device_binding')=='cpu-only-v1'
    if cpu:
        if gpus is not None:raise ValueError('CPU build cannot advertise selected CUDA devices')
        workers=[0];environment={'CUDA_VISIBLE_DEVICES':''}
        hardware=dict(host=os.uname().nodename,backend='CPU',machine=os.uname().machine)
    else:
        from context_time_travel.pools import normalize_gpu_selection
        if gpus is None:raise ValueError('Explicit physical CUDA workers required')
        workers=list(normalize_gpu_selection(gpus))
        if any(type(v) is not int for v in workers):raise ValueError('Dense grouped checkpoints are not qualified')
        environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(map(str,workers)))
        hardware=dict(host=os.uname().nodename,gpu=subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip())
    with model.open('rb') as f:weights=hashlib.file_digest(f,'sha256').hexdigest()
    eos=md.get('tokenizer.ggml.eos_token_id');vocab=len(md['tokenizer.ggml.tokens'])
    if type(eos) is not int or not 0<=eos<vocab or vocab!=130560:raise ValueError('Qualified MiniCPM vocabulary required')
    execution=dict(environment=environment,GPU=workers,eos_tokens=terminal_tokens(md),text_template_kwargs={'enable_thinking':False},
        generated_operator='llama-forward-consumed-v1',selected_token_protocol='consumed-selected-receipt-v1',
        sampling_protocol='full-vocab-min-p-u53-v2',sampling_location='native-cpu-frontier',
        capacity=capacity,prefill_chunk=batch,threads=threads,kv_dtype='FP16',flash_attention=True,
        state_format='llama-dense-frontier-v1',resume_contract='identical-native-batches-v1',
        cuda_cublas_compute_type=os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE','auto'),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE','unset'),physical_sharing=False,embedding_width=2048)
    if not cpu:execution['device_binding']='cuda-single-device-v1'
    if math_profile not in ('native','f32-activations','f32-reference'):raise ValueError('Unsupported math profile')
    if math_profile!='native':
        if not build.get('f32_activations_patch_sha256'):raise ValueError('Explicit FP32 activation build required')
        execution.update(f32_activations=True,cuda_cublas_compute_type='f32',nvidia_tf32_override='0')
        environment.update(GGML_CUDA_F32_ACTIVATIONS='1',GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',NVIDIA_TF32_OVERRIDE='0')
    if math_profile=='f32-reference':
        execution.update(f32_attention=True,kv_dtype='FP32',flash_attention=False)
        environment.update(COW_QWEN35_F32_ATTENTION='1',GGML_CUDA_DISABLE_FUSION='1')
    runtime_root=ROOT.parents[1]
    prefixes=['serve','tools','experimental/branching_context/src','experimental/branching_context/scripts']
    closure={str(f.relative_to(runtime_root)):hashlib.sha256(f.read_bytes()).hexdigest()
        for prefix in prefixes for f in (runtime_root/prefix).rglob('*.py') if '__pycache__' not in f.parts}
    build['selected_runtime_source_closure']=dict(source_commit=subprocess.check_output(
        ['git','-C',str(runtime_root),'rev-parse','HEAD'],text=True).strip(),files_sha256=closure)
    contract=dict(weights_sha256=weights,architecture='llama',state_kind='llama-dense-frontier-v1',
        positions='absolute-zero-based',kv_dtype=execution['kv_dtype'],input_kind='token_ids')
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    profile=dict(backend='llama-dense-llama.cpp',model_path=str(model),model_weights_sha256=weights,
        model_contract_id=digest(contract),execution_profile_id=digest(dict(build=build,execution=execution,hardware=hardware)),
        native_build=build,execution=execution,hardware=hardware,geometry=geometry,
        bridge_library=str((build_dir/'libllamadensectt.so').resolve()),vocab_size=vocab,
        oracle_binary=str((build_dir/'llama-dense-fresh-oracle').resolve()))
    if lora is not None:
        from tools.pcc_cow_profile import bind_dense_adapter
        path=Path(lora).absolute()
        with path.open('rb') as f:adapter_sha=hashlib.file_digest(f,'sha256').hexdigest()
        profile=bind_dense_adapter(profile,adapter_path=path,adapter_sha256=adapter_sha,scale=lora_scale)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(profile,f,indent=2);f.write('\n')
    output.chmod(0o600);return profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=16384)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--math-profile',default='native')
    p.add_argument('--lora',type=Path);p.add_argument('--lora-scale',type=float,default=1.)
    a=p.parse_args();make_profile(a.model,a.build,a.output,a.context,a.batch,a.threads,gpus=a.gpus,
        lora=a.lora,lora_scale=a.lora_scale,math_profile=a.math_profile)
    print(a.output)

if __name__=='__main__':main()
