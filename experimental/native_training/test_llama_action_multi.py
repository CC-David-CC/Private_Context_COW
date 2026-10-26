"""Multi-GPU ordinary backprop qualification using generated tiny models only."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from .make_llama_fixture import make_fixture
from .test_llama_profile import run as short_qualification
from .test_llama_action_profile import example
from tools.pcc_native_training import validate_devices

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(executable,output,devices,reference,reference_sha256):
    executable=Path(executable).resolve(strict=True);root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    devices=validate_devices(list(devices))
    if len(devices)<2 or devices!=['CUDA'+str(i)for i in range(len(devices))]:
        raise ValueError('At least two distinctly mapped CUDA devices required')
    reference=Path(reference);assert sha(reference)==reference_sha256
    expected=json.loads(reference.read_bytes())
    assert expected['status']=='GRADIENT_PROBE_PASS'and expected['model_layers']==42
    short=short_qualification(executable,root/'existing-short-qualification',devices)
    assert short['status']=='PASS'
    model=make_fixture(root/'tiny-deep-42-f32.gguf',dtype='f32',layers=42,context_length=8192)
    e=example(1024);e['loss_mask']=[i>=1024-256 for i in range(1024)]
    request=dict(model=str(model),base_sha256=sha(model),mode='full',rank=1,devices=devices,
        sequence_profile='dense-llama-action-8k-v1',memory_profile='allocator-v1',
        device_budget_bytes={d:4<<30 for d in devices},reserve_bytes=128<<20,host_reserve_bytes=128<<20,
        examples=[e],gradient_probe_only=True,candidate=str(root/'never-written.gguf'),deadline_seconds=600)
    path=root/'request-private.json';path.write_text(json.dumps(request,indent=2)+'\n')
    process=subprocess.run([str(executable),'--probe-gradients',str(path)],capture_output=True,text=True,timeout=615)
    (root/'stdout-private.json').write_text(process.stdout);(root/'stderr-private.log').write_text(process.stderr)
    assert process.returncode==0,process.stderr[-2400:]
    actual=json.loads(process.stdout)
    assert actual['status']=='GRADIENT_PROBE_PASS'and actual['backprop_mode']=='standard'
    assert actual['model_layers']==42 and actual['weights_updated']is False and not Path(request['candidate']).exists()
    left={r['name']:r['l2']for r in expected['gradients']};right={r['name']:r['l2']for r in actual['gradients']}
    assert len(left)==len(right)==381 and left.keys()==right.keys()
    assert {r['device']for r in actual['gradients']}==set(devices)
    worst=max(abs(left[n]-right[n])for n in left)
    assert all(np.isclose(left[n],right[n],rtol=1e-4,atol=1e-5)for n in left),worst
    result=dict(schema=1,status='PASS',scope='Generated tiny multi-GPU standard backprop and long/deep gradient agreement; not real model learning',
        devices=devices,existing_short_qualification=short,context_tokens=1024,layers=42,selected_targets=256,
        gradient_tensors=381,maximum_l2_norm_difference_from_single_gpu=worst,
        reference_sha256=reference_sha256,reference_scope='Prior independent single-GPU tiny fixture; device generations can differ',
        allocation_peaks=actual['allocation_peaks'],native_executable_sha256=sha(executable),
        test_source_sha256=sha(__file__),real_model_updates=0,experimental_section_backprop_used=False,promoted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--devices',nargs='+',required=True)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--reference-sha256',required=True)
    args=parser.parse_args();print(json.dumps(run(args.executable,args.output,args.devices,args.reference,args.reference_sha256)))
