"""Run the two uncached batch layouts and compare finite complete traces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from compare_batched_layer_trace import compare
from durable_resume import publish, sha, vector, numerical_record
from context_time_travel.numerics import compare_logits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    profile=json.loads(a.profile.read_text())
    result=dict(status='RUNNING',scope='batches-only activation diagnosis; no restore qualification',
                binary_sha256=sha(a.binary),profile_sha256=sha(a.profile),observations=[])
    histories=[[100+i%17 for i in range(128)]+[117,118,119,120],
               [100+i%17 for i in range(128)]+[117,118,119,120,198]]
    for i,ids in enumerate(histories):
        tokens=a.output/(str(i)+'.tokens');tokens.write_text(','.join(map(str,ids)))
        frontier=a.output/(str(i)+'.f32');trace=a.output/(str(i)+'.trace')
        cmd=list(profile['execution']['oracle_command']);cmd[0]=str(a.binary)
        cmd[cmd.index('--max-new')+1]='1';cmd+=['--tokens-file',str(tokens)]
        env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))}
        env.update(profile['execution']['environment'],STRATA_DUMP_FIRST_LOGITS=str(frontier),
                   COW_BATCH_LAYER_TRACE=str(trace),COW_BATCH_TRACE_POSITION='132')
        env['COW_BATCH_GDN_TRACE']=str(a.output/(str(i)+'.gdn'))
        started=time.monotonic()
        run=subprocess.run(cmd,env=env,capture_output=True,timeout=900)
        (a.output/(str(i)+'.stdout')).write_bytes(run.stdout)
        (a.output/(str(i)+'.stderr')).write_bytes(run.stderr)
        if run.returncode: raise RuntimeError('Layer probe failed: '+str(i))
        match=re.search(rb'batch trace expected_layers=(\d+) width=(\d+)',run.stderr)
        if not match: raise ValueError('Native layer-count witness missing')
        result['observations'].append(dict(command=cmd,work=json.loads(run.stdout),seconds=time.monotonic()-started,
            layers=int(match[1]),width=int(match[2]),trace_sha256=sha(trace),frontier_sha256=sha(frontier)))
        publish(a.output/'result.json',result)
    if result['observations'][0]['layers']!=result['observations'][1]['layers']: raise ValueError('Layer counts differ')
    actual=a.output/'0.f32.decode.f32';reference=a.output/'1.f32'
    unchanged=sha(actual)==sha(a.preflight/'batch-preflight-oracle-0.f32.decode.f32') and \
              sha(reference)==sha(a.preflight/'batch-preflight-oracle-1.f32')
    result.update(status='DIAGNOSED' if unchanged else 'INSTRUMENTATION_CHANGED_OUTPUT',
        instrumentation_preserved_frontiers=unchanged,
        numerical_comparison=numerical_record(compare_logits(vector(actual),vector(reference),atol=0,rtol=0)),
        layer_comparison=compare(a.output/'0.trace',a.output/'1.trace',132,result['observations'][0]['layers']))
    result['gdn_comparison']=compare(a.output/'0.gdn',a.output/'1.gdn',132,1,tuple(range(2,10)))
    images=[a.output/(str(i)+'.trace.input-f16-bf16') for i in range(2)]
    for image,observation in zip(images,result['observations']):
        if image.stat().st_size!=observation['width']*4: raise ValueError('Incomplete projection input image')
    result['projection_input_images']=dict(bit_identical=images[0].read_bytes()==images[1].read_bytes(),
        bytes_per_trace=images[0].stat().st_size,sha256={p.name:sha(p) for p in images})
    publish(a.output/'result.json',result)
    print(json.dumps(dict(status=result['status'],first_difference=result['layer_comparison']['first_difference'])))


if __name__=='__main__': main()
