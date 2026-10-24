"""Owner-operated immutable GGUF conversion through the pinned native quantizer.

No training, model upload or remote-client executable grants. The operator owns
resource admission; every output is a separate weight/storage identity.
"""
from pathlib import Path
import argparse,collections,hashlib,json,os,subprocess,time
from tools.gguf_reader import GGUFFile

FORMATS=('F32','F16','Q8_0','Q5_K_M','Q4_K_M','Q3_K_M','Q2_K','Q1_0')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def convert(executable,source,source_sha256,output,format,*,threads=4,timeout=1800):
 if format not in FORMATS or type(threads) is not int or not 1<=threads<=64 or type(timeout) is not int or not 1<=timeout<=7200:raise ValueError('Explicit supported conversion and bounded CPU resources required')
 executable=Path(executable).resolve(strict=True);source=Path(source).resolve(strict=True)
 if sha(source)!=source_sha256:raise ValueError('Pinned source model differs')
 before=GGUFFile(source)
 if any(t.type_name not in ('F16','F32') for t in before.tensors):raise ValueError('Convert directly from floating-point master, not another lossy quant')
 root=Path(output);root.mkdir(mode=0o700,parents=True,exist_ok=False);target=root/('model-'+format+'.gguf')
 argv=[str(executable),str(source),str(target),format,str(threads)]
 write(root/'intent.json',dict(schema=1,source_sha256=source_sha256,quantizer_sha256=sha(executable),wrapper_sha256=sha(__file__),requested_format=format,threads=threads,timeout_seconds=timeout,command=argv,weights_published=False))
 start=time.monotonic()
 with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
  child=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  stat=Path('/proc')/str(child.pid)/'stat';write(root/'process.json',dict(pid=child.pid,start_ticks=stat.read_text().split()[21] if stat.exists() else None))
  try:code=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:child.kill();child.wait();write(root/'exit.json',dict(status='TIMEOUT',completion_unknown=True));raise
 write(root/'exit.json',dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,conversion_wall_seconds=time.monotonic()-start))
 if code:raise RuntimeError('Native conversion failed; preserve original run, no automatic retry')
 after=GGUFFile(target)
 if len(before.tensors)!=len(after.tensors) or len({t.name for t in before.tensors})!=len(before.tensors) or len({t.name for t in after.tensors})!=len(after.tensors) or {t.name: t.shape for t in before.tensors}!={t.name: t.shape for t in after.tensors}:raise ValueError('Conversion changed tensor names/shapes')
 before_contract={k:v for k,v in before.metadata.items() if k.startswith('tokenizer.') or k.startswith(before.metadata['general.architecture']+'.') or k=='general.architecture'}
 if before_contract!={k:after.metadata.get(k) for k in before_contract}:raise ValueError('Conversion changed model/tokenizer architecture contract')
 if sha(source)!=source_sha256:raise ValueError('Source changed during conversion')
 result=dict(schema=1,status='CONVERTED_NOT_EVALUATED',source_sha256=source_sha256,model_sha256=sha(target),requested_format=format,bytes=target.stat().st_size,tensor_types=after.by_type(),tensor_type_ids=dict(collections.Counter(str(t.type_id) for t in after.tensors)),tensors=len(after.tensors),conversion_wall_seconds=time.monotonic()-start,behavior_measured=False,weights_published=False)
 write(root/'result.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('executable','source','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source-sha256',required=True);p.add_argument('--format',choices=FORMATS,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--timeout-seconds',type=int,default=1800)
 a=p.parse_args();print(json.dumps(convert(a.executable,a.source,a.source_sha256,a.output,a.format,threads=a.threads,timeout=a.timeout_seconds)))
