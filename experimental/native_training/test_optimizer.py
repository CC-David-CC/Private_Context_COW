"""Exercise actual native optimizer tensors against independent FP64 math."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
from .optimizer_reference import verify

def requests(backend):
 rng=np.random.default_rng(20261024)
 shapes=[('blk.0.attn_q.weight',(3,2)),('blk.0.attn_k.weight',(2,3)),('blk.0.attn_v.weight',(3,3)),('blk.0.ffn_up.weight',(4,1)),('blk.0.ffn_down.weight',(1,4)),('blk.0.ffn_gate.weight',(3,2)),('token_embd.weight',(3,4)),('output.weight',(3,4)),('blk.0.attn_norm.weight',(4,1)),('blk.0.mhc_gate.weight',(2,2))]
 params=[dict(name=n,shape=list(s),values=rng.normal(0,.1,np.prod(s)).tolist()) for n,s in shapes]
 gradients=[]
 for i in range(4):
  row={n:rng.normal(0,.2,np.prod(s)).tolist() for n,s in shapes}
  row['blk.0.attn_v.weight']=(np.outer([1.,2.,3.],[.1,.2,.3])*(i+1)).ravel().tolist()
  if i==0:row['blk.0.attn_q.weight']=[0.]*6
  if i==2:row['blk.0.ffn_up.weight']=[0.]*4
  gradients.append(row)
 settings=[('adamw',{}),('adamw',dict(weight_decay=.1,aux_weight_decay=.03)),('muon',dict(nesterov=False)),('muon',dict(nesterov=True)),('muon',dict(scaling='rms',weight_decay=.1,aux_weight_decay=.03)),('muon',dict(projection='fast-refine-v1',refine_start_step=2,refine_steps=5,scaling='rms',weight_decay=.1))]
 for index,(kind,options) in enumerate(settings):yield str(index),dict(optimizer=kind,optimizer_settings=dict(schema=1,**options),learning_rate=.003,backend=backend,parameters=params,gradients=gradients)

def run(executable,output,backend):
 output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False);results=[]
 for label,request in requests(backend):
  path=output/(label+'-request.json');path.write_text(json.dumps(request)+'\n')
  result=subprocess.run([str(executable),'--optimizer-diagnostic',str(path)],capture_output=True,text=True,timeout=120)
  (output/(label+'-stdout.json')).write_text(result.stdout);(output/(label+'-stderr.log')).write_text(result.stderr)
  if result.returncode:raise RuntimeError(f'Native optimizer diagnostic {label} failed: {result.stderr[-1200:]}')
  report=json.loads(result.stdout);proof=verify(request,report);(output/(label+'-independent.json')).write_text(json.dumps(proof,indent=2)+'\n')
  results.append(dict(label=label,optimizer=request['optimizer'],settings=request['optimizer_settings'],status=proof['status'],scalar_comparisons=proof['scalar_comparisons'],max_absolute_error=proof['max_absolute_error'],failures=proof['failures']))
 summary=dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',backend=backend,scope='Native optimizer tiny-tensor mechanics only; not MiniCPM training or learning',cases=results,promoted=False)
 (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',default='CPU');a=p.parse_args()
 result=run(a.executable,a.output,a.backend);print(json.dumps(result));raise SystemExit(result['status']!='PASS')
