"""Independent FP64 ordinary-Llama forward and finite differences; NumPy only.

Reads exported fixture values or floating-point GGUF bytes, never native cached
activations or native differentiation code. No PyTorch or GGML imports.
"""
from pathlib import Path
import argparse,json,struct
import numpy as np

ATOL=5e-5
RTOL=5e-4

def norm(x,w,eps):return x/np.sqrt(np.mean(x*x,axis=0,keepdims=True)+eps)*w.reshape(-1,1)
def layer(x,weights,index,h):
 p='blk.'+str(index)+'.';n=x.shape[1];d=h['head'];heads=h['heads'];kh=h['kvheads']
 def w(name):return weights[p+name]
 z=norm(x,w('attn_norm.weight'),h['eps'])
 q=(w('attn_q.weight')@z).reshape(heads,d,n)
 k=(w('attn_k.weight')@z).reshape(kh,d,n)
 v=(w('attn_v.weight')@z).reshape(kh,d,n)
 positions=np.arange(n,dtype=np.float64)
 for offset in range(0,h['rotary'],2):
  angle=positions/(h['freq']**(offset/h['rotary']));co=np.cos(angle);si=np.sin(angle)
  for value in [q,k]:
   a=value[:,offset,:].copy();b=value[:,offset+1,:].copy()
   value[:,offset,:]=a*co-b*si;value[:,offset+1,:]=a*si+b*co
 out=np.zeros((heads,d,n),dtype=np.float64)
 for head in range(heads):
  key_head=head//(heads//kh)
  for t in range(n):
   scores=k[key_head,:,:t+1].T@q[head,:,t]/np.sqrt(d)
   prob=np.exp(scores-np.max(scores));prob/=prob.sum()
   out[head,:,t]=v[key_head,:,:t+1]@prob
 residual=x+w('attn_output.weight')@out.reshape(heads*d,n)
 z=norm(residual,w('ffn_norm.weight'),h['eps'])
 gate=w('ffn_gate.weight')@z;silu=gate/(1+np.exp(-gate))
 return residual+w('ffn_down.weight')@((w('ffn_up.weight')@z)*silu)

def model_value(example,weights,h):
 x=weights['token_embd.weight'][example['tokens']].T
 for i in range(h['layers']):x=layer(x,weights,i,h)
 logits=weights['output.weight']@norm(x,weights['output_norm.weight'],h['eps'])
 shifted=logits-np.max(logits,axis=0,keepdims=True)
 logs=shifted-np.log(np.sum(np.exp(shifted),axis=0,keepdims=True))
 losses=[-logs[t,i] for i,t in enumerate(example['targets']) if example['loss_mask'][i]]
 return float(np.mean(losses)),logits

def derivatives(weights,native,value):
 worst=0.;sensitivity=0.;scalars=0;failures=[]
 for name,values in weights.items():
  estimates=[]
  for step in [1e-6,2e-6]:
   gradient=np.zeros_like(values)
   for index in np.ndindex(values.shape):
    saved=values[index]
    values[index]=saved+step;plus=value()
    values[index]=saved-step;minus=value()
    values[index]=saved;gradient[index]=(plus-minus)/(2*step)
   estimates.append(gradient)
  error=float(np.max(np.abs(estimates[0]-native[name])))
  delta=float(np.max(np.abs(estimates[0]-estimates[1])))
  worst=max(worst,error);sensitivity=max(sensitivity,delta);scalars+=values.size
  if not np.allclose(estimates[0],native[name],rtol=RTOL,atol=ATOL):failures.append(dict(tensor=name,max_absolute_error=error))
  if not np.allclose(estimates[0],estimates[1],rtol=1e-5,atol=1e-7):failures.append(dict(tensor=name,finite_difference_sensitivity=delta))
 return dict(scalar_gradients=scalars,max_absolute_gradient_error=worst,step_sensitivity=sensitivity,failures=failures)

def arrays(document,field):
 return {name:np.array(data[field],dtype=np.float64).reshape(data['shape'][1],data['shape'][0]) for name,data in document.items()}

def verify_chain(report):
 cases=[]
 for case in report['reports']:
  f=case['fixture'];parameters=arrays(f['parameters'],'values');native=arrays(f['parameters'],'gradient')
  if case['mode']=='full':
   def effective():return parameters
  elif case['mode']=='lora':
   base=arrays(f['base_parameters'],'values')
   def effective():
    return {n:w+parameters[n+'.lora_b']@parameters[n+'.lora_a'] if n+'.lora_a' in parameters else w for n,w in base.items()}
  else:raise ValueError('Explicit full/LoRA fixture mode required')
  value=lambda:model_value(f['example'],effective(),f['geometry'])[0]
  loss,logits=model_value(f['example'],effective(),f['geometry'])
  observed=np.array(f['logits'],dtype=np.float64).reshape(logits.shape[1],logits.shape[0]).T
  result=derivatives(parameters,native,value)
  result.update(mode=case['mode'],fp64_loss=loss,native_loss=case['before'],max_absolute_logit_error=float(np.max(np.abs(logits-observed))))
  if not np.isclose(loss,case['before'],rtol=RTOL,atol=5e-6):result['failures'].append(dict(forward_loss_disagreement=True))
  if not np.allclose(logits,observed,rtol=RTOL,atol=ATOL):result['failures'].append(dict(forward_logit_disagreement=True))
  cases.append(result)
 if len(cases)!=2 or {c['mode'] for c in cases}!={'full','lora'}:raise ValueError('Complete full-weight and LoRA fixtures required')
 return dict(status='PASS' if all(not c['failures'] for c in cases) else 'FAIL',scope='All scalar derivatives of tiny complete ordinary Llama full-weight and LoRA graphs',checks=cases)

def verify_layers(report):
 cases=[]
 for case in report['checks']:
  f=case['fixture'];weights=arrays(f['weights'],'values')
  native={entry['name']:np.array(entry['gradient'],dtype=np.float64).reshape(weights[entry['name']].shape) for entry in case['directions']}
  x=np.array(f['input'],dtype=np.float64).reshape(f['tokens'],-1).T
  up=np.array(f['upstream'],dtype=np.float64).reshape(x.shape[1],x.shape[0]).T
  value=lambda:float(np.sum(layer(x,weights,case['layer'],f['geometry'])*up))
  result=derivatives(weights,native,value);result.update(layer=case['layer'],fp64_value=value(),native_value=case['baseline'])
  if not np.isclose(value(),case['baseline'],rtol=RTOL,atol=5e-6):result['failures'].append(dict(forward_disagreement=True))
  cases.append(result)
 return dict(status='PASS' if cases and all(not c['failures'] for c in cases) else 'FAIL',scope='Independent tiny ordinary Llama layer scalar derivatives',checks=cases)

def read_gguf(path):
 from tools.gguf_reader import GGUFFile
 path=Path(path);file=GGUFFile(path);md=file.metadata;weights={}
 if md['general.architecture']!='llama':raise ValueError('Ordinary Llama source required')
 with path.open('rb') as source:
  for t in file.tensors:
   if t.type_name not in ['F16','F32']:raise ValueError('Independent tiny oracle admits F16/F32 only')
   source.seek(file.data_start+t.offset);raw=source.read(t.expected_bytes())
   values=struct.unpack('<'+('e' if t.type_name=='F16' else 'f')*t.elements,raw)
   weights[t.name]=np.array(values,dtype=np.float64).reshape(t.shape[1] if len(t.shape)>1 else 1,t.shape[0])
 h={key:md['llama.'+field] for key,field in [('hidden','embedding_length'),('layers','block_count'),('head','attention.key_length'),('heads','attention.head_count'),('kvheads','attention.head_count_kv'),('rotary','rope.dimension_count'),('freq','rope.freq_base'),('eps','attention.layer_norm_rms_epsilon')]}
 return weights,h

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['layers','chain']);p.add_argument('report',type=Path);a=p.parse_args()
 result=(verify_layers if a.kind=='layers' else verify_chain)(json.loads(a.report.read_text()))
 print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
