"""Compare retained native logits to a lazy independent full-model FP64 forward.

No model update, PyTorch, native cached activations or protected test solution.
Run the native trainer's evaluate-only mode first and retain its exact request.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from .llama_reference import read_lazy_gguf,model_value

def run(model,requests,native_logits,output):
 if len(requests)!=len(native_logits) or not requests:raise ValueError('Paired request/logits artifacts required')
 output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
 with Path(model).open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
 weights,h=read_lazy_gguf(model);start=time.monotonic();checks=[]
 # These bounds are declared before comparing the actual model, not fit to it.
 atol,rms_limit=2e-4,5e-5
 for index,(request_path,logits_path) in enumerate(zip(requests,native_logits)):
  request=json.loads(Path(request_path).read_text())
  if request['base_sha256']!=identity or request.get('evaluate_only') is not True or len(request['examples'])!=1:raise ValueError('Pinned single-example evaluate-only request required')
  example=request['examples'][0]
  if not 2<=len(example['tokens'])<=512:raise ValueError('Bounded independent forward required')
  loss,logits=model_value(example,weights,h)
  actual=np.fromfile(logits_path,dtype='<f4').reshape(len(example['tokens']),h['vocab'] if 'vocab' in h else logits.shape[0]).T
  if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(actual)):raise ValueError('Nonfinite forward result')
  delta=actual[:,-1]-logits[:,-1];whole=actual-logits
  row=dict(case=index,tokens=len(example['tokens']),vocabulary=logits.shape[0],max_absolute_error=float(np.max(np.abs(delta))),rms_error=float(np.sqrt(np.mean(delta**2))),all_positions_max_absolute_error=float(np.max(np.abs(whole))),all_positions_rms_error=float(np.sqrt(np.mean(whole**2))),same_top_token=bool(np.argmax(actual[:,-1])==np.argmax(logits[:,-1])),fp64_loss=loss)
  row['passed']=row['max_absolute_error']<=atol and row['rms_error']<=rms_limit and row['same_top_token'] and row['all_positions_max_absolute_error']<=atol and row['all_positions_rms_error']<=rms_limit;checks.append(row)
  (output/(str(index)+'-logits.f64')).write_bytes(logits[:,-1].astype('<f8').tobytes())
 result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',scope='Actual ordinary-Llama short FP64 forward versus retained native logits; not learning',model_sha256=identity,max_absolute_tolerance=atol,rms_tolerance=rms_limit,checks=checks,seconds=time.monotonic()-start,weights_updated=False,promoted=False)
 (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--requests',type=Path,nargs='+',required=True);p.add_argument('--native-logits',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.model,a.requests,a.native_logits,a.output);print(json.dumps(r));raise SystemExit(r['status']!='PASS')
