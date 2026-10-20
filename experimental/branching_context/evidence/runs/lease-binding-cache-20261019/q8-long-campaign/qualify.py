from pathlib import Path
import os,sys,json,time,subprocess,threading,struct,hashlib,traceback
D=Path(__file__).parent
BASE=D.parent
PY='/home/dflanag3/src/Strata/.venv/bin/python'
def write(path,data):
 t=path.with_suffix('.partial');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
def worker(kind):
 runtime=BASE/'lease-binding-cache-q8-20261019-v1/runtime' if kind!='baseline' else BASE/'runtime-grammar-rebuild-v1'
 profile=D/'profile.json' if kind!='baseline' else BASE/'standalone-SC16-q8-worker-profile-dev16-v1/profile.json'
 deps=Path('/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps')
 sys.path[:0]=list(map(str,[runtime,runtime/'tools',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',deps]))
 from context_time_travel.real import BranchingModel
 from context_time_travel.strata_backend import StrataBackend
 from run_q4_publication_fault import _runtime_closure
 config=json.loads(profile.read_bytes());_runtime_closure(config,runtime)
 out=D/kind;out.mkdir(mode=0o700)
 os.environ['CTT_NATIVE_TRACE']=str(out/'native.jsonl')
 result=dict(status='RUNNING',kind=kind,python_pid=os.getpid(),execution_profile_id=config['execution_profile_id'],branches={})
 backend=None;start=time.monotonic()
 try:
  backend=StrataBackend(config,gpu_id=0);result['native_pid']=backend.process.pid
  store=D/('candidate/store' if kind=='restart' else kind+'/store')
  api=BranchingModel(store,backend,config)
  if kind=='restart':
   parent=json.loads((D/'candidate/result.json').read_text())['prefix_snapshot']
   result['restore']=api.restore_checkpoint(parent);assert result['restore']['forward_tokens']==0
  else:
   prefix=json.loads((D/'fixture.json').read_text())['prefix']
   parent=api.create_inputs([dict(kind='text',tokens=prefix)],artifacts=None)
   api.checkpoint(parent)
  result['prefix_snapshot']=parent
  values=api.next_logits(parent);(out/'prefix.f32').write_bytes(struct.pack('<'+'f'*len(values),*values))
  parent_audit=backend.audit(api.cache[parent].blob)
  for label,suffix in [(name+str(i),ids) for name,ids in json.loads((D/'fixture.json').read_text())['suffixes'].items() for i in range(3)]:
   before=backend.work_counters()
   answer=api.run_greedy_request(parent,suffix,128,stream='cache-qualification-'+label)
   after=backend.work_counters()
   assert after['forward_tokens']-before['forward_tokens']==len(suffix)+len(answer['tokens'])
   values=api.next_logits(answer['snapshot']);(out/(label+'.f32')).write_bytes(struct.pack('<'+'f'*len(values),*values))
   controller=api.controller(answer['snapshot'])
   audit=backend.audit(api.cache[parent].blob)
   assert all(audit[k]==parent_audit[k] for k in ['sha256','consumed','bytes'])
   result['branches'][label]=dict(**answer,controller=controller,work_before=before,work_after=after)
   write(out/'result.json',result)
  result['status']='PASS'
 except BaseException as exc:
  result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  if backend is not None:
   pid=backend.process.pid;backend.close();result['native_exit']={'pid':pid,'exit_code':backend.process.returncode,'absent':not Path('/proc/'+str(pid)).exists()}
  result['elapsed_seconds']=time.monotonic()-start;write(out/'result.json',result)
def campaign():
 result={'status':'RUNNING','scope':'Independent-process baseline versus candidate regression, not independent full-history oracle','started':time.time(),'stages':[]}
 write(D/'campaign.json',result)
 for kind in ['baseline','candidate','restart']:
  env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA_','NVIDIA_','STRATA_','GGML_'))};env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
  with (D/(kind+'.stdout')).open('x') as o,(D/(kind+'.stderr')).open('x') as e:
   child=subprocess.Popen([PY,'-B',__file__,kind],env=env,stdout=o,stderr=e)
   started=time.monotonic()
   while True:
    try:code=child.wait(timeout=30);break
    except subprocess.TimeoutExpired:
     write(D/'heartbeat.json',dict(stage=kind,pid=child.pid,elapsed_seconds=time.monotonic()-started,observed_live=child.poll() is None))
   result['stages'].append(dict(kind=kind,pid=child.pid,exit_code=code,absent=not Path('/proc/'+str(child.pid)).exists()));write(D/'campaign.json',result)
   if code:result['status']='FAIL';write(D/'campaign.json',result);return 1
 records={k:json.loads((D/k/'result.json').read_text()) for k in ['baseline','candidate','restart']}
 comparisons=[]
 for kind in ['candidate','restart']:
  for label in ['prefix','A0','A1','A2','B0','B1','B2']:
   a=(D/'baseline'/(label+'.f32')).read_bytes();b=(D/kind/(label+'.f32')).read_bytes();assert a==b,(kind,label)
   comparisons.append(dict(kind=kind,label=label,bytes=len(a),exact=True,max_abs_error=0))
  for label in ['A0','A1','A2','B0','B1','B2']:
   a=records['baseline']['branches'][label];b=records[kind]['branches'][label]
   assert a['tokens']==b['tokens'];assert a['controller']==b['controller']
 result.update(status='PASS',comparisons=comparisons,scope='8K-plus coding-reference prefix, two coding questions, three repetitions each, up to128 generated tokens. Native API timing and exact regression, not autonomous coding completion or full G1-G5.')
 write(D/'campaign.json',result)
 return 0
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1])
 else:raise SystemExit(campaign())
