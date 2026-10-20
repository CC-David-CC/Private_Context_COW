from pathlib import Path
import sys,os,json,subprocess,time,secrets,hashlib
b=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1');d=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/standalone-SC01-q8-dynamic-v1');cow=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/runtime-grammar-admission-v1');profile=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1/standalone-SC01-q8-dynamic-v1/profile.json');deps=Path('/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps')
sys.path[:0]=[str(cow),str(cow/'tools'),str(cow/'experimental/branching_context/src'),str(cow/'experimental/branching_context/scripts'),str(deps)]
from run_cow_client_recovery import ProgressTracker,ServiceProcess,_trace_worker
from run_q4_publication_fault import _runtime_closure
p=json.loads(profile.read_bytes());_runtime_closure(p,cow)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
cache=d/'store';cache.mkdir(mode=0o700,exist_ok=False)
trace=d/'native.jsonl';trace.touch(mode=0o600,exist_ok=False)
secret=secrets.token_urlsafe(32)
bootstrap=d/'server_bootstrap.py';bootstrap.write_bytes((d/'bootstrap-template.py').read_bytes())
env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))};env.update(p['execution']['environment'])
env.update(PYTHONDONTWRITEBYTECODE='1',CTT_NATIVE_TRACE=str(trace),PYTHONPATH=os.pathsep.join(map(str,[deps,cow,cow/'tools',cow/'experimental/branching_context/src'])))
from context_time_travel.strata_backend import _validate_execution
_validate_execution(p,environment=env)
argv=['/home/dflanag3/src/Strata/.venv/bin/python','-B',str(bootstrap),str(profile),str(cow),str(d/'grammar.json'),str(deps),'--profile',str(profile),'--store',str(cache),'--host','127.0.0.1','--port','0','--text','--grammar-store',str(d/'grammar-store'),'--api-key='+secret,'--ram-gib','96','--disk-gib','16','--checkpoint-idle-seconds','1800','--pool-trace-output',str(d/'pool.json')]
tracker=ProgressTracker(d,'cross-host-api-server').start();service=ServiceProcess(argv,cwd=cow,env=env,stdout_path=d/'server.stdout',stderr_path=d/'server.stderr',timeout=180,progress=tracker)
result=dict(status='RUNNING',server_host=os.uname().nodename,client_host='llm-49',profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),model_contract_id=p['model_contract_id'],execution_profile_id=p['execution_profile_id'],model_weights_sha256=p.get('model_weights_sha256'),eos_tokens=p['execution']['eos_tokens'],grammar_sha256=hashlib.sha256((d/'grammar.json').read_bytes()).hexdigest(),service_argv_redacted=[v.replace(secret,'<REDACTED>') for v in argv],numerical_oracle='NOT_RUN')
def write():
 (d/'result.partial').write_text(json.dumps(result,indent=2)+'\n');(d/'result.partial').replace(d/'result.json')
error=None
try:
 t=time.monotonic();ready=service.start();result.update(server_pid=service.process.pid,ready=ready,startup_seconds=time.monotonic()-t);write()
 credentials=dict(api_key=secret,port=ready['address'][1],model_contract_id=p['model_contract_id'],execution_profile_id=p['execution_profile_id'],eos_tokens=p['execution']['eos_tokens'])
 with (d/'credentials.json').open('x') as f:os.chmod(f.name,0o600);json.dump(credentials,f)
 started=time.monotonic();last=started
 while not (d/'STOP').exists():
  if service.process.poll() is not None:raise RuntimeError('server exited before requested stop')
  if time.monotonic()-started>3600:raise TimeoutError('bounded cross-host server expired after3600s')
  if time.monotonic()-last>=30:print(json.dumps(dict(phase='await-client-api-run',elapsed_seconds=time.monotonic()-started,server_pid=service.process.pid)),flush=True);last=time.monotonic()
  time.sleep(.2)
 result['status']='CLIENT_FINISHED_SERVER_STOP_REQUESTED'
except BaseException as exc:error=exc;result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
finally:
 try:
  result['stop']=service.stop();result['stop']['command']=[v.replace(secret,'<REDACTED>') for v in result['stop']['command']];result['worker_shutdown']=_trace_worker(d/'pool.json',service.process.pid)
  if result['stop']['exit_code']!=0 or result['worker_shutdown']['native_exit_code']!=0:raise RuntimeError('nonzero server/native shutdown')
 except BaseException as exc:error=error or exc;result.update(status='FAIL',shutdown_error=dict(type=type(exc).__name__,message=str(exc)))
 tracker.finish('PASS' if error is None else 'FAIL',None if error is None else str(error));tracker.close();write()
 (d/'credentials.json').unlink(missing_ok=True)
raise SystemExit(error is not None)
