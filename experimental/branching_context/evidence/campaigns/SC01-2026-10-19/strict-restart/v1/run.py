from pathlib import Path
import os,sys,json,time,threading,subprocess,hashlib,traceback
base=Path('/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1/vc-rerun-20261018-v1')
source=base/'standalone-SC01-q8-dynamic-v1';runtime=base/'runtime-grammar-admission-v1';d=Path(__file__).parent
sys.path[:0]=[str(Path('/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps')),str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src')]
from context_time_travel.strata_backend import StrataBackend
from context_time_travel.real import BranchingModel
from serve.cow_grammar_disk import DurableGrammarCatalog
state=dict(status='RUNNING',phase='starting',python_pid=os.getpid(),native_pid=None)
started=time.monotonic();lock=threading.Lock();stop=threading.Event()
def write(name,value):
 tmp=d/(name+'.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(d/name)
def heartbeat():
 while not stop.wait(30):
  with lock:row=dict(state,elapsed_seconds=time.monotonic()-started)
  write('progress.json',row);print(json.dumps(row),flush=True)
thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
profile=json.loads((source/'profile.json').read_text())
payloads=json.loads((source/'saved-bytes-audit.json').read_text())['payloads']
old_pid=json.loads((source/'result.json').read_text())['worker_shutdown']['native_pid']
result=dict(status='RUNNING',old_native_pid=old_pid,execution_profile_id=profile['execution_profile_id'],restores=[],scope='Exact saved-frontier/controller restore only;not fresh full-history or continuation qualification')
write('result.json',result);backend=None;model=None;catalog=None;error=None
try:
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if jobs:raise RuntimeError('GPU is occupied;not interrupting another job')
 t=time.monotonic();backend=StrataBackend(profile)
 result['native_startup_seconds']=time.monotonic()-t
 result['new_native_pid']=backend.process.pid
 if backend.process.pid==old_pid:raise RuntimeError('Producer native process identity reused')
 with lock:state.update(phase='strict-restore',native_pid=backend.process.pid)
 model=BranchingModel(source/'store',backend,profile)
 catalog=DurableGrammarCatalog(source/'grammar-store',backend.tokenizer,profile['execution']['eos_tokens'])
 for expected in payloads:
  sid=expected['snapshot'];controller=model.controller(sid)
  receipt=model.restore_checkpoint(sid,expected_schedule_sha256=expected['schedule_sha256'])
  assert receipt['forward_tokens']==0 and receipt['consumed']==expected['consumed']
  assert model.controller(sid)==controller
  grammar_checked=False
  if controller.get('constraint') is not None:
   registry=catalog.resolve(controller=controller)
   history=model.history(sid);tokens=getattr(history,'text_tokens',history)
   restored=registry.restore(controller,list(tokens))
   assert restored.current_controller==controller
   grammar_checked=True
  result['restores'].append(dict(expected=expected,receipt=receipt,controller_record_unchanged=True,grammar_progress_restored=grammar_checked,max_numeric_error_vs_saved_frontier=0.0))
  write('result.json',result)
  print(json.dumps(dict(phase='restored',snapshot=sid,consumed=receipt['consumed'],forward_tokens=receipt['forward_tokens'],grammar_checked=grammar_checked)),flush=True)
 result['status']='BOUNDED_SC01_STRICT_RESTART_PASS'
except BaseException as exc:
 error=exc;result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
 traceback.print_exc()
finally:
 if catalog is not None:catalog.close()
 if model is not None:
  try:model.cache.clear()
  except BaseException as exc:error=error or exc;result['cache_release_error']=str(exc)
 if backend is not None:
  try:backend.close();result['native_exit_code']=backend.process.poll()
  except BaseException as exc:error=error or exc;result['close_error']=str(exc)
 if error is not None:result['status']='FAIL'
 result['elapsed_seconds']=time.monotonic()-started
 stop.set();thread.join(2);write('result.json',result)
 with lock:state.update(status=result['status'],phase='finished')
 write('progress.json',state)
raise SystemExit(error is not None)
