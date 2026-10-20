"""Cross-host CLI audit through real authenticated HTTP.

Text fixtures use an outer client filesystem namespace without native files or
GPU devices. Browser fixtures use the normal trusted host CLI with Python
import/file/process audits; registered browser work retains its own OS sandbox.
The outer browser-client namespace remains unsupported on this host. No model
or tool choices are supplied. Audit hooks are not adversarial containment.
"""
import argparse,hashlib,json,os,shlex,signal,socket,subprocess,sys,time
from pathlib import Path
from fault_proxy import LostAckProxy
from branchwrite.cow_client import OperationJournal

GUARD=r"""import atexit,hashlib,importlib.abc,json,os,sys
from pathlib import Path
blocked={'context_time_travel','serve','torch','llama_cpp'}
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked:raise ImportError('server/native implementation forbidden on client: '+fullname)
sys.meta_path.insert(0,NoNative())
for path in ['/home/dflanag3/models','/home/dflanag3/src/Strata','/dev/nvidia0','/dev/dri']:
 if Path(path).exists():os._exit(78)
def done():
 result=dict(schema=1,namespace_hostname=os.uname().nodename,python=sys.version.split()[0],native_modules_loaded=sorted(blocked.intersection(sys.modules)),gpu_devices_visible=[str(p) for p in Path('/dev').glob('nvidia*')],server_and_model_paths_absent=True,branchwrite_modules={k:str(getattr(v,'__file__','')) for k,v in sys.modules.items() if k=='branchwrite' or k.startswith('branchwrite.')})
 Path(os.environ['BW_CLIENT_PROOF']).write_text(json.dumps(result,indent=2)+'\n')
atexit.register(done)
"""

HOST_GUARD=r"""import atexit,importlib.abc,json,os,sys
from pathlib import Path
blocked={'context_time_travel','serve','torch','llama_cpp'}
blocked_paths=('/home/dflanag3/models','/home/dflanag3/src/Strata','/home/dflanag3/spatial-artifacts/cow-vision-20261016-v1','/dev/nvidia','/dev/dri','/home/dflanag3/src/branchwrite-shell-admission-20261019','/home/dflanag3/src/branchwrite-real-repository-20261019','/home/dflanag3/src/branchwrite-prefix-admission-20261019','/home/dflanag3/src/branchwrite-project-admission-20261019','/home/dflanag3/src/branchwrite-inline-task-20261019','/home/dflanag3/src/branchwrite-model-reconciliation-20261019','/home/dflanag3/src/branchwrite-client-package-20261019','/home/dflanag3/src/branchwrite-coding-entrypoint-20261019','/home/dflanag3/src/branchwrite-grammar-client-20261019')
violations=[]
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked:
   violations.append(dict(kind='import',name=fullname));raise ImportError('server/native implementation forbidden on client: '+fullname)
sys.meta_path.insert(0,NoNative())
def audit(event,args):
 if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
  path=os.path.abspath(os.fsdecode(args[0]))
  if any(path==prefix or path.startswith(prefix+'/') or (prefix=='/dev/nvidia' and path.startswith(prefix)) for prefix in blocked_paths):
   violations.append(dict(kind='file',path=path));raise PermissionError('model/server/GPU file access forbidden on client')
 if event=='subprocess.Popen':
  path=str(Path(os.fsdecode(args[0])).resolve())
  if path not in {str(Path(sys.executable).resolve()),'/usr/bin/prlimit','/usr/bin/bwrap'}:
   violations.append(dict(kind='process',path=path));raise PermissionError('unregistered client subprocess executable')
sys.addaudithook(audit)
def done():
 result=dict(schema=1,layout='trusted-host-cli-with-audit-and-isolated-tools',hostname=os.uname().nodename,python=sys.version.split()[0],native_modules_loaded=sorted(blocked.intersection(sys.modules)),native_file_and_process_audit=True,violations=violations,branchwrite_modules={k:str(getattr(v,'__file__','')) for k,v in sys.modules.items() if k=='branchwrite' or k.startswith('branchwrite.')},sys_path=list(sys.path),scope='Python audit of trusted CLI; not an adversarial OS security proof. Browser/candidate execution retains its separate OS sandbox.')
 Path(os.environ['BW_CLIENT_PROOF']).write_text(json.dumps(result,indent=2)+'\n')
atexit.register(done)
"""

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['checkout','fixture','state','credentials','ssh-key','known-hosts','output']:ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--server-host',default='llm-60');args=ap.parse_args()
 checkout=args.checkout.resolve(strict=True);fixture=json.loads(args.fixture.read_bytes());credentials=json.loads(args.credentials.read_bytes())
 args.output.mkdir(mode=0o700,exist_ok=False);out=args.output.resolve();guard=out/'guard';guard.mkdir(mode=0o700);host_tools_task=('browser' in fixture or 'shell' in fixture or fixture['grant']['allow_tests']);(guard/'sitecustomize.py').write_text(HOST_GUARD if host_tools_task else GUARD)
 result=dict(schema=1,status='RUNNING',client_host=os.uname().nodename,server_host=args.server_host,client_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=checkout,text=True,capture_output=True,check=True).stdout.strip(),source_sha256={str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkout/'branchwrite').glob('*.py')},fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),python=sys.version.split()[0],numerical_oracle='NOT_RUN',sampler=fixture.get('profiles',{}).get('control',{}).get('settings',{}),thinking_budget=fixture['thinking_tokens'])
 def write():
  q=out/'result.partial';q.write_text(json.dumps(result,indent=2)+'\n');q.replace(out/'result.json')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));local_port=sock.getsockname()[1]
 forward=['ssh','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=bw-llm60','-o','UserKnownHostsFile='+str(args.known_hosts),'-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-i',str(args.ssh_key),'-L','127.0.0.1:'+str(local_port)+':127.0.0.1:'+str(credentials['port']),'dflanag3@'+args.server_host]
 tunnel=None;error=None;proxy=None
 def stop_forward():
  nonlocal tunnel
  if tunnel is None:return
  tunnel.terminate()
  try:tunnel.wait(timeout=10)
  except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait(timeout=10)
  result['ssh_tunnel_exit']=tunnel.returncode;result['ssh_tunnel_absent']=not Path('/proc',str(tunnel.pid)).exists();tunnel=None
 def jail(command,state,label,key):
  if not state.exists():state.mkdir(mode=0o700)
  proof=out/(label+'-admission.json')
  argv=['/usr/bin/bwrap','--die-with-parent','--new-session','--unshare-all','--share-net','--unshare-user','--cap-drop','ALL','--hostname','bw-client-llm49','--clearenv']
  for path in [Path('/usr'),Path('/lib'),Path('/lib64'),Path(sys.prefix),checkout,Path(fixture['source']).parent,guard]:
   argv+=['--ro-bind',str(path),str(path)]
  argv+=['--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',str(state),str(state),'--bind',str(out),str(out),'--chdir',str(checkout),
   '--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/tmp','--setenv','PYTHONPATH',os.pathsep.join([str(guard),str(checkout)]),'--setenv','PYTHONDONTWRITEBYTECODE','1','--setenv','BW_CLIENT_PROOF',str(proof),'--setenv','BW_COW_API_KEY',key,*command]
  child_env=None
  if host_tools_task:
   argv=command
   child_env=dict(PATH='/usr/bin:/bin',HOME='/tmp',LANG='C.UTF-8',PYTHONPATH=str(guard),
       PYTHONDONTWRITEBYTECODE='1',BW_CLIENT_PROOF=str(proof),BW_COW_API_KEY=key)
  with (out/(label+'.stdout')).open('xb') as stdout,(out/(label+'.stderr')).open('xb') as stderr:
   started=time.monotonic();process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,start_new_session=True,env=child_env,cwd=out);last=started
   while process.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>1800:
     os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10);raise TimeoutError('client audit phase exceeded1800s')
    if time.monotonic()-last>=30:result.update(phase=label,client_pid=process.pid,elapsed_seconds=elapsed);write();print(json.dumps(dict(phase=label,pid=process.pid,elapsed_seconds=elapsed)),flush=True);last=time.monotonic()
    time.sleep(.2)
  row=dict(exit_code=process.returncode,seconds=time.monotonic()-started,pid=process.pid,process_absent=not Path('/proc',str(process.pid)).exists(),proof=None if not proof.exists() else json.loads(proof.read_bytes()))
  if row['proof'] is not None:
   assert not row['proof']['native_modules_loaded']
   if not host_tools_task:assert not row['proof']['gpu_devices_visible']
  return row
 try:
  with (out/'ssh.stderr').open('xb') as stderr:tunnel=subprocess.Popen(forward,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=stderr,close_fds=True)
  result['ssh_tunnel_pid']=tunnel.pid;result['local_forward_port']=local_port;write();deadline=time.monotonic()+20
  while True:
   if tunnel.poll() is not None:raise RuntimeError('owned SSH forward exited; see ssh.stderr')
   try:
    with socket.create_connection(('127.0.0.1',local_port),timeout=1):break
   except OSError:
    if time.monotonic()>deadline:raise TimeoutError('SSH listener not ready')
    time.sleep(.1)
  proxy=LostAckProxy('http://127.0.0.1:'+str(local_port),None)
  base=dict(schema=1,source_policy='metadata-free-project-v1',source=fixture['source'],run_id=fixture.get('run_id','ubuntu-shell-'+fixture['target']),grant=fixture['grant'],verification=fixture['verification'],bindings=dict(task=dict(type='Text',value=Path(fixture['task_file']).read_text())),cow=dict(endpoint_url=proxy.url,endpoint_pin='ssh-pinned-owned-llm60-api',model_contract_id=credentials['model_contract_id'],execution_profile_id=credentials['execution_profile_id'],eos_tokens=credentials['eos_tokens'],branch=fixture.get('branch','ubuntu-shell-'+fixture['target'].lower()),mode='ask',bootstrap='coding'))
  if 'browser' in fixture:base['browser']=fixture['browser']
  if 'shell' in fixture:base['shell']=fixture['shell']
  if 'profiles' in fixture:base['profiles']=fixture['profiles']
  if 'grammar_registration' in fixture:base['cow']['grammar_registration']=fixture['grammar_registration']
  if 'context_pack' in fixture:base['context_pack']=fixture['context_pack']
  result['client_layout']='trusted-host-cli-with-audit-and-isolated-tools' if host_tools_task else 'filesystem-namespace-no-server-files-or-gpu'
  config=out/'host.json';config.write_text(json.dumps(base,indent=2)+'\n')
  python=str(Path(sys.executable));program=fixture['workflow']
  # Verify the import guard actually refuses a native implementation import.
  blocked=jail([python,'-B','-c','import context_time_travel'],out/'native-denied-state','native-import-negative','not-a-key')
  result['native_import_negative']=blocked;assert blocked['exit_code']!=0 and blocked['proof'] is not None;write()
  negative_state=out/'unauthorized-state'
  negative=jail([python,'-B','-m','branchwrite','run',program,'--config',str(config),'--state',str(negative_state)],negative_state,'unauthorized','wrong-ephemeral-key')
  result['unauthorized']=negative;assert negative['exit_code']==2 and negative['proof'] is not None
  assert 'COW HTTP 401' in (out/'unauthorized.stderr').read_text()
  if host_tools_task:
   assert negative['proof']['violations']==[]
   file_denied=jail([python,'-B','-c',"open('/dev/nvidia0','rb')"],out/'gpu-denied-state','gpu-open-negative','not-a-key')
   assert file_denied['exit_code']!=0 and any(v['kind']=='file' for v in file_denied['proof']['violations'])
   result['gpu_open_negative']=file_denied
   process_denied=jail([python,'-B','-c',"import subprocess;subprocess.run(['/bin/true'])"],out/'process-denied-state','process-negative','not-a-key')
   assert process_denied['exit_code']!=0 and any(v['kind']=='process' for v in process_denied['proof']['violations'])
   result['process_negative']=process_denied
  write()
  (out/'NEGATIVE_COMPLETE').write_text('await server-side zero-work audit\n');deadline=time.monotonic()+180
  while not (out/'ALLOW_POSITIVE').exists():
   if time.monotonic()>deadline:raise TimeoutError('controller zero-work audit barrier expired')
   time.sleep(.2)
  connection=out/'connection.json'
  connection.write_text(json.dumps({k:base['cow'][k] for k in ('endpoint_url','endpoint_pin','model_contract_id','execution_profile_id','eos_tokens')}))
  command=[python,'-B','-m','branchwrite','code',fixture['source'],'--task',Path(fixture['task_file']).read_text(),'--connection',str(connection),'--state',str(args.state),'--branch',base['cow']['branch'],'--from-prefix','standalone-sc17-q8-binding-cache-dev16-v1','--from-snapshot','4164a409a37ba67670189cb7bbbf43d66fa89341c8aaff428f9a63e5c2d6f506','--shell-timeout','30']
  for value in fixture['read_dirs']:command += ['--read-dir',value]
  for value in fixture['edit_dirs']:command += ['--edit-dir',value]
  for key,flag in [('read_paths','--read'),('edit_paths','--edit'),('create_paths','--create')]:
   if key=='read_paths':continue
   for value in fixture['grant'][key]:command += [flag,value]
  for row in fixture['verification']['commands']:command += ['--test-command',row['command']]
  result['lost_reply']=jail(command,args.state,'lost-reply-cli',credentials['api_key']);write()
  assert result['lost_reply']['exit_code'] in (2,4) and proxy.dropped.is_set()
  result['lost_operation_id']=proxy.operation_id
  result['dropped_complete_sha256']=hashlib.sha256(json.dumps(proxy.dropped_completion,sort_keys=True).encode()).hexdigest()
  ledger=lambda:{str(p.relative_to(args.state/'ledger')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.state/'ledger').rglob('*') if p.is_file()}
  before_ledger=ledger()
  (out/'BEFORE_RECONCILE').write_text(proxy.operation_id+'\n');deadline=time.monotonic()+180
  while not (out/'ALLOW_RECONCILE').exists():
   if time.monotonic()>deadline:raise TimeoutError('reconcile measurement barrier expired')
   time.sleep(.2)
  result['reconcile']=jail([python,'-B','-m','branchwrite','reconcile',str(args.state),'--operation-id',proxy.operation_id],args.state,'reconcile-cli',credentials['api_key']);write()
  assert result['reconcile']['exit_code']==0 and ledger()==before_ledger
  record=OperationJournal(args.state/'cow-operations').read(proxy.operation_id)
  assert record['answer']==proxy.dropped_completion['answer'] and record['receipt']==proxy.dropped_completion['receipt']
  result['recovered_exact_answer']=True
  result['workflow_ledger_unchanged_by_reconcile']=True
  (out/'RECONCILED').write_text(proxy.operation_id+'\n');deadline=time.monotonic()+180
  while not (out/'ALLOW_RESUME').exists():
   if time.monotonic()>deadline:raise TimeoutError('resume measurement barrier expired')
   time.sleep(.2)
  result['positive']=jail([python,'-B','-m','branchwrite','resume',str(args.state)],args.state,'autonomous-cli',credentials['api_key']);write()
  assert result['positive']['exit_code']==0 and result['positive']['proof'] is not None
  if host_tools_task:assert result['positive']['proof']['violations']==[]
  result['cli_report']=json.loads((out/'autonomous-cli.stdout').read_bytes());assert result['cli_report']['verified']
  result['proxy_requests']=list(proxy.requests)
  proxy.close();proxy=None
  stop_forward()
  result['completed_run_reopen_without_transport']=jail(command,args.state,'completed-reopen',credentials['api_key']);write()
  assert result['completed_run_reopen_without_transport']['exit_code']==0
  assert (out/'autonomous-cli.stdout').read_bytes()==(out/'completed-reopen.stdout').read_bytes()
  # The transport is deliberately stopped above. Parent audit is independently
  # performed server-side after settlement; do not reconnect for offline replay.
  result['status']='BOUNDED_CROSS_HOST_AUTONOMOUS_CLI_PASS'
 except BaseException as exc:error=exc;result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
 finally:
  if proxy is not None:proxy.close()
  stop_forward();write()
 return int(error is not None)
if __name__=='__main__':raise SystemExit(main())
