from pathlib import Path
import time,json,subprocess,traceback,os
D=Path(__file__).parent
PY='/home/dflanag3/spatial-artifacts/branchwrite-dev17-package-20261019-v1/installation/venv/bin/python'
r={'status':'WAITING_FOR_TERMINAL_CLIENT','stages':[]}
def write():
 t=D/'settled-audit.partial';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(D/'settled-audit-result.json')
write();start=time.monotonic()
try:
 while not (D/'command.exit').exists():
  if time.monotonic()-start>1900:raise TimeoutError('No terminal client result; no ledger inspection performed')
  time.sleep(2)
 assert (D/'command.exit').read_text().strip()=='0','Client failed; do not audit its ledger as a successful run'
 r['status']='AUDITING_TERMINAL_OUTPUT';write()
 env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
 for name in ['audit-output.py','audit-ledger.py','audit-public-flow.py']:
  with (D/(name+'.stdout')).open('x') as out,(D/(name+'.stderr')).open('x') as err:
   p=subprocess.run([PY,'-I','-B',str(D/name)],cwd=D,env=env,stdout=out,stderr=err,timeout=180)
  r['stages'].append({'script':name,'exit_code':p.returncode});write()
  assert p.returncode==0, name+' failed; retain evidence'
 r['status']='PASS'
except BaseException as exc:r.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
r['seconds']=time.monotonic()-start;write()
raise SystemExit(r['status']!='PASS')
