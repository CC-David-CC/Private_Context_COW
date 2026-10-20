from pathlib import Path
import subprocess,os,json,hashlib,shutil
from branchwrite.execution import Executor,Grant
D=Path(__file__).parent
assert (D/'command.exit').read_text().strip()=='0'
assert json.loads((D/'audit-output-result.json').read_text())['status']=='PASS'
PY='/home/dflanag3/spatial-artifacts/branchwrite-dev17-package-20261019-v1/installation/venv/bin/python'
def hashes(root):return {str(p.relative_to(root)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mode&0o777) for p in root.rglob('*') if p.is_file()}
before=hashes(D/'state');env={k:v for k,v in os.environ.items() if k!='PYTHONPATH' and not k.startswith(('BW_COW_','COW_'))}
def call(label,args):
 r=subprocess.run([PY,'-I','-B','-m','branchwrite',*args],cwd=D/'output',env=env,capture_output=True,text=True,timeout=60)
 (D/('public-'+label+'.stdout')).write_text(r.stdout);(D/('public-'+label+'.stderr')).write_text(r.stderr);assert r.returncode==0,(label,r.stderr)
 return json.loads(r.stdout),r.stdout
status,_=call('status',['status',str(D/'state'),'--json'])
review,_=call('review',['review',str(D/'state'),'--json']);assert review['complete_project_patch']
assert sorted(c['path'] for c in review['changes'])==['intervals.py']
export,_=call('export',['export',str(D/'state'),str(D/'reviewed-changes.zip')])
args=['apply',str(D/'reviewed-changes.zip'),'--source',str(D/'source'),'--destination',str(D/'applied-project'),'--sha256',export['sha256'],'--operation-id','sc18-reviewed-copy']
applied,one=call('apply',args);_,two=call('apply-retry',args);assert one==two
assert hashes(D/'applied-project')==hashes(D/'state/world/workspace')
resume,_=call('resume',['resume',str(D/'state')]);assert resume==json.loads((D/'output/autonomous-cli.stdout').read_text())
assert hashes(D/'state')==before
source=D/'applied-independent-source';shutil.copytree(D/'applied-project',source)
shutil.copy2(D/'independent-source/tests/test_independent.py',source/'tests/test_independent.py')
e=Executor.create(source,D/'applied-independent-world');r=e.run_tests('sc18-applied',Grant((),True),timeout=30)
(D/'applied-independent.stdout').write_bytes(r.stdout.read_bytes());(D/'applied-independent.stderr').write_bytes(r.stderr.read_bytes());assert r.exit_code==0
result=dict(status='PASS',review_export_apply_retry=True,offline_resume=True,state_bytes_and_modes_unchanged=True,applied_independent_cases=1000)
(D/'public-flow-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
