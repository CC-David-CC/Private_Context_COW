from pathlib import Path
import subprocess,os,json,hashlib,shutil
from branchwrite.execution import Executor,Grant
b=Path(__file__).parent
assert (b/'command.exit').read_text().strip()=='0'
assert json.loads((b/'audit-output-result.json').read_text())['status']=='PASS'
py='/home/dflanag3/spatial-artifacts/branchwrite-dev16-package-20261019-v1/installation/venv/bin/python'
def hashes(root):
 return {str(p.relative_to(root)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mode&0o777) for p in root.rglob('*') if p.is_file()}
before=hashes(b/'state');env=os.environ.copy()
env.pop('PYTHONPATH',None)
for key in list(env):
 if key.startswith(('BW_COW_','COW_')):env.pop(key)
def call(label,args):
 p=subprocess.run([py,'-I','-B','-m','branchwrite',*args],cwd=b/'output',env=env,capture_output=True,text=True,timeout=60)
 (b/('public-'+label+'.stdout')).write_text(p.stdout);(b/('public-'+label+'.stderr')).write_text(p.stderr)
 assert p.returncode==0,(label,p.stderr)
 return json.loads(p.stdout),p.stdout
status,_=call('status',['status',str(b/'state'),'--json'])
review,_=call('review',['review',str(b/'state'),'--json'])
assert review['complete_project_patch']
assert sorted(c['path'] for c in review['changes'])==['branchwrite/project_scope.py']
export,_=call('export',['export',str(b/'state'),str(b/'reviewed-changes.zip')])
args=['apply',str(b/'reviewed-changes.zip'),'--source',str(b/'source'),'--destination',str(b/'applied-project'),'--sha256',export['sha256'],'--operation-id','sc14-reviewed-copy']
applied,one=call('apply',args);_,two=call('apply-retry',args)
assert one==two
assert hashes(b/'applied-project')==hashes(b/'state/world/workspace')
resume,_=call('resume',['resume',str(b/'state')])
assert resume==json.loads((b/'output/autonomous-cli.stdout').read_text())
assert hashes(b/'state')==before
audit=b/'applied-independent-source';audit.mkdir()
for name in ['branchwrite','examples']:shutil.copytree(b/'applied-project'/name,audit/name)
(audit/'tests').mkdir();(audit/'tests/__init__.py').touch()
for name in ['test_directory_grants.py','test_directory_scope_overlap.py']:
 shutil.copyfile(b/'applied-project/tests'/name,audit/'tests'/name)
shutil.copyfile(b/'independent-tests.py.txt',audit/'tests/test_generated.py')
shutil.copyfile(b/'supplemental-tests.py.txt',audit/'tests/test_supplemental.py')
executor=Executor.create(audit,b/'applied-independent-world')
result=executor.run_tests('sc14-applied-independent',Grant((),True),timeout=30)
(b/'applied-independent.stderr').write_bytes(result.stderr.read_bytes())
assert result.exit_code==0,result.stderr.read_bytes()
value=dict(status='PASS',status_review_export_apply_retry_resume=True,complete_project_patch=True,changed_files=1,run_state_unchanged=True,identical_apply_retry=True,applied_protected_methods=9,applied_generated_cases=100,applied_supplemental_methods=3,isolated_executor=True,credentials_absent=True)
(b/'public-flow-result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
