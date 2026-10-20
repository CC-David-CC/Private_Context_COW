from pathlib import Path
import json,hashlib,shutil
from branchwrite.execution import Executor,Grant
b=Path(__file__).parent
assert (b/'command.exit').read_text().strip()=='0'
baseline=json.loads((b/'baseline.json').read_text());source=b/'source';workspace=b/'state/world/workspace'
for name,sha in baseline['files'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,('original changed',name)
files={p.relative_to(workspace).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.rglob('*') if p.is_file()}
assert set(files)==set(baseline['files'])
changed=sorted(n for n in files if files[n]!=baseline['files'][n])
assert changed==baseline['expected_changes'],changed
audit=b/'independent-source';audit.mkdir();shutil.copytree(workspace/'branchwrite',audit/'branchwrite');shutil.copytree(workspace/'examples',audit/'examples')
(audit/'tests').mkdir();(audit/'tests/__init__.py').touch()
for name in ['test_directory_grants.py','test_directory_scope_overlap.py']:
 shutil.copyfile(source/'tests'/name,audit/'tests'/name)
shutil.copyfile(b/'independent-tests.py.txt',audit/'tests/test_independent.py')
shutil.copyfile(b/'supplemental-tests.py.txt',audit/'tests/test_supplemental.py')
executor=Executor.create(audit,b/'independent-world')
result=executor.run_tests('sc14-independent',Grant((),True),timeout=30)
(b/'independent.stdout').write_bytes(result.stdout.read_bytes())
(b/'independent.stderr').write_bytes(result.stderr.read_bytes())
value=dict(status='PASS' if result.exit_code==0 else 'FAIL',exit_code=result.exit_code,changed=changed,source_files=len(files),independent_cases=100,seed=141019,original_source_unchanged=True,protected_tests_unchanged=True,isolated_executor=True)
(b/'audit-output-result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
raise SystemExit(result.exit_code!=0)
