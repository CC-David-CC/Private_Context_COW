import unittest,json,sys,time,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[3];sys.path[:0]=[str(r/'src'),str(r/'tests/implementation'),str(r.parents[1]/'serve'),str(r.parents[1]/'tools'),str(r/'scripts'),'/home/dflanag3/spatial-artifacts/grammar-first-20261016-v1/deps']
modules=['test_input_projection_cache','test_input_history_views','test_input_prefix_comparison','test_vision_input_history','test_vision_execution','test_vision_schedule_cursor','test_vision_schedule_projection','test_vision_binding_cache','test_schedule_binding_cache','test_vision_dry_projection','test_vision_backend_binding']
rows=[]
class Result(unittest.TextTestResult):
 def addSuccess(self,t):super().addSuccess(t);rows.append(dict(id=t.id(),status='PASS'))
 def addFailure(self,t,e):super().addFailure(t,e);rows.append(dict(id=t.id(),status='FAIL'))
 def addError(self,t,e):super().addError(t,e);rows.append(dict(id=t.id(),status='ERROR'))
 def addSkip(self,t,e):super().addSkip(t,e);rows.append(dict(id=t.id(),status='SKIP',reason=e))
def fingerprints():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'src/context_time_travel').glob('*.py')}
before=fingerprints();start=time.monotonic()
with (Path(__file__).parent/'IP02-regressions.txt').open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
v=dict(status='PASS' if result.wasSuccessful() and not result.skipped and before==fingerprints() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),source_unchanged=before==fingerprints(),source_sha256=before,seconds=time.monotonic()-start,scope='CPU structural/reference and host-protocol tests, no GPU numerics',tests=rows)
(Path(__file__).parent/'IP02-regressions.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['status','tests_run','failures','errors','skips','seconds']}));raise SystemExit(0 if v['status']=='PASS' else 1)
