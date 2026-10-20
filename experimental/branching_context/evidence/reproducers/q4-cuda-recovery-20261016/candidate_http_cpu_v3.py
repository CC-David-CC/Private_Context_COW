"""Synthetic HTTP regressions against the frozen candidate runtime."""
import hashlib,importlib,json,os
from pathlib import Path
import sys,unittest
runtime=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1/http-control-v3/source')
sys.path[:0]=[str(runtime/'experimental/branching_context/src'),str(runtime),str(runtime/'serve'),str(runtime/'experimental/branching_context/scripts')]
profile=json.loads(Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json').read_bytes())
names=["test_cow_adapter","test_cow_browser_fixture","test_cow_context_root_capability","test_cow_http","test_cow_namespace_http","test_cow_prefix","test_cow_presentation_adapter","test_cow_presentation_flow","test_cow_prompt_encoder","test_cow_renderer","test_cow_reusable_context_smoke","test_cow_status_http","test_cow_text_adapter","test_cow_workspace","test_cow_workspace_graph","test_cow_workspace_graph_http","test_cow_workspace_http"]
def hashes():return {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime.rglob('*.py'))}
before=hashes()
modules=[importlib.import_module(name) for name in names]
imports={}
for name,module in tuple(sys.modules.items()):
    if name=='context_time_travel' or name.startswith('context_time_travel.') or name in names:
        path=Path(module.__file__).resolve();value=hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.is_relative_to(runtime),name
        if name.startswith('context_time_travel'):
            expected_path=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime')/path.relative_to(runtime)
            assert profile['native_build']['sha256'][str(expected_path)]==value,name
        imports[name]=dict(path=str(path),sha256=value)
print(json.dumps(dict(phase='before_tests',imports=imports),sort_keys=True),flush=True)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=hashes()
r=dict(status='PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures and before==after else 'FAIL',tests=result.testsRun,failures=[t.id() for t,_ in result.failures],errors=[t.id() for t,_ in result.errors],skipped=[t.id() for t,_ in result.skipped],imports=imports,source_unchanged=before==after,profile_id=profile['execution_profile_id'],scope='CPU synthetic HTTP/control only; no GPU model or numerical qualification',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with Path(sys.argv[1]).open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(r,sort_keys=True),flush=True)
sys.exit(0 if r['status']=='PASS' else 1)