"""Integrated CPU regressions; synthetic native work, real Store/HTTP controls."""
import hashlib,importlib,json,os,sys,time,unittest
from pathlib import Path
stage=Path('/home/dflanag3/spatial-artifacts/q4-cuda-recovery-20261016-v1/integrated-62b4e466-v2')
source=stage/'source';mode=sys.argv[1]
sys.path[:0]=[str(source/'experimental/branching_context/src'),str(source/'experimental/branching_context/tests/implementation'),str(source/'experimental/branching_context/scripts'),str(source/'serve'),str(source)]
focus=['test_checkpoint_operation_receipts','test_native_request_protocol','test_native_terminal_error_recovery','test_poisoned_worker_replacement','test_pool_coordinator','test_pool_spill_reload_trace','test_q4_cuda_recovery']
http=['test_cow_adapter','test_cow_browser_fixture','test_cow_context_root_capability','test_cow_http','test_cow_namespace_http','test_cow_prefix','test_cow_presentation_adapter','test_cow_presentation_flow','test_cow_prompt_encoder','test_cow_renderer','test_cow_reusable_context_smoke','test_cow_status_http','test_cow_text_adapter','test_cow_workspace','test_cow_workspace_graph','test_cow_workspace_graph_http','test_cow_workspace_http']
assert mode in ('focus','http')
names=focus if mode=='focus' else http
profile=json.loads(Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json').read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():return {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
before=hashes();modules=[importlib.import_module(name) for name in names]
imports={}
for name,module in tuple(sys.modules.items()):
    if name=='context_time_travel' or name.startswith('context_time_travel.') or name in names:
        path=Path(module.__file__).resolve();assert path.is_relative_to(source),name
        digest=sha(path)
        if name.startswith('context_time_travel'):
            expected_path=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime')/path.relative_to(source)
            assert profile['native_build']['sha256'][str(expected_path)]==digest,name
        imports[name]=dict(path=str(path),sha256=digest)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from flatten(item)
        else:yield item.id()
ids=list(flatten(suite));started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=hashes()
record=dict(schema='q4-recovery-integrated-cpu-v1',source_commit='62b4e466',mode=mode,
    status='PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures and before==after else 'FAIL',
    tests=result.testsRun,test_ids=ids,failures=[t.id() for t,_ in result.failures],errors=[t.id() for t,_ in result.errors],
    skipped=[t.id() for t,_ in result.skipped],xfails=[t.id() for t,_ in result.expectedFailures],
    seconds=time.monotonic()-started,imports=imports,source_unchanged=before==after,source_manifest=before,
    script_sha256=sha(Path(__file__)),scope='CPU synthetic backend and real filesystem/HTTP controls; not GPU inference')
with (stage/(mode+'-result.json')).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('source_manifest','test_ids','imports')},sort_keys=True))
raise SystemExit(0 if record['status']=='PASS' else 1)
