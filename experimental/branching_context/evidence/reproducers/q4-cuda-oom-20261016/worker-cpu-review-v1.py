import hashlib, importlib, json, os
from pathlib import Path
import sys, unittest
source=Path(sys.argv[1]).resolve(strict=True)
report_path=Path(sys.argv[2])
sys.path[:0]=[str(source/'src'),str(source/'scripts'),str(source/'tests/implementation')]
names=['test_poisoned_worker_replacement','test_native_terminal_error_recovery','test_pool_coordinator','test_native_request_protocol','test_pool_spill_reload_trace','test_checkpoint_operation_receipts']
def hashes(): return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
before=hashes()
modules=[importlib.import_module(name) for name in names]
imports={m.__name__:{'path':str(Path(m.__file__).resolve()),'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()} for m in modules}
for name in ('context_time_travel.coordinator','context_time_travel.strata_backend'):
    m=importlib.import_module(name)
    imports[name]={'path':str(Path(m.__file__).resolve()),'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
assert all(Path(row['path']).is_relative_to(source) for row in imports.values()), imports
print(json.dumps({'phase':'before_tests','imports':imports,'source_files':len(before)},sort_keys=True),flush=True)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=hashes()
report={'status':'PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures and before==after else 'FAIL','count':result.testsRun,'failures':[t.id() for t,_ in result.failures],'errors':[t.id() for t,_ in result.errors],'skipped':[t.id() for t,_ in result.skipped],'imports':imports,'source_unchanged':before==after,'source_manifest':before,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with report_path.open('x') as f: json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_manifest'},sort_keys=True),flush=True)
sys.exit(0 if report['status']=='PASS' else 1)