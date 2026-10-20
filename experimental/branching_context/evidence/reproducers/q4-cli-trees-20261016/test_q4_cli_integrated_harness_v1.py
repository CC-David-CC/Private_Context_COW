"""Run combined harness controls from the exact private source package."""
import hashlib,importlib,json,sys,unittest
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1/integrated-harness-v1')
source=root/'source'
sys.path[:0]=[str(source/'experimental/branching_context/tests/implementation'),str(source/'experimental/branching_context/scripts'),str(source)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes():return {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
before=hashes()
assert before['tools/cow_cli.py']=='798fe8706b0e5bf8b6ebdf956d10d59ef6bc1e5647b98efe3590e098afb04b91'
assert before['experimental/branching_context/scripts/observe_cow_cli.py']=='20149f9face52ffa0afa2962824f364b9765e7b954a3da90e6084a8c974d7b28'
assert before['experimental/branching_context/scripts/run_q4_cli_trees.py']=='3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2'
modules=[importlib.import_module(name) for name in ('test_observe_cow_cli','test_q4_cli_trees')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
def ids(suite):
    for child in suite:
        if isinstance(child,unittest.TestSuite):yield from ids(child)
        else:yield child.id()
test_ids=list(ids(suite))
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(schema='cow-cli-integrated-harness-cpu-v1',status='PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures and hashes()==before else 'FAIL',
    tests=result.testsRun,test_ids=test_ids,failures=[test.id() for test,_ in result.failures],errors=[test.id() for test,_ in result.errors],
    skips=[test.id() for test,_ in result.skipped],xfails=[test.id() for test,_ in result.expectedFailures],source_unchanged=hashes()==before,
    source_manifest=before,scope='CPU measurement-harness controls only; no model/GPU',script_sha256=sha(Path(__file__)))
with (root/'result.json').open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:record[key] for key in ('status','tests','failures','errors','skips','xfails','source_unchanged')}))
raise SystemExit(0 if record['status']=='PASS' else 1)
