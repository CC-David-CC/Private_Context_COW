"""Frozen-source CPU regression ledger for bounded tree retention."""
import hashlib,importlib,json,sys,unittest
from pathlib import Path

source=Path(sys.argv[1]).resolve(strict=True)
output=Path(sys.argv[2]);assert not output.exists()
api=source/'experimental/branching_context'
sys.path[:0]=[str(source),str(source/'tools'),str(api),str(api/'src'),str(api/'tests/implementation'),str(api/'scripts')]
def hashes():
    return {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob('*')) if p.is_file()}
before=hashes()
assert before['experimental/branching_context/src/context_time_travel/coordinator.py']=='7042b31f1f4bb23580cd21213052ad27e8ab861b011dca98a95ede9db2602f62'
assert before['experimental/branching_context/tests/implementation/test_bounded_tree_retention.py']=='a93064a33a14b3b4d611243e16944cdfb5c5453c7e96301006e1338fb839f896'
modules=[importlib.import_module(name) for name in sys.argv[3:]]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
def ids(items):
    for item in items:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()
names=list(ids(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(schema='cow-bounded-tree-retention-cpu-v1',status='PASS' if result.wasSuccessful()
    and not result.skipped and not result.expectedFailures and hashes()==before else 'FAIL',
    scope='Synthetic CPU admission/ownership/retention, not native GPU or performance evidence',
    tests=result.testsRun,test_ids=names,failures=[item.id() for item,_ in result.failures],
    errors=[item.id() for item,_ in result.errors],skips=[item.id() for item,_ in result.skipped],
    xfails=[item.id() for item,_ in result.expectedFailures],source_unchanged=hashes()==before,source_manifest=before)
with output.open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:record[key] for key in ('status','tests','failures','errors','skips','xfails','source_unchanged')}))
raise SystemExit(0 if record['status']=='PASS' else 1)
