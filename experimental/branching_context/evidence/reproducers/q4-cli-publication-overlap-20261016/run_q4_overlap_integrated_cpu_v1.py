"""Combined scheduler and evidence guards; no native/model execution."""
import hashlib,importlib,json,sys,unittest
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-overlap-20261016-v1/integrated-cpu-v1')
source=root/'source';api=source/'experimental/branching_context'
sys.path[:0]=[str(api/'src'),str(api/'scripts'),str(api/'tests/implementation')]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes():return {str(path.relative_to(source)):sha(path) for path in sorted(source.rglob('*')) if path.is_file()}
before=hashes()
assert before['experimental/branching_context/scripts/run_q4_cli_overlap.py']=='9e146f3cc6abed0df2ef4d30ccf53b8f3289172ab7f60067a0866af3a2b41e3a'
assert before['experimental/branching_context/tests/implementation/test_publication_overlap.py']=='d91ce8c65310e24b3ab5cd02e04014fb90ed0b6130b3bebe7223764d2b575f2d'
modules=[importlib.import_module(name) for name in ('test_publication_overlap','test_q4_cli_overlap')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()
names=list(ids(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(schema='cow-q4-overlap-integrated-cpu-v1',status='PASS' if result.wasSuccessful() and not result.skipped and not result.expectedFailures and hashes()==before else 'FAIL',
            tests=result.testsRun,test_ids=names,failures=[item.id() for item,_ in result.failures],errors=[item.id() for item,_ in result.errors],
            skips=[item.id() for item,_ in result.skipped],xfails=[item.id() for item,_ in result.expectedFailures],source_manifest=before,source_unchanged=hashes()==before,
            scope='Synthetic CPU scheduler controls and measurement admission guards; not GPU inference')
with (root/'result.json').open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({key:record[key] for key in ('status','tests','failures','errors','skips','xfails','source_unchanged')}))
raise SystemExit(0 if record['status']=='PASS' else 1)
