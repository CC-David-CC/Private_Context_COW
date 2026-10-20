"""Record combined CPU regressions; retain any failure without reclassification."""
import hashlib
import importlib
import json
import sys
import unittest
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
assert not output.exists()
api = source / 'experimental/branching_context'
sys.path[:0] = [str(source), str(source / 'tools'), str(api), str(api / 'src'),
               str(api / 'tests/implementation'), str(api / 'scripts')]
def hashes():
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob('*')) if p.is_file() and not p.is_symlink()
            and 'evidence' not in p.relative_to(source).parts}
before = hashes()
assert before['experimental/branching_context/src/context_time_travel/coordinator.py'] == 'a45638c8aa159c3ab91213d085e70822b8433d16ec9ad826e9c2a963d19d48dd'
assert before['experimental/branching_context/src/context_time_travel/store.py'] == 'db2e595f2b54e43b99ea6ae3de27f4790be4ce423475f289da33e75d7288db27'
assert before['tools/cow_cli.py'] == 'd8c5e652184a54af0509f8fb82674def06505dc234495496219574bd446f677a'
modules = [importlib.import_module(name) for name in sys.argv[3:]]
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
def ids(items):
    for item in items:
        if isinstance(item, unittest.TestSuite):
            yield from ids(item)
        else:
            yield item.id()
names = list(ids(suite))
assert len(names) == len(set(names)), 'duplicate test IDs'
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = hashes()
record = dict(schema='cow-native-combined-cpu-v2',
    source_commit='0a156901c356e7cecd49be63bbdf922d90362048',
    status='PASS' if result.wasSuccessful() and not result.skipped
        and not result.expectedFailures and after == before else 'FAIL',
    scope='Combined synthetic CPU/filesystem/HTTP controls; no GPU/performance claim',
    tests=result.testsRun, test_ids=names,
    failures=[item.id() for item, _ in result.failures],
    errors=[item.id() for item, _ in result.errors],
    skips=[item.id() for item, _ in result.skipped],
    xfails=[item.id() for item, _ in result.expectedFailures],
    source_unchanged=after == before, source_manifest=before)
with output.open('x') as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps({key: record[key] for key in
    ('status', 'tests', 'failures', 'errors', 'skips', 'xfails', 'source_unchanged')}))
raise SystemExit(0 if record['status'] == 'PASS' else 1)
