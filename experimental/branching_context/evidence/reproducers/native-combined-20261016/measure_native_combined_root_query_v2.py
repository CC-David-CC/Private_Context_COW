"""Compare full-catalog and targeted root reads on the same retained stores."""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
runtime = root / 'gpu-candidate-v2/runtime'
sys.path.insert(0, str(runtime / 'experimental/branching_context/src'))
from context_time_travel.store import Store
import context_time_travel.store as implementation
assert hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest() == 'db2e595f2b54e43b99ea6ae3de27f4790be4ce423475f289da33e75d7288db27'
pair_root = Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
pair = json.loads((pair_root / 'policy-pair-v2.json').read_bytes())
def metadata(directory):
    return {str(path.relative_to(directory)): (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file() and not path.name.endswith(('.lock', '.lck'))
            and 'checkpoints' not in path.relative_to(directory).parts}
rows = []
for arm, data in sorted(pair['arms'].items()):
    campaign = json.loads(Path(data['result_path']).read_bytes())
    directory = Path(campaign['store'])
    before = metadata(directory)
    store = Store(directory)
    for index, entry in sorted(campaign['roots'].items()):
        namespace = store.branch_namespace(entry['branch'])
        sid = entry['head']['root']
        observations = []
        expected = None
        for repeat in range(4):
            order = ('catalog', 'root') if repeat % 2 == 0 else ('root', 'catalog')
            values = {}
            for kind in order:
                counters = store.work_counters()
                start = time.perf_counter()
                if kind == 'root':
                    result = store.root_presentations(namespace, sid)
                else:
                    catalog = store.presentation_catalog(namespace)
                    result = [event['presentation'] for event in catalog['events']
                              if event['result'].get('tip') == sid and isinstance(event.get('presentation'), dict)
                              and event['presentation']['kind'] == 'root']
                elapsed = time.perf_counter() - start
                after = store.work_counters()
                canonical = sorted(json.dumps(item, sort_keys=True, separators=(',', ':')) for item in result)
                if expected is None:
                    expected = canonical
                assert canonical == expected, (arm, index, kind)
                values[kind] = dict(seconds=elapsed, counter_deltas={key: value-counters[key]
                    for key, value in after.items()})
            observations.append(dict(repetition=repeat, warmup=repeat == 0, order=list(order), values=values))
        row = dict(arm=arm, context=int(index), namespace=namespace, root=sid,
            descriptor_count=len(expected), observations=observations,
            medians={kind: statistics.median(item['values'][kind]['seconds'] for item in observations[1:])
                     for kind in ('catalog', 'root')})
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert metadata(directory) == before, 'canonical metadata changed'
receipt = dict(schema='cow-root-query-readonly-comparison-v1', status='PASS', contexts=rows,
    canonical_file_metadata_unchanged=True, native_executions=0,
    source_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=['CPU metadata query only, not inference or whole-request throughput',
        'One warmup plus three alternating-order pairs per context; filesystem cache uncontrolled',
        'Compares queries on the same current Store implementation; old full-catalog implementation is unchanged',
        'Guard excludes OS lock files and numerical checkpoints; metadata guard is inode/size/mtime, not a byte hash'])
with (root / 'root-query-measurement-v2.json').open('x') as output:
    json.dump(receipt, output, sort_keys=True, indent=2)
    output.write('\n')
