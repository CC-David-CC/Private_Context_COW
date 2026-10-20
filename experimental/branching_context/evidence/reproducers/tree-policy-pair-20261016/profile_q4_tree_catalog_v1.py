"""CPU-only profiling of existing canonical metadata; no backend/model loading."""
import cProfile
import hashlib
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
runtime=root/'gpu-candidate-v1/runtime';sys.path.insert(0,str(runtime/'experimental/branching_context/src'))
from context_time_travel.store import Store
import context_time_travel.store as store_module
pair=json.loads((root/'policy-pair-v2.json').read_bytes());arms={}
profile=json.loads((root/'gpu-candidate-v1/profile.json').read_bytes())
assert hashlib.sha256(Path(store_module.__file__).read_bytes()).hexdigest()==profile['native_build']['sha256'][str(Path(store_module.__file__).resolve())]
def files(directory):
    return {str(path.relative_to(directory)):(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
        for path in directory.rglob('*') if path.is_file() and 'checkpoints' not in path.relative_to(directory).parts}
for arm,data in pair['arms'].items():
    campaign=json.loads(Path(data['result_path']).read_bytes());directory=Path(campaign['store'])
    before=files(directory);store=Store(directory);contexts=[]
    for index,entry in sorted(campaign['roots'].items()):
        namespace=store.branch_namespace(entry['branch']);measurements=[]
        profiler=cProfile.Profile()
        for repeat in range(3):
            counters=store.work_counters();started=time.perf_counter()
            if repeat==2:profiler.enable()
            catalog=store.presentation_catalog(namespace)
            if repeat==2:profiler.disable()
            elapsed=time.perf_counter()-started;after=store.work_counters()
            measurements.append(dict(repetition=repeat,seconds=elapsed,profiled=repeat==2,
                counter_deltas={key:value-counters[key] for key,value in after.items()}))
        stats=pstats.Stats(profiler)
        top=[]
        for (filename,line,function),(primitive,calls,own,total,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:12]:
            top.append(dict(file=filename,line=line,function=function,calls=calls,self_seconds=own,inclusive_seconds=total))
        contexts.append(dict(context=int(index),namespace=namespace,heads=len(catalog['heads']),events=len(catalog['events']),
            measurements=measurements,median_unprofiled_seconds=statistics.median(row['seconds'] for row in measurements if not row['profiled']),
            top_profiled_inclusive_calls=top))
    assert files(directory)==before,'Canonical metadata changed during read-only profile'
    arms[arm]=dict(contexts=contexts,canonical_file_metadata_unchanged=True)
receipt=dict(schema='cow-readonly-catalog-attribution-v1',status='PASS',arms=arms,
    selected_store_source_sha256=hashlib.sha256(Path(store_module.__file__).read_bytes()).hexdigest(),
    source_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=['CPU-only canonical query measurements, not inference or GPU qualification',
        'Two unprofiled repetitions plus one cProfile repetition per namespace; no isolated machine benchmark',
        'Final stores contain different amounts of earlier history; results do not isolate the retention patch'])
with (root/'catalog-attribution-v1.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
for arm,data in arms.items():
    print(json.dumps(dict(arm=arm,contexts=[dict(context=row['context'],heads=row['heads'],events=row['events'],
        unprofiled_seconds=row['median_unprofiled_seconds'],snapshot_read_bytes=row['measurements'][1]['counter_deltas']['snapshot_read_bytes'],
        top=row['top_profiled_inclusive_calls'][:5]) for row in data['contexts']])))
