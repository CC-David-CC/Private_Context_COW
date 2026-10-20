"""Bounded read-only phase summary for the policy comparison."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);assert root.is_relative_to('/home/dflanag3/spatial-artifacts')
for name in sys.argv[2:]:
    directory=root/name;result=directory/'result.json'
    if result.exists():
        data=json.loads(result.read_bytes());rows=data.get('generations',[])
        print(json.dumps(dict(run=name,status=data['status'],comparisons=data.get('comparisons'),error=data.get('numerical_max_error'),
            generations=len(rows),reference_runs=len(data.get('references',[])),failures=data.get('failures'))))
        measured=[row for row in rows if str(row.get('tag','')).startswith('round')]
        if measured:print(json.dumps(dict(measured=[dict(context=row['context_index'],question=row['question'],tag=row['tag'],
            seconds=row['process_seconds'],loaded=row['counter_deltas']['loaded_file_bytes']) for row in measured])))
        services=data.get('services',[])
        if services:print(json.dumps(dict(services=[dict(pid=row['http_pid'],live=Path('/proc',str(row['http_pid'])).exists(),native=row.get('worker',{}).get('native_pid')) for row in services])))
    progress=directory/'progress.json'
    if progress.exists():
        data=json.loads(progress.read_bytes());print(json.dumps(dict(progress={key:data.get(key) for key in ('status','phase','elapsed_seconds','last_completed_step','error')})))
    live=[]
    for path in directory.glob('*/*.process.json'):
        data=json.loads(path.read_bytes())
        if Path('/proc',str(data['pid'])).exists():live.append(dict(pid=data['pid'],status=data['status'],name=path.parent.name))
    if live:print(json.dumps(dict(live_references=live)))
