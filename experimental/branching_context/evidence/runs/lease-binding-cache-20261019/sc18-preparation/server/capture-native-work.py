from pathlib import Path
import json,sys,time
D=Path(__file__).parent
phase=sys.argv[1]
assert phase in ('before-reconcile','after-reconcile')
rows=[]
with (D/'native.jsonl').open() as f:
 for raw in f:
  row=json.loads(raw)
  if row.get('command')=='stats':rows.append(row)
assert rows,'no native work witness'
last=rows[-1]
r=dict(phase=phase,time_ns=time.time_ns(),native_pid=last['pid'],forward_tokens=last['response']['forward_tokens'],stats_samples=len(rows),trace_bytes=(D/'native.jsonl').stat().st_size)
if phase=='after-reconcile':
 before=json.loads((D/'before-reconcile.json').read_text())
 assert before['native_pid']==r['native_pid'],'native process changed'
 r['forward_work_delta']=r['forward_tokens']-before['forward_tokens']
 assert r['forward_work_delta']==0,'recovery performed forward work'
(D/(phase+'.json')).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
