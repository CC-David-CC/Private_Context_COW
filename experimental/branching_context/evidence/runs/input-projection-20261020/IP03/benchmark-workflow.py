from pathlib import Path
import subprocess,os,json,statistics
root=Path(__file__).parent;r=root.parents[3]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'src'))
results=[]
for size in [8192,65536,131072]:
 row=dict(prefix=size,steps=64,samples={'baseline':[],'packed':[]},runs=[],scope='CPU metadata only; original scalar cursor still recalculates every new boundary')
 for repeat,arms in enumerate([['baseline','packed'],['packed','baseline']]):
  for arm in arms:
   p=subprocess.run(['/home/dflanag3/src/Strata/.venv/bin/python','-B',str(root/'benchmark-child.py')],cwd=r,env=dict(env,BENCH_SIZE=str(size),BENCH_ARM=arm,BASELINE_SOURCE=str(root/'baseline_projection.py')),capture_output=True,text=True,timeout=120)
   assert p.returncode==0,p.stderr
   value=json.loads(p.stdout);row['runs'].append(dict(arm=arm,repeat=repeat,result=value));row['samples'][arm].extend(value['samples'])
 for arm in row['samples']:
  row[arm+'_total']=statistics.median(v['total_seconds'] for v in row['samples'][arm])
  row[arm+'_finish']=statistics.median(v['finish_seconds'] for v in row['samples'][arm])
  assert len(set(v['schedule_digest'] for v in row['samples'][arm]))==1
 assert row['samples']['baseline'][0]['schedule_digest']==row['samples']['packed'][0]['schedule_digest']
 assert row['samples']['baseline'][0]['history_digest']==row['samples']['packed'][0]['history_digest']
 row['total_change_percent']=100*(row['packed_total']/row['baseline_total']-1)
 row['finish_change_percent']=100*(row['packed_finish']/row['baseline_finish']-1)
 results.append(row)
 print(json.dumps({k:v for k,v in row.items() if k not in ['samples','runs']}),flush=True)
(root/'benchmark-workflow.json').write_text(json.dumps(dict(status='PASS',same_history_and_schedule_digests=True,order='baseline/packed then packed/baseline, fresh warmup per child, identical empty cache ownership per run',results=results),indent=2)+'\n')
