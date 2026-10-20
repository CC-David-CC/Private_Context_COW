from pathlib import Path
import subprocess,json,os,time,hashlib
r=Path(__file__).parent;rows=[]
roots=dict(IP01=str(r/'IP01-package'),IP02='/home/dflanag3/src/context-cow-input-projection-20261020/experimental/branching_context/src')
for label,root in roots.items():
 for n in [1024,8192,32768,65536]:
  env=os.environ.copy();env.update(PYTHONPATH=root,BENCH_SIZE=str(n))
  p=subprocess.run(['/home/dflanag3/src/Strata/.venv/bin/python',str(r/'IP02-child.py')],env=env,capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr
  v=json.loads(p.stdout);v.update(source=label,source_root=root);rows.append(v);print(json.dumps(v),flush=True)
  (r/'IP02-benchmark.json').write_text(json.dumps(dict(status='RUNNING',rows=rows),indent=2)+'\n')
for n in [1024,8192,32768,65536]:
 for key in ['history_digest','schedule_digest']:assert len({sample[key] for row in rows if row['prefix']==n for sample in row['samples']})==1
(r/'IP02-benchmark.json').write_text(json.dumps(dict(status='PASS',scope='CPU-only backend expected-binding + fresh cursor check + complete publication audit; empty hash cache each repetition; no GPU inference',warmup=1,repetitions=3,rows=rows),indent=2)+'\n')
