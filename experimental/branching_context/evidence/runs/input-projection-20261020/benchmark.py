from pathlib import Path
import subprocess,json,os,hashlib,time
r=Path(__file__).parent;rows=[];start=time.monotonic()
roots=dict(serving='/home/dflanag3/src/context-cow-client-setup-20261020',pause_base='/home/dflanag3/src/context-cow-consumed-pause-20261020',candidate='/home/dflanag3/src/context-cow-input-projection-20261020')
for label,root in roots.items():
 for n in [1024,8192,32768,65536]:
  for mixed in [False,True]:
   env=os.environ.copy();env.update(PYTHONPATH=root+'/experimental/branching_context/src',BENCH_SIZE=str(n),BENCH_MIXED='1' if mixed else '0')
   p=subprocess.run(['/home/dflanag3/src/Strata/.venv/bin/python',str(r/'child.py')],env=env,capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr
   v=json.loads(p.stdout);v.update(source=label,source_root=root,input_spans_sha256=hashlib.sha256((Path(root)/'experimental/branching_context/src/context_time_travel/input_spans.py').read_bytes()).hexdigest());rows.append(v);print(json.dumps(v),flush=True)
   (r/'benchmark.json').write_text(json.dumps(dict(status='RUNNING',scope='CPU-only history/projection microbenchmark, not GPU throughput',rows=rows),indent=2)+'\n')
for n in [1024,8192,32768,65536]:
 for mixed in [False,True]:assert len({v['final_digest'] for v in rows if v['prefix']==n and v['mixed']==mixed})==1
(r/'benchmark.json').write_text(json.dumps(dict(status='PASS',scope='CPU-only history/projection microbenchmark; exact final digest across all source arms; no GPU speed claim',seconds=time.monotonic()-start,rows=rows),indent=2)+'\n')
