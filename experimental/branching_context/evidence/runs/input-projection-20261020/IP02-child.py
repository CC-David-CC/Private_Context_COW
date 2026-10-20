import json,time,statistics,os,hashlib
from context_time_travel.input_spans import InputHistory
from context_time_travel import input_schedule_projection as projection
from context_time_travel.execution_receipt import _InputScheduleCursor,tokens_digest,schedule_digest
from context_time_travel.strata_backend import input_binding as backend_binding
n=int(os.environ['BENCH_SIZE']);steps=64
spans=[dict(kind='text',tokens=[(i*7919+17)%200000 for i in range(n)])]
def event(h,start):
 delta=h.segment(start).native_tokens
 return dict(operator='normal-verifier-decode' if start else 'batched-prefill',position=start,count=len(delta),tokens_sha256=tokens_digest(delta),chunks=[[i,min(512,len(h)-i)] for i in range(start,len(h),512)],inputs_sha256=h.digest,**backend_binding(h))
def run():
 projection._BINDINGS=projection._BindingCache(65536)
 h=InputHistory(spans);cur=_InputScheduleCursor([event(h,0)],h,512);start=time.perf_counter()
 for i in range(steps):
  old=len(h);h=h+(41+i,);cur.append(event(h,old),h)
 loop=time.perf_counter()-start;start=time.perf_counter();events=cur.finish();finish=time.perf_counter()-start
 return dict(loop_seconds=loop,finish_seconds=finish,total_seconds=loop+finish,history_digest=h.digest,schedule_digest=schedule_digest(events))
run();samples=[run() for _ in range(3)]
print(json.dumps(dict(prefix=n,steps=steps,samples=samples,median_total=statistics.median(v['total_seconds'] for v in samples),median_loop=statistics.median(v['loop_seconds'] for v in samples),median_finish=statistics.median(v['finish_seconds'] for v in samples))))
