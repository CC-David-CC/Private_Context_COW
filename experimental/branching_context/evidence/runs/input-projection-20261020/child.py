import time,json,statistics,hashlib,os
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding
n=int(os.environ['BENCH_SIZE']);mixed=os.environ['BENCH_MIXED']=='1';steps=20
spans=[dict(kind='text',tokens=[(i*7919+17)%200000 for i in range(n)])]
if mixed:spans.extend([dict(kind='image',original_sha256='a'*64,normalized_sha256='b'*64,encoder_profile_id='c'*64,embedding_sha256='d'*64,rows=6,width=4,grid_width=3,grid_height=2,pad_token_id=77,embedding_bytes=116),dict(kind='text',tokens=[31])])
def run():
 h=InputHistory(spans);start=time.perf_counter()
 for i in range(steps):
  old=len(h);new=h+(41+i,)
  assert new.starts_with(h) if hasattr(new,'starts_with') else new.prefix(old)==h
  assert new.segment(old).native_tokens==(41+i,)
  input_binding(new);input_binding(new);h=new
 return time.perf_counter()-start,h.digest
run();samples=[run() for _ in range(3)]
print(json.dumps(dict(prefix=n,mixed=mixed,steps=steps,samples=[v[0] for v in samples],median=statistics.median(v[0] for v in samples),final_digest=samples[0][1])))
