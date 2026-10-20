import time,statistics,json
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding
from context_time_travel.packed_input_binding import fresh_input_binding
results=[]
for size in [8192,65536,131072]:
 h=InputHistory([dict(kind='text',tokens=[(i*7919+17)%200000 for i in range(size)])])
 assert input_binding(h)==fresh_input_binding(h)
 row=dict(prefix=size,hashes=input_binding(h),unit='seconds per fresh complete binding; CPU only',samples={})
 for label,fn in [('reference',input_binding),('packed',fresh_input_binding)]:
  fn(h);samples=[]
  for _ in range(9):
   t=time.perf_counter();fn(h);samples.append(time.perf_counter()-t)
  row['samples'][label]=samples;row[label+'_median']=statistics.median(samples)
 row['speedup']=row['reference_median']/row['packed_median'];results.append(row)
print(json.dumps(dict(status='PASS',scope='CPU metadata byte construction only, no GPU throughput claim',results=results),indent=2))
