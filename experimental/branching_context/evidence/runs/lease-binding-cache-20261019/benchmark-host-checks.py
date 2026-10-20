import hashlib,json,platform,statistics,threading,time
from types import SimpleNamespace
from context_time_travel.input_spans import InputHistory
import context_time_travel.strata_backend as module
from context_time_travel.vision_execution import input_binding,VISION_PROTOCOL
from test_vision_execution import spans
backend=object.__new__(module.StrataBackend)
calls=0;queries=0;actual={}
original=module.input_binding
def counted(history):
 global calls
 calls+=1;return original(history)
module.input_binding=counted
def request(command):
 global queries
 assert command=='vision_info';queries+=1;return dict(actual)
backend._request=request
result={}
for kind in ['text','mixed']:
 ids=[(i*17+5)%16 for i in range(17857)]
 if kind=='text':history=InputHistory([dict(kind='text',tokens=ids)])
 else:
  source=spans();source[0]['tokens']=ids[:-7];history=InputHistory(source)
 histories=[]
 for i in range(128):
  history=history+[(i*31+3)%16];histories.append(history)
 measures=[]
 for repeat in range(3):
  lease=module._Lease(backend,bound=True,consumed=0,state='ACTIVE',_owner_thread=threading.get_ident())
  backend._active=lease;calls=0;queries=0;elapsed=0
  for history in histories:
   actual=dict(enabled=True,vision_protocol=VISION_PROTOCOL,consumed=len(history),**input_binding(history))
   lease.inputs=history;lease.consumed=len(history)
   started=time.perf_counter()
   for _ in range(3):backend._check_input_binding(history)
   elapsed+=time.perf_counter()-started
  measures.append(dict(seconds=elapsed,expected_binding_computations=calls,native_echo_queries=queries))
 result[kind]=dict(repetitions=measures,median_seconds=statistics.median(row['seconds'] for row in measures))
print(json.dumps(dict(scope='Isolated host binding checks against explicit CPU native-response stub; not GPU/whole-request speed',source=module.__file__,source_sha256=hashlib.sha256(open(module.__file__,'rb').read()).hexdigest(),python=platform.python_version(),prefix_cells=17857,steps=128,checks_per_step=3,results=result),indent=2))
