import hashlib,importlib,json,sys
from pathlib import Path
profile=json.loads(Path(sys.argv[1]).read_bytes())
runtime=Path(sys.argv[2]).resolve()
registry=Path(sys.argv[3]).resolve()
dependency=Path(sys.argv[4]).resolve() if sys.argv[4] else None
sys.path[:0]=([str(dependency)] if dependency else [])+[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src')]
rows={}
for name in ('context_time_travel.coordinator','context_time_travel.strata_backend','context_time_travel.real','context_time_travel.store','context_time_travel.grammar_first','serve.cow_grammar','serve.cow_http'):
 m=importlib.import_module(name);p=Path(m.__file__).resolve();rows[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'event':'runtime_imports','modules':rows}),flush=True)
sys.argv=['cow-http',*sys.argv[5:]]
importlib.import_module('serve.cow_http').main()
