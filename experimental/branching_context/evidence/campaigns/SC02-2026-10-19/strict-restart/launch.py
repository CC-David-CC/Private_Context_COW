from pathlib import Path
import subprocess,os,json
d=Path(__file__).parent
source=d.parent/'standalone-SC02-q8-retained-v2'
profile=json.loads((source/'profile.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))}
env.update(profile['execution']['environment'])
env['PYTHONDONTWRITEBYTECODE']='1'
env['CTT_NATIVE_TRACE']=str(d/'native.jsonl')
with (d/'stdout').open('xb') as o,(d/'stderr').open('xb') as e:
 p=subprocess.run(['flock','-n','/home/dflanag3/branching-g3/native-client-speed.gpu0.lock','/home/dflanag3/src/Strata/.venv/bin/python','-B',str(d/'run.py')],stdout=o,stderr=e,env=env)
(d/'command.exit').write_text(str(p.returncode))
raise SystemExit(p.returncode)
