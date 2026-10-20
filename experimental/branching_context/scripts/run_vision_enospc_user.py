import argparse,json,os,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
c=json.loads(a.config.read_bytes());b=Path(c['artifact_base']);runtime=Path(c['runtime']);bw=Path(c['branchwrite']);profile=Path(c['profile'])
assert hashlib.sha256(profile.read_bytes()).hexdigest()==c['profile_sha256']
pr=json.loads(profile.read_bytes())
env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))};env.update(pr['execution']['environment'])
paths=[bw,runtime,runtime/'tools',runtime/'serve',runtime/'experimental/branching_context/src',runtime/'experimental/branching_context/scripts',Path(c['dependency_root'])]
env.update(PYTHONPATH=os.pathsep.join(map(str,paths)),PYTHONDONTWRITEBYTECODE='1',COW_PYTHON=c['python'])
args=[c['python'],'-B',c['probe']]
for name in ['runtime','branchwrite','profile','dependency_root','vision_config','fixtures','output','store','loop_image','parent_mount_namespace','expected_image_uuid','expected_image_inode']:
 args.extend(['--'+name.replace('_','-'),str(c[name])])
if a.preflight_only:args.append('--preflight-only')
os.execve(c['python'],args,env)
