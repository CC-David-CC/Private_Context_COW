"""Bounded private evidence backup; excludes model and numerical state payloads."""
import hashlib, io, json, os
from pathlib import Path
import subprocess, tarfile
base=Path('/home/dflanag3/spatial-artifacts')
root=base/'q4-cuda-oom-20261016-v1'
worker=base/'q4-terminal-error-cpu-20261016-v1'
output=root/'compact-baseline-v2'
output.mkdir(mode=0o700)
selected={}
def add(path, name):
    if path.is_symlink() or not path.is_file(): raise RuntimeError(str(path))
    if name in selected: raise RuntimeError('duplicate '+name)
    if path.suffix in ('.state','.gguf','.pack'): raise RuntimeError('forbidden payload')
    data=path.read_bytes()
    if len(data)>32*1024**2: raise RuntimeError('unexpected large file '+str(path))
    selected[name]=data
for directory in ('gpu-command-v1','gpu-command-v2','root-preflight-v1','root-preflight-v2','root-audit-v1','gpu-q4-v2/campaign'):
    for path in sorted((root/directory).rglob('*')):
        if path.is_file(): add(path,'oom/'+str(path.relative_to(root)))
for name in ('preparation.json','primary-audit-v2.json','preconditions-primary-v1.json','preconditions-backup-primary-v1.json','launch-v1.sh','launch-v2.sh','audit-baseline-v1.py'):
    add(root/name,'oom/'+name)
for rel in ('run/tests-v1','source/experimental/branching_context/scripts','source/experimental/branching_context/tests'):
    for path in sorted((root/'cpu-lifecycle-v5'/rel).rglob('*')):
        if path.is_file(): add(path,'oom/'+str(path.relative_to(root)))
for path in sorted((root/'input-store-v1').rglob('*')):
    if path.is_file() and path.suffix != '.state' and path.name not in ('pool.lock','store.lock'):
        add(path,'non-numerical-store/'+str(path.relative_to(root/'input-store-v1')))
for stage, folders in (('replacement-root-red-v4',('run',)),('replacement-root-green-v4',('run','g0','source'))):
    for folder in folders:
        for path in sorted((worker/stage/folder).rglob('*')):
            if path.is_file(): add(path,'worker/'+str(path.relative_to(worker)))
    add(worker/stage/'report.json','worker/'+stage+'/report.json')
add(worker/'replacement-root-red-v4/source/src/context_time_travel/coordinator.py','worker/red-coordinator.py')
for name in ('worker_recovery_cpu_review_v1.py','stage_worker_recovery_root_v2.sh'):
    add(worker/name,'worker/'+name)
add(Path(__file__),'backup-script.py')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(selected.items())]
manifest=dict(schema=1,member_count=len(members),total_member_bytes=sum(row['size_bytes'] for row in members),members=members,exclusions=['Model files','All numerical .state payloads','Complete historical artifact stages; selected receipts only'],scope='Actual Q4 OOM baseline and exact frozen CPU candidate; no GPU qualification of candidate')
manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as tar:
    for name,data in list(sorted(selected.items()))+[('member-manifest.json',manifest_bytes)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);tar.addfile(info,io.BytesIO(data))
os.chmod(archive,0o600)
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-cuda-oom-baseline-cpu-replacement-v1'
subprocess.run(['ssh','box-git','test ! -e '+destination],check=True)
subprocess.run(['ssh','box-git','mkdir -m 700 '+destination],check=True)
subprocess.run(['scp',str(archive),'box-git:'+destination+'/evidence.tar.part'],check=True)
received=subprocess.check_output(['ssh','box-git','sha256sum '+destination+'/evidence.tar.part'],text=True).split()[0]
if received!=archive_sha: raise RuntimeError('receiver checksum')
subprocess.run(['ssh','box-git','chmod 600 '+destination+'/evidence.tar.part'],check=True)
subprocess.run(['ssh','box-git','mv '+destination+'/evidence.tar.part '+destination+'/evidence.tar'],check=True)
receipt=dict(schema='q4-cuda-oom-compact-backup-v1',status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',remote_archive=destination+'/evidence.tar',archive_bytes=archive.stat().st_size,archive_sha256=archive_sha,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),member_count=len(members),total_member_bytes=manifest['total_member_bytes'])
with (output/'transfer.json').open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(receipt,sort_keys=True),flush=True)