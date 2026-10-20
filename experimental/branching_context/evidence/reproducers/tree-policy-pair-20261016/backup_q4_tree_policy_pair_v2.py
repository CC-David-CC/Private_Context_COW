"""Archive the quiescent pair and retain provenance of the earlier mixed result."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

parent=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
control=Path('/home/dflanag3/spatial-artifacts/q4-tree-policy-control-20261016-v1')
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pair=read(parent/'policy-pair-v2.json');final=read(parent/'policy-final-consistency-v1.json')
assert final['status']=='PASS' and pair['all_owned_pids_absent']
assert all(not Path('/proc',str(pid)).exists() for pid in pair['owned_pids'])
files={}
def add(path,name):
    assert path.is_file() and not path.is_symlink(),str(path)
    if path.suffix in ('.state','.gguf','.pack','.pyc','.tar') or path.name.endswith(('.lock','.lck')):return
    data=path.read_bytes();assert name not in files and len(data)<40*1024**2,(name,len(data))
    files[name]=data
def tree(directory,name):
    assert directory.is_dir(),str(directory)
    for path in sorted(directory.rglob('*')):
        if path.is_file():add(path,name+'/'+str(path.relative_to(directory)))
for root,names,prefix in (
    (control,('runtime','campaign-quiet-v2','campaign-quiet-command-v2','reference-quiet-v2','reference-quiet-command-v2','store-v1'),'control'),
    (parent,('gpu-quiet-v3','gpu-quiet-command-v3','gpu-quiet-reference-v3','gpu-quiet-reference-command-v3',
        'pair-export-command-v2','pair-final-audit-command-v1','pair-plot-command-v1',
        'pair-protocol-attribution-command-v1','pair-event-attribution-command-v1',
        'pair-catalog-attribution-command-v1','pair-catalog-attribution-command-v2',
        'pair-eviction-plan-command-v1','pair-eviction-command-v1','workspace-store-v1'),'candidate')):
    for name in names:tree(root/name,prefix+'/'+name)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in ('.json','.py','.sh','.png'):
            add(path,prefix+'/root-files/'+path.name)
for name in ('transfer.json','readback.json'):
    add(parent/'pair-backup-v1'/name,'prior-backup/'+name)
for reference,prefix in ((control/'reference-v1','reused-control'),(parent/'gpu-campaign-reference-v1','reused-candidate')):
    add(reference/'result.json',prefix+'/result.json')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(files.items())]
manifest=dict(schema='cow-tree-policy-pair-backup-v2',members=members,member_count=len(members),
    total_member_bytes=sum(row['size_bytes'] for row in members),
    scope='Quiescent pair traces, exactness receipts, final full-payload audit, source pins, canonical histories, checkpoint manifests, graphs and earlier mixed-pair results/backup/eviction receipts',
    exclusions=['Model files and native .state payloads',
        'Full per-token oracle arrays remain on llm-60; all hashes and selected vectors are in the preceding verified archives',
        'Separate unintegrated publication ownership work'])
assert manifest['total_member_bytes']<320*1024**2
output=parent/'pair-backup-v2';output.mkdir(mode=0o700)
manifest_raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(files.items()))+[('member-manifest.json',manifest_raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
archive.chmod(0o600);digest=sha(archive)
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-tree-policy-pair-v2'
subprocess.run(['ssh','box-git','test ! -e '+destination],check=True)
subprocess.run(['ssh','box-git','mkdir -m 700 '+destination],check=True)
subprocess.run(['scp',str(archive),'box-git:'+destination+'/evidence.tar.part'],check=True)
assert subprocess.check_output(['ssh','box-git','sha256sum '+destination+'/evidence.tar.part'],text=True).split()[0]==digest
subprocess.run(['ssh','box-git','chmod 600 '+destination+'/evidence.tar.part'],check=True)
subprocess.run(['ssh','box-git','mv '+destination+'/evidence.tar.part '+destination+'/evidence.tar'],check=True)
receipt=dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',archive=destination+'/evidence.tar',
    archive_bytes=archive.stat().st_size,archive_sha256=digest,manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    member_count=len(members),total_member_bytes=manifest['total_member_bytes'])
with (output/'transfer.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(receipt,sort_keys=True))
