"""Archive terminal policy comparison and cleanup evidence, excluding KV payloads."""
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
pair=read(parent/'policy-pair-v1.json')
assert pair['status']=='CORRECTNESS-PASS; TIMING-REPORTED'
assert all(not Path('/proc',str(pid)).exists() for pid in pair['owned_pids'])
files={}
def raw(data,name):
    assert name not in files and len(data)<40*1024**2,(name,len(data))
    files[name]=data
def add(path,name):
    assert path.is_file() and not path.is_symlink(),str(path)
    if path.suffix in ('.state','.gguf','.pack','.pyc','.tar') or path.name.endswith(('.lock','.lck')):return
    raw(path.read_bytes(),name)
def tree(directory,name):
    assert directory.is_dir(),str(directory)
    for path in sorted(directory.rglob('*')):
        if path.is_file():add(path,name+'/'+str(path.relative_to(directory)))
for root,names,prefix in (
    (control,('runtime','campaign-v1','campaign-command-v1','reference-command-v1'), 'control'),
    (parent,('gpu-repeated-v2','gpu-repeated-reference-v2','gpu-repeated-command-v2',
        'gpu-repeated-reference-command-v2','pair-export-command-v1','eviction-plan-command-v1',
        'eviction-command-v1','prepare-control-command-v1'), 'candidate')):
    for name in names:tree(root/name,prefix+'/'+name)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in ('.json','.py','.sh'):
            add(path,prefix+'/root-files/'+path.name)
for directory,prefix in ((control/'store-v1','control/store'),(parent/'workspace-store-v1','candidate/store')):
    tree(directory,prefix)
# Preserve every receipt/hash but only selected full vectors from the new
# independent control references. All per-token arrays remain on llm-60.
directory=control/'reference-v1';receipt=read(directory/'result.json')
for path in directory.iterdir():
    if path.is_file():add(path,'control/reference-v1/'+path.name)
for row in receipt['references']:
    name=row['name'];base=directory/name
    for suffix in ('.process.json','.tokens','.stdout','.stderr','.f32.schedule.json'):
        add(base/(name+suffix),'control/reference-v1/'+name+'/'+name+suffix)
    for suffix,key in (('.f32.requests.f32','prompt_frontier_sha256'),('.f32.decode.f32','generated_frontier_sha256')):
        with (base/(name+suffix)).open('rb') as stream:
            indices=range(len(row[key])) if key.startswith('prompt') else (len(row[key])-1,)
            for index in indices:
                stream.seek(index*993280);data=stream.read(993280)
                assert hashlib.sha256(data).hexdigest()==row[key][index]
                raw(data,'control/reference-v1/'+name+f'/{key}-{index}.f32')
# Candidate reference arrays are in the preceding verified compact archive and
# remain on llm-60. Preserve the exact reused result that binds those hashes.
add(parent/'gpu-campaign-reference-v1/result.json','candidate/reused-original-reference-result.json')
fixtures=read(control/'fixture-set.json')
for row in fixtures['contexts']:
    path=Path(row['fixture']);assert sha(path)==row['fixture_sha256']
    add(path,'fixtures/'+str(row['index'])+'/fixture.json')
    case=read(path);context=Path(case['context_file']);assert sha(context)==case['context_sha256']
    add(context,'fixtures/'+str(row['index'])+'/context.txt')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(files.items())]
manifest=dict(schema='cow-tree-policy-pair-backup-v1',members=members,member_count=len(members),
    total_member_bytes=sum(row['size_bytes'] for row in members),
    scope='Terminal equal-budget control/candidate runs, raw CLI and native traces, canonical history, checkpoint manifests, source pins, selected exact reference vectors, audited cache eviction',
    exclusions=['Model files and native checkpoint .state payloads',
        'Full per-token oracle arrays retained on llm-60; hashes and selected boundaries preserved',
        'Separate unintegrated publication ownership work'])
assert manifest['total_member_bytes']<320*1024**2
output=parent/'pair-backup-v1';output.mkdir(mode=0o700)
manifest_raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(files.items()))+[('member-manifest.json',manifest_raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
archive.chmod(0o600);digest=sha(archive)
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-tree-policy-pair-v1'
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
