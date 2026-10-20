"""Compact private backup of terminal overlap experiments and exact references."""
import hashlib,io,json,os,subprocess,tarfile
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-overlap-20261016-v1')
base=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1')
primary=json.loads((root/'primary-v1.json').read_bytes())
assert primary['status']=='FOCUSED-PASS' and primary['numerical_max_error']==0
output=root/'compact-backup-v2';output.mkdir(mode=0o700)
selected={}
def add_bytes(data,name):
    assert name not in selected and len(data)<40*1024**2,(name,len(data))
    selected[name]=data
def add(path,name):
    assert path.is_file() and not path.is_symlink()
    if path.suffix in ('.state','.tar','.gguf','.pack','.pyc') or path.name.endswith('.lock'):return
    add_bytes(path.read_bytes(),name)
for item in sorted(root.rglob('*')):
    relative=item.relative_to(root)
    if relative.parts[0].startswith(('compact-backup-','backup-command','backup-readback-command')):continue
    if item.is_file():add(item,'overlap/'+str(relative))
for directory,prefix in ((base/'fixtures-v1','fixtures'),(base/'workspace-store-v1','canonical-store')):
    for item in sorted(directory.rglob('*')):
        if item.is_file():add(item,prefix+'/'+str(item.relative_to(directory)))
reference=base/'campaign-reference-v1';receipt=json.loads((reference/'result.json').read_bytes())
add(reference/'result.json','independent-reference/result.json')
for index,question in ((0,'A'),(1,'A'),(2,'A'),(0,'B')):
    name=f'tree{index}-{question}-fresh0';saved=next(row for row in receipt['references'] if row['name']==name)
    directory=reference/name
    for suffix in ('.process.json','.tokens','.stdout','.stderr','.f32.schedule.json'):
        add(directory/(name+suffix),'independent-reference/'+name+'/'+name+suffix)
    with (directory/(name+'.f32.decode.f32')).open('rb') as stream:
        for token in (8,128):
            stream.seek((token-1)*993280);raw=stream.read(993280)
            assert hashlib.sha256(raw).hexdigest()==saved['generated_frontier_sha256'][token-1]
            add_bytes(raw,'independent-reference/'+name+f'/consumed-output-{token}.f32')
profile=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json')
add(profile,'profile/profile.json')
cpu=Path('/home/dflanag3/spatial-artifacts/publication-overlap-cpu-20261016-v1')
for directory in ('green-v1','green-v2'):
    for item in sorted((cpu/directory).rglob('*')):
        if item.is_file():add(item,'worker-cpu/'+directory+'/'+str(item.relative_to(cpu/directory)))
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(selected.items())]
manifest=dict(schema='cow-q4-overlap-backup-v1',members=members,member_count=len(members),total_member_bytes=sum(row['size_bytes'] for row in members),
    scope='Controlled GPU publication hold and unmodified-server natural queue benchmark, exact reference boundaries, canonical metadata, CPU/G0 controls and preserved setup failures',
    exclusions=['Model and native numerical .state files','Full reference arrays already retained on llm-60 and selected in prior CLI backup',
                'Current runtime source already backed up in qualified recovery archive SHA 9ee31f352a332522b7510e07356d676be1032a5161777d9d487d967e0ccb7fd3'])
# The canonical objects alone occupy about 104 MiB; retain them all.
assert manifest['total_member_bytes']<256*1024**2
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(selected.items()))+[('member-manifest.json',raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
os.chmod(archive,0o600)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-cli-publication-overlap-v1'
subprocess.run(['ssh','box-git','test ! -e '+destination],check=True)
subprocess.run(['ssh','box-git','mkdir -m 700 '+destination],check=True)
subprocess.run(['scp',str(archive),'box-git:'+destination+'/evidence.tar.part'],check=True)
assert subprocess.check_output(['ssh','box-git','sha256sum '+destination+'/evidence.tar.part'],text=True).split()[0]==digest
subprocess.run(['ssh','box-git','chmod 600 '+destination+'/evidence.tar.part'],check=True)
subprocess.run(['ssh','box-git','mv '+destination+'/evidence.tar.part '+destination+'/evidence.tar'],check=True)
report=dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',archive=destination+'/evidence.tar',archive_bytes=archive.stat().st_size,
            archive_sha256=digest,manifest_sha256=hashlib.sha256(raw).hexdigest(),member_count=len(members),total_member_bytes=manifest['total_member_bytes'])
with (output/'transfer.json').open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(report,sort_keys=True))
