"""Private compact backup of fresh candidate fixture/reference evidence."""
import hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
base=Path('/home/dflanag3/spatial-artifacts')
root=base/'q4-cuda-recovery-20261016-v1'
output=root/'compact-recovery-v1';output.mkdir(mode=0o700)
selected={}
def add(path,name):
    assert path.is_file() and not path.is_symlink()
    if path.suffix in ('.state','.tar','.gguf','.pack','.pyc') or path.name.endswith('.lock'):return
    data=path.read_bytes();assert len(data)<40*1024**2,(name,len(data))
    assert name not in selected,name
    selected[name]=data
def tree(directory,prefix):
    assert directory.is_dir(),directory
    for p in sorted(directory.rglob('*')):
        if p.is_file():add(p,prefix+'/'+str(p.relative_to(directory)))
for name in ('fixture-source','fixture-source-v2','inputs','tiny-prepared-v1','tiny-reference-v1','tiny-reference-v2','tiny-input-store-v1','tiny-recovery-v1','tiny-recovery-v2','target-prepared-v1','target-reference-v2','target-input-store-v1','target-recovery-v1'):
    tree(root/name,'fixture/'+name)
for p in sorted(root.iterdir()):
    if p.is_file():add(p,'fixture/'+p.name)
    elif p.is_dir() and ('command-v' in p.name or 'preflight-v' in p.name) and not p.name.startswith('backup-recovery'):tree(p,'fixture/'+p.name)
for name in ('http-control-v2','http-control-v3'):
    for p in sorted((root/name).iterdir()):
        if p.is_file():add(p,'fixture/'+name+'/'+p.name)
        elif p.name.startswith('run'):tree(p,'fixture/'+name+'/'+p.name)
for name in ('integrated-62b4e466','integrated-62b4e466-v2'):
    for p in sorted((root/name).iterdir()):
        if p.is_file():add(p,'integrated/'+name+'/'+p.name)
        elif p.name.endswith('-command'):tree(p,'integrated/'+name+'/'+p.name)
harness=base/'q4-cuda-recovery-harness-20261016-v1'
for name in ('cpu-final-v4','cpu-token-only-v2'):
    tree(harness/name,'recovery-harness/'+name)
profile=base/'q4-terminal-error-profile-20261016-v3'
tree(profile,'profile-v3')
for version in (1,2,3):
    stage=base/('q4-terminal-error-profile-20261016-v'+str(version))
    for p in sorted(stage.iterdir()):
        if p.is_file():add(p,'profile-preparation-v'+str(version)+'/'+p.name)
        elif p.name=='profile' and version!=3:tree(p,'profile-preparation-v'+str(version)+'/profile')
    run=base/('q4-terminal-error-profile-run-20261016-v'+str(version))
    if run.is_dir():tree(run,'profile-preparation-run-v'+str(version))
add(Path(__file__),'backup-script.py')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(selected.items())]
manifest=dict(schema=1,member_count=len(members),total_member_bytes=sum(x['size_bytes'] for x in members),members=members,
    scope='Qualified Q4 tiny and 7,999-token CUDA OOM recovery, independent fresh/reference raw vectors, coherent profile/runtime and integrated CPU/HTTP/G0 controls. No cumulative gate certification.',
    exclusions=['Model and numerical .state payloads','Duplicated HTTP test source trees','Redundant integrated source trees (exact main Git commit and full source manifests retained)'])
mraw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as f:
    for name,data in list(sorted(selected.items()))+[('member-manifest.json',mraw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);f.addfile(info,io.BytesIO(data))
os.chmod(archive,0o600)
asha=hashlib.sha256(archive.read_bytes()).hexdigest()
dest='/srv/data/git/private_context_cow.evidence/20261016-q4-cuda-recovery-qualified-v1'
subprocess.run(['ssh','box-git','test ! -e '+dest],check=True)
subprocess.run(['ssh','box-git','mkdir -m 700 '+dest],check=True)
subprocess.run(['scp',str(archive),'box-git:'+dest+'/evidence.tar.part'],check=True)
assert subprocess.check_output(['ssh','box-git','sha256sum '+dest+'/evidence.tar.part'],text=True).split()[0]==asha
subprocess.run(['ssh','box-git','chmod 600 '+dest+'/evidence.tar.part'],check=True)
subprocess.run(['ssh','box-git','mv '+dest+'/evidence.tar.part '+dest+'/evidence.tar'],check=True)
r=dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',schema='qualified-recovery-backup-v1',remote_archive=dest+'/evidence.tar',archive_sha256=asha,archive_bytes=archive.stat().st_size,manifest_sha256=hashlib.sha256(mraw).hexdigest(),member_count=len(members),total_member_bytes=manifest['total_member_bytes'])
with (output/'transfer.json').open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(r,sort_keys=True))