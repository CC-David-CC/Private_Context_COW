"""Compact private CLI evidence backup; excludes model/native payloads."""
import hashlib,io,json,os,subprocess,tarfile
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1')
report=json.loads((root/'qualified-primary-v1.json').read_bytes())
assert report['status']=='FOCUSED-PASS' and report['numerical_max_error']==0
output=root/'compact-backup-v1';output.mkdir(mode=0o700)
selected={}
def add_bytes(data,name):
    assert name not in selected,name
    assert len(data)<=40*1024**2,(name,len(data))
    selected[name]=data
def add(path,name):
    assert path.is_file() and not path.is_symlink()
    if path.suffix in ('.state','.tar','.gguf','.pack','.pyc') or path.name.endswith('.lock'):return
    if path.name.endswith('.decode.f32'):
        with path.open('rb') as stream:
            size=path.stat().st_size
            assert size%993280==0
            count=size//993280
            for token in sorted({1,count,*[n for n in (16,64) if n<=count]}):
                stream.seek((token-1)*993280)
                add_bytes(stream.read(993280),name+f'.selected-token-{token}.f32')
        return
    add_bytes(path.read_bytes(),name)
def tree(path,prefix):
    assert path.is_dir(),path
    for item in sorted(path.rglob('*')):
        if item.is_file():add(item,prefix+'/'+str(item.relative_to(path)))
for name in ('fixtures-v1','fixture-source-v1','campaign-source-v1','campaign-source-v2','reference-source-v1','reference-source-v2','reference-source-v3',
             'guard-red-source-v1','guard-green-source-v1','smoke-v1','smoke-v2','smoke-observed-v1','smoke-reference-v1','smoke-reference-v2',
             'smoke-observed-reference-v1','campaign-v1','campaign-reference-v1','workspace-store-v1','integrated-harness-v1'):
    tree(root/name,'experiment/'+name)
for item in sorted(root.iterdir()):
    if item.is_file():add(item,'experiment/'+item.name)
    elif item.is_dir() and 'command-v' in item.name and not item.name.startswith('backup-'):
        tree(item,'experiment/'+item.name)
observer=Path('/home/dflanag3/spatial-artifacts/q4-cli-event-timing-20261016-v1')
for name in ('green-v16','green-v17','red-v13','red-v14'):
    tree(observer/name,'observer/'+name)
profile=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/profile/profile.json')
add(profile,'profile/profile.json')
add(Path(__file__),'backup-script.py')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(selected.items())]
manifest=dict(schema='cow-q4-cli-compact-backup-v1',member_count=len(members),total_member_bytes=sum(row['size_bytes'] for row in members),members=members,
    scope='Executed CLI/timing/controller/history/byte/work evidence, native reference receipts and selected raw reference boundaries',
    selected_reference_generated_frontiers=[1,16,64,128],
    exclusions=['Model and native numerical .state files','Full per-token fresh decode arrays (all hashes retained; selected boundaries kept)',
                'Duplicate current runtime already backed up in qualified recovery archive SHA 9ee31f352a332522b7510e07356d676be1032a5161777d9d487d967e0ccb7fd3'])
assert manifest['total_member_bytes']<=256*1024**2
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(selected.items()))+[('member-manifest.json',raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
os.chmod(archive,0o600)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-cli-trees-v1'
subprocess.run(['ssh','box-git','test ! -e '+destination],check=True)
subprocess.run(['ssh','box-git','mkdir -m 700 '+destination],check=True)
subprocess.run(['scp',str(archive),'box-git:'+destination+'/evidence.tar.part'],check=True)
assert subprocess.check_output(['ssh','box-git','sha256sum '+destination+'/evidence.tar.part'],text=True).split()[0]==digest
subprocess.run(['ssh','box-git','chmod 600 '+destination+'/evidence.tar.part'],check=True)
subprocess.run(['ssh','box-git','mv '+destination+'/evidence.tar.part '+destination+'/evidence.tar'],check=True)
receipt=dict(status='TRANSFER-HASH-VERIFIED; MEMBER-READBACK-PENDING',archive=destination+'/evidence.tar',archive_bytes=archive.stat().st_size,
             archive_sha256=digest,manifest_sha256=hashlib.sha256(raw).hexdigest(),member_count=len(members),total_member_bytes=manifest['total_member_bytes'])
with (output/'transfer.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(receipt,sort_keys=True))
