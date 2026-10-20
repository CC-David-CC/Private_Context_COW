"""Compact terminal evidence, retaining canonical history and exact boundaries."""
import hashlib,io,json,os,subprocess,tarfile
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
def read(path):return json.loads(path.read_bytes())
primary=read(root/'primary-v1.json')
assert primary['status']=='FOCUSED-GPU-PASS; NOT-INTEGRATED' and primary['all_owned_pids_absent']
assert all(not Path('/proc',str(pid)).exists() for pid in primary['owned_pids'])
selected={}
def add_bytes(raw,name):
    assert name not in selected and len(raw)<40*1024**2,(name,len(raw))
    selected[name]=raw
def add(path,name):
    assert path.is_file() and not path.is_symlink(),str(path)
    if path.suffix in ('.state','.tar','.gguf','.pack','.pyc') or path.name.endswith(('.lock','.lck')):return
    add_bytes(path.read_bytes(),name)
def tree(directory,name):
    for item in sorted(directory.rglob('*')):
        if item.is_file():add(item,name+'/'+str(item.relative_to(directory)))
for name in ('red-stage-v3/source','green-stage-v1/source','green-stage-v4/source',
    'red-command-v2','red-command-v3','green-command-v1','green-command-v2','green-command-v3','green-command-v4',
    'baseline-branch-command-v1','focus-command-v5','focus-command-v6','g0-command-v6','g0-command-v7','g0-evidence-v7',
    'prepare-command-v1','gpu-harness-v1','gpu-smoke-v1','gpu-campaign-v1','gpu-candidate-v1',
    'gpu-smoke-command-v1','gpu-smoke-reference-command-v1','gpu-campaign-command-v1','gpu-campaign-reference-command-v1'):
    tree(root/name,'retention/'+name)
for item in sorted(root.iterdir()):
    if item.is_file() and item.suffix in ('.py','.sh','.json','.png','.svg'):add(item,'retention/'+item.name)
store=root/'workspace-store-v1'
for item in sorted(store.rglob('*')):
    if item.is_file() and 'checkpoints' not in item.relative_to(store).parts:
        add(item,'canonical-store/'+str(item.relative_to(store)))
fixtures=read(root/'gpu-candidate-v1/fixture-set.json')
for entry in fixtures['contexts']:
    path=Path(entry['fixture']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['fixture_sha256']
    add(path,'fixtures/'+str(entry['index'])+'/fixture.json')
    case=read(path);context=Path(case['context_file']);assert hashlib.sha256(context.read_bytes()).hexdigest()==case['context_sha256']
    add(context,'fixtures/'+str(entry['index'])+'/context.txt')
for phase in ('gpu-smoke-reference-v1','gpu-campaign-reference-v1'):
    directory=root/phase;receipt=read(directory/'result.json')
    add(directory/'result.json','references/'+phase+'/result.json')
    for row in receipt['references']:
        name=row['name'];base=directory/name
        for suffix in ('.process.json','.tokens','.stdout','.stderr','.f32.schedule.json'):
            add(base/(name+suffix),'references/'+phase+'/'+name+'/'+name+suffix)
        for suffix,key in (('.f32.requests.f32','prompt_frontier_sha256'),('.f32.decode.f32','generated_frontier_sha256')):
            with (base/(name+suffix)).open('rb') as stream:
                indices=range(len(row[key])) if key.startswith('prompt') else (len(row[key])-1,)
                for index in indices:
                    stream.seek(index*993280);raw=stream.read(993280)
                    assert hashlib.sha256(raw).hexdigest()==row[key][index]
                    add_bytes(raw,'references/'+phase+'/'+name+f'/{key}-{index}.f32')
members=[dict(path=name,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in sorted(selected.items())]
manifest=dict(schema='cow-bounded-tree-retention-private-backup-v1',members=members,member_count=len(members),
    total_member_bytes=sum(item['size_bytes'] for item in members),scope='Candidate code, CPU failures/passes, fresh profile/runtime, actual CLI traces, canonical history and selected independent full-vocabulary boundaries',
    exclusions=['Models and native .state payloads','Full per-token oracle arrays retained on llm-60; all hashes and selected exact boundaries included',
                'Red-v1 incomplete extraction/source archive; source transfer race recorded in report, no test executed'])
assert manifest['total_member_bytes']<256*1024**2,manifest['total_member_bytes']
output=root/'compact-backup-v1';output.mkdir(mode=0o700)
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(selected.items()))+[('member-manifest.json',raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
archive.chmod(0o600);digest=hashlib.sha256(archive.read_bytes()).hexdigest()
destination='/srv/data/git/private_context_cow.evidence/20261016-q4-bounded-tree-retention-v1'
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
