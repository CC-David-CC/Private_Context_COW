"""Compact reviewed CPU evidence; preserve failed packagers without executing them."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/publication-handoff-ownership-20261016-v1')
review=root/'controller-review-v1'
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
qualified=read(review/'PUBLICATION-HANDOFF-OWNERSHIP-CPU-2026-10-16.json')
assert qualified['status']=='PASS' and [row['tests'] for row in qualified['qualification']]==[8,37,51]
for row in qualified['qualification']:
    assert row['passed']==row['tests'] and row['failed']==row['skipped']==0
    for name in ('command.exit','wrapper.exit'):
        assert (root/row['stage']/name).read_text().strip()=='0',(row['stage'],name)
for name,expected in qualified['final_source']['files'].items():
    assert sha(Path(qualified['final_source']['path'])/name)==expected
red=root/'red-final-principal-v1/run-tests'
assert all((red/name).read_text().strip()=='1' for name in ('command.exit','wrapper.exit'))
assert 'retry discarded the principal-bound model' in (red/'stderr.log').read_text()
files={}
def add(path,name):
    assert path.is_file() and not path.is_symlink(),str(path)
    if path.suffix in ('.state','.tar','.gguf','.pack','.pyc'):return
    data=path.read_bytes();assert name not in files and len(data)<16*1024**2
    files[name]=data
def tree(path,prefix):
    for item in sorted(path.rglob('*')):
        if item.is_file():add(item,prefix+'/'+str(item.relative_to(path)))
tree(root/'backup-input-v3/evidence-package','worker-evidence-package')
tree(review,'controller-review')
for row in qualified['source_mappings']:
    for name,key in (('experimental/branching_context/src/context_time_travel/coordinator.py','coordinator_sha256'),
        ('experimental/branching_context/tests/implementation/test_publication_handoff_ownership.py','test_sha256')):
        path=root/row['stage']/'source'/name
        assert sha(path)==row[key],(row['stage'],name)
        add(path,'stage-owned-sources/'+row['stage']+'/'+name)
tree(root/'red-final-principal-v1/run-tests','unchanged-final-test-red')
for source in ('backup-build-command-v1','backup-build-command-v2','backup-build-command-v3'):
    if (root/source).is_dir():tree(root/source,'packaging-attempts/'+source)
for item in root.iterdir():
    if item.is_file() and item.suffix in ('.json','.md','.py','.sh','.sha256'):
        add(item,'root-files/'+item.name)
# Include selected failed packagers/tools and their receipts, never their
# partially duplicated bundles. Their failure is distinct from tested code.
for stage in ('backup-input-v1','backup-input-v2','backup-input-v3'):
    directory=root/stage
    if not directory.exists():continue
    for item in directory.iterdir():
        if item.is_file() and item.suffix in ('.py','.sh','.json','.sha256'):add(item,'failed-packagers/'+stage+'/'+item.name)
    if (directory/'backup-tools').is_dir():tree(directory/'backup-tools','failed-packagers/'+stage+'/backup-tools')
members=[dict(path=name,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(files.items())]
manifest=dict(schema='cow-publication-ownership-private-backup-v1',members=members,member_count=len(members),
    total_member_bytes=sum(row['size_bytes'] for row in members),base_commit=qualified['base_commit'],
    scope='Reviewed CPU8/regressions37/G0-51, unchanged final-test RED, all owned source versions, prior failures/interrupted tests and packaging setup failures',
    exclusions=['Model/native state; no GPU work was executed','Unchanged full source closure available from private Git base plus owned-source overlays',
        'Partially duplicated packaging bundles; their scripts and failure receipts retained'])
assert manifest['total_member_bytes']<64*1024**2
output=root/'controller-backup-v1';output.mkdir(mode=0o700)
manifest_raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
archive=output/'evidence.tar'
with tarfile.open(archive,'x') as stream:
    for name,data in list(sorted(files.items()))+[('member-manifest.json',manifest_raw)]:
        info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(data);stream.addfile(info,io.BytesIO(data))
archive.chmod(0o600);digest=sha(archive)
destination='/srv/data/git/private_context_cow.evidence/20261016-publication-handoff-ownership-controller-v1'
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
