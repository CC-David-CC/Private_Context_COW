"""Stream private archive back and verify every member without extraction."""
import hashlib,json,subprocess,tarfile
from pathlib import Path,PurePosixPath

root=Path('/home/dflanag3/spatial-artifacts/q4-cli-overlap-20261016-v1/compact-backup-v2')
transfer=json.loads((root/'transfer.json').read_bytes())
class Reader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.bytes=0
    def read(self,size=-1):
        raw=self.stream.read(size);self.digest.update(raw);self.bytes+=len(raw);return raw

child=subprocess.Popen(['ssh','box-git','cat '+transfer['archive']],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stream=Reader(child.stdout);actual={};manifest=None;manifest_sha=None
try:
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        for member in archive:
            assert member.isfile() and not member.issym() and not member.islnk(),member.name
            name=PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts and member.name not in actual
            source=archive.extractfile(member);digest=hashlib.sha256();count=0;pieces=[]
            while raw:=source.read(1024**2):
                digest.update(raw);count+=len(raw)
                if member.name=='member-manifest.json':pieces.append(raw)
            assert count==member.size
            if member.name=='member-manifest.json':
                assert manifest is None
                manifest=json.loads(b''.join(pieces));manifest_sha=digest.hexdigest()
            else:actual[member.name]=dict(size_bytes=count,sha256=digest.hexdigest())
    while stream.read(1024**2):pass
    error=child.stderr.read();code=child.wait(timeout=20)
    assert code==0,error.decode('utf-8','replace')
    assert stream.bytes==transfer['archive_bytes'] and stream.digest.hexdigest()==transfer['archive_sha256']
    assert manifest is not None and manifest_sha==transfer['manifest_sha256']
    expected={row['path']:dict(size_bytes=row['size_bytes'],sha256=row['sha256']) for row in manifest['members']}
    assert len(expected)==len(manifest['members'])==transfer['member_count']
    assert actual==expected
    total=sum(row['size_bytes'] for row in actual.values())
    assert total==transfer['total_member_bytes']
    receipt=dict(schema='cow-q4-overlap-private-backup-readback-v1',status='PASS',archive=transfer['archive'],archive_bytes=stream.bytes,
        archive_sha256=stream.digest.hexdigest(),manifest_sha256=manifest_sha,verified_members=len(actual),verified_member_bytes=total,
        extracted_files=0,symlinks_or_hardlinks=0,ssh_exit_code=code)
    with (root/'readback.json').open('x') as output:json.dump(receipt,output,sort_keys=True,indent=2);output.write('\n')
    print(json.dumps(receipt,sort_keys=True))
finally:
    if child.poll() is None:
        child.terminate();child.wait(timeout=20)
