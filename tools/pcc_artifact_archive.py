"""Owner-only verified SSH artifact copies. This command never retires a source."""
from pathlib import Path
import argparse,hashlib,json,os,re,shlex,stat,subprocess

REMOTE=r'''
def require(value,message):
 if not value:raise ValueError(message)
from pathlib import Path
import hashlib, json, os, shutil, stat, subprocess, sys
os.umask(63)
s = json.loads(sys.argv[1])
mount = Path(s['mount'])
target = Path(s['destination'])
require(mount.is_absolute() and mount.resolve(strict=True) == mount, 'Archive invariant failed')
device = json.loads(subprocess.check_output(['findmnt', '-J', '-o', 'TARGET,UUID', '--mountpoint', str(mount)], text=True))['filesystems'][0]
require(device['target'] == str(mount) and device['uuid'] == s['mount_uuid'], 'Archive mount identity differs')
require(target.is_absolute() and target != mount and target.is_relative_to(mount), 'Archive invariant failed')
require(target.resolve() == target, 'Archive path contains a link or parent traversal')
target.parent.mkdir(parents=True, exist_ok=True, mode=448)
require(target.parent.stat().st_uid == os.getuid(), 'Archive directory is not owned by this account')
pending = target.with_name(target.name + '.partial')
receipt = target.with_name(target.name + '.archive.json')

def regular(p):
    st = p.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid(), 'Archive invariant failed')
    return st

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def syncdir():
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
if target.exists() or receipt.exists():
    require(target.exists() and receipt.exists(), 'Incomplete archive publication; inspect original copy')
    require(regular(target).st_size == s['bytes'], 'Archive invariant failed')
    regular(receipt)
    value = json.loads(receipt.read_bytes())
    require(value['sha256'] == s['sha256'] and value['bytes'] == s['bytes'] and (value['destination'] == str(target)), 'Archive invariant failed')
    require(value['mount_uuid'] == s['mount_uuid'] and value['status'] == 'VERIFIED_ARCHIVE', 'Archive invariant failed')
    require(sha(target) == s['sha256'], 'Archived bytes differ from their receipt')
    value['reused'] = True
    print(json.dumps(value))
    sys.exit(0)
require(not pending.exists() and (not pending.is_symlink()), 'Original transfer is incomplete; never overwrite it')
require(shutil.disk_usage(mount).free >= s['bytes'] + (1 << 20), 'Archive storage admission refused')
if s['action'] == 'probe':
    print(json.dumps(dict(status='READY')))
    sys.exit(0)
require(s['action'] == 'store', 'Archive invariant failed')
with pending.open('xb') as f:
    remaining = s['bytes']
    while remaining:
        chunk = sys.stdin.buffer.read(min(4 << 20, remaining))
        require(chunk, 'Truncated archive input')
        f.write(chunk)
        remaining -= len(chunk)
    require(not sys.stdin.buffer.read(1), 'Archive input exceeds pinned size')
    f.flush()
    os.fsync(f.fileno())
require(regular(pending).st_size == s['bytes'] and sha(pending) == s['sha256'], 'Archive byte verification failed')
value = dict(schema=1, status='VERIFIED_ARCHIVE', destination=str(target), mount_uuid=s['mount_uuid'], sha256=s['sha256'], bytes=s['bytes'], reused=False, source_retired=False)
os.link(pending, target)
pending.unlink()
syncdir()
temporary = receipt.with_name(receipt.name + '.partial')
with temporary.open('x') as f:
    json.dump(value, f, sort_keys=True)
    f.write('\n')
    f.flush()
    os.fsync(f.fileno())
os.link(temporary, receipt)
temporary.unlink()
syncdir()
print(json.dumps(value))
'''


def copy_artifact(source,expected_sha256,*,host,mount,mount_uuid,destination,receipt,timeout=3600):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256):raise ValueError('Pinned SHA256 required')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@:-]{0,253}',host):raise ValueError('Explicit SSH host required')
    if type(timeout) is not int or not 1<=timeout<=86400:raise ValueError('Bounded copy lifecycle required')
    source=Path(source).absolute();receipt=Path(receipt).absolute()
    if source.resolve(strict=True)!=source or source.is_symlink():raise ValueError('Source link rejected')
    if receipt.exists():raise FileExistsError('Inspect the existing local archive receipt')
    def identity():
        s=source.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid():raise ValueError('Owned regular source required')
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before=identity()
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha256:raise ValueError('Source SHA256 differs')
    if identity()!=before:raise ValueError('Source changed during preflight')
    specification=dict(mount=str(mount),mount_uuid=mount_uuid,destination=str(destination),
                       sha256=expected_sha256,bytes=before[2],action='probe')
    def remote(action,stream=None):
        args=dict(specification,action=action)
        command='python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(args))
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',host,command],
                              stdin=stream,capture_output=True,text=True,timeout=timeout)
        if result.returncode:raise RuntimeError('Archive operation refused: '+result.stderr[-2000:])
        return json.loads(result.stdout)
    value=remote('probe')
    if value['status']=='READY':
        with source.open('rb') as stream:value=remote('store',stream)
    if identity()!=before:raise ValueError('Source changed while copying; archive not admitted locally')
    if value['status']!='VERIFIED_ARCHIVE' or value['sha256']!=expected_sha256 or value['bytes']!=before[2]:
        raise ValueError('Archive receipt differs from pinned source')
    value=dict(value,source=str(source),host=host,source_retired=False)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','sha256','host','mount','mount-uuid','destination','receipt'):p.add_argument('--'+key,required=True)
    p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
    print(json.dumps(copy_artifact(a.source,a.sha256,host=a.host,mount=a.mount,mount_uuid=a.mount_uuid,
                                  destination=a.destination,receipt=a.receipt,timeout=a.timeout)))


if __name__=='__main__':main()
