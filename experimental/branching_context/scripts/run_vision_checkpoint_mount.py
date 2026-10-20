"""Root-only mount setup; inference is delegated to the ordinary user.

Use through sudo unshare --mount --fork. All filesystem stress is confined to
one pinned512MiB loop image; canonical history stays outside it. Diagnostic evidence stays outside that volume.
"""
import argparse,hashlib,json,os,pwd,subprocess,sys,time
from pathlib import Path

def _image_hash(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 cfg=json.loads(a.config.read_bytes());base=Path(cfg['fixture']).resolve(strict=True)
 assert os.geteuid()==0 and base.owner()==cfg['user']
 assert os.readlink('/proc/self/ns/mnt')!=cfg['parent_mount_namespace']
 user=pwd.getpwnam(cfg['user']);assert user.pw_uid!=0
 def owned(name):
  x=Path(cfg[name]).resolve(strict=True);assert x.is_relative_to(base) and x.stat().st_uid==user.pw_uid;return x
 image=owned('loop_image');store=owned('store');mount=store/'checkpoints'
 assert mount.is_dir() and mount.stat().st_uid==user.pw_uid and mount.resolve().is_relative_to(base)
 assert image.stat().st_size==512*1024**2 and image.stat().st_ino==cfg['expected_image_inode']
 assert subprocess.check_output(['blkid','-p','-s','UUID','-o','value',str(image)],text=True).strip()==cfg['expected_image_uuid']
 assert not subprocess.check_output(['losetup','-j',str(image)],text=True).strip()
 assert not list(mount.iterdir())
 subprocess.run(['mount','--make-rprivate','/'],check=True)
 loop=None;mounted=False;bound=False;code=1;failure=None
 started=time.time();kind='preflight' if a.preflight_only else 'gpu'
 try:
  loop=subprocess.check_output(['losetup','--find','--show','--',str(image)],text=True).strip()
  assert loop.startswith('/dev/loop') and loop[9:].isdigit()
  subprocess.run(['mount','-o','nosuid,nodev,noexec',loop,str(mount)],check=True);mounted=True
  os.chown(mount,user.pw_uid,user.pw_gid);os.chmod(mount,0o700)
  command=['runuser','-u',cfg['user'],'--',cfg['python'],'-B',cfg['user_launcher'],'--config',str(a.config)]
  if a.preflight_only:command.append('--preflight-only')
  code=subprocess.run(command,stdin=subprocess.DEVNULL,check=False).returncode
 except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc))
 finally:
  cleanup=[]
  for mounted_flag,target in [(mounted,mount)]:
   if mounted_flag:
    q=subprocess.run(['umount','--',str(target)],capture_output=True,text=True);cleanup.append(dict(target=str(target),exit=q.returncode,stderr=q.stderr))
  if loop:
   # Never detach an unverified or still-mounted loop device.
   associated=subprocess.check_output(['losetup','-j',str(image)],text=True)
   if associated.startswith(loop+':') and not any(x['exit'] for x in cleanup):
    q=subprocess.run(['losetup','-d',loop],capture_output=True,text=True);cleanup.append(dict(loop=loop,exit=q.returncode,stderr=q.stderr))
   else:cleanup.append(dict(loop=loop,exit=1,stderr='loop identity changed or unmount failed; left for inspection'))
  remaining=subprocess.check_output(['losetup','-j',str(image)],text=True)
  receipt=dict(schema=1,kind=kind,child_exit=code,failure=failure,cleanup=cleanup,remaining_loop=remaining,
   mountpoint_empty=not list(mount.iterdir()),parent_namespace=cfg['parent_mount_namespace'],private_namespace=os.readlink('/proc/self/ns/mnt'),
   elapsed_seconds=time.time()-started,image_sha256=_image_hash(image))
  path=base/(kind+'-mount-receipt.json');fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'w') as stream:json.dump(receipt,stream,indent=2);stream.flush();os.fsync(stream.fileno())
  os.chown(path,user.pw_uid,user.pw_gid)
  if failure or remaining or not receipt['mountpoint_empty'] or any(x['exit'] for x in cleanup):code=1
 return code
if __name__=='__main__':raise SystemExit(main())
