"""Start one configured Context COW worker without putting its API key in OS argv."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import shutil
from contextlib import contextmanager,ExitStack

def clean_abandoned_sessions(root):
    """Remove only marked, unlocked directories directly inside our session root."""
    root=Path(root).resolve(strict=True)
    for session in root.iterdir():
        if not session.name.startswith('session-') or session.is_symlink() or not session.is_dir():continue
        marker=session/'session.json';lease=session/'lease.lock'
        if marker.is_symlink() or lease.is_symlink() or not marker.is_file() or not lease.is_file():continue
        try:
            if json.loads(marker.read_text())!={'schema':1,'session':session.name,'uid':os.getuid()}:continue
            fd=os.open(lease,os.O_RDWR|os.O_NOFOLLOW)
            with os.fdopen(fd,'r+') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if session.resolve().parent!=root:raise ValueError('Session escaped its cleanup root')
                shutil.rmtree(session)
        except (FileNotFoundError,json.JSONDecodeError):continue

@contextmanager
def session_storage(directory,config):
    """Ephemeral stores contain both numerical checkpoints and session history."""
    ephemeral=config.get('ephemeral',False)
    if type(ephemeral) is not bool:raise ValueError('ephemeral must be a boolean')
    if not ephemeral:
        yield Path(directory),config
        return
    root=Path(directory)/'.ephemeral-sessions'
    if root.is_symlink():raise ValueError('Ephemeral session root cannot be a symlink')
    root.mkdir(mode=0o700,exist_ok=True)
    stat=root.stat()
    if stat.st_uid!=os.getuid() or stat.st_mode&0o077:
        raise ValueError('Ephemeral session root must be private and owned by this account')
    clean_abandoned_sessions(root)
    with tempfile.TemporaryDirectory(prefix='session-',dir=root) as folder:
        session=Path(folder)
        with (session/'lease.lock').open('x+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (session/'session.json').write_text(json.dumps({'schema':1,'session':session.name,'uid':os.getuid()}))
            yield session,dict(config,store=str(session/'store'))

@contextmanager
def gpu_locks(directory,selection):
    """Lock every physical GPU, in stable order; unwind partial acquisition."""
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'experimental/branching_context/src')]
    from context_time_travel.pools import normalize_gpu_selection
    workers=normalize_gpu_selection(selection)
    physical=sorted(g for worker in workers for g in ((worker,) if type(worker) is int else worker))
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    with ExitStack() as stack:
        for gpu in physical:
            lock=stack.enter_context((directory/f'gpu-{gpu}.lock').open('a'))
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError(f'Another configured COW server owns GPU {gpu}.') from None
        yield

def configured_argv(directory,config,key):
    """Keep text-only defaults; opt in to the existing pinned image preparer.

    Native --vision consumes embeddings. It does not encode uploaded images.
    serve.cow_http validates the producer and its file hashes before GPU startup.
    """
    argv=["cow-http","--profile",config["profile"],"--store",config["store"],
              "--host",config["host"],"--port",str(config["port"]),"--api-key",key,
              "--ram-gib",str(config["ram_gib"]),"--disk-gib",str(config["disk_gib"]),
              "--min-available-gib",str(config.get("min_available_gib",16)),
              "--checkpoint-idle-seconds",str(config["checkpoint_idle_seconds"]),
              "--text","--grammar-store",str(directory/"grammars")]
    vision=config.get("vision_config")
    if vision is not None:
        if not isinstance(vision,str):raise ValueError("vision_config must be an absolute file path")
        path=Path(vision)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("vision_config must name an existing absolute regular file")
        argv += ["--vision-config",str(path)]
    return argv

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--ephemeral',action='store_true',help='Discard session history and checkpoints when this server exits')
    args=parser.parse_args()
    directory=args.directory.resolve(strict=True)
    config=json.loads((directory/"server.json").read_text())
    if args.ephemeral:config=dict(config,ephemeral=True)
    key_file=Path(config["api_key_file"])
    if key_file.is_symlink() or key_file.stat().st_mode&0o077:
        raise ValueError("The API key file must be private (chmod 600).")
    key=key_file.read_text().strip()
    if not key or "\n" in key: raise ValueError("The API key file must hold one nonempty key.")
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/"experimental/branching_context/src")]
    for name in list(os.environ):
        if name.startswith(("STRATA_","CUDA_","NVIDIA_","GGML_")):
            del os.environ[name]
    profile=json.loads(Path(config["profile"]).read_text())
    os.environ.update(profile["execution"]["environment"])
    os.umask(0o077)
    locks=Path.home()/".local/state/context-cow"
    locks.mkdir(mode=0o700,parents=True,exist_ok=True)
    with gpu_locks(locks,profile['execution']['GPU']):
        with session_storage(directory,config) as (runtime,session_config):
            sys.argv=configured_argv(runtime,session_config,key)
            runpy.run_module("serve.cow_http",run_name="__main__")

if __name__=="__main__": main()
