"""Start one configured Context COW worker without putting its API key in OS argv."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys

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
    if len(sys.argv)!=2:
        raise SystemExit("Usage: python tools/cow_server.py /path/to/server-folder")
    directory=Path(sys.argv[1]).resolve(strict=True)
    config=json.loads((directory/"server.json").read_text())
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
    with (locks/"gpu-0.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("Another configured COW server owns GPU 0.")
        sys.argv=configured_argv(directory,config,key)
        try: runpy.run_module("serve.cow_http",run_name="__main__")
        finally: fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=="__main__": main()
