"""Download the pinned Q4 model and prepare its native COW pack. No MTP.

Run from the checkout with its server venv. This reuses Strata's resumable
resource downloader and its actual iq_pack.py; it never launches ordinary serving.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
MODEL='UD-Q4_K_XL'
PACK_NAME='pack-ud-q4-k-xl'
RECEIPT='cow-prepared.json'

@dataclass(frozen=True)
class Asset:
    name:str
    url:str
    size:int
    sha256:str

_setup_module=None
def setup_source():
    global _setup_module
    if _setup_module is None:
        spec=importlib.util.spec_from_file_location('_cow_resource_setup',ROOT/'setup.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        _setup_module=module
    return _setup_module

def model_plan():
    setup=setup_source();family=setup.FAMILIES['unsloth'];assets=[]
    for index in range(1,setup.model_shards(family,MODEL)+1):
        name=setup.model_file(family,MODEL,index)
        size,digest=family['sha256'][name]
        assets.append(Asset(name,family['hf'].format(q=MODEL)+name,size,digest))
    return tuple(assets)

def digest_file(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Refusing symlink: '+str(path))
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());digest=hashlib.sha256();count=0;last=time.monotonic()
        while chunk:=stream.read(16*1024*1024):
            digest.update(chunk);count+=len(chunk)
            if time.monotonic()-last>=30:
                print(f'Checking {path.name}: {count/1024**3:.1f} / {before.st_size/1024**3:.1f} GiB',flush=True)
                last=time.monotonic()
        after=os.fstat(stream.fileno())
    current=path.lstat()
    key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if key(before)!=key(after) or key(before)!=key(current):
        raise ValueError('File changed during verification: '+str(path))
    return digest.hexdigest()

def fetch_verified(asset,directory,*,downloader=None,existing_only=False):
    if Path(asset.name).name!=asset.name:raise ValueError('Asset name must be a basename')
    path=Path(directory)/asset.name
    if path.is_symlink():raise ValueError('Refusing model symlink: '+str(path))
    if not path.exists():
        if existing_only:raise ValueError('Missing existing model shard: '+str(path))
        download=downloader or setup_source().download
        download(asset.url,path,asset.name)
    if path.stat().st_size!=asset.size:
        raise ValueError('Wrong model file size; existing file retained: '+str(path))
    if digest_file(path)!=asset.sha256:
        raise ValueError('Wrong model SHA-256; existing file retained: '+str(path))
    return path

def recipe():
    names=('tools/iq_pack.py','tools/gguf_reader.py','tools/strata_tokenizer.py')
    gguf=ROOT/'third_party/llama.cpp/gguf-py'
    return dict(model=MODEL,arguments=['--compat-bf16'],
        gguf_py_revision=setup_source().LLAMA_CPP_COMMIT,
        gguf_python_sha256={str(p.relative_to(gguf)):digest_file(p) for p in sorted(gguf.rglob('*.py'))},
        python_packages={n:importlib.metadata.version(n) for n in ('numpy','pyyaml')},
        source_sha256={n:digest_file(ROOT/n) for n in names})

def pack_manifest(pack):
    values={}
    for path in sorted(pack.rglob('*')):
        if path.is_symlink():raise ValueError('Refusing pack symlink: '+str(path))
        if path.is_file() and path!=pack/RECEIPT:
            values[str(path.relative_to(pack))]=dict(bytes=path.stat().st_size,sha256=digest_file(path))
    required={'dense.bin','index.txt','native_experts.txt','tokenizer/vocab.json'}
    if not required<=values.keys():raise ValueError('Pack builder did not produce the required files')
    return values

def publish_pack(data,assets,build):
    data=Path(data).resolve(strict=True);pack=data/PACK_NAME
    pins=[dict(name=a.name,bytes=a.size,sha256=a.sha256) for a in assets]
    selected_recipe=recipe()
    if pack.exists() or pack.is_symlink():
        if pack.is_symlink():raise ValueError('Refusing pack symlink')
        receipt=pack/RECEIPT
        if not receipt.is_file() or receipt.is_symlink():raise ValueError('Existing pack has no trusted preparation receipt; choose a new data folder')
        saved=json.loads(receipt.read_text())
        if (saved.get('schema')!=1 or saved.get('recipe')!=selected_recipe
                or saved.get('shards')!=pins or saved.get('files')!=pack_manifest(pack)):
            raise ValueError('Prepared pack changed or uses another recipe; existing files retained')
        return pack
    stage=Path(tempfile.mkdtemp(prefix='.cow-pack-',dir=data))
    try:
        build(stage)
        receipt=dict(schema=1,recipe=selected_recipe,shards=pins,files=pack_manifest(stage))
        with (stage/RECEIPT).open('x') as stream:
            json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        stage.rename(pack)
    finally:
        if stage.exists():
            assert stage.parent==data and stage.name.startswith('.cow-pack-')
            shutil.rmtree(stage)
    return pack

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path.home()/'ContextCOW-data')
    parser.add_argument('--existing-models',type=Path,help='Verify and read existing shards; never download or write into this folder')
    parser.add_argument('--yes',action='store_true',help='Approve the displayed model download without a question')
    args=parser.parse_args(argv);assets=model_plan()
    try:
        for name in ('numpy','yaml','requests','tqdm'):__import__(name)
    except ImportError as error:
        raise ValueError('Install the server requirements first: '+str(sys.executable)+' -m pip install -r '+str(ROOT/'requirements-cow-server.txt')) from error
    data=args.data_dir.expanduser().absolute()
    models=args.existing_models.expanduser().resolve(strict=True) if args.existing_models else data/'models'/MODEL
    missing=sum(a.size for a in assets if not (models/a.name).exists())
    if args.existing_models and missing:raise ValueError('The existing model folder is incomplete; no files were changed')
    print(f'Context COW Q4: four pinned model files, {sum(a.size for a in assets)/1e9:.1f} GB total.',flush=True)
    print('Downloads and prepares the model only. No MTP, model server or driver installation.',flush=True)
    if missing and not args.yes and input(f'Download up to {missing/1e9:.1f} GB? Type YES: ').strip()!='YES':
        print('Canceled.');return 1
    ancestor=data
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<missing+4*1024**3:
        raise ValueError('Not enough free disk space for missing model bytes plus 4 GiB of packing space')
    if data.is_symlink():raise ValueError('Use a real data folder, not a symlink')
    data.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (data/'.prepare.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another model preparation owns this data folder')
        if not args.existing_models:models.mkdir(mode=0o700,parents=True,exist_ok=True)
        print('Preparing pinned GGUF packing tools...',flush=True)
        gguf_py=setup_source().get_llama_cpp()/'gguf-py'
        if not gguf_py.is_dir():raise ValueError('Pinned GGUF Python tools are missing')
        packing_environment=dict(os.environ,STRATA_GGUF_PY=str(gguf_py))
        shards=[]
        for number,asset in enumerate(assets,1):
            print(f'Model file {number}/{len(assets)}: {asset.name}',flush=True)
            shards.append(fetch_verified(asset,models,existing_only=bool(args.existing_models)))
        def build(stage):
            print('Preparing the native model pack...',flush=True)
            subprocess.run([sys.executable,str(ROOT/'tools/iq_pack.py'),'--gguf',str(shards[0]),'--out',str(stage),'--compat-bf16'],check=True,cwd=ROOT,env=packing_environment)
        pack=publish_pack(data,assets,build)
    print('Model preparation complete.',flush=True)
    print('Model: '+str(shards[0]));print('Pack: '+str(pack))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit('Model preparation stopped: '+str(error))
