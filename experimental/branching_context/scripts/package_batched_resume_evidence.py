"""Package executed batch diagnostics; omit weights, native binaries and server secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for name in ('preflight','layers','gdn-layers','input-layers'):
        directory=a.root/name
        if not directory.is_dir(): raise ValueError('Missing executed diagnostic: '+name)
        candidates.extend(p for p in directory.rglob('*') if p.is_file())
    for name in ('binaries','layer-binaries','layer-binaries-v2','layer-binaries-v3','layer-binaries-v4','layer-binaries-v5'):
        candidates.extend(p for p in (a.root/name).rglob('*')
                          if p.is_file() and p.suffix in ('.json','.cpp','.hpp','.stdout','.stderr'))
    candidates.extend(p for p in a.root.glob('*') if p.is_file() and p.suffix in ('.json','.log','.stdout','.stderr','.py'))
    candidates.extend(p for p in (a.root/'experimental/branching_context').rglob('*.py') if '__pycache__' not in p.parts)
    candidates.extend((a.root/'experimental/branching_context/evidence/runs').glob('*G0.json'))
    native_source=Path('/home/dflanag3/branching-g3/durable-resume-20261013/source/src/prefill/prefill.cpp')
    candidates=sorted(set(candidates))
    inventory=[];archive=a.output/'raw-batch-diagnostics.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in candidates:
            name=path.relative_to(a.root).as_posix()
            if path.suffix in ('.state','.gguf') or path.name=='config.json': raise ValueError('Forbidden payload')
            z.write(path,name);inventory.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
        z.write(native_source,'frozen-native/prefill.cpp')
        inventory.append(dict(path='frozen-native/prefill.cpp',bytes=native_source.stat().st_size,sha256=sha(native_source)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC failure')
    manifest=dict(scope='executed batches-only preflight and diagnostics; no restart PASS',
                  archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=inventory)
    (a.output/'inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=manifest['bytes'],sha256=manifest['sha256'],files=len(inventory))))


if __name__=='__main__':main()
