"""Back up terminal native-resident evidence without model weights or server secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def collect_files(campaigns, probe, parts, apis):
    files = {}
    def include(root, prefix):
        if root.is_symlink(): raise ValueError('Archive only explicitly selected real roots')
        if not root.is_dir(): raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                if not path.resolve().is_relative_to(root.resolve()): raise ValueError('File escaped selected root')
                name = prefix+'/'+str(path.relative_to(root)).replace('\\','/')
                if name in files: raise ValueError('Duplicate archive name: '+name)
                files[name] = path
    for prefix,root in campaigns: include(root,prefix)
    for name in parts+apis:
        if not name or Path(name).name != name or '/' in name or '\\' in name or name in ('.','..'):
            raise ValueError('Select a probe directory basename')
    for name in parts: include(probe/name,'probe/'+name)
    for name in apis:
        for section in ('src','scripts'): include(probe/name/section,'probe/'+name+'/'+section)
    if not files: raise ValueError('No evidence selected')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primitive',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--campaign',type=Path,action='append',default=[],help='Additional terminal campaign')
    parser.add_argument('--probe-part',action='append',help='Explicit frozen probe directory; replaces v3 defaults')
    parser.add_argument('--api-name',action='append',help='Explicit frozen API version; replaces api-v3 default')
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    campaigns = [(label,root) for label,root in (('primitive',args.primitive),('resume',args.resume)) if root is not None]
    campaigns += [('campaigns/'+root.name,root) for root in args.campaign]
    if not campaigns: parser.error('Select at least one campaign')
    for _,root in campaigns:
        worker = json.loads((root/'worker.json').read_text())
        if (worker['status'] not in ('PASS','FAIL') or worker.get('server_http_status') != 200
                or Path('/proc/'+str(worker['pid'])).exists()):
            raise RuntimeError('Campaign must be terminal, exited and server restored before archival')
    if args.output.exists(): raise FileExistsError(args.output)
    parts = args.probe_part or ['binaries-v3','q4-profile-v3','q8-profile-v3']
    apis = args.api_name or ['api-v3']
    files = collect_files(campaigns,args.probe,parts,apis)
    # pathlib does not descend into the campaigns' API symlink, and files above
    # are checked again. Never copy the external web config or model directories.
    manifest = {name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in files.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
        for name,path in files.items(): archive.write(path,name)
    receipt = dict(status='ARCHIVED',path=str(args.output),bytes=args.output.stat().st_size,
                   sha256=sha(args.output),files=len(files),source_bytes=sum(v['bytes'] for v in manifest.values()),
                   campaigns=[dict(label=label,path=str(root)) for label,root in campaigns],
                   probe_parts=parts,api_versions=apis)
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__': main()
