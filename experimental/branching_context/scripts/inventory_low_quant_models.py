"""Read-only low-quantization inventory; existence/metadata are not loader qualification."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,action='append',required=True)
    parser.add_argument('--reader',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('cow_inventory_gguf',args.reader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    pattern = re.compile(r'(?:IQ[123][_-]|Q[123]_|TQ[12]_)',re.I)
    candidates, packs, missing = [], [], []
    for root in args.root:
        if not root.is_dir():
            missing.append(str(root)); continue
        for parent,dirs,names in os.walk(root,followlinks=False):
            dirs[:] = [name for name in dirs if name not in ('.git','__pycache__')]
            for name in names:
                path = Path(parent)/name
                if name == 'conversions.json':
                    try:
                        record = json.loads(path.read_text())
                        shards = record.get('source_shards',[])
                        if any(pattern.search(item['name']) for item in shards):
                            packs.append(dict(path=str(path),source_shards=shards))
                    except (ValueError,KeyError,TypeError) as error:
                        packs.append(dict(path=str(path),error=str(error)))
                if not name.lower().endswith('.gguf') or not pattern.search(name): continue
                item = dict(path=str(path),bytes=path.stat().st_size,
                    header_cache='quant-headers' in path.parts,loader_status='NOT_TESTED')
                try:
                    metadata = module.GGUFFile(path).metadata
                    item.update(architecture=metadata.get('general.architecture'),
                        name=metadata.get('general.name'),file_type=metadata.get('general.file_type'),
                        tokenizer=metadata.get('tokenizer.ggml.model'),
                        split_count=metadata.get('split.count',1),split_no=metadata.get('split.no',0))
                except Exception as error:
                    item['metadata_error'] = str(error)
                candidates.append(item)
    for item in candidates:
        filename = Path(item['path']).name
        matches = []
        for pack in packs:
            for shard in pack.get('source_shards',[]):
                if filename == shard['name']:
                    matches.append(dict(pack=pack['path'],expected_shard_bytes=shard.get('size'),
                                        shard_size_matches=shard.get('size') == item['bytes']))
        item['matching_pack_receipts'] = matches
    record = dict(host=socket.gethostname(),status='INVENTORIED_NOT_QUALIFIED',models=candidates,packs=packs,
        missing_roots=missing,reader_sha256=hashlib.sha256(args.reader.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(host=record['host'],models=len(candidates),packs=len(packs),output=str(args.output))))


if __name__ == '__main__': main()
