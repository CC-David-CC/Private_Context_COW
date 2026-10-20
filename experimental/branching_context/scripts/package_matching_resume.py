"""Package executed restricted-resume evidence; retain native payloads on the host."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    root,out = args.root,args.output
    out.mkdir(parents=True,exist_ok=False)
    def read(name): return json.loads((root/name).read_text())
    prepared,resumed,boundary,comparison = [read(n+'.json') for n in ('prepare','resume','boundary','compare')]
    profile,result = read('profile.json'),read('result.json')
    sid = prepared['root']['snapshot']
    manifest_path = root/'store/checkpoints'/(sid+'.json')
    manifest = json.loads(manifest_path.read_text())['manifest']
    payload = manifest_path.parent/(manifest['payload_sha256']+'.state')
    if sha(payload) != manifest['payload_sha256']: raise ValueError('Retained root payload changed')
    receipts = resumed['restore_receipts']+[boundary['receipt']]
    repeats = [json.loads(path.read_text()) for path in sorted(root.glob('restore-repeat-*.json'))]
    summary = dict(schema=1,status=result['status'],contract='identical-native-batches-v1',
        certification='G1-G5 NOT QUALIFIED',prefix_tokens=result['prefix'],suffix_tokens=result['suffix'],
        output_limit=result['output_bound'],generated={name:len(row['generated']) for name,row in resumed['branches'].items()},
        comparisons=len(comparison['comparisons']),max_logit_error=max(row['numerics']['max_absolute_error'] for row in comparison['comparisons']),
        frozen_atol=profile['atol'],frozen_rtol=profile['rtol'],controller_scope='deterministic greedy only',
        sampling='every generated frontier' if result['prefix'] <= 1024 else 'prefix, suffix, outputs 1/16/64/final',
        model_contract_id=profile['model_contract_id'],model_weights_sha256=profile['model_weights_sha256'],
        execution_profile_id=profile['execution_profile_id'],hardware=profile['hardware'],
        sut_sha256=profile['native_build']['sha256'][profile['execution']['sut_command'][0]],
        oracle_sha256=profile['native_build']['sha256'][profile['execution']['oracle_command'][0]],
        checkpoint=dict(snapshot=sid,manifest_sha256=sha(manifest_path),payload_sha256=manifest['payload_sha256'],
            bytes=payload.stat().st_size,native_state_bytes=manifest['state_bytes'],schedule_sha256=manifest['schedule_sha256']),
        process_ids=dict(prepare=dict(python=prepared['python_pid'],native=prepared['native_pid']),
            resume=dict(python=resumed['python_pid'],native=resumed['native_pid']),
            boundary=dict(python=boundary['python_pid'],native=boundary['receipt']['native_pid'])),
        restore_receipts=receipts,additional_restore_processes=repeats,
        startup_seconds=dict(prepare=prepared['startup_seconds'],resume=resumed['startup_seconds'],boundary=boundary['startup_seconds']),
        prefill_seconds=prepared['prefill_seconds'],save_seconds=prepared['save_seconds'],
        branches={name:{key:row[key] for key in ('suffix_seconds','suffix_work','suffix_prefill_chunks',
            'first_token_seconds','decode_seconds','end_to_end_decode_tps','stopping_reason')}
            for name,row in resumed['branches'].items()},
        memory='Peak RAM/VRAM not measured by this harness; checkpoint bytes are measured',
        durability='Linux local filesystem; clean process restart; no power-loss qualification',
        filesystem_cache=result['filesystem_cache'],limitations=['cross-batch equivalence excluded',
            'nonzero RNG, grammar, pending decoder, native crash matrix unqualified'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    inventory=[]
    archive=out/'raw-matching-resume.zip'
    candidates=sorted(path for path in root.rglob('*') if path.is_file()
        and path.suffix in ('.json','.jsonl','.f32','.tokens','.stdout','.stderr','.log')
        and '__pycache__' not in path.parts and path.name != 'config.json')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in candidates:
            name=path.relative_to(root).as_posix()
            package.write(path,name)
            inventory.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None: raise ValueError('Archive CRC failure')
    (out/'inventory.json').write_text(json.dumps(dict(archive=archive.name,bytes=archive.stat().st_size,
        sha256=sha(archive),files=inventory,native_payload_retained=str(payload)),indent=2)+'\n')
    print(json.dumps(dict(summary=str(out/'summary.json'),archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive),comparisons=summary['comparisons'],max_logit_error=summary['max_logit_error'])))


if __name__ == '__main__': main()
