"""Package executed measurements; omit native payloads, weights and credentials."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def test_counts(path):
    text = path.read_text()
    runs = re.findall(r'Ran (\d+) tests? in',text)
    if not runs: raise ValueError('No executed unittest count: '+str(path))
    executed = int(runs[-1])
    endings = re.findall(r'(?:FAILED|OK)(?: \(([^\n]+)\))?',text)
    if not endings: raise ValueError('No terminal unittest result')
    counts = {name:int(value) for name,value in re.findall(r'(failures|errors|skipped)=(\d+)',endings[-1])}
    failed,errors,skips = (counts.get(name,0) for name in ('failures','errors','skipped'))
    return dict(executed=executed,passed=max(0,executed-failed-errors-skips),failed=failed,errors=errors,skips=skips,
                log_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('output',type=Path)
    for name in ('control','report','real-audit','g0','cumulative'):
        parser.add_argument('--'+name+'-log',type=Path,required=True)
    args = parser.parse_args()
    root,out = args.campaign,args.output
    out.mkdir(parents=True,exist_ok=False)
    prepared = json.loads((root/'prepare.json').read_text())
    resumed = json.loads((root/'resume.json').read_text())
    boundary = json.loads((root/'boundary.json').read_text())
    diagnostic = json.loads((root/'frontier-diagnosis.json').read_text())
    profile = json.loads((root/'profile.json').read_text())
    checkpoint = root/'store/checkpoints'
    sid = prepared['root']['snapshot']
    manifest = json.loads((checkpoint/(sid+'.json')).read_text())['manifest']
    payload = checkpoint/(manifest['payload_sha256']+'.state')
    if sha(payload) != manifest['payload_sha256']: raise ValueError('Saved payload changed')
    summary = dict(schema=1,status='BLOCKED_EXECUTION_EQUIVALENCE',
        real_audit=test_counts(args.real_audit_log),
        supplemental_tests=dict(control_plane=test_counts(args.control_log),evidence_serialization=test_counts(args.report_log)),
        g0=dict(status='PASS',**test_counts(args.g0_log)),
        cumulative=dict(g1=dict(status='FAIL',reason='missing CTT_PROFILE',**test_counts(args.cumulative_log)),g2='NOT RUN'),
        scope=dict(prefix_tokens=len(prepared['root']['history']),suffix_tokens={label:row['suffix_tokens'] for label,row in resumed['branches'].items()},
                   output_tokens_per_branch={label:len(row['generated']) for label,row in resumed['branches'].items()},
                   larger_runs='1K and 8K NOT RUN after tiny numerical failure'),
        model_contract_id=profile['model_contract_id'],model_weights_sha256=profile['model_weights_sha256'],
        execution_profile_id=profile['execution_profile_id'],hardware=profile['hardware'],
        sut_sha256=profile['native_build']['sha256'][profile['execution']['sut_command'][0]],
        oracle_sha256=profile['native_build']['sha256'][profile['execution']['oracle_command'][0]],
        atol=profile['atol'],rtol=profile['rtol'],
        max_full_batched_logit_error=diagnostic['restored_vs_uncached_full_batched']['max_absolute_error'],
        max_normal_decode_logit_error=diagnostic['restored_vs_uncached_normal_decode']['max_absolute_error'],
        root_snapshot=sid,checkpoint_manifest_sha256=sha(checkpoint/(sid+'.json')),
        checkpoint_payload_sha256=manifest['payload_sha256'],checkpoint_payload_bytes=payload.stat().st_size,
        native_state_bytes=manifest['state_bytes'],frontier_bytes=4*manifest['logits_count'],
        timings=dict(prefix_api_seconds=prepared['prefill_seconds'],save_seconds=prepared['save_seconds'],
                     startup_seconds=[prepared['startup_seconds'],resumed['startup_seconds'],boundary['startup_seconds']]),
        processes=dict(prepare=prepared['native_pid'],resume=resumed['native_pid'],boundary=boundary['native_pid'],
                       all_exits_observed=all(row['native_exit_observed'] for row in (prepared,resumed,boundary))),
        restores=[dict(consumed=r['consumed'],forward_tokens=r['forward_tokens'],seconds=r['seconds'],
            loaded_file_bytes=r['counter_deltas']['loaded_file_bytes'],
            completed_cuda_copy_bytes=r['counter_deltas']['cold_restore_copy_bytes'],
            completed_cuda_copy_calls=r['counter_deltas']['cold_restore_copy_calls']) for r in resumed['restore_receipts']],
        branches={label:dict(generated_tokens=len(row['generated']),suffix_forward_tokens=row['suffix_work'],
            suffix_seconds=row['suffix_seconds'],decode_seconds=row['decode_seconds'],
            audit_inclusive_tps=row['end_to_end_decode_tps']) for label,row in resumed['branches'].items()},
        native_counters=resumed['final_counters'],
        limits=['Not a G1/G2/G3 certification','No nonzero RNG/grammar/pending UTF8 real-resume campaign',
                'Process-kill publication tests use synthetic native bytes; power loss untested',
                'Whole-process RAM/VRAM peaks uninstrumented','GPU evidence predates final controller-rejection/reporting guards',
                'Audit-heavy Python timings are not a serving throughput benchmark',
                'Native checkpoint payloads remain on llm-60; archive carries checksums and manifests'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in
             ('.json','.jsonl','.f32','.tokens','.stdout','.stderr') and not p.name.endswith('.tmp')]
    inventory = {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    archives = []
    archive = None
    try:
        for path in files:
            if archive is None or archive.fp.tell() > 32*1024**2:
                if archive is not None: archive.close()
                name = 'raw-evidence-'+str(len(archives)+1).zfill(2)+'.zip'
                archives.append(out/name)
                archive = zipfile.ZipFile(out/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)
            archive.write(path,str(path.relative_to(root)))
    finally:
        if archive is not None: archive.close()
    (out/'archives.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in archives},indent=2)+'\n')
    print(json.dumps(dict(summary=str(out/'summary.json'),files=len(files),
                         archive_bytes=sum(p.stat().st_size for p in archives),archives=len(archives))))


if __name__=='__main__': main()
