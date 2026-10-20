"""Summarize executed artifacts; retain failing numerical tests and omitted work."""
import argparse
import json
from pathlib import Path
import re
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args()
    preflight=json.loads((a.results/'preflight.json').read_text())
    trace=json.loads((a.results/'input-layers.json').read_text())
    inventory=json.loads((a.results/'inventory.json').read_text())
    with zipfile.ZipFile(a.results/inventory['archive']) as z:
        profile=json.loads(z.read('profile.json'))
        build=json.loads(z.read('binaries/build.json'))
        diagnostic=json.loads(z.read('layer-binaries-v5/build.json'))
        g0_names=[name for name in z.namelist() if name.endswith('-G0.json')]
        if len(g0_names)!=1:raise ValueError('Expected one executed G0 report')
        g0=json.loads(z.read(g0_names[0]))
    logs={name:(a.results/name).read_text() for name in ('control-tests.log','batch-audit.log')}
    methods={name:re.findall(r'^(test_\S+) \(([^)]+)\) \.\.\. (ok|FAIL|ERROR|skipped.*)$',text,re.M)
             for name,text in logs.items()}
    if len(methods['control-tests.log'])!=sum(map(int,re.findall(r'Ran (\d+) tests?',logs['control-tests.log']))):
        raise ValueError('Unittest method count differs from summary')
    summary=dict(status='BLOCKED',milestone='batches-only durable resume',
        blocker='Native dense projection does not preserve per-row results across batch sizes 1 and 5 at frozen tolerance',
        gates=dict(G0=g0,G1='NOT_QUALIFIED',G2='BLOCKED_BEFORE_CURRENT_PROFILE_RESTART',G3='NOT_QUALIFIED'),
        executed_methods=methods,
        supplemental=dict(passed=sum(m[2]=='ok' for m in methods['control-tests.log']),failures=0,skips=0),
        real_artifact_audit=dict(passed=sum(m[2]=='ok' for m in methods['batch-audit.log']),
                                failed=sum(m[2]=='FAIL' for m in methods['batch-audit.log']),skips=0),
        preflight=dict(status=preflight['status'],max_logit_error=preflight['batch_continuation_vs_full_history']['max_absolute_error'],
            fresh_repeat_error=preflight['fresh_repeat']['max_absolute_error'],atol=preflight['atol'],rtol=preflight['rtol'],
            work=[r['work'] for r in preflight['observations']]),
        projection_input_images=trace['projection_input_images'],
        first_projection_difference=trace['gdn_comparison']['first_difference'],
        instrumentation_preserved_frontiers=trace['instrumentation_preserved_frontiers'],
        hardware=profile['hardware'],model_weights_sha256=profile['model_weights_sha256'],
        model_contract_id=profile['model_contract_id'],execution_profile_id=profile['execution_profile_id'],
        backend_binaries_sha256=build['binaries'],diagnostic_binary_sha256=diagnostic['binary_sha256'],
        model_hash_verification='Earlier full hash receipt reused; no model content rehash in this campaign',
        current_profile_restore_performed=False,
        unexecuted=['current-profile restart','1K/8K fixtures','nonzero RNG/grammar/decoder continuation',
                    'real GPU crash/OOM/eviction matrix','power-loss qualification','peak RAM/VRAM measurements'],
        normal_decode='Excluded; further research/fixes not started. Earlier exploratory measurements retained.',
        remote='box-git:/srv/data/git/private_context_cow.git',archive_sha256=inventory['sha256'])
    (a.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
