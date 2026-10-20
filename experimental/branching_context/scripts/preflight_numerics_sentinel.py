from argparse import Namespace
from pathlib import Path
import sys

root = Path('/home/dflanag3/spatial-artifacts/cli-workflow-harness-20261016-v1')
stage = root / 'audit-root-preflight-v3/source'
runtime = root / 'credential-green-v1/source'
sys.path[:0] = [str(stage / 'scripts'), str(runtime / 'tools'),
                str(runtime / 'experimental/branching_context/src'),
                str(runtime / 'experimental/branching_context/scripts')]
import verify_cli_workflow_numerics as audit

class ReachedFirstReference(Exception):
    pass

def no_launch(profile, outdir, name, ids, prefill_requests, segments, binary):
    if len(ids) != 181 or prefill_requests != [181] or segments != [8]:
        raise AssertionError('unexpected first reference plan')
    print('SENTINEL_FIRST_REFERENCE plan=prefix181 prefill_requests=[181] segments=[8]')
    raise ReachedFirstReference('sentinel prevented native launch')

audit.reference_segmented = no_launch
args = Namespace(
    runtime_root=runtime,
    profile=root / 'credential-green-v1/profile.json',
    campaign=root / 'runs/g4-cli-q4-tiny-redaction-v1/campaign/result.json',
    oracle_binary=root / 'audit-oracle-compile-v1/strata-batched-fresh-segmented-audit',
    original_build_receipt=Path('/home/dflanag3/spatial-artifacts/cow-seal-stream-candidate-20261015-v1/native-probe-v3/build.json'),
    build_result=root / 'audit-oracle-compile-v1/compile-link.json',
    input_verification=root / 'audit-oracle-compile-v1/input-verification-final.json',
    outputs=8,
    continue_outputs=8,
    output=root / 'runs/preflight-sentinel-v1/audit-output',
)
try:
    audit.run_audit(args)
except ReachedFirstReference:
    print('PREFLIGHT_RESULT=PASS_REACHED_FIRST_REFERENCE_NO_CHILD_LAUNCHED')
else:
    raise SystemExit('sentinel was not reached')