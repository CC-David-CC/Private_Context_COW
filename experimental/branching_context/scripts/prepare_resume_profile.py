"""Pin an existing model receipt and newly built batched resume executables."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

API = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(API/'src'))
from context_time_travel.strata_backend import _digest, _validate_execution, _validate_model


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''): h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--binaries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunk',type=int,required=True)
    parser.add_argument('--host-source',type=Path,help='New host: verify all model bytes and rebuild every host-specific identity')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--pack',type=Path)
    parser.add_argument('--expert-profile',type=Path)
    parser.add_argument('--resident-budget-gib',type=int,default=16)
    parser.add_argument('--baseline-calibration',type=Path,
                        help='Reuse measured fresh repeats only with identical native/math/model inputs')
    args = parser.parse_args()
    profile = json.loads(args.base.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    if args.host_source:
        if not all((args.model,args.pack,args.expert_profile)):
            parser.error('New host requires --model, --pack and --expert-profile')
        from prepare_speed_profile import model_inputs, hash_inventory
        inventory = hash_inventory(model_inputs(args.model,args.pack,args.expert_profile))
        # Verify donor GGUF content, not merely that the destination contains files.
        donor = {Path(name).name:item['sha256'] for name,item in profile['model_files'].items() if name.endswith('.gguf')}
        for name,item in inventory['model_files'].items():
            if name.endswith('.gguf') and donor.get(Path(name).name) != item['sha256']:
                raise ValueError('Destination model shard differs from pinned donor')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        receipt_path = args.output.with_suffix('.model-receipt.json')
        receipt_path.write_text(json.dumps(inventory,indent=2)+'\n')
        profile.update(model_path=str(args.model),gguf_reader=str(args.host_source/'tools/gguf_reader.py'),
            model_files=inventory['model_files'],model_hash_receipt=str(receipt_path),
            model_hash_receipt_sha256=sha(receipt_path))
        profile['model_weights_sha256'] = _digest({name:dict(bytes=item['bytes'],sha256=item['sha256'])
            for name,item in inventory['model_files'].items()})
        profile['model_contract']['weights_sha256'] = profile['model_weights_sha256']
        profile['model_contract']['tokenizer_identity'] = profile['model_weights_sha256']
        profile['model_contract_id'] = _digest(profile['model_contract'])
        profile['hardware'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader'],text=True).strip()
        for key in ('sut_command','oracle_command'):
            command = profile['execution'][key]
            for flag,value in [('--pack',str(args.pack)),('--native',str(args.model)),
                ('--expert-profile',str(args.expert_profile)),('--resident-budget-gib',str(args.resident_budget_gib))]:
                command[command.index(flag)+1] = value
        verification = dict(model_bytes_hashed_this_process=inventory['model_bytes_hashed'],
                            donor_model_content_verified=True,model_file_hash_receipts_reused=0)
        profile['native_build']['sha256'] = {}
    else:
        _validate_execution(profile)
        verification = _validate_model(profile)
    receipt = json.loads((args.binaries/'build.json').read_text())
    if receipt['status'] != 'BUILT_NOT_EXECUTED' or not receipt['long_demo']:
        raise ValueError('Successful long-demo build receipt required')
    for category in ('source_sha256','generated_sha256'):
        for path,expected in receipt[category].items():
            if sha(path) != expected: raise ValueError('Build source changed: '+path)
    profile = copy.deepcopy(profile)
    for name,binary in [('sut_command','strata-batched-branch'),('oracle_command','strata-batched-fresh')]:
        path = args.binaries/binary
        if sha(path) != receipt['binaries'][binary]: raise ValueError('Binary changed')
        command = profile['execution'][name]
        command[0] = str(path)
        command[command.index('--prefill')+1] = str(args.chunk)
        command[command.index('--max-new')+1] = '0' if name=='oracle_command' else '128'
    continuation_operator = receipt.get('continuation_operator', 'normal-verifier-decode')
    if continuation_operator not in ('batched-prefill', 'normal-verifier-decode'):
        raise ValueError('Unknown native continuation operator')
    profile['execution'].update(operator='batched-prefill/durable-resume-experimental',
        generated_operator=continuation_operator)
    if receipt.get('resume_contract') == 'identical-native-batches-v1':
        profile['execution'].update(resume_contract=receipt['resume_contract'],prefill_chunk=args.chunk)
    build = profile['native_build']
    build['revision'] = 'durable-resume-on-c8e9c09b'
    build['resume_build'] = receipt
    for category in ('source_sha256','generated_sha256'):
        build['sha256'].update(receipt[category])
    build['sha256'][profile['gguf_reader']] = sha(profile['gguf_reader'])
    for path in [args.binaries/'build.json',*(args.binaries/p for p in receipt['binaries']),
                 *list((API/'src/context_time_travel').glob('*.py')),Path(__file__).resolve(),
                 API/'scripts/durable_resume.py']:
        build['sha256'][str(path)] = sha(path)
    profile.update(numerical_gate_qualified=False,calibration_status='NOT_RUN')
    # Never inherit the different operator's numerical admission.
    profile.pop('atol',None); profile.pop('rtol',None)
    if args.baseline_calibration:
        baseline = json.loads(args.baseline_calibration.read_text())
        old = json.loads((args.baseline_calibration.parent/'profile.json').read_text())
        if (baseline['status'] != 'PASS' or old['model_contract_id'] != profile['model_contract_id']
                or old['hardware'] != profile['hardware']
                or old['execution']['environment'] != profile['execution']['environment']):
            raise ValueError('Baseline model/hardware/environment mismatch')
        expected_command = list(profile['execution']['oracle_command'])
        expected_command[expected_command.index('--max-new')+1] = '1'
        maximum = 0.0
        from context_time_travel.strata_backend import _unpack_frontier
        for row in baseline['comparisons']:
            values = []
            for observation in (row['first'],row['second']):
                command = list(observation['command'])
                token_index = command.index('--tokens-file')
                tokens = Path(command[token_index+1])
                del command[token_index:token_index+2]
                if command != expected_command:
                    raise ValueError('Baseline numerical command differs')
                binary = command[0]
                if old['native_build']['sha256'][binary] != sha(binary):
                    raise ValueError('Baseline oracle executable changed')
                path = args.baseline_calibration.parent/observation['frontier']
                if sha(path) != observation['logits_sha256']:
                    raise ValueError('Baseline frontier changed')
                values.append(_unpack_frontier(path.read_bytes(),248320))
                build['sha256'][str(path)] = sha(path)
                build['sha256'][str(tokens)] = sha(tokens)
            measured = max(abs(a-b) for a,b in zip(*values))
            if measured != row['max_absolute_error']:
                raise ValueError('Baseline measurement differs from its raw frontiers')
            maximum = max(maximum,measured)
        if baseline['atol'] != 2*maximum or baseline['rtol'] != 0:
            raise ValueError('Baseline tolerance does not follow frozen 2x-repeat rule')
        build['sha256'][str(args.baseline_calibration)] = sha(args.baseline_calibration)
        profile.update(atol=baseline['atol'],rtol=baseline['rtol'],
            calibration_status='REUSED_IDENTICAL_NATIVE_FRESH_BASELINE',
            baseline_calibration=str(args.baseline_calibration),
            baseline_profile_sha256=sha(args.baseline_calibration.parent/'profile.json'),
            baseline_scope='Native oracle and math inputs unchanged; control-plane code/budget changed; unqualified')
    profile['execution_profile_id'] = _digest(dict(build=build,execution=profile['execution'],hardware=profile['hardware']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(dict(profile=str(args.output),model_verification=verification,
                         execution_profile_id=profile['execution_profile_id'])))


if __name__=='__main__': main()
