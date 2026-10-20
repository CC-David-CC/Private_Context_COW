"""Bind a new model/chunk configuration to an existing pinned batched build.

Creates an unqualified profile. It never inherits numerical qualification.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

from prepare_speed_profile import digest, hash_inventory, model_inputs, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--receipt', type=Path)
    p.add_argument('--api', type=Path, help='Bind the actual Python API source used by this variant')
    a = p.parse_args()
    base = json.loads(a.base.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')): del os.environ[key]
    os.environ.update(base['execution']['environment'])
    _validate_execution(base)
    _validate_model(base)
    if base['execution'].get('resume_contract') != 'identical-native-batches-v1':
        raise ValueError('Require the qualified-schedule build, not normal decode')
    if a.chunk not in (128, 512, 2048, 8192): raise ValueError('Unsupported campaign chunk')
    command = base['execution']['sut_command']
    experts = Path(command[command.index('--expert-profile')+1])
    inputs = model_inputs(a.model, a.pack, experts)
    spec = importlib.util.spec_from_file_location('variant_gguf', base['gguf_reader'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = module.GGUFFile(a.model).metadata
    vocab = metadata['tokenizer.ggml.tokens']
    if metadata['general.architecture'] != 'qwen4exp' or len(vocab) != 248320:
        raise ValueError('Native probe does not support this geometry')
    eos = [vocab.index(t) for t in ('<|endoftext|>', '<|im_end|>')]
    if eos != base['execution']['eos_tokens']: raise ValueError('EOS geometry mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads(a.receipt.read_text()) if a.receipt else None
    receipt = hash_inventory(inputs, prior)
    receipt_path = a.output.resolve()/'model-receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    profile = copy.deepcopy(base)
    profile['model_path'] = str(a.model.resolve())
    profile['model_files'] = receipt['model_files']
    profile['model_hash_receipt'] = str(receipt_path)
    profile['model_hash_receipt_sha256'] = sha(receipt_path)
    weights = {k:dict(bytes=v['bytes'], sha256=v['sha256']) for k,v in receipt['model_files'].items()}
    profile['model_weights_sha256'] = digest(weights)
    profile['model_contract'].update(weights_sha256=digest(weights), tokenizer_identity=digest(weights),
        rope_base=str(metadata['qwen4exp.rope.freq_base']))
    profile['model_contract_id'] = digest(profile['model_contract'])
    for name in ('sut_command', 'oracle_command'):
        cmd = profile['execution'][name]
        for flag,value in (('--native',str(a.model.resolve())),('--pack',str(a.pack.resolve())),('--prefill',str(a.chunk))):
            cmd[cmd.index(flag)+1] = value
    profile['execution']['prefill_chunk'] = a.chunk
    if a.api:
        for path in sorted((a.api/'src/context_time_travel').glob('*.py')):
            profile['native_build']['sha256'][str(path.resolve())] = sha(path)
    profile['execution_profile_id'] = digest(dict(build=profile['native_build'],
        execution=profile['execution'], hardware=profile['hardware']))
    for key in list(profile):
        if key.startswith(('calibration', 'baseline', 'admission')) or key in ('atol','rtol','tolerance_rule'):
            profile.pop(key)
    profile.update(calibration_status='NOT_RUN', numerical_gate_qualified=False,
        profile_derivation='New model/chunk; same pinned native batched build; exactness must be executed')
    _validate_execution(profile)
    _validate_model(profile)
    (a.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', profile=str(a.output/'profile.json'),
        model_bytes_hashed=receipt['model_bytes_hashed'])),flush=True)


if __name__ == '__main__': main()
