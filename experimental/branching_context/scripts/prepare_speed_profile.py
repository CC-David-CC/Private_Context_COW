"""Prepare a local, unqualified profile and fixture for the bounded CUDA demo.

This records actual files and hardware. It neither calibrates tolerances nor
admits a certification gate. A prior model receipt is an explicit local trust
input; changed file metadata causes the bytes to be hashed again.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_inputs(model, pack, experts):
    model, pack, experts = Path(model).absolute(), Path(pack).resolve(), Path(experts).resolve()
    for name in ('dense.bin', 'index.txt', 'native_experts.txt', 'conversions.json'):
        if not (pack / name).is_file():
            raise FileNotFoundError(pack / name)
    shards = json.loads((pack / 'conversions.json').read_text())['source_shards']
    if not shards or model.name not in [item['name'] for item in shards]:
        raise ValueError('Model is absent from the pack conversion manifest')
    if len({item['name'] for item in shards}) != len(shards):
        raise ValueError('Duplicate source shard')
    paths = [experts]
    for item in shards:
        name = item['name']
        if not isinstance(name, str) or '/' in name or '\\' in name or ':' in name or name in ('.', '..'):
            raise ValueError('Shard names must be plain filenames')
        path = model.parent / name
        if path.stat().st_size != item['size']:
            raise ValueError('Shard size differs from pack manifest: ' + name)
        paths.append(path.resolve())
    paths.extend(p.resolve() for p in pack.rglob('*') if p.is_file())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(paths))


def hash_inventory(paths, prior=None):
    measured = (prior or {}).get('model_files', {})
    result = dict(model_files={}, model_bytes_hashed=0, model_file_hash_receipts_reused=0)
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        entry = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
        previous = measured.get(str(path), {})
        fingerprint = previous.get('sha256', '')
        if (all(previous.get(k) == v for k, v in entry.items()) and
                len(fingerprint) == 64 and all(c in '0123456789abcdef' for c in fingerprint)):
            result['model_file_hash_receipts_reused'] += 1
        else:
            print(json.dumps(dict(event='hash', file=path.name, bytes=stat.st_size)), flush=True)
            fingerprint = sha(path)
            result['model_bytes_hashed'] += stat.st_size
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            raise RuntimeError('File changed during fingerprinting: ' + str(path))
        result['model_files'][str(path)] = dict(entry, sha256=fingerprint)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='This Strata/Context COW checkout')
    parser.add_argument('--binaries', type=Path, required=True, help='Output of build_fast_prefill_probe.py --long-demo')
    parser.add_argument('--model', type=Path, required=True, help='First GGUF shard')
    parser.add_argument('--pack', type=Path, required=True, help='Existing tools/iq_pack.py output')
    parser.add_argument('--expert-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the pack')
    parser.add_argument('--model-receipt', type=Path, help='Trusted existing local model_files receipt; optional')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The real-backend demo currently requires Linux, CUDA and POSIX file locking')
    source, binaries, pack = args.source.resolve(), args.binaries.resolve(), args.pack.resolve()
    model, experts, output = args.model.absolute(), args.expert_profile.resolve(), args.output.resolve()
    if output == pack or pack in output.parents:
        parser.error('Keep output outside the model pack')
    inputs = model_inputs(model, pack, experts)
    from speed_8k import command, fixture
    # Check tokenizer availability and build receipts before any expensive hash.
    tokens = fixture(source, pack)
    build_receipt = json.loads((binaries / 'build.json').read_text())
    if build_receipt.get('status') != 'BUILT_NOT_EXECUTED' or not build_receipt.get('long_demo'):
        raise ValueError('A successfully built --long-demo probe is required')
    for category in ('source_sha256', 'generated_sha256'):
        for name, expected in build_receipt[category].items():
            if sha(name) != expected:
                raise ValueError('Probe source changed: ' + name)
    for name in ('strata-batched-branch', 'strata-batched-fresh'):
        if sha(binaries / name) != build_receipt['binaries'][name]:
            raise ValueError('Probe binary changed: ' + name)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    if len(hardware.splitlines()) != 1:
        raise ValueError('This bounded demo requires one visible physical GPU')
    helper = source / 'tools/gguf_reader.py'
    spec = importlib.util.spec_from_file_location('speed_profile_gguf', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = module.GGUFFile(model).metadata
    vocabulary = metadata['tokenizer.ggml.tokens']
    if metadata['general.architecture'] != 'qwen4exp' or len(vocabulary) != 248320:
        raise ValueError('Demo driver requires the tested qwen4exp / 248320-token geometry')
    eos = [vocabulary.index(token) for token in ('<|endoftext|>', '<|im_end|>')]
    if eos != tokens['eos_tokens']:
        raise ValueError('Model EOS differs from the demo driver')
    prior = json.loads(args.model_receipt.read_text()) if args.model_receipt else None
    output.mkdir(parents=True, exist_ok=False)
    receipt = hash_inventory(inputs, prior)
    if args.model_receipt:
        receipt['prior_receipt_sha256'] = sha(args.model_receipt)
    receipt['timestamp_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    receipt_path = output / 'model-receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    weights = {name: dict(bytes=item['bytes'], sha256=item['sha256']) for name, item in receipt['model_files'].items()}
    model_sha = digest(weights)
    contract = dict(weights_sha256=model_sha, architecture=metadata['general.architecture'],
                    rope_base=str(metadata['qwen4exp.rope.freq_base']), yarn='model-default',
                    tokenizer_identity=model_sha, template_identity='exact-token-ids/no-template',
                    attention='causal-qsa-indexed', positions='absolute-zero-based', input_kind='token_ids',
                    adapters=[], state_kind='qwen4exp-gdn-qsa-ple-v1')
    seed = dict(execution=dict(oracle_command=['unused', '--pack', str(pack), '--native', str(model),
                                              '--expert-profile', str(experts)]))
    common = command(seed, 'normal') + ['--pcie-frac', '0.55', '--max-new', '128']
    execution = dict(sut_command=[str(binaries/'strata-batched-branch'), *common, '--tokens', '1'],
                     oracle_command=[str(binaries/'strata-batched-fresh'), *common],
                     capacity=12288, kv_dtype='FP16', eos_tokens=eos,
                     operator='aligned-batched-prefix/normal-resources/fixed-pcie-0.55',
                     environment={k: v for k, v in os.environ.items()
                                  if k.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))})
    execution['write_integrity_policy'] = build_receipt.get('write_integrity_policy', 'per-write-sha256')
    execution['private_page_access'] = build_receipt.get('private_page_access', 'set-each-write')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    paths = [source/'CMakeLists.txt', helper, binaries/'build.json',
             binaries/'strata-batched-branch', binaries/'strata-batched-fresh']
    paths.extend(Path(name) for category in ('source_sha256', 'generated_sha256')
                 for name in build_receipt[category])
    paths.extend((source/'experimental/branching_context/src/context_time_travel').glob('*.py'))
    build = dict(revision=revision, source_sha256=build_receipt['source_sha256'],
                 sha256={str(p): sha(p) for p in sorted(set(paths))},
                 probe_build_sha256=sha(binaries/'build.json'),
                 base_build_files_sha256={str(source/'build'/name): sha(source/'build'/name)
                                          for name in ('CMakeCache.txt', 'libstrata_engine.a')})
    profile = dict(backend='strata', model_path=str(model), gguf_reader=str(helper),
                   model_weights_sha256=model_sha, model_files=receipt['model_files'],
                   model_hash_receipt=str(receipt_path), model_hash_receipt_sha256=sha(receipt_path),
                   model_contract=contract, model_contract_id=digest(contract),
                   backend_revision=revision, native_build=build, execution=execution, hardware=hardware,
                   execution_profile_id=digest(dict(build=build, execution=execution, hardware=hardware)),
                   numerical_gate_qualified=False, calibration_status='EXPERIMENTAL_UNQUALIFIED',
                   oracle_binary=str(binaries/'strata-batched-fresh'), oracle_revision=revision,
                   fixtures=tokens, token_fixtures_sha256=digest(tokens))
    (output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    (output/'fixture.json').write_text(json.dumps(tokens)+'\n')
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', profile=str(output/'profile.json'),
                          fixture=str(output/'fixture.json'), numerical_gate_qualified=False,
                          model_bytes_hashed=receipt['model_bytes_hashed'],
                          model_file_hash_receipts_reused=receipt['model_file_hash_receipts_reused'])), flush=True)


if __name__ == '__main__':
    main()
