"""Measure ordinary serial native serving; no COW or numerical qualification claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def replace_option(arguments, name, value):
    result = list(arguments)
    positions = [i for i, item in enumerate(result) if item == name]
    if len(positions) > 1:
        raise ValueError('Duplicate engine option: ' + name)
    if positions:
        position = positions[0]
        if position + 1 == len(result) or result[position + 1].startswith('--'):
            raise ValueError('Missing engine option value: ' + name)
        result[position + 1] = str(value)
    else:
        result += [name, str(value)]
    return result


def option_integer(arguments, name):
    positions = [i for i, item in enumerate(arguments) if item == name]
    if len(positions) > 1:
        raise ValueError('Duplicate engine option: ' + name)
    if not positions:
        return None
    position = positions[0]
    if position + 1 == len(arguments) or arguments[position + 1].startswith('--'):
        raise ValueError('Missing engine option value: ' + name)
    try:
        return int(arguments[position + 1])
    except ValueError as error:
        raise ValueError('Invalid integer engine option: ' + name) from error


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish(path, record):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_fixture(fixture_path, synthetic_prefix, synthetic_suffix):
    if fixture_path is None:
        fixture = dict(prefix=[100+i % 17 for i in range(synthetic_prefix)],
                       A=[117+i % 13 for i in range(synthetic_suffix)],
                       B=[150+i % 11 for i in range(synthetic_suffix)])
        return fixture, dict(fixture_kind='synthetic', model_contract_id=None,
                             source_fixture_sha256=None)
    fixture_path = Path(fixture_path)
    supplied = json.loads(fixture_path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError('Fixture JSON must be an object')
    fixture = {}
    for name in ('prefix', 'A', 'B'):
        values = supplied.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f'Fixture requires nonempty {name} token IDs')
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError(f'Fixture {name} must contain nonnegative integer token IDs')
        fixture[name] = values
    contract = supplied.get('model_contract_id')
    if not isinstance(contract, str) or not contract:
        raise ValueError('Code fixture requires model_contract_id')
    return fixture, dict(fixture_kind='supplied-code-fixture', model_contract_id=contract,
                         source_fixture=str(fixture_path.resolve()),
                         source_fixture_sha256=sha(fixture_path))


def native_exit_record(process):
    code = process.poll()
    valid_code = type(code) is int
    return dict(pid=process.pid, exit_code=code, observed=code is not None,
                success=valid_code and code == 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-json', type=Path,
                        help='use exact pretokenized prefix/A/B token arrays from this JSON fixture')
    parser.add_argument('--prefix', type=int, default=8192)
    parser.add_argument('--suffix', type=int, default=512)
    parser.add_argument('--generated', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=16384)
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['fresh', 'resident'], default=['fresh', 'resident'])
    args = parser.parse_args(argv)
    if min(args.prefix, args.suffix, args.generated, args.chunk, args.repetitions) <= 0:
        parser.error('Counts must be positive')
    fixture, fixture_metadata = load_fixture(args.fixture_json, args.prefix, args.suffix)
    branch_suffix_lengths = {name: len(fixture[name]) for name in ('A', 'B')}
    if any(len(fixture['prefix']) + length + args.generated > args.capacity
           for length in branch_suffix_lengths.values()):
        parser.error('Fixture exceeds capacity')
    sys.path.insert(0, str(args.serve_root))
    from serve.server import StrataEngine, engine_args
    config = json.loads(args.config.read_text())
    if config.get('parallel', 1) not in (0, 1):
        raise ValueError('Only one native request at a time is supported')
    executable = Path(config['exe'])
    if not executable.is_absolute():
        executable = Path(config.get('cwd') or '.') / executable
    arguments = engine_args(config)
    if '--batch' in arguments:
        raise ValueError('Multi-request batch option is outside this campaign')
    for name, value in [('--max-context', args.capacity), ('--prefill', args.chunk),
                        ('--kv', 'fp16'), ('--spec', 2), ('--suffix-draft', 0)]:
        arguments = replace_option(arguments, name, value)
    if '--mtp' in arguments:
        raise ValueError('MTP must be separately qualified')
    args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output/'fixture.json', fixture)
    record = dict(status='RUNNING', qualification='NOT_QUALIFIED',
                  fixture_scope=('synthetic deterministic token workload; not a codebase benchmark'
                                 if fixture_metadata['fixture_kind'] == 'synthetic'
                                 else 'supplied exact code-fixture token IDs; no text repacking'),
                  fixture_kind=fixture_metadata['fixture_kind'],
                  model_contract_id=fixture_metadata['model_contract_id'],
                  source_fixture=fixture_metadata.get('source_fixture'),
                  source_fixture_sha256=fixture_metadata['source_fixture_sha256'],
                  source_sha256=sha(__file__), executable_sha256=sha(executable),
                  serving_adapter_sha256=sha(args.serve_root/'serve/server.py'),
                  config_sha256=sha(args.config), fixture_sha256=sha(args.output/'fixture.json'),
                  capacity=args.capacity, chunk=args.chunk, prefix=len(fixture['prefix']),
                  suffix=max(branch_suffix_lengths.values()), suffix_lengths=branch_suffix_lengths,
                  output_bound=args.generated,
                  operator='ordinary-native-serving', controller_scope='greedy',
                  filesystem_cache='uncontrolled; no cold-filesystem claim', runs=[])
    publish(args.output/'result.json', record)
    engine = None
    try:
        for mode in args.modes:
            actual = list(arguments)
            if mode == 'fresh':
                actual = replace_option(actual, '--prompt-cache', 0)
                actual = replace_option(actual, '--conversation-cache-mib', 0)
            else:
                # A pinned shared prefix is disabled by prompt-cache values below 3.
                actual = replace_option(actual, '--prompt-cache',
                                        max(3, option_integer(actual, '--prompt-cache') or 0))
            record.setdefault('engine_arguments', {})[mode] = actual
            startup = time.monotonic()
            engine = StrataEngine(str(executable), actual, cwd=config.get('cwd'),
                log=str(args.output/(mode+'.native.log')),
                env=dict(os.environ, **config.get('env', {})))
            startup_seconds = time.monotonic()-startup
            record.setdefault('engine_startups', []).append(dict(
                mode=mode, pid=engine.proc.pid, seconds=startup_seconds))
            first_request = True
            try:
                for repetition in ([0] if args.warmup else []) + list(range(1, args.repetitions+1)):
                    for request_index, name in enumerate(('A', 'B', 'A')):
                        started = time.monotonic()
                        tokens, timestamps = [], []
                        sampling = dict(temperature=0, strata_prefix={'tokens': len(fixture['prefix'])}) if mode == 'resident' else dict(temperature=0, strata_checkpoint=False)
                        for token in engine.generate(fixture['prefix']+fixture[name], args.generated,
                                                      sampling, threading.Event()):
                            if token is not None:
                                tokens.append(token)
                                timestamps.append(time.monotonic())
                        completed = time.monotonic()
                        stream_seconds = timestamps[-1]-timestamps[0] if len(tokens)>1 else None
                        row = dict(mode=mode, repetition=repetition, warmup=repetition == 0,
                            branch=name, native_pid=engine.proc.pid,
                            request_seconds=completed-started, output_tokens=len(tokens), tokens=tokens,
                            first_token_seconds=timestamps[0]-started if tokens else None,
                            stream_tokens_per_second=(len(tokens)-1)/stream_seconds if stream_seconds else None,
                            drain_seconds=completed-timestamps[-1] if tokens else None,
                            native=dict(engine.last), reused_prompt_tokens=engine.reused,
                            engine_info=dict(engine.info))
                        if mode == 'resident':
                            row['expected_shared_prefix_tokens'] = len(fixture['prefix'])
                            row['shared_prefix_reuse_verified'] = None if first_request else (
                                engine.reused >= len(fixture['prefix']))
                            row['resident_prefix_state'] = 'initial-fill' if first_request else 'reused'
                            if not first_request and not row['shared_prefix_reuse_verified']:
                                raise RuntimeError('Resident request after initial fill did not reuse the complete pinned prefix')
                        if not tokens or engine.last.get('generated') != len(tokens):
                            raise RuntimeError('Output or native completion accounting is incomplete')
                        if mode == 'fresh' and engine.reused:
                            raise RuntimeError('Fresh measurement unexpectedly reused prompt state')
                        record['runs'].append(row)
                        publish(args.output/'result.json', record)
                        print(json.dumps(dict(mode=mode, repetition=repetition, branch=name,
                            outputs=len(tokens), stream_tps=row['stream_tokens_per_second'],
                            reused=engine.reused)), flush=True)
                        first_request = False
            finally:
                process = engine.proc
                engine.close()
                exit_record = native_exit_record(process)
                record.setdefault('exits', []).append(dict(mode=mode, **exit_record))
                engine = None
                publish(args.output/'result.json', record)
            if not exit_record['success']:
                record.update(status='FAIL', error='Native process did not exit with code zero')
                publish(args.output/'result.json', record)
                raise RuntimeError('Native process did not exit with code zero')
        record['status'] = 'MEASURED'
    except BaseException as error:
        record.update(status='FAIL', error=str(error))
        raise
    finally:
        if engine:
            engine.close()
        publish(args.output/'result.json', record)


if __name__ == '__main__':
    main()
