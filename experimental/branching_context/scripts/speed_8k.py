"""Bounded 8K + 512 + at most 128 output experiment; native stdio, no server.

Each result keeps its exact tokens, command, binary hash and native timing.
Resource profiles are experiments, not inherited numerical certifications.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command(profile, resources):
    original = profile['execution']['oracle_command']
    value = lambda flag: original[original.index(flag) + 1]
    args = ['--pack', value('--pack'), '--native', value('--native'),
            '--max-context', '12288', '--prefill', '8192', '--spec', '2',
            '--suffix-draft', '0', '--kv', 'fp16', '--expert-profile', value('--expert-profile'),
            '--prompt-cache', '0', '--conversation-cache-mib', '0', '--stats',
            '--no-prefill-borrow']
    if resources == 'restricted':
        args += ['--mmap-experts', '--pool-workers', '2', '--expert-cache', '512', '--pcie-frac', '0.55']
    elif resources == 'normal':
        args += ['--resident-budget-gib', '40', '--expert-cache', 'auto',
                 '--ple-io', 'direct', '--ple-row-cache', '1048576', '--vram-reserve-mib', '2048']
    else:
        raise ValueError(resources)
    return args


def fixture(source, pack):
    sys.path[:0] = [str(source), str(source / 'tools')]
    from conversation_cache_parity import load_tokenizer
    tok = load_tokenizer(pack / 'tokenizer')
    encode = lambda text: list(tok.encode(text, parse_special=True))
    opening = encode('<|im_start|>user\nThese are records from a fictional warehouse.\n')
    records = encode(''.join(f'Record {i}: a blue crate holds {i % 97 + 1} books in aisle {i % 31}.\n'
                             for i in range(1500)))
    prefix = (opening + records)[:8192]
    suffixes = {}
    for label, subject in [('A', 'a careful engineer'), ('B', 'an adventurous librarian')]:
        ending = encode('\nWrite a detailed story of at least 600 words about ' + subject +
                        ' organizing this warehouse. Describe the work step by step.\n<|im_end|>\n'
                        '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        body = encode(('Additional facts for version ' + label + ': the warehouse has glass skylights.\n') * 80)
        suffixes[label] = body[:512 - len(ending)] + ending
    assert len(prefix) == 8192 and all(len(v) == 512 for v in suffixes.values())
    assert suffixes['A'] != suffixes['B']
    return dict(prefix=prefix, suffixes=suffixes, output_bound=128, eos_tokens=[248044, 248046])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resources', choices=('restricted', 'normal'), required=True)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--labels', default='A,B,A')
    parser.add_argument('--fixed-dispatch', action='store_true',
                        help='Pin PCIe share and disable adaptive expert replacement for parity controls')
    parser.add_argument('--threaded-launch', action='store_true',
                        help='Exercise the existing CPU-serviced graph launch fix')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(options.profile.read_text())
    args = command(profile, options.resources)
    if options.fixed_dispatch:
        args += ['--pcie-frac', '0.55', '--adapt-swaps', '0']
    pack = Path(args[args.index('--pack') + 1])
    tokens = json.loads(options.fixture.read_text()) if options.fixture else fixture(options.source, pack)
    token_path = options.output / 'fixture.json'
    token_path.write_text(json.dumps(tokens) + '\n')
    sys.path.insert(0, str(options.source))
    from serve.server import StrataEngine
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))}
    env.update(profile['execution']['environment'])
    if options.threaded_launch:
        env['STRATA_VERIFY_ASYNC_LAUNCH'] = '1'
    report = dict(schema=1, status='RUNNING', phase='startup', resource_profile=options.resources,
                  binary=str(options.engine), binary_sha256=sha(options.engine), args=args,
                  environment={k: v for k, v in env.items() if k.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))},
                  script_sha256=sha(__file__), fixture_sha256=sha(token_path),
                  model_identity=profile['model_weights_sha256'], hardware=profile['hardware'], requests=[],
                  limitations=['Timing and token agreement experiment; not full-logit certification',
                               'Native serving may leave the final output unconsumed',
                               'No MTP; FP16 KV; exact fixture shared across compared arms'])

    def save():
        report['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        pending = options.output / 'result.json.tmp'
        pending.write_text(json.dumps(report, indent=2) + '\n')
        pending.replace(options.output / 'result.json')

    engine = None
    save()
    try:
        started = time.perf_counter()
        engine = StrataEngine(str(options.engine), args, cwd=str(options.source),
                              log=str(options.output / 'engine.log'), env=env)
        report.update(startup_seconds=time.perf_counter() - started, engine_info=engine.info)
        for index, label in enumerate(options.labels.split(',')):
            prompt = tokens['prefix'] + tokens['suffixes'][label]
            report.update(phase='request', request_index=index, label=label, accepted_outputs=0)
            save()
            print(json.dumps(dict(event='request', index=index, label=label, prompt_tokens=len(prompt))), flush=True)
            started = time.perf_counter()
            ids = []
            first = None
            for token in engine.generate(prompt, tokens['output_bound'],
                                         {'temperature': 0, 'repetition_penalty': 1.0}, threading.Event()):
                if token is None:
                    continue
                if first is None:
                    first = time.perf_counter() - started
                ids.append(token)
                if len(ids) % 16 == 0:
                    report['accepted_outputs'] = len(ids)
                    save()
                    print(json.dumps(dict(event='decode', count=len(ids), seconds=time.perf_counter()-started)), flush=True)
            native = dict(engine.last)
            row = dict(label=label, prompt_tokens=len(prompt), ids=ids, actual_outputs=len(ids),
                       seconds=time.perf_counter()-started, first_token_seconds=first, native=native,
                       native_decode_tps=(native.get('generated', 0)*1000/native['decode_ms']
                                          if native.get('decode_ms') else None))
            report['requests'].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k != 'ids'}), flush=True)
            if not ids or not 0 < native.get('generated', 0) <= tokens['output_bound']:
                raise AssertionError('No completed generation or output bound exceeded')
            if native.get('reused', 0) != 0:
                raise AssertionError('Uncached serving control unexpectedly reused prompt state')
        report.update(status='MEASURED', phase='complete')
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        if engine is not None:
            engine.close()
        save()


if __name__ == '__main__':
    main()
