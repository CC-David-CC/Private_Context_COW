"""Real GPU qualification of the resident request primitive, before store integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API/'src'))
from context_time_travel.strata_backend import StrataBackend, _validate_execution, _validate_model, _unpack_frontier


def publish(path, record):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _record_native_shutdown(record, processes, close):
    """Close native workers and make their numeric exit status part of PASS."""
    close_error = None
    try:
        close()
    except BaseException as exc:
        close_error = str(exc)
    codes = {}
    pids = {}
    poll_errors = {}
    for label, process in processes.items():
        key = str(label)
        pids[key] = process.pid
        try:
            codes[key] = process.poll()
        except BaseException as exc:
            codes[key] = None
            poll_errors[key] = str(exc)
    observed = bool(codes) and all(type(code) is int for code in codes.values())
    success = observed and all(code == 0 for code in codes.values()) and close_error is None and not poll_errors
    record['native_pids'] = pids
    record['native_exit_codes'] = codes
    record['native_exit_observed'] = observed
    record['native_exit_success'] = success
    if close_error is not None:
        record['cleanup_error'] = close_error
    if poll_errors:
        record['native_exit_poll_errors'] = poll_errors
    if not success:
        record['status'] = 'FAIL'
        if close_error is None:
            record['native_exit_error'] = 'Native worker shutdown did not complete successfully'
    return success


def frontier(values):
    return struct.pack('<'+'f'*len(values), *values)


def reference(profile, output, name, ids, requests, generated):
    tokens = output/(name+'.tokens')
    target = output/(name+'.f32')
    tokens.write_text(','.join(map(str, ids)))
    command = list(profile['execution']['oracle_command'])
    command[command.index('--max-new')+1] = str(generated)
    command += ['--tokens-file', str(tokens)]
    started = time.monotonic()
    with (output/(name+'.stdout')).open('wb') as out, (output/(name+'.stderr')).open('wb') as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=900,
            env=dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(target),
                     STRATA_PREFILL_REQUESTS=','.join(map(str, requests))))
    if result.returncode:
        raise AssertionError('Uncached native reference failed: '+name)
    work = json.loads((output/(name+'.stdout')).read_text())
    schedule = json.loads(Path(str(target)+'.schedule.json').read_text())
    outputs = work['generated']
    if (schedule['requests'] != requests or work['prefill_tokens'] != len(ids)
            or work['normal_forward_tokens'] != len(outputs)
            or work['consumed'] != len(ids)+len(outputs)
            or not 0 < len(outputs) <= generated):
        raise AssertionError('Independent native work accounting differs')
    width = 4*248320
    raw = Path(str(target)+'.decode.f32').read_bytes()
    prompt_raw = Path(str(target)+'.requests.f32').read_bytes()
    if len(raw) != width*len(outputs) or len(prompt_raw) != width*len(requests):
        raise AssertionError('Incomplete independent frontier evidence')
    return dict(tokens=outputs, generated_frontiers=[raw[i:i+width] for i in range(0,len(raw),width)],
        prompt_frontiers=[prompt_raw[i:i+width] for i in range(0,len(prompt_raw),width)],
        work=work, seconds=time.monotonic()-started, command=command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=int, default=128)
    parser.add_argument('--suffix', type=int, default=4)
    parser.add_argument('--generated', type=int, default=8)
    parser.add_argument('--check-boundaries', choices=['every','sample'], default='every')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    model_identity = _validate_model(profile)
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = dict(prefix=[100+i%17 for i in range(args.prefix)],
        A=[117+i%13 for i in range(args.suffix)], B=[150+i%11 for i in range(args.suffix)])
    publish(args.output/'fixture.json', fixture)
    record = dict(status='RUNNING', scope='native request primitive only; canonical publication not qualified',
        profile_id=profile['execution_profile_id'], model_contract_id=profile['model_contract_id'],
        model_verification=model_identity, comparisons=0, numerical_max_error=0., branches={},
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checked_boundaries=args.check_boundaries,
        fixture_sizes=dict(prefix=args.prefix, suffix=args.suffix, output_limit=args.generated))
    def compare(actual, expected):
        if actual != expected:
            (args.output/'mismatch-actual.f32').write_bytes(actual)
            (args.output/'mismatch-expected.f32').write_bytes(expected)
            a, b = _unpack_frontier(actual, 248320), _unpack_frontier(expected, 248320)
            record['numerical_max_error'] = max(record['numerical_max_error'], max(abs(x-y) for x,y in zip(a,b)))
            raise AssertionError('Matching-schedule frontier differs')
        record['comparisons'] += 1
    backend = None
    shutdown_ok = True
    try:
        references = {}
        for name in ('A', 'B', 'A-repeat'):
            branch = name.split('-')[0]
            references[name] = reference(profile, args.output, name,
                fixture['prefix']+fixture[branch], [args.prefix,args.suffix], args.generated)
            publish(args.output/'result.json', record)
        if references['A']['tokens'] != references['A-repeat']['tokens']:
            raise AssertionError('Independent fixed-profile greedy repeat differs')
        for field in ('prompt_frontiers', 'generated_frontiers'):
            for a,b in zip(references['A'][field], references['A-repeat'][field]):
                compare(a,b)
        startup = time.monotonic()
        backend = StrataBackend(profile)
        record['startup_seconds'] = time.monotonic()-startup
        with backend.lease() as lease:
            backend.consume(lease, fixture['prefix'])
            root = backend.export(lease)
            root_frontier = backend.logits(lease)
            compare(frontier(root_frontier), references['A']['prompt_frontiers'][0])
            original = hashlib.sha256(backend.serialize(lease,root)).hexdigest()
        for name in ('A','B'):
            with backend.lease() as lease:
                before = backend.work_counters()
                backend.restore(lease,root,history=fixture['prefix'],frontier=root_frontier)
                restored = backend.work_counters()
                if restored['forward_tokens'] != before['forward_tokens']:
                    raise AssertionError('Root activation recomputed saved tokens')
                backend.consume(lease,fixture[name])
                compare(frontier(backend.logits(lease)), references[name]['prompt_frontiers'][-1])
                before = backend.work_counters()
                started = time.monotonic()
                with backend.generate_request(lease,args.generated) as request:
                    outputs = list(request)
                elapsed = time.monotonic()-started
                after = backend.work_counters()
                if outputs != references[name]['tokens']:
                    raise AssertionError('Native request tokens differ from fresh reference')
                if (after['forward_tokens']-before['forward_tokens'] != len(outputs)
                        or after['prefill_tokens'] != before['prefill_tokens']
                        or after['captures'] != before['captures'] or after['restores'] != before['restores']):
                    raise AssertionError('Generation repeated state work or omitted normal forward work')
                compare(frontier(backend.logits(lease)), references[name]['generated_frontiers'][-1])
                boundary = backend.export(lease)
                record['branches'][name] = dict(outputs=outputs, seconds=elapsed,
                    tokens_per_second=len(outputs)/elapsed, consumed=lease.consumed,
                    request_receipt=backend.last_native_request,
                    work_delta={k:v-before.get(k,0) for k,v in after.items() if type(v) is int},
                    boundary_state_bytes=len(boundary))
                publish(args.output/'result.json', record)
            # Separate resident reruns terminate at each checked boundary. This
            # verifies every tiny/medium frontier without exporting vectors in
            # the timed generation loop or introducing per-token snapshots.
            total = len(references[name]['tokens'])
            checks = range(1,total+1) if args.check_boundaries == 'every' else sorted({1,total,*[n for n in (16,64) if n <= total]})
            record['branches'][name]['checked_generated_boundaries'] = list(checks)
            for count in checks:
                with backend.lease() as lease:
                    before = backend.work_counters()
                    backend.restore(lease,root,history=fixture['prefix'],frontier=root_frontier)
                    if backend.work_counters()['forward_tokens'] != before['forward_tokens']:
                        raise AssertionError('Boundary check replayed the saved prefix')
                    backend.consume(lease,fixture[name])
                    with backend.generate_request(lease,count) as request:
                        tokens = list(request)
                    if tokens != references[name]['tokens'][:count]:
                        raise AssertionError('Boundary rerun tokens differ')
                    compare(frontier(backend.logits(lease)), references[name]['generated_frontiers'][count-1])
                publish(args.output/'result.json', record)
        with backend.lease() as lease:
            backend.restore(lease,root,history=fixture['prefix'],frontier=root_frontier)
            backend.consume(lease,fixture['A'])
            with backend.generate_request(lease,args.generated) as request:
                token = next(request)
            if token != references['A']['tokens'][0]:
                raise AssertionError('Cancelled request selected a different first token')
            compare(frontier(backend.logits(lease)), references['A']['generated_frontiers'][0])
            record['cancellation_consumed'] = lease.consumed
            backend.restore(lease,root,history=fixture['prefix'],frontier=root_frontier)
            if hashlib.sha256(backend.serialize(lease,root)).hexdigest() != original:
                raise AssertionError('Saved parent bytes changed')
        record.update(status='PASS', parent_byte_audit='PASS', final_counters=backend.work_counters())
    except BaseException as error:
        record.update(status='FAIL', error=str(error))
        raise
    finally:
        if backend:
            process = backend.process
            shutdown_ok = _record_native_shutdown(record, {backend.gpu_id if backend.gpu_id is not None else 0: process}, backend.close)
        publish(args.output/'result.json', record)
    if not shutdown_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
