"""Measure aligned batched prefix reuse, retaining exact fresh-oracle checks.

This is a bounded experiment with the existing isolated native probe, not a
qualified execution profile. Separate timers expose validation/controller cost.
"""
import argparse
import array
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from speed_8k import command, sha


def main(request_lanes=(('resident', 'A'), ('resident', 'B'), ('host-ram', 'A'), ('disk', 'B'))):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle-evidence', type=Path,
                        help='Reuse already executed independent fresh traces with matching fingerprints')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.api / 'src'))
    from context_time_travel import controller
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _digest, _validate_execution, _validate_model
    from context_time_travel.checkpoint_cache import CheckpointCache
    original = json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(original['execution']['environment'])
    _validate_execution(original)
    verification = _validate_model(original)
    profile = copy.deepcopy(original)
    build_receipt = json.loads((args.binaries / 'build.json').read_text())
    write_integrity = build_receipt.get('write_integrity_policy', 'per-write-sha256')
    private_access = build_receipt.get('private_page_access', 'set-each-write')
    assert write_integrity in ('per-write-sha256', 'externally-managed')
    for path in (args.api / 'src/context_time_travel').glob('*.py'):
        profile['native_build']['sha256'][str(path)] = sha(path)
    common = command(original, 'normal') + ['--pcie-frac', '0.55', '--max-new', '128']
    for key, binary in [('sut_command', 'strata-batched-branch'), ('oracle_command', 'strata-batched-fresh')]:
        path = args.binaries / binary
        profile['execution'][key] = [str(path), *common]
        if key == 'sut_command':
            # The CLI parser requires an input; the branch driver starts empty.
            profile['execution'][key] += ['--tokens', '1']
        profile['native_build']['sha256'][str(path)] = sha(path)
    profile['execution'].update(capacity=12288, operator='aligned-batched-prefix/normal-resources/fixed-pcie-0.55')
    profile['execution'].update(write_integrity_policy=write_integrity, private_page_access=private_access)
    profile['execution_profile_id'] = _digest(dict(build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))
    profile['numerical_gate_qualified'] = False
    # A changed resource profile is not admitted by the old numerical baseline.
    profile['calibration_status'] = 'EXPERIMENTAL_UNQUALIFIED'
    (args.output / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    tokens = json.loads(args.fixture.read_text())
    prefix = tuple(tokens['prefix'])
    assert len(prefix) == 8192 and all(len(t) == 512 for t in tokens['suffixes'].values())
    report = dict(status='RUNNING', phase='start', schema=1, model_verification=verification,
                  model_identity=profile['model_weights_sha256'], hardware=profile['hardware'],
                  profile_sha256=sha(args.output / 'profile.json'), fixture_sha256=sha(args.fixture),
                  script_sha256=sha(__file__), oracle={}, requests=[], numerical_max_error=0.0,
                  request_sequence=request_lanes,
                  write_integrity_policy=write_integrity, private_page_access=private_access,
                  limitations=['Aligned 8192-token prefill boundary only; arbitrary split remains open',
                               'Each native fresh process has no branch/checkpoint restore',
                               'Timings separate native calls, full-vector validation, controller and durable publication',
                               'Warm filesystem disk restore; no full G1/G2 qualification'])
    if write_integrity == 'externally-managed':
        report['limitations'].append('Caller guarantees unchanged sealed GPU bytes; write preparation does not detect an external violation')

    def save():
        report['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        pending = args.output / 'result.json.tmp'
        pending.write_text(json.dumps(report, indent=2) + '\n')
        pending.replace(args.output / 'result.json')

    def stage(name, **fields):
        report.update(phase=name, **fields)
        save()
        print(json.dumps(dict(event=name, **fields)), flush=True)

    def read_vector(path):
        values = array.array('f')
        values.frombytes(Path(path).read_bytes())
        if len(values) != 248320:
            raise AssertionError('Incomplete full-vocabulary vector')
        return values

    def equal(actual, expected, row):
        if len(actual) != 248320 or len(expected) != 248320:
            raise AssertionError('Incomplete logits')
        if not all(math.isfinite(v) for vector in (actual, expected) for v in vector):
            raise AssertionError('Nonfinite logits')
        error = max(abs(a-b) for a, b in zip(actual, expected))
        row['comparisons'] = row.get('comparisons', 0) + 1
        row['max_absolute_error'] = max(row.get('max_absolute_error', 0), error)
        report['numerical_max_error'] = max(report['numerical_max_error'], error)
        if error != 0:
            save()
            raise AssertionError('Fresh full-history disagreement: ' + str(error))

    backend = None
    save()
    try:
        # Two separate full-history forwards, before reuse; no cached prefix oracle.
        for label, suffix in tokens['suffixes'].items():
            prompt = args.output / ('fresh-' + label + '.tokens')
            prompt.write_text(','.join(map(str, prefix + tuple(suffix))))
            vector = args.output / ('fresh-' + label + '.f32')
            argv = profile['execution']['oracle_command'] + ['--tokens-file', str(prompt)]
            stage('fresh ' + label)
            started = time.perf_counter()
            if args.oracle_evidence:
                prior = json.loads((args.oracle_evidence / 'result.json').read_text())
                prior_profile = json.loads((args.oracle_evidence / 'profile.json').read_text())
                assert prior['fixture_sha256'] == report['fixture_sha256']
                previous_command = prior_profile['execution']['oracle_command']
                current_command = profile['execution']['oracle_command']
                assert previous_command[1:] == current_command[1:]
                assert prior_profile['native_build']['sha256'][previous_command[0]] == profile['native_build']['sha256'][current_command[0]]
                assert prior_profile['model_contract_id'] == profile['model_contract_id']
                assert prior_profile['hardware'] == profile['hardware']
                assert prior['oracle'][label]['logits_sha256'] == sha(args.oracle_evidence / vector.name)
                assert prior['oracle'][label]['trace_sha256'] == sha(args.oracle_evidence / (vector.name + '.decode.f32'))
                for extension in ('.stdout', '.stderr', '.f32', '.f32.decode.f32'):
                    name = 'fresh-' + label + extension
                    shutil.copyfile(args.oracle_evidence / name, args.output / name)
            else:
                with (args.output / ('fresh-' + label + '.stdout')).open('wb') as out, (args.output / ('fresh-' + label + '.stderr')).open('wb') as err:
                    done = subprocess.run(argv, env=dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(vector)),
                                          stdout=out, stderr=err, timeout=600)
                if done.returncode:
                    raise RuntimeError('Fresh process failed: ' + str(done.returncode))
            native = json.loads((args.output / ('fresh-' + label + '.stdout')).read_text())
            count = len(native['generated'])
            assert native['prefill_tokens'] == 8704 and native['decode_tokens'] == count
            assert native['consumed'] == 8704 + count and 0 < count <= 128
            assert count == 128 or native['generated'][-1] in tokens['eos_tokens']
            assert Path(str(vector) + '.decode.f32').stat().st_size == count * 248320 * 4
            report['oracle'][label] = dict(native=native, seconds=time.perf_counter()-started,
                                           logits_sha256=sha(vector), trace_sha256=sha(str(vector)+'.decode.f32'),
                                           native_decode_tps=count/native['decode_seconds'])
            if args.oracle_evidence:
                report['oracle'][label]['reused_executed_evidence'] = str(args.oracle_evidence)
                report['oracle'][label]['seconds'] = prior['oracle'][label]['seconds']
            stage('fresh complete', label=label, native_decode_tps=count/native['decode_seconds'],
                  prefill_seconds=native['prefill_seconds'], outputs=count)
        stage('COW startup')
        backend = StrataBackend(profile)
        backend.new_cache = lambda: CheckpointCache(2 * 1024**3, 256)
        sut = BranchingModel(args.output / 'store', backend, profile)
        started = time.perf_counter()
        root = sut.create(prefix)
        report['prefix_prepare_seconds'] = time.perf_counter()-started
        held = sut.cache[root].blob
        root_bytes = sut.snapshot_bytes(root)
        audit = backend.audit(held)
        started = time.perf_counter()
        files = sut.checkpoint(root)
        report['save_seconds'] = time.perf_counter()-started
        report['checkpoint'] = dict(bytes=files['payload_path'].stat().st_size, sha256=sha(files['payload_path']))
        for lane, label in request_lanes:
            stage(lane + ' ' + label)
            suffix = tuple(tokens['suffixes'][label])
            oracle = report['oracle'][label]['native']
            if lane == 'disk':
                sut.cache.clear()
                backend.release_checkpoints()
            before = backend._request('stats')
            before_ram = backend._request('ram_stats')
            row = dict(lane=lane, label=label, status='RUNNING', timers={})
            report['requests'].append(row)
            started = time.perf_counter()
            manager = backend.lease() if lane == 'host-ram' else sut.materialize(root)
            with manager as materialized:
                lease = materialized if lane == 'host-ram' else materialized[0]
                if lane == 'host-ram' and not backend.restore_cached(lease, prefix):
                    raise AssertionError('No actual RAM restore')
                row['restore_seconds'] = time.perf_counter()-started
                t = time.perf_counter()
                backend.consume(lease, suffix)
                row['suffix_prefill_seconds'] = time.perf_counter()-t
                logits = backend.logits(lease)
                equal(logits, read_vector(args.output / ('fresh-' + label + '.f32')), row)
                after_prefill = backend._request('stats')
                assert after_prefill['prefill_tokens']-before['prefill_tokens'] == 512
                decoder = controller.start(dict(temperature=0, eos_tokens=tokens['eos_tokens']),
                                           'speed-8k', prefix+suffix, backend.vocab_size)
                sid = sut.store.append(root, suffix, decoder)
                generated = []
                timers = row['timers']
                decode_start = time.perf_counter()
                with (args.output / ('fresh-' + label + '.f32.decode.f32')).open('rb') as trace:
                    for index in range(128):
                        if decoder['status'] == 'stopped':
                            break
                        t = time.perf_counter()
                        token, next_decoder = controller.choose(logits, decoder)
                        timers['choose'] = timers.get('choose', 0) + time.perf_counter()-t
                        t = time.perf_counter()
                        response = backend._request('decode ' + str(token))
                        timers['native_decode_call'] = timers.get('native_decode_call', 0) + time.perf_counter()-t
                        assert response['consumed'] == lease.consumed+1
                        lease.consumed += 1
                        lease.image_identity = None
                        t = time.perf_counter()
                        logits = backend.logits(lease)
                        timers['read_logits'] = timers.get('read_logits', 0) + time.perf_counter()-t
                        t = time.perf_counter()
                        ref = array.array('f')
                        ref.frombytes(trace.read(248320*4))
                        equal(logits, ref, row)
                        assert index < len(oracle['generated']) and token == oracle['generated'][index]
                        timers['validate_logits'] = timers.get('validate_logits', 0) + time.perf_counter()-t
                        t = time.perf_counter()
                        decoder = controller.consume(next_decoder, token, backend.piece(token), backend.is_eog(token))
                        timers['consume_controller'] = timers.get('consume_controller', 0) + time.perf_counter()-t
                        t = time.perf_counter()
                        sid = sut.store.append(sid, (token,), decoder)
                        timers['durable_append'] = timers.get('durable_append', 0) + time.perf_counter()-t
                        generated.append(token)
                        if index % 16 == 15:
                            stage('COW decode', lane=lane, label=label, outputs=len(generated), timers=timers)
                row['decode_seconds'] = time.perf_counter()-decode_start
                row['native_call_tps'] = len(generated)/timers['native_decode_call']
                row['instrumented_api_tps'] = len(generated)/row['decode_seconds']
                assert generated == oracle['generated']
                after_decode = backend._request('stats')
                row['decode_integrity'] = {key: after_decode[key]-after_prefill[key]
                                          for key in ('integrity_kernel_calls', 'integrity_device_hashed_bytes', 'integrity_checked_pages')}
                if write_integrity == 'externally-managed' or lane == 'host-ram':
                    assert all(value == 0 for value in row['decode_integrity'].values())
                else:
                    assert row['decode_integrity']['integrity_kernel_calls'] == len(generated)
                saved = sut._save(lease, prefix+suffix+tuple(generated))
            sut.cache[sid] = saved
            row.update(seconds=time.perf_counter()-started, ids=generated, actual_outputs=len(generated))
            after = backend._request('stats')
            keys = ('forward_tokens', 'prefill_tokens', 'windows', 'loaded_file_bytes', 'cow_pages', 'copied_bytes',
                    'integrity_kernel_calls', 'integrity_device_hashed_bytes', 'integrity_checked_pages',
                    'decode_prepare_ns', 'decode_compute_ns', 'decode_finish_ns', 'decode_timed_tokens')
            row['counters'] = {key: after[key]-before[key] for key in keys}
            row['counters']['ram_hits'] = backend._request('ram_stats')['hits']-before_ram['hits']
            assert row['counters']['forward_tokens'] == 512+len(generated)
            assert row['counters']['windows'] == len(generated)
            assert row['counters']['decode_timed_tokens'] == len(generated)
            assert all(row['counters'][key] > 0 for key in ('decode_prepare_ns', 'decode_compute_ns', 'decode_finish_ns'))
            row['native_decode_seconds'] = {name: row['counters']['decode_' + name + '_ns']/1e9
                                            for name in ('prepare', 'compute', 'finish')}
            assert (row['counters']['loaded_file_bytes'] > 0) == (lane == 'disk')
            if lane == 'host-ram':
                assert row['counters']['ram_hits'] == 1
                assert row['counters']['cow_pages'] == row['counters']['copied_bytes'] == 0
            assert sut.history(sid) == prefix+suffix+tuple(generated)
            assert sut.snapshot_bytes(root) == root_bytes
            current = backend.audit(held)
            assert current['sha256'] == audit['sha256']
            row['parent_audit'] = dict(bytes=current['bytes'], sha256=current['sha256'])
            assert sha(files['payload_path']) == report['checkpoint']['sha256']
            row['gpu_used_mib_after'] = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            row['status'] = 'PASS'
            stage('COW request complete', lane=lane, label=label, native_call_tps=row['native_call_tps'], instrumented_api_tps=row['instrumented_api_tps'])
        report['status'] = 'PASS'
        stage('complete')
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        stage('failed')
        raise
    finally:
        if backend is not None:
            backend._log.seek(0)
            (args.output / 'native.log').write_bytes(backend._log.read())
            backend.close()
        save()


if __name__ == '__main__':
    main()
