"""Real-backend restart harness. Strict restore never recomputes saved inputs."""
import argparse
import array
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API/'src'))
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import _validate_execution, _validate_model, _unpack_frontier
from context_time_travel.numerics import compare_logits
from context_time_travel.store import IntegrityError, _json_bytes


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def numerical_record(report):
    """JSON has no infinity; retain that diagnostic as an explicit string."""
    record = asdict(report)
    if not math.isfinite(record['max_normalized_error']):
        record['max_normalized_error'] = 'infinity'
    return record


def vector(path, values=None):
    if values is not None:
        out = array.array('f', values)
        if sys.byteorder != 'little': out.byteswap()
        path.write_bytes(out.tobytes())
        return str(path)
    out = array.array('f')
    out.frombytes(path.read_bytes())
    if sys.byteorder != 'little': out.byteswap()
    return tuple(out)


def open_model(args):
    profile = json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    os.environ['CTT_NATIVE_TRACE'] = str(args.output/(args.phase+'-native.jsonl'))
    start = time.monotonic()
    backend = BranchingModel.load_backend(profile)
    return BranchingModel(args.output/'store', backend, profile), time.monotonic()-start


def capture(model, sid, output, label):
    path = output/(label+'.f32')
    vector(path, model.next_logits(sid))
    saved = model.cache[sid]
    from context_time_travel.execution_receipt import schedule_digest
    return dict(snapshot=sid, history=list(model.history(sid)), controller=model.controller(sid),
                frontier=path.name, frontier_sha256=sha(path), consumed=model.consumed(sid),
                execution_schedule=list(saved.execution_schedule),
                schedule_sha256=schedule_digest(saved.execution_schedule))


def assert_batched_generation_work(before, after, count):
    if (after['prefill_tokens']-before['prefill_tokens'] != count
            or after['consumed_tokens']-before['consumed_tokens'] != count
            or after['decode_timed_tokens']-before['decode_timed_tokens'] != 0):
        raise AssertionError('Batches-only generation used decode, replayed history or omitted work')


def save(model, branch, sid, operation):
    head = model.store.head(branch)
    return model.save_branch(branch, head.revision, sid, operation)


class BatchedOracle:
    """Fresh Prefill session, independent of BranchSession and checkpoint loading."""
    def __init__(self, profile, output, namespace='compare'):
        self.profile, self.output, self.requests = profile, output, 0
        self.namespace = namespace
        _validate_execution(profile)
        _validate_model(profile)

    def forward(self, ids):
        label = self.namespace+'-oracle-'+str(self.requests)
        tokens = self.output/(label+'.tokens')
        frontier = self.output/(label+'.f32')
        tokens.write_text(','.join(map(str, ids)))
        command = list(self.profile['execution']['oracle_command'])
        # The production argument parser requires a positive generation bound.
        # This independent driver writes its prompt frontier BEFORE that decode.
        # Account for the extra consumed token rather than hiding its work.
        command[command.index('--max-new')+1] = '1'
        command += ['--tokens-file', str(tokens)]
        started = time.monotonic()
        result = subprocess.run(command, env=dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(frontier)),
                                capture_output=True, timeout=900)
        (self.output/(label+'.stdout')).write_bytes(result.stdout)
        (self.output/(label+'.stderr')).write_bytes(result.stderr)
        if result.returncode: raise RuntimeError('Fresh oracle failed: '+label)
        work = json.loads(result.stdout)
        if (work['prefill_tokens'] != len(ids) or work['prompt_consumed'] != len(ids)
                or work['consumed'] != len(ids)+work['decode_tokens'] or work['decode_tokens'] != 1):
            raise IntegrityError('Fresh oracle did not evaluate exactly the full history')
        raw = frontier.read_bytes()
        self.last = dict(command=command, work=work, seconds=time.monotonic()-started,
                         logits_sha256=sha(frontier), frontier=frontier.name,
                         frontier_boundary=len(ids), extra_decode_tokens=work['decode_tokens'])
        self.requests += 1
        return _unpack_frontier(raw, 248320)

    def close(self):
        pass

    def continuation(self, ids, requests, generated):
        """One independently initialized native run; no checkpoint imports."""
        if sum(requests) != len(ids) or any(type(n) is not int or n <= 0 for n in requests):
            raise IntegrityError('Invalid reference request schedule')
        label = self.namespace+'-scheduled-'+str(self.requests)
        tokens,frontier = self.output/(label+'.tokens'),self.output/(label+'.f32')
        tokens.write_text(','.join(map(str,ids)))
        command = list(self.profile['execution']['oracle_command'])
        command[command.index('--max-new')+1] = str(generated)
        command += ['--tokens-file',str(tokens)]
        started = time.monotonic()
        result = subprocess.run(command,env=dict(os.environ,
            STRATA_DUMP_FIRST_LOGITS=str(frontier),STRATA_PREFILL_REQUESTS=','.join(map(str,requests))),
            capture_output=True,timeout=900)
        (self.output/(label+'.stdout')).write_bytes(result.stdout)
        (self.output/(label+'.stderr')).write_bytes(result.stderr)
        if result.returncode: raise RuntimeError('Matching-schedule reference failed: '+label)
        work = json.loads(result.stdout)
        schedule = json.loads(Path(str(frontier)+'.schedule.json').read_text())
        outputs = work['generated']
        if (schedule['requests'] != requests or schedule['operator'] != 'batched-prefill'
                or work['continuation_operator'] != 'batched-prefill'
                or work['prefill_tokens'] != len(ids)
                or work['total_prefill_tokens'] != len(ids)+len(outputs)
                or work['consumed'] != len(ids)+len(outputs)):
            raise IntegrityError('Reference schedule/work witness mismatch')
        from context_time_travel.execution_receipt import tokens_digest,validate_schedule
        events,position,chunk_index = [],0,0
        all_ids = list(ids)+outputs
        chunks = schedule['completed_chunks']
        for count in requests+[1]*len(outputs):
            used = []
            end = position+count
            while chunk_index < len(chunks) and chunks[chunk_index][0] < end:
                used.append(chunks[chunk_index]);chunk_index += 1
            events.append(dict(operator='batched-prefill',position=position,count=count,
                tokens_sha256=tokens_digest(all_ids[position:end]),chunks=used))
            position = end
        if chunk_index != len(chunks): raise IntegrityError('Unaccounted reference native chunks')
        try: validate_schedule(events,all_ids,self.profile['execution']['prefill_chunk'])
        except ValueError as error: raise IntegrityError('Invalid independent native chunk witness') from error
        width = 4*248320
        request_bytes = Path(str(frontier)+'.requests.f32').read_bytes()
        generated_bytes = Path(str(frontier)+'.decode.f32').read_bytes()
        if len(request_bytes) != width*len(requests) or len(generated_bytes) != width*len(outputs):
            raise IntegrityError('Incomplete reference frontiers')
        frontiers = [_unpack_frontier(raw[i:i+width],248320)
                     for raw in (request_bytes,generated_bytes) for i in range(0,len(raw),width)]
        self.last = dict(command=command,work=work,schedule=schedule,
            execution_schedule=events,
            seconds=time.monotonic()-started,request_frontiers_sha256=sha(Path(str(frontier)+'.requests.f32')),
            generated_frontiers_sha256=sha(Path(str(frontier)+'.decode.f32')))
        self.requests += 1
        return outputs,frontiers


def continuation(model, root, suffix, args, label):
    partials = []
    bound = model.profile['execution'].get('resume_contract') == 'identical-native-batches-v1'
    if args.prefix <= 128 and not bound:
        for length in range(1,len(suffix)):
            partial = model.feed(root,suffix[:length])
            partials.append(capture(model,partial,args.output,label+'-suffix-'+str(length)))
    before = model.backend.work_counters()
    start = time.monotonic()
    tip = model.feed(root, suffix)
    elapsed = time.monotonic()-start
    after = model.backend.work_counters()
    if after['consumed_tokens']-before['consumed_tokens'] != len(suffix):
        raise AssertionError('Suffix recomputed saved prefix or omitted input work')
    if after.get('prefill_tokens', 0)-before.get('prefill_tokens', 0) != len(suffix):
        raise AssertionError('Suffix did not use native batched prefill')
    states = partials+[capture(model, tip, args.output, label+'-suffix')]
    policy = dict(temperature=0)
    tip = model.start_generation(tip, policy, 'durable-greedy')
    tokens = []
    generated_before = model.backend.work_counters()
    begin = time.monotonic()
    first_token_seconds = None
    for tip, token in model.stream(tip, args.generated):
        tokens.append(token)
        if first_token_seconds is None: first_token_seconds = time.monotonic()-begin
        # Tiny fixtures check every generated frontier; longer fixtures sample.
        if args.prefix <= 1024 or len(tokens) in (1, 16, 64, args.generated):
            states.append(capture(model, tip, args.output, label+'-generated-'+str(len(tokens))))
        if len(tokens) == min(4, args.generated):
            model.store.fork(label+'-mid', tip)
            save(model, label+'-mid', tip, label+'-save-mid')
            publish(args.output/(label+'-mid.json'), dict(snapshot=tip, tokens=tokens[:]))
        if model.controller(tip)['status'] == 'stopped': break
    if not states or states[-1]['snapshot'] != tip:
        states.append(capture(model, tip, args.output, label+'-final'))
    decode_elapsed = time.monotonic()-begin
    generated_after = model.backend.work_counters()
    if model.profile['execution'].get('generated_operator') == 'batched-prefill':
        assert_batched_generation_work(generated_before, generated_after, len(tokens))
    return dict(suffix_tokens=len(suffix), suffix_seconds=elapsed,
        suffix_work=after['consumed_tokens']-before['consumed_tokens'],
        suffix_prefill_chunks=after['prefill_chunks']-before['prefill_chunks'],
        generated=tokens, stopping_reason=model.controller(tip)['status'], states=states,
        generated_work_before=generated_before, generated_work_after=generated_after,
        first_token_seconds=first_token_seconds, decode_seconds=decode_elapsed,
        end_to_end_decode_tps=len(tokens)/decode_elapsed if decode_elapsed else None)


def prepare(args):
    model, startup = open_model(args)
    result = dict(status='RUNNING', python_pid=os.getpid(), native_pid=model.backend.process.pid,
                  startup_seconds=startup, branches={})
    try:
        fixture = json.loads((args.output/'fixture.json').read_text())
        before = model.backend.work_counters()
        start = time.monotonic()
        root = model.create(fixture['prefix'])
        result['prefill_seconds'] = time.monotonic()-start
        result['prefill_counters_before'] = before
        result['prefill_counters_after'] = model.backend.work_counters()
        assert result['prefill_counters_after']['prefill_tokens']-before['prefill_tokens'] == args.prefix
        result['root'] = capture(model, root, args.output, 'prepared-root')
        model.store.fork('saved', root)
        start = time.monotonic()
        result['save_receipt'] = save(model, 'saved', root, 'prepare-save')
        result['save_seconds'] = time.monotonic()-start
        publish(args.output/'prepare.json', result)
        for name in ('A', 'B'):
            result['branches'][name] = continuation(model, root, fixture[name], args, 'uninterrupted-'+name)
            publish(args.output/'prepare.json', result)
        result['counters'] = model.backend.work_counters()
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        raise
    finally:
        model.backend.close()
        result['final_counters'] = dict(model.backend.stats)
        result['native_exit_code'] = model.backend.process.returncode
        result['native_exit_observed'] = model.backend.process.poll() is not None
        publish(args.output/'prepare.json', result)


def resume(args):
    model, startup = open_model(args)
    result = dict(status='RUNNING', python_pid=os.getpid(), native_pid=model.backend.process.pid,
                  startup_seconds=startup, branches={})
    try:
        prior = json.loads((args.output/'prepare.json').read_text())
        assert prior['native_exit_observed'] and prior['native_pid'] != model.backend.process.pid
        root = model.store.head('saved').tip
        fixture = json.loads((args.output/'fixture.json').read_text())
        for name in ('A', 'B'):
            receipt = model.restore_checkpoint(root,
                expected_schedule_sha256=prior['root'].get('schedule_sha256') if
                model.profile['execution'].get('resume_contract') == 'identical-native-batches-v1' else None)
            if receipt['counter_deltas'].get('loaded_file_bytes',0) <= 0:
                raise AssertionError('Missing native disk-byte witness')
            if receipt['counter_deltas'].get('cold_restore_copy_bytes',0) <= 0:
                raise AssertionError('Missing completed native restore-copy witness')
            result.setdefault('restore_receipts', []).append(receipt)
            actual = capture(model, root, args.output, 'restored-'+name+'-root')
            assert actual['frontier_sha256'] == prior['root']['frontier_sha256']
            result['branches'][name] = continuation(model, root, fixture[name], args, 'resumed-'+name)
            publish(args.output/'resume.json', result)
        assert sha(model.checkpoint_directory/(root+'.json')) == fixture['manifest_sha256']
        result['counters'] = model.backend.work_counters()
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        raise
    finally:
        model.backend.close()
        result['final_counters'] = dict(model.backend.stats)
        result['native_exit_code'] = model.backend.process.returncode
        result['native_exit_observed'] = model.backend.process.poll() is not None
        publish(args.output/'resume.json', result)


def boundary(args):
    model, startup = open_model(args)
    result = dict(status='RUNNING', python_pid=os.getpid(), native_pid=model.backend.process.pid,
                  startup_seconds=startup)
    try:
        previous = json.loads((args.output/'resume.json').read_text())
        assert previous['native_exit_observed'] and previous['native_pid'] != model.backend.process.pid
        mid = json.loads((args.output/'resumed-A-mid.json').read_text())
        receipt = model.restore_checkpoint(mid['snapshot'])
        result['receipt'] = receipt
        generated = model.generate(mid['snapshot'], args.generated-len(mid['tokens']))
        expected = previous['branches']['A']['generated'][len(mid['tokens']):]
        assert list(generated.accepted_tokens) == expected
        expected_final = previous['branches']['A']['states'][-1]
        result['final'] = capture(model, generated.snapshot_id, args.output, 'boundary-final')
        assert result['final']['frontier_sha256'] == expected_final['frontier_sha256']
        assert result['final']['controller'] == expected_final['controller']
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        raise
    finally:
        model.backend.close()
        result['final_counters'] = dict(model.backend.stats)
        result['native_exit_observed'] = model.backend.process.poll() is not None
        publish(args.output/'boundary.json', result)


def compare(args):
    profile = json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    oracle = BatchedOracle(profile, args.output)
    result = dict(status='RUNNING', comparisons=[], oracle='fresh independent process; no branch/cache imports')
    try:
        prepared = json.loads((args.output/'prepare.json').read_text())
        resumed = json.loads((args.output/'resume.json').read_text())
        if profile['execution'].get('resume_contract') == 'identical-native-batches-v1':
            for name in ('A','B'):
                a,b = prepared['branches'][name],resumed['branches'][name]
                assert a['generated'] == b['generated']
                assert [s['controller'] for s in a['states']] == [s['controller'] for s in b['states']]
                assert [s['frontier_sha256'] for s in a['states']] == [s['frontier_sha256'] for s in b['states']]
                assert [s['schedule_sha256'] for s in a['states']] == [s['schedule_sha256'] for s in b['states']]
                suffix_state = b['states'][0]
                requests = [event['count'] for event in suffix_state['execution_schedule']]
                outputs,frontiers = oracle.continuation(suffix_state['history'],requests,args.generated)
                assert outputs == b['generated']
                reference = {args.prefix:frontiers[len(requests)-2],
                    len(suffix_state['history']):frontiers[len(requests)-1]}
                reference.update({len(suffix_state['history'])+i+1:frontiers[len(requests)+i]
                                  for i in range(len(outputs))})
                for state in [prepared['root']]+b['states']:
                    from context_time_travel.execution_receipt import schedule_digest
                    expected_events = [event for event in oracle.last['execution_schedule']
                                       if event['position']+event['count'] <= state['consumed']]
                    if schedule_digest(expected_events) != state['schedule_sha256']:
                        raise IntegrityError('Reference native chunk schedule differs from tested continuation')
                    report = compare_logits(vector(args.output/state['frontier']),reference[state['consumed']],atol=0,rtol=0)
                    result['comparisons'].append(dict(branch=name,frontier=state['frontier'],consumed=state['consumed'],
                        numerics=numerical_record(report),oracle_work=oracle.last))
                    publish(args.output/'compare.json',result)
                    if not report.pass_elementwise:
                        raise AssertionError('Matching-schedule reference disagreement: '+state['frontier'])
            result.update(status='PASS',contract='identical-native-batches-v1')
            return
        states = [prepared['root']]
        for name in ('A','B'):
            a,b = prepared['branches'][name],resumed['branches'][name]
            assert a['generated'] == b['generated']
            assert [s['controller'] for s in a['states']] == [s['controller'] for s in b['states']]
            assert [s['frontier_sha256'] for s in a['states']] == [s['frontier_sha256'] for s in b['states']]
            states.extend(b['states'])
        for state in states:
            expected = oracle.forward(state['history'])
            actual = vector(args.output/state['frontier'])
            report = compare_logits(actual, expected, atol=profile['atol'], rtol=profile['rtol'])
            result['comparisons'].append(dict(frontier=state['frontier'], consumed=state['consumed'],
                numerics=numerical_record(report), oracle_work=oracle.last))
            publish(args.output/'compare.json', result)
            if not report.pass_elementwise:
                raise AssertionError('Fresh oracle disagreement: '+state['frontier']+' max='+str(report.max_absolute_error))
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        raise
    finally:
        oracle.close()
        publish(args.output/'compare.json', result)


def calibrate(args):
    profile = json.loads(args.profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    oracle = BatchedOracle(profile,args.output,'calibration')
    fixture = json.loads((args.output/'fixture.json').read_text())
    result = dict(status='RUNNING', baseline='independent fresh/fresh repeats only', comparisons=[])
    try:
        if profile['execution'].get('resume_contract') == 'identical-native-batches-v1':
            for label in ('A','B'):
                ids = fixture['prefix']+fixture[label]
                requests = [args.prefix,args.suffix]
                ta,a = oracle.continuation(ids,requests,args.generated); first = oracle.last.copy()
                tb,b = oracle.continuation(ids,requests,args.generated); second = oracle.last.copy()
                maximum = max(abs(x-y) for va,vb in zip(a,b) for x,y in zip(va,vb))
                if ta != tb or len(a) != len(b) or maximum != 0:
                    raise AssertionError('Fresh matching-schedule repeats are not exact')
                result['comparisons'].append(dict(label=label,max_absolute_error=maximum,first=first,second=second))
                publish(args.output/'calibration.json',result)
            result.update(status='PASS',atol=0,rtol=0,scope='exact matching-schedule repeats only')
            profile.update(atol=0,rtol=0,calibration_status='EXACT_MATCHING_SCHEDULE_REPEATS',
                           calibration_sha256=hashlib.sha256(_json_bytes(result)).hexdigest())
            publish(args.profile,profile)
            return
        for label,ids in [('P',fixture['prefix']),('PA',fixture['prefix']+fixture['A']),
                          ('PB',fixture['prefix']+fixture['B'])]:
            a = oracle.forward(ids); first = oracle.last.copy()
            b = oracle.forward(ids); second = oracle.last.copy()
            maximum = max(abs(x-y) for x,y in zip(a,b))
            result['comparisons'].append(dict(label=label,max_absolute_error=maximum,first=first,second=second))
            publish(args.output/'calibration.json',result)
        bound = 2*max(row['max_absolute_error'] for row in result['comparisons'])
        result.update(status='PASS',atol=bound,rtol=0,scope='bounded repeats; not full G1 qualification')
        profile.update(atol=bound,rtol=0,calibration_status='BOUNDED_FRESH_REPEAT_BASELINE',
                       calibration_sha256=hashlib.sha256(_json_bytes(result)).hexdigest())
        publish(args.profile,profile)
    finally:
        oracle.close()
        publish(args.output/'calibration.json',result)


def run_child(args, phase):
    command = [sys.executable, str(Path(__file__).resolve()), phase, '--profile',str(args.profile),
        '--output',str(args.output), '--prefix',str(args.prefix),'--suffix',str(args.suffix),
        '--generated',str(args.generated)]
    started = time.monotonic()
    with (args.output/(phase+'.stdout')).open('w') as out, (args.output/(phase+'.stderr')).open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                state_path = args.output/(phase+'.json')
                progress = json.loads(state_path.read_text()) if state_path.exists() else {}
                print(json.dumps(dict(phase=phase,pid=child.pid,native_pid=progress.get('native_pid'),
                    elapsed=time.monotonic()-started,last_completed=list(progress.get('branches',{})))), flush=True)
                if time.monotonic()-started > args.timeout:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise TimeoutError(phase+' timed out; worker process group terminated')
    if code:
        raise RuntimeError(phase+' failed; see '+str(args.output/(phase+'.stderr')))
    return dict(phase=phase,python_pid=child.pid,exit_code=code,seconds=time.monotonic()-started)


def campaign(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = dict(prefix=[100+i%17 for i in range(args.prefix)],
                   A=[117+i%13 for i in range(args.suffix)],
                   B=[150+i%11 for i in range(args.suffix)])
    fixture['tokens_sha256'] = hashlib.sha256(_json_bytes(fixture)).hexdigest()
    publish(args.output/'fixture.json', fixture)
    result = dict(status='RUNNING', gate_status='NOT_QUALIFIED', phases=[],
        profile_sha256=sha(args.profile), source_sha256=sha(Path(__file__)),
        prefix=args.prefix,suffix=args.suffix,output_bound=args.generated,
        filesystem_cache='uncontrolled warm OS cache; no cold-disk claim')
    try:
        # Freeze a local campaign copy, never edit the supplied profile in place.
        profile_copy = args.output/'profile.json'
        shutil.copyfile(args.profile,profile_copy)
        args.profile = profile_copy
        pinned = json.loads(args.profile.read_text())
        if pinned.get('calibration_status') == 'REUSED_IDENTICAL_NATIVE_FRESH_BASELINE':
            result['baseline_reuse'] = dict(receipt=pinned['baseline_calibration'],
                prior_profile_sha256=pinned['baseline_profile_sha256'],scope=pinned['baseline_scope'])
        else:
            result['phases'].append(run_child(args,'calibrate'))
        result['profile_sha256'] = sha(args.profile)
        result['phases'].append(run_child(args, 'prepare'))
        prepared = json.loads((args.output/'prepare.json').read_text())
        sid = prepared['root']['snapshot']
        fixture['manifest_sha256'] = sha(args.output/'store/checkpoints'/(sid+'.json'))
        publish(args.output/'fixture.json', fixture)
        for phase in ('resume','boundary','compare'):
            result['phases'].append(run_child(args, phase))
            publish(args.output/'result.json', result)
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL', error=str(error))
        raise
    finally:
        publish(args.output/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','resume','boundary','compare','calibrate','campaign'])
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=int, default=128)
    parser.add_argument('--suffix', type=int, default=4)
    parser.add_argument('--generated', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if min(args.prefix,args.suffix,args.generated) <= 0:
        parser.error('Lengths must be positive')
    globals()[args.phase](args)
