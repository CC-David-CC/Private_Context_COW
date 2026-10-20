"""Private CLI client timing of consumed-progress events and durable boundaries.

Uses an already qualified resident_resume case. This is a local subprocess
client, not an HTTP serving benchmark. No per-token numerical export is enabled.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from durable_resume import open_model, sha, publish
from native_request_probe import frontier, _record_native_shutdown
from resident_resume import checked_restore, save


def _finalize_native_worker(record, process, close):
    """Close the worker and record its numeric exit status in the result."""
    return _record_native_shutdown(record, {"worker": process}, close)


def _native_worker_succeeded(record):
    codes = record.get("native_exit_codes")
    return (record.get("native_exit_success") is True
            and isinstance(codes, dict) and bool(codes)
            and all(type(code) is int and code == 0 for code in codes.values()))


class RequestTimeline:
    def __init__(self):
        self.start = self.snapshot = self.durable = self.last = None
        self.tokens, self.arrivals = [], []

    def accept(self, event, now):
        if self.last is not None and now < self.last:
            raise ValueError('Client clock moved backwards')
        kind = event['event']
        if kind == 'request_started':
            if self.start is not None: raise ValueError('Repeated request start')
            self.start = now
        elif kind == 'token':
            if (self.start is None or self.snapshot is not None
                    or event['index'] != len(self.tokens) or type(event['token']) is not int):
                raise ValueError('Invalid consumed-progress sequence')
            self.tokens.append(event['token']); self.arrivals.append(now)
        elif kind == 'snapshot':
            if self.start is None or self.snapshot is not None or event['tokens'] != self.tokens:
                raise ValueError('Snapshot does not commit the observed token sequence')
            self.snapshot = now
        elif kind == 'durable':
            if self.snapshot is None or self.durable is not None:
                raise ValueError('Premature or repeated durable acknowledgment')
            self.durable = now
        else:
            raise ValueError('Unknown timing event')
        self.last = now

    def summary(self):
        if self.snapshot is None: raise ValueError('Request has not completed')
        n = len(self.tokens)
        span = self.arrivals[-1]-self.arrivals[0] if n > 1 else 0.
        gaps = [b-a for a,b in zip(self.arrivals,self.arrivals[1:])]
        elapsed = self.snapshot-self.start
        durable_elapsed = self.durable-self.start if self.durable is not None else None
        return dict(outputs=n,time_to_first_token_seconds=(self.arrivals[0]-self.start if n else None),
            steady_received_tps=(n-1)/span if span > 0 else None,
            token_gap_count=len(gaps),mean_tpot_seconds=statistics.mean(gaps) if gaps else None,
            median_tpot_seconds=statistics.median(gaps) if gaps else None,
            p95_tpot_seconds=sorted(gaps)[math.ceil(.95*len(gaps))-1] if gaps else None,
            max_tpot_seconds=max(gaps) if gaps else None,
            request_seconds=elapsed,request_tps=n/elapsed if elapsed > 0 else None,
            durable=self.durable is not None,durable_request_seconds=durable_elapsed,
            durable_request_tps=n/durable_elapsed if durable_elapsed and durable_elapsed > 0 else None)


def emit(kind, **fields):
    print(json.dumps(dict(event=kind,**fields),allow_nan=False),flush=True)


def counter_delta(before, after):
    result = {}
    for key,value in after.items():
        old = before[key]
        if type(value) in (int,float) and type(old) in (int,float): result[key] = value-old
        elif type(old) != type(value): raise ValueError('Native counter type changed: '+key)
    return result


def clone_store(case, output, prepared):
    """Copy canonical files; share only the immutable audited prefix payload."""
    shutil.copytree(case/'store',output/'store',ignore=shutil.ignore_patterns('checkpoints'))
    source = case/'store/checkpoints'
    destination = output/'store/checkpoints'
    destination.mkdir(mode=0o700)
    sid = prepared['root']['snapshot']
    record = json.loads((source/(sid+'.json')).read_text())
    payload = record['manifest']['payload_sha256']+'.state'
    if sha(source/payload) != prepared['parent_payload_sha256']:
        raise ValueError('Qualified prefix payload changed')
    shutil.copy2(source/(sid+'.json'),destination/(sid+'.json'))
    os.link(source/payload,destination/payload)
    return source/payload


def worker(args):
    result = dict(status='RUNNING',scope='CLI consumed-progress; greedy matching native operators',requests=[])
    model = None
    try:
        case_result = json.loads((args.case/'result.json').read_text())
        prepared = json.loads((args.case/'prepare.json').read_text())
        fixture = json.loads((args.case/'fixture.json').read_text())
        profile = json.loads(args.profile.read_text())
        if (case_result['status'] != 'PASS' or case_result['profile_id'] != profile['execution_profile_id']
                or prepared['status'] != 'PASS' or not prepared['native_exit_observed']):
            raise ValueError('Require an executed exact matching-profile case')
        payload = clone_store(args.case,args.output,prepared)
        model,startup = open_model(SimpleNamespace(profile=args.profile,output=args.output,phase='client-speed'))
        emit('ready',startup_seconds=startup,native_pid=model.backend.process.pid)
        root = prepared['root']['snapshot']
        initial = checked_restore(model,root,prepared['root']['schedule_sha256'])
        emit('loaded',receipt=initial)
        for repeat in range(args.repetitions+int(args.warmup)):
            warmup = bool(args.warmup and repeat == 0)
            for name in ('A','B'):
                label = name+'-'+str(repeat)
                expected = prepared['branches'][name]
                observed = []
                emit('request_started',request=label,warmup=warmup,mode=args.mode)
                disk = checked_restore(model,root,prepared['root']['schedule_sha256']) if args.mode == 'disk' else None
                if root not in model.cache:
                    raise AssertionError('Prefix not retained; cannot claim resident execution')
                before = model.backend.work_counters()
                def progress(token):
                    observed.append(token)
                    emit('token',request=label,index=len(observed)-1,token=token)
                answer = model.run_greedy_request(root,fixture[name],case_result['fixture_sizes']['output_limit'],
                                                  stream='resident-'+name,on_token=progress)
                after = model.backend.work_counters()
                emit('snapshot',request=label,**answer)
                began = time.monotonic()
                state = expected['state']
                sid = answer['snapshot']
                if (answer['tokens'] != expected['result']['tokens'] or observed != answer['tokens']
                        or model.controller(sid) != state['controller']
                        or model.history(sid) != tuple(state['history'])
                        or hashlib.sha256(frontier(model.next_logits(sid))).hexdigest() != state['frontier_sha256']):
                    raise AssertionError('Client request differs from the qualified reference')
                delta = counter_delta(before,after)
                if delta['forward_tokens'] != len(fixture[name])+len(observed):
                    raise AssertionError('Unexpected prefix replay or incomplete consumed work')
                if delta['loaded_file_bytes'] != 0:
                    raise AssertionError('Hidden disk restore during a resident request')
                validation = time.monotonic()-began
                (args.output/(label+'.txt')).write_text(
                    b''.join(model.backend.piece(t) for t in observed).decode('utf-8',errors='replace'),
                    encoding='utf-8')
                branch = 'client-'+label
                model.store.fork(branch,sid)
                began = time.monotonic()
                receipt = save(model,branch,sid)
                durable_seconds = time.monotonic()-began
                emit('durable',request=label,receipt=receipt)
                result['requests'].append(dict(request=label,warmup=warmup,mode=args.mode,
                    outputs=len(observed),native_generation_seconds=answer['generation_seconds'],
                    native_request_seconds=answer['request_seconds'],disk_restore=disk,
                    counter_deltas=delta,validation_seconds=validation,durable_save_seconds=durable_seconds))
                # Retain only the useful prefix image. Completed answers are
                # durable; keeping them resident would evict the measured root.
                if sid != root: model.cache.pop(sid,None)
                publish(args.output/'worker.json',result)
        if sha(payload) != prepared['parent_payload_sha256']:
            raise AssertionError('Original prefix payload mutated')
        result.update(status='PASS',profile_id=profile['execution_profile_id'],
                      startup_seconds=startup,initial_restore=initial,harness_sha256=sha(__file__))
    except BaseException as error:
        result.update(status='FAIL',error=str(error))
        raise
    finally:
        if model is not None:
            process = model.backend.process
            _finalize_native_worker(result, process, model.backend.close)
            if not _native_worker_succeeded(result):
                result['status'] = 'FAIL'
        publish(args.output/'worker.json',result)


def client(args):
    args.output.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,str(Path(__file__).resolve()),'worker','--case',str(args.case),
        '--profile',str(args.profile),'--output',str(args.output),'--mode',args.mode,
        '--repetitions',str(args.repetitions)]+(['--warmup'] if args.warmup else [])
    timelines,metadata = {},{}
    record = dict(status='RUNNING',scope='same-host subprocess client; not HTTP or remote network speed',
        tpot_scope='received gaps after first token; p95 uses nearest rank; startup excluded from request TTFT',
        requests={})
    with (args.output/'worker.stderr').open('wb') as err,(args.output/'events.jsonl').open('w') as log:
        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
        try:
            for line in child.stdout:
                arrived = time.monotonic()
                event = json.loads(line)
                log.write(json.dumps(dict(received_monotonic=arrived,event=event))+'\n')
                label = event.get('request')
                if label is not None:
                    if event['event'] == 'request_started':
                        if label in timelines: raise ValueError('Duplicate request identity')
                        timelines[label] = RequestTimeline();metadata[label] = event
                    timelines[label].accept(event,arrived)
                    if event['event'] == 'durable':
                        row = dict(timelines[label].summary(),warmup=metadata[label]['warmup'],mode=args.mode)
                        record['requests'][label] = row
                        publish(args.output/'client.json',record)
                        print(json.dumps(dict(request=label,**row)),flush=True)
            code = child.wait(timeout=30)
            worker_result = json.loads((args.output/'worker.json').read_text())
            if (code or worker_result['status'] != 'PASS' or not _native_worker_succeeded(worker_result)
                    or len(record['requests']) != 2*(args.repetitions+int(args.warmup))):
                raise AssertionError('Client campaign incomplete or native worker failed')
            record.update(status='PASS',worker_sha256=sha(args.output/'worker.json'))
        except BaseException as error:
            record.update(status='FAIL',error=str(error))
            if child.poll() is None:
                child.terminate(); child.wait(timeout=30)
            raise
        finally:
            publish(args.output/'client.json',record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['client','worker'])
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['resident','disk'],required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--warmup',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 3: parser.error('Require one to three measured repetitions')
    (client if args.phase == 'client' else worker)(args)


if __name__ == '__main__': main()
