"""Run the real COW CLI while privately recording streamed NDJSON arrival times.

This observer is a harness. It patches only ``tools.cow_cli.urlopen`` in-process,
wraps responses advertising ``application/x-ndjson``, and yields each response
line unchanged to the CLI. It does not modify the client, request, or backend.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit


class ObservationError(RuntimeError):
    pass


def make_request(url, body, accept):
    """Small helper shared by tests; the production CLI builds urllib Requests itself."""
    from urllib.request import Request
    raw=json.dumps(body,ensure_ascii=False).encode('utf-8')
    return Request(url,data=raw,headers={'Accept':accept,'Content-Type':'application/json'},method='POST')


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _request_binding(request):
    url=request.full_url if hasattr(request,'full_url') else str(request)
    parsed=urlsplit(url)
    body=getattr(request,'data',None)
    if body is None:
        body=getattr(request,'data',b'')
    try:
        decoded=json.loads(body.decode('utf-8')) if isinstance(body,bytes) else {}
    except (UnicodeError,ValueError):
        decoded={}
    return {'method':request.get_method() if hasattr(request,'get_method') else None,
            'request_path':parsed.path,
            'operation_id':decoded.get('operation_id') if isinstance(decoded,dict) else None,
            'branch':decoded.get('branch') if isinstance(decoded,dict) else None,
            'revision':decoded.get('revision') if isinstance(decoded,dict) else None}


class _ObservedResponse:
    def __init__(self,response,row):
        self._response=response
        self._row=row

    def __enter__(self):
        enter=getattr(self._response,'__enter__',None)
        if enter is not None:
            enter()
        return self

    def __exit__(self,exc_type,exc,tb):
        exit_method=getattr(self._response,'__exit__',None)
        if exit_method is not None:
            return exit_method(exc_type,exc,tb)
        self.close()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            raw=next(self._response)
        except StopIteration:
            self._row['eof_monotonic_ns']=time.monotonic_ns()
            raise
        except BaseException as exc:
            self._row['read_error']={'type':type(exc).__name__,'message':str(exc)}
            self._row['read_error_monotonic_ns']=time.monotonic_ns()
            raise
        arrival=time.monotonic_ns()
        if not isinstance(raw,bytes):
            self._row['observer_error']='urllib response yielded a non-bytes line'
            raise TypeError('NDJSON response yielded a non-bytes line')
        self._row['events'].append({'arrival_monotonic_ns':arrival,
                                    'raw_line_b64':base64.b64encode(raw).decode('ascii')})
        return raw

    def close(self):
        return self._response.close()

    def __getattr__(self,name):
        return getattr(self._response,name)


class _Observer:
    def __init__(self,original_urlopen):
        self.original_urlopen=original_urlopen
        self.requests=[]

    def urlopen(self,request,*args,**kwargs):
        headers=getattr(request,'headers',{}) or {}
        accept=(request.get_header('Accept') if hasattr(request,'get_header') else None)
        if accept is None:
            accept=next((v for k,v in headers.items() if k.lower()=='accept'), '')
        if 'application/x-ndjson' not in str(accept).lower():
            return self.original_urlopen(request,*args,**kwargs)
        row=_request_binding(request)
        row.update(request_started_monotonic_ns=time.monotonic_ns(),events=[])
        self.requests.append(row)
        try:
            response=self.original_urlopen(request,*args,**kwargs)
        except BaseException as exc:
            row['open_error']={'type':type(exc).__name__,'message':str(exc)}
            row['open_error_monotonic_ns']=time.monotonic_ns()
            raise
        row['headers_monotonic_ns']=time.monotonic_ns()
        row['http_status']=getattr(response,'status',None)
        response_headers=getattr(response,'headers',None)
        row['content_type']=(response_headers.get('Content-Type')
                             if response_headers is not None else None)
        return _ObservedResponse(response,row)


def _decode_events(request):
    events=[]
    errors=[]
    previous=None
    for index,row in enumerate(request.get('events',[])):
        stamp=row.get('arrival_monotonic_ns')
        if type(stamp) is not int or stamp<0:
            errors.append(f'event {index} has invalid arrival timestamp')
        elif previous is not None and stamp<previous:
            errors.append(f'event {index} arrival timestamp moved backwards')
        previous=stamp if type(stamp) is int else previous
        try:
            raw=base64.b64decode(row['raw_line_b64'],validate=True)
            event=json.loads(raw.decode('utf-8'))
        except (KeyError,ValueError,UnicodeError,TypeError) as exc:
            errors.append(f'event {index} is malformed NDJSON: {type(exc).__name__}')
            continue
        if not isinstance(event,dict) or not isinstance(event.get('event'),str):
            errors.append(f'event {index} has no event type')
            continue
        events.append({'sequence':index,'arrival_monotonic_ns':stamp,'event':event})
    return events,errors


def _duration(end,start):
    if type(end) is not int or type(start) is not int or end<start:
        return None
    return end-start


def summarize_request(request):
    """Validate one captured stream and derive event-based timing metrics."""
    decoded,errors=_decode_events(request)
    process_start=request.get('process_started_monotonic_ns')
    request_start=request.get('request_started_monotonic_ns')
    headers=request.get('headers_monotonic_ns')
    eof=request.get('eof_monotonic_ns')
    process_end=request.get('process_finished_monotonic_ns')
    stamps=(process_start,request_start,headers)
    if any(type(stamp) is not int or stamp<0 for stamp in stamps):
        errors.append('process start, request start, and response headers require nonnegative integer timestamps')
    elif not process_start<=request_start<=headers:
        errors.append('process start, request start, and response headers are out of order')
    if type(process_end) is not int or process_end<0:
        errors.append('process end requires a nonnegative integer timestamp')
    if request.get('open_error'):
        errors.append('request open failed before a stream was established')
        open_at=request.get('open_error_monotonic_ns')
        if (type(open_at) is not int or type(process_start) is not int
                or type(request_start) is not int or type(process_end) is not int
                or not process_start<=request_start<=open_at<=process_end):
            errors.append('open error timestamp is missing or out of process/request order')
    elif request.get('read_error'):
        errors.append('stream ended with a transport read error')
        error_at=request.get('read_error_monotonic_ns')
        if type(error_at) is not int or type(headers) is not int or type(process_end) is not int or not headers<=error_at<=process_end:
            errors.append('read error timestamp is missing or out of response/process order')
        if eof is not None:
            errors.append('read-error stream cannot also claim clean EOF')
    if request.get('observer_error'):
        errors.append('observer failed to capture the stream faithfully')
    elif type(eof) is not int:
        errors.append('stream has no clean EOF timestamp')
    else:
        if type(headers) is int and eof<headers:
            errors.append('clean EOF preceded response headers')
        if type(process_end) is int and eof>process_end:
            errors.append('clean EOF followed process end')
    for index,row in enumerate(decoded):
        stamp=row['arrival_monotonic_ns']
        if type(stamp) is not int:
            continue
        if type(headers) is int and stamp<headers:
            errors.append(f'event {index} preceded response headers')
        if type(eof) is int and stamp>eof:
            errors.append(f'event {index} arrived after EOF')
        if request.get('read_error') and type(request.get('read_error_monotonic_ns')) is int \
                and stamp>request['read_error_monotonic_ns']:
            errors.append(f'event {index} arrived after stream read error')
        if type(process_end) is int and stamp>process_end:
            errors.append(f'event {index} arrived after process end')
    types=[row['event']['event'] for row in decoded]
    token_rows=[]
    text_empty=0
    durable_rows=[]
    complete_rows=[]
    failure_rows=[]
    snapshots=[]
    terminal_receipts=[]
    completed=False
    for index,row in enumerate(decoded):
        event=row['event']; kind=event['event']
        if completed:
            errors.append(f'event {index} arrived after complete')
        if kind in ('token','text','turn_complete') and 'token_id' in event:
            token=event.get('token_id')
            if type(token) is not int or token<0:
                errors.append(f'token event {index} has invalid nonnegative token_id')
            else:
                token_rows.append(row)
        if kind=='text':
            if not isinstance(event.get('text'),str):
                errors.append(f'text event {index} has no text string')
            elif event['text']=='' and type(event.get('token_id')) is int and event['token_id']>=0:
                text_empty+=1
        elif kind=='token' and 'text' in event:
            if not isinstance(event['text'],str):
                errors.append(f'token event {index} has invalid text')
            elif event['text']=='' and type(event.get('token_id')) is int and event['token_id']>=0:
                text_empty+=1
        elif kind=='durable':
            durable_rows.append(row)
            receipt=event.get('receipt')
            if isinstance(receipt,dict):
                snapshots.append({'sequence':row['sequence'],'kind':'durable',
                                  'snapshot':receipt.get('snapshot')})
                terminal_receipts.append({'kind':'durable','sequence':row['sequence'],
                    'receipt_sha256':hashlib.sha256(json.dumps(receipt,sort_keys=True,
                        separators=(',',':')).encode('utf-8')).hexdigest()})
        elif kind=='complete':
            complete_rows.append(row); completed=True
            receipt=event.get('receipt'); answer=event.get('answer')
            if isinstance(answer,dict):
                snapshots.append({'sequence':row['sequence'],'kind':'complete',
                                  'snapshot':answer.get('snapshot')})
            if isinstance(receipt,dict):
                terminal_receipts.append({'kind':'complete','sequence':row['sequence'],
                    'receipt_sha256':hashlib.sha256(json.dumps(receipt,sort_keys=True,
                        separators=(',',':')).encode('utf-8')).hexdigest()})
        elif kind=='snapshot':
            snapshots.append({'sequence':row['sequence'],'kind':'snapshot',
                              'snapshot':event.get('snapshot')})
        elif kind in ('error','failure'):
            failure_rows.append(row)

    if len(durable_rows)>1:
        errors.append('stream contains duplicate durable events')
    if len(complete_rows)>1:
        errors.append('stream contains duplicate complete events')
    if failure_rows:
        errors.append('stream contains an error/failure event')
    if len(durable_rows)==1 and len(complete_rows)==1:
        durable=durable_rows[0]; complete=complete_rows[0]
        if (durable['sequence']>=complete['sequence']
                or durable['arrival_monotonic_ns']>complete['arrival_monotonic_ns']):
            errors.append('durable event did not precede complete')
        d_receipt=durable['event'].get('receipt')
        c_receipt=complete['event'].get('receipt')
        if not isinstance(d_receipt,dict) or not isinstance(c_receipt,dict) or d_receipt!=c_receipt:
            errors.append('durable and complete receipts are missing or differ')
        answer=complete['event'].get('answer')
        if not isinstance(answer,dict) or not isinstance(answer.get('tokens'),list):
            errors.append('complete event lacks answer token list')
        else:
            answer_tokens=answer['tokens']
            if any(type(token) is not int or token<0 for token in answer_tokens):
                errors.append('complete answer contains invalid token IDs')
            elif answer_tokens!=[row['event']['token_id'] for row in token_rows]:
                errors.append('streamed token IDs differ from completed answer')
        if (isinstance(answer,dict) and isinstance(c_receipt,dict)
                and answer.get('snapshot')!=c_receipt.get('snapshot')):
            errors.append('complete answer snapshot differs from durable receipt')
        request_branch=request.get('branch')
        receipt_branch=c_receipt.get('branch') if isinstance(c_receipt,dict) else None
        if not isinstance(receipt_branch,dict):
            errors.append('completion receipt lacks structured branch identity')
        else:
            name=receipt_branch.get('name'); tip=receipt_branch.get('tip')
            root=receipt_branch.get('root'); revision=receipt_branch.get('revision')
            if not isinstance(name,str) or not name or not isinstance(tip,str) or not tip or not isinstance(root,str) or not root:
                errors.append('completion receipt has invalid branch name, tip, or root')
            if type(revision) is not int or revision<0:
                errors.append('completion receipt has invalid branch revision')
            if request_branch is not None and name!=request_branch:
                errors.append('completion receipt branch differs from request binding')
            request_revision=request.get('revision')
            if request_revision is not None:
                if type(request_revision) is not int or request_revision<0:
                    errors.append('request binding has invalid branch revision')
                elif type(revision) is int and revision!=request_revision+1:
                    errors.append('completion receipt revision is not request revision plus one')
            if isinstance(answer,dict) and isinstance(tip,str) and answer.get('snapshot')!=tip:
                errors.append('completion answer snapshot differs from receipt branch tip')
            if isinstance(c_receipt,dict) and c_receipt.get('snapshot')!=tip:
                errors.append('completion receipt snapshot differs from branch tip')
        request_operation=request.get('operation_id')
        if (request_operation is not None and isinstance(c_receipt,dict)
                and 'operation_id' in c_receipt and c_receipt['operation_id']!=request_operation):
            errors.append('completion receipt operation differs from request binding')
    elif not failure_rows:
        errors.append('stream lacks exactly one durable event and one complete event')

    snapshot_rows=[row for row in decoded if row['event']['event']=='snapshot']
    if len(snapshot_rows)>1:
        errors.append('stream contains duplicate snapshot events')
    if snapshot_rows and len(complete_rows)==1 and len(durable_rows)==1:
        snapshot_row=snapshot_rows[0]
        answer=snapshot_row['event']
        final_answer=complete_rows[0]['event'].get('answer')
        if not isinstance(answer,dict) or not isinstance(answer.get('tokens'),list):
            errors.append('snapshot event lacks answer token list')
        else:
            if any(type(token) is not int or token<0 for token in answer['tokens']):
                errors.append('snapshot answer contains invalid token IDs')
            if isinstance(final_answer,dict) and (answer.get('tokens')!=final_answer.get('tokens')
                    or answer.get('snapshot')!=final_answer.get('snapshot')):
                errors.append('snapshot answer differs from complete answer')
        token_sequences=[row['sequence'] for row in token_rows]
        durable_sequence=durable_rows[0]['sequence']
        if ((token_sequences and snapshot_row['sequence']<=max(token_sequences))
                or snapshot_row['sequence']>=durable_sequence):
            errors.append('snapshot event must follow tokens and precede durable')

    if errors:
        status='invalid'
        if failure_rows and not any('malformed' in e or 'moved backwards' in e for e in errors):
            status='error'
        elif request.get('open_error') or request.get('read_error'):
            status='transport_error'
        elif not complete_rows and not failure_rows and not any('malformed' in e or 'moved backwards' in e for e in errors):
            status='truncated'
    else:
        status='valid'

    token_times=[row['arrival_monotonic_ns'] for row in token_rows]
    intervals=[b-a for a,b in zip(token_times,token_times[1:])]
    start=request_start
    durable_time=durable_rows[0]['arrival_monotonic_ns'] if len(durable_rows)==1 else None
    metrics={
        'request_to_first_token':_duration(token_times[0],start) if token_times else None,
        'token_interval_mean':statistics.mean(intervals) if intervals else None,
        'token_interval_median':statistics.median(intervals) if intervals else None,
        'token_interval_p95':sorted(intervals)[math.ceil(.95*len(intervals))-1] if intervals else None,
        'token_interval_max':max(intervals) if intervals else None,
        'last_token_to_durable':_duration(durable_time,token_times[-1]) if token_times else None,
        'request_to_durable':_duration(durable_time,start),
        'process_elapsed':_duration(process_end,process_start),
    }
    generation_tps=None
    if status=='valid' and len(token_times)>=2 and token_times[-1]>token_times[0]:
        generation_tps=(len(token_times)-1)*1_000_000_000/(token_times[-1]-token_times[0])
    if request.get('open_error'):
        stream_end_status='open_error'
    elif request.get('read_error'):
        stream_end_status='read_error'
    elif type(eof) is int:
        stream_end_status='clean_eof'
    else:
        stream_end_status='missing_eof'
    return {'validation':{'status':status,'errors':errors},'event_types':types,
            'stream_end_status':stream_end_status,
            'token_count':len(token_rows),'token_ids':[row['event']['token_id'] for row in token_rows
                if type(row['event'].get('token_id')) is int],
            'empty_text_token_event_count':text_empty,'eos_event_count':sum(kind=='turn_complete' for kind in types),
            'snapshots':snapshots,'terminal_receipts':terminal_receipts,
            'metrics_ns':metrics,'generation_tokens_per_second':generation_tps}


def _write_trace(path,trace):
    path=Path(path).expanduser().absolute()
    parent=path.parent
    parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if parent.is_symlink():
        raise ObservationError('trace directory may not be a symlink')
    if os.name!='nt' and parent.stat().st_mode & 0o077:
        raise ObservationError('trace directory must be private (mode 0700 or stricter)')
    payload=(json.dumps(trace,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    fd=os.open(path,flags,0o600)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        if os.name!='nt': os.chmod(path,0o600)
    except BaseException:
        try: path.unlink()
        except OSError: pass
        raise


def run_cli_with_observer(cli_module,cli_args,trace_path,*,runtime_root,cli_path=None):
    """Invoke ``cli_module.main`` with only its module-local urlopen instrumented."""
    runtime_root=Path(runtime_root).expanduser().resolve(strict=True)
    cli_path=Path(cli_path or (runtime_root/'tools'/'cow_cli.py')).resolve(strict=True)
    try:
        cli_path.relative_to(runtime_root)
    except ValueError:
        raise ObservationError('CLI source must be inside the pinned runtime root') from None
    cli_sha=_sha256(cli_path)
    original=cli_module.urlopen
    observer=_Observer(original)
    process_start=time.monotonic_ns(); code=None; exception=None
    try:
        cli_module.urlopen=observer.urlopen
        code=cli_module.main(list(cli_args))
        if type(code) is not int:
            raise ObservationError('CLI main did not return an integer exit code')
        return code
    except BaseException as exc:
        exception={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        process_end=time.monotonic_ns()
        cli_module.urlopen=original
        try:
            requests=[]
            for request in observer.requests:
                request['process_started_monotonic_ns']=process_start
                request['process_finished_monotonic_ns']=process_end
                request['summary']=summarize_request(request)
                requests.append(request)
            trace={'schema':'cow-cli-event-observation-v1','runtime_root':str(runtime_root),
                'cli_source_path':str(cli_path),'cli_source_sha256':cli_sha,
                'process_started_monotonic_ns':process_start,'process_finished_monotonic_ns':process_end,
                'process_elapsed_ns':process_end-process_start,'cli_exit_code':code,
                'cli_exception':exception,'requests':requests}
            _write_trace(trace_path,trace)
        except BaseException as exc:
            print(f'cow-cli-observer: trace write failed: {exc}',file=sys.stderr,flush=True)


def load_cli(runtime_root):
    runtime_root=Path(runtime_root).expanduser().resolve(strict=True)
    candidate=runtime_root/'tools'/'cow_cli.py'
    if candidate.is_symlink():
        raise ObservationError('pinned runtime tools/cow_cli.py may not be a symlink')
    cli_path=candidate.resolve(strict=True)
    if not cli_path.is_file():
        raise ObservationError('pinned runtime tools/cow_cli.py must be a regular file')
    spec=importlib.util.spec_from_file_location('_cow_cli_observed',cli_path)
    if spec is None or spec.loader is None:
        raise ObservationError('could not load the pinned cow_cli.py entrypoint')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,cli_path


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    try:
        separator=args.index('--')
    except ValueError:
        raise SystemExit('usage: observe_cow_cli.py --runtime-root ROOT --trace-file PATH -- CLI_ARGS...')
    prefix=args[:separator]; cli_args=args[separator+1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',required=True)
    parser.add_argument('--trace-file',required=True)
    options=parser.parse_args(prefix)
    module,cli_path=load_cli(options.runtime_root)
    return run_cli_with_observer(module,cli_args,options.trace_file,
        runtime_root=options.runtime_root,cli_path=cli_path)


if __name__=='__main__':
    raise SystemExit(main())
