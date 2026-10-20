#!/usr/bin/env python3
"""CPU-only receipt validation timings; no inference or model speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from unittest.mock import patch
from context_time_travel import input_schedule_projection as projection
from context_time_travel.input_spans import InputHistory
from context_time_travel.execution_receipt import validate_input_schedule, tokens_digest


def reset_cache():
    if hasattr(projection,'_BindingCache'):
        projection._BINDINGS=projection._BindingCache(65536)


def fixture(prefix,generated):
    # Deterministic, nonconstant IDs; no model/tokenizer interpretation claimed.
    history=InputHistory([dict(kind='text',tokens=[(n*7919+17)%200000 for n in range(prefix+generated)])])
    ends=[prefix]+list(range(prefix+1,len(history)+1))
    projected=projection._InputScheduleProjection(history,ends)
    events=[];start=0;ids=history.native_tokens
    for end in ends:
        chunks=[[pos,min(512,end-pos)] for pos in range(start,end,512)]
        events.append(dict(operator='batched-prefill',position=start,count=end-start,chunks=chunks,
            tokens_sha256=tokens_digest(ids[start:end]),**projected.binding(end)))
        start=end
    return history,events


def run_case(prefix,generated,repeats):
    history,events=fixture(prefix,generated)
    reset_cache()
    before=time.perf_counter();expected=validate_input_schedule(events,history,512)
    cold=time.perf_counter()-before
    timings=[]
    for _ in range(repeats):
        start=time.perf_counter();actual=validate_input_schedule(events,history,512)
        timings.append(time.perf_counter()-start)
        if actual!=expected:raise AssertionError('schedule digest changed')
    # Count actual SHA update bytes separately, not inside timed samples.
    original=hashlib.sha256;counts=dict(native_calls=0,position_calls=0,native_bytes=0,position_bytes=0)
    class MeasuredHash:
        def __init__(self,inner,kind):self.inner=inner;self.kind=kind
        def update(self,data):
            counts[self.kind+'_bytes']+=len(data);self.inner.update(data)
        def hexdigest(self):return self.inner.hexdigest()
    def measured(data=b'',**kwargs):
        inner=original(data,**kwargs)
        kind={b'COW-VISION-INPUT-v1\0':'native',b'COW-MROPE-v1\0':'position'}.get(data)
        if kind is None:return inner
        counts[kind+'_calls']+=1;counts[kind+'_bytes']+=len(data)
        return MeasuredHash(inner,kind)
    with patch.object(projection.hashlib,'sha256',side_effect=measured):
        if validate_input_schedule(events,history,512)!=expected:raise AssertionError('instrumented changed result')
    # Separate memory sample; cache contains metadata, not histories/payloads.
    reset_cache();tracemalloc.start()
    validate_input_schedule(events,history,512)
    retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(prefix=prefix,generated=generated,receipts=len(events),schedule_sha256=expected,
        canonical_input_sha256=history.digest,cold_seconds=cold,warm_seconds=timings,
        warm_median_seconds=statistics.median(timings),warm_hash_work=counts,
        traced_retained_bytes=retained,traced_peak_bytes=peak,
        cache_entries=len(projection._BINDINGS) if hasattr(projection,'_BINDINGS') else 0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--generated',type=int,default=1024)
    ap.add_argument('--prefixes',type=int,nargs='+',default=[8192,65536]);args=ap.parse_args()
    result=dict(scope='CPU receipt validator, not GPU or whole-agent throughput',host=platform.node(),
        python=platform.python_version(),source_file=str(Path(projection.__file__).resolve()),
        source_sha256=hashlib.sha256(Path(projection.__file__).read_bytes()).hexdigest(),
        cases=[])
    for prefix in args.prefixes:
        result['cases'].append(run_case(prefix,args.generated,args.repeats))
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['cases'][-1]),flush=True)

if __name__=='__main__':main()
