"""Strict summaries of executed lifecycle samples and sanitizer diagnostics.

This module does not execute inference or establish instrumentation provenance.
Callers must retain actual counters, process samples and instrumented build/tool
receipts. Summaries alone never certify a gate.
"""
import math
import re

PAIRS=(('allocations','releases'),('handle_allocations','handle_releases'),
       ('snapshot_allocations','snapshot_releases'),('mappings','unmappings'),
       ('address_reservations','address_releases'),
       ('integrity_workspace_allocations','integrity_workspace_releases'))
LIVE=('live_bytes','handle_live_entries','handle_live_bytes','snapshot_live_images',
      'snapshot_payload_capacity_bytes','integrity_workspace_live_bytes')
REQUIRED=set(LIVE)|{name for pair in PAIRS for name in pair}|{'host_heap_allocated_bytes',
    'integrity_mismatches','vmm_driver_failures','vmm_out_of_memory_failures'}


def memory_histogram(values):
    values=list(values);edges=[1<<k for k in range(10,41)]
    bins=[dict(lower_bytes=0,upper_bytes=edges[0],count=sum(x<edges[0] for x in values))]
    bins.extend(dict(lower_bytes=low,upper_bytes=high,count=sum(low<=x<high for x in values))
                for low,high in zip(edges,edges[1:]))
    bins.append(dict(lower_bytes=edges[-1],upper_bytes=None,count=sum(x>=edges[-1] for x in values)))
    return bins


def summarize(rows, *, warmup_cycles):
    rows=list(rows)
    if not rows or type(warmup_cycles) is not int or not 0<=warmup_cycles<len(rows):
        raise ValueError('Need executed samples beyond the declared warmup')
    durations=[];heap_bound=None
    for index,row in enumerate(rows):
        for phase in ('before','after'):
            counters=row[phase]
            if REQUIRED-set(counters):raise ValueError('Missing native instrumentation')
            if any(type(counters[k]) is not int or counters[k]<0 for k in REQUIRED):
                raise ValueError('Invalid native counter')
        before,after=row['before'],row['after']
        if any(after[k]!=before[k] for k in LIVE):raise ValueError('Ownership did not return to baseline')
        for allocated,released in PAIRS:
            delta=after[allocated]-before[allocated]
            if delta<0 or delta!=after[released]-before[released]:
                raise ValueError('Unbalanced native resource operations')
        if any(after[k] for k in ('integrity_mismatches','vmm_driver_failures','vmm_out_of_memory_failures')):
            raise ValueError('Corruption or native allocation failure')
        duration=row['seconds']
        if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:
            raise ValueError('Invalid measured latency')
        for name in ('rss_bytes','gpu_bytes'):
            if type(row[name]) is not int or row[name]<0:raise ValueError('Missing or invalid process memory sample')
        if index==warmup_cycles and warmup_cycles:
            heap_bound=max(r['after']['host_heap_allocated_bytes'] for r in rows[:warmup_cycles])
        if heap_bound is not None and after['host_heap_allocated_bytes']>heap_bound:
            raise ValueError('Native heap exceeded frozen warm-cycle bound')
        durations.append(float(duration))
    values=sorted(durations)
    percentile=lambda p:values[max(0,math.ceil(p*len(values))-1)]
    edges=[2.0**k for k in range(-10,13)]
    bins=[dict(lower_seconds=0,upper_seconds=edges[0],count=sum(x<edges[0] for x in values))]
    bins.extend(dict(lower_seconds=low,upper_seconds=high,count=sum(low<=x<high for x in values))
                for low,high in zip(edges,edges[1:]))
    bins.append(dict(lower_seconds=edges[-1],upper_seconds=None,count=sum(x>=edges[-1] for x in values)))
    return dict(status='MEASURED',cycles=len(rows),warmup_cycles=warmup_cycles,
        frozen_host_heap_bound=heap_bound,latency_seconds=dict(minimum=values[0],maximum=values[-1],
        p50=percentile(.5),p95=percentile(.95),p99=percentile(.99)),
        latency_histogram=bins,
        latency_samples=durations,peak_rss_bytes=max(r['rss_bytes'] for r in rows),
        peak_gpu_bytes=max(r['gpu_bytes'] for r in rows),
        rss_histogram=memory_histogram(r['rss_bytes'] for r in rows),
        gpu_histogram=memory_histogram(r['gpu_bytes'] for r in rows))


def sanitizer_status(returncode, diagnostics, *, instrumented):
    if type(returncode) is not int:raise ValueError('Executed exit code required')
    errors=r'ERROR: (?:AddressSanitizer|LeakSanitizer)|runtime error:|WARNING: ThreadSanitizer|ERROR SUMMARY: [1-9][0-9]* errors'
    if returncode!=0 or re.search(errors,diagnostics):return 'FAIL'
    return 'PASS' if instrumented is True else 'BLOCKED'
