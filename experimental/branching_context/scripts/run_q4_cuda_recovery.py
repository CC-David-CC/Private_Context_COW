"""Strict same-service recovery probe after a real Q4 CUDA VMM allocation OOM.

This version is separate from run_q4_cuda_oom.py: it requires the first HTTP
service to replace its poisoned native worker and serve a same-service
continuation before testing an independently forked branch after clean restart.
It does not change engine/API code or claim numerical-oracle coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid

import run_q4_cuda_oom as base

OOMProbeError = base.OOMProbeError


def _need(condition, message):
    if not condition:
        raise OOMProbeError(message)


def validate_same_service_recovery(row, *, source_snapshot, source_payload_sha256,
                                   http_pid, old_native_pid):
    """Validate lifecycle/work evidence without inferring recovery from HTTP status alone."""
    row = base._object(row, 'same-service recovery evidence')
    replacement = base._object(row.get('replacement'), 'replacement worker')
    followup = base._object(row.get('followup'), 'same-service follow-up')
    source = base._object(row.get('source_after'), 'source after recovery')
    old_exit = base._object(replacement.get('old_native_exit'), 'old native exit')
    new_worker = base._object(replacement.get('new_native_process'), 'new native process')
    _need(row.get('http_pid') == http_pid, 'same-service recovery changed the HTTP PID')
    _need(replacement.get('old_native_pid') == old_native_pid
          and old_exit.get('pid') == old_native_pid
          and type(old_exit.get('exit_code')) is int
          and old_exit.get('pid_absent') is True,
          'old native worker was not observed exited before replacement')
    _need(type(new_worker.get('pid')) is int and new_worker['pid'] != old_native_pid
          and new_worker.get('parent_pid') == http_pid,
          'replacement native worker identity is not a distinct child of the same HTTP process')
    _need(replacement.get('factory_after_old_exit') is True,
          'replacement factory was not observed after old native exit')
    worker_status = base._object(replacement.get('worker_status'), 'replacement worker status')
    _need(replacement.get('restart_count') == 1 and worker_status.get('health') == 'healthy'
          and worker_status.get('restarts') == 1 and worker_status.get('pid') == new_worker.get('pid'),
          'same-service worker replacement is not healthy with exactly one restart')
    _need(followup.get('event') == 'complete', 'same-service continuation did not complete')
    answer = base._object(followup.get('answer'), 'same-service answer')
    tokens = answer.get('tokens')
    _need(type(tokens) is list and len(tokens) == 8 and all(type(t) is int for t in tokens),
          'same-service continuation must contain exactly eight generated token IDs')
    work = base._object(followup.get('counter_deltas'), 'same-service counter deltas')
    _need(work.get('forward_tokens') == 8 and work.get('consumed_tokens') == 8
          and work.get('prefill_tokens') == 0,
          'same-service continuation did not do exactly eight forward tokens and zero prefix work')
    durable = validate_durable_completion(followup.get('events'),
        branch_before=base._object(followup.get('branch_before'), 'fault branch before continuation'),
        branch_after=base._object(followup.get('branch_after'), 'fault branch after continuation'),
        branch_name=followup.get('branch'))
    _need(durable['output_tokens'] == tokens,
          'Same-service answer tokens differ from the durable completion event')
    _need(source.get('tip') == source_snapshot
          and source.get('payload_sha256') == source_payload_sha256,
          'canonical source snapshot changed during same-service recovery')
    return {'http_pid': http_pid, 'old_native_pid': old_native_pid,
            'new_native_pid': new_worker['pid'], 'restart_count': 1,
            'followup_tokens': list(tokens), 'forward_tokens': 8, 'prefill_tokens': 0}


def validate_clean_restart_branch(row, *, source_snapshot, source_payload_sha256):
    row = base._object(row, 'clean-restart branch evidence')
    before = base._object(row.get('branch_before'), 'restart branch before')
    after = base._object(row.get('source_after'), 'source after clean restart')
    followup = base._object(row.get('followup'), 'clean-restart follow-up')
    _need(before.get('tip') == source_snapshot,
          'independent restart branch was not still rooted at the original source snapshot')
    _need(before.get('branch') != row.get('fault_branch'),
          'restart continuation reused the advanced fault branch')
    _need(after.get('tip') == source_snapshot and after.get('payload_sha256') == source_payload_sha256,
          'canonical source changed during clean-restart continuation')
    _need(followup.get('event') == 'complete', 'clean-restart continuation did not complete')
    answer = base._object(followup.get('answer'), 'restart answer')
    tokens = answer.get('tokens')
    work = base._object(followup.get('counter_deltas'), 'restart counter deltas')
    _need(type(tokens) is list and len(tokens) == 8 and all(type(t) is int for t in tokens),
          'clean-restart continuation must contain exactly eight generated token IDs')
    _need(work.get('forward_tokens') == 8 and work.get('consumed_tokens') == 8
          and work.get('prefill_tokens') == 0,
          'clean-restart continuation did not do exactly eight forward tokens and zero prefix work')
    durable = validate_durable_completion(followup.get('events'),
        branch_before=base._object(followup.get('branch_before'), 'restart branch before continuation'),
        branch_after=base._object(followup.get('branch_after'), 'restart branch after continuation'),
        branch_name=followup.get('branch'))
    _need(durable['output_tokens'] == tokens,
          'Restart answer tokens differ from the durable completion event')
    return {'tokens': list(tokens), 'forward_tokens': 8, 'prefill_tokens': 0,
            'source_tip': source_snapshot}


def validate_worker_transition_trace(document, *, old_pid, new_pid, http_pid):
    """Require the coordinator's persisted replacement event, not PID polling alone."""
    doc = base._object(document, 'worker transition trace')
    events = doc.get('events')
    if (type(events) is not list or doc.get('overflowed') is True
            or doc.get('process_id') != http_pid or doc.get('event_count') != len(events)):
        raise OOMProbeError('Worker transition trace is missing, malformed, or overflowed')
    closes = [event for event in events if type(event) is dict
              and event.get('event') == 'native_worker_closed' and event.get('pid') == old_pid]
    rows = [event for event in events if type(event) is dict
            and event.get('event') == 'native_worker_replaced']
    if len(closes) != 1 or len(rows) != 1:
        raise OOMProbeError('Trace must contain exactly one old-worker close and one replacement')
    closed, event = closes[0], rows[0]
    replacement_closes = [row for row in events if type(row) is dict
        and row.get('event') == 'native_worker_closed' and row.get('pid') == new_pid]
    if (event.get('old_pid') != old_pid or event.get('new_pid') != new_pid
            or event.get('process_id') != http_pid
            or type(event.get('old_returncode')) is not int
            or type(event.get('restart_count')) is not int or event['restart_count'] != 1
            or type(closed.get('sequence')) is not int or type(event.get('sequence')) is not int
            or type(closed.get('returncode')) is not int
            or closed.get('returncode') != event.get('old_returncode')
            or closed.get('sequence') >= event.get('sequence')
            or len(replacement_closes)!=1 or replacement_closes[0].get('returncode')!=0
            or replacement_closes[0].get('close_succeeded') is not True):
        raise OOMProbeError('Persisted worker replacement does not bind the exact old/new process identities')
    return dict(old_pid=old_pid, old_returncode=event['old_returncode'],
                new_pid=new_pid, restart_count=1, close_sequence=closed.get('sequence'),
                event_sequence=event.get('sequence'),
                replacement_close_sequence=replacement_closes[0].get('sequence'))


def validate_worker_status(status, *, new_pid):
    row = base._object(status, 'same-service worker status')
    workers = base._object(row.get('workers'), 'worker status map')
    worker = workers.get('0')
    if worker is None:
        raise OOMProbeError('Status lacks the configured GPU 0 worker')
    worker = base._object(worker, 'GPU 0 worker status')
    if (worker.get('health') != 'healthy' or worker.get('restarts') != 1
            or worker.get('pid') != new_pid):
        raise OOMProbeError('Live service status does not report the exact healthy replacement worker')
    return dict(health=worker['health'], restarts=worker['restarts'], pid=worker['pid'])


def validate_clean_worker_exit_trace(document, *, http_pid, native_pid):
    doc = base._object(document, 'clean worker shutdown trace')
    events = doc.get('events')
    if (type(events) is not list or doc.get('overflowed') is True
            or doc.get('process_id') != http_pid or doc.get('event_count') != len(events)):
        raise OOMProbeError('Clean worker shutdown trace is missing or not bound to the HTTP process')
    rows = [event for event in events if type(event) is dict
            and event.get('event') == 'native_worker_closed' and event.get('pid') == native_pid]
    if (len(rows) != 1 or rows[0].get('returncode') != 0
            or rows[0].get('close_succeeded') is not True):
        raise OOMProbeError('Native child did not close exactly once with exit code zero')
    return dict(pid=native_pid, returncode=0, close_succeeded=True, sequence=rows[0].get('sequence'))


def validate_durable_completion(events, *, branch_before, branch_after, branch_name, output_count=8):
    """Bind the ordered durable/complete pair to answer and actual branch head."""
    if type(events) is not list or any(type(event) is not dict for event in events):
        raise OOMProbeError('Continuation event list is malformed')
    terminal=[(index,event) for index,event in enumerate(events)
              if event.get('event') in ('complete','durable','error','failure')]
    durable_rows=[(index,event) for index,event in terminal if event.get('event')=='durable']
    complete_rows=[(index,event) for index,event in terminal if event.get('event')=='complete']
    if (len(terminal)!=2 or len(durable_rows)!=1 or len(complete_rows)!=1
            or durable_rows[0][0]>=complete_rows[0][0]):
        raise OOMProbeError('Continuation requires one durable event before one complete acknowledgement')
    durable=durable_rows[0][1]; complete=complete_rows[0][1]
    durable_receipt=base._object(durable.get('receipt'),'durable event receipt')
    receipt=base._object(complete.get('receipt'),'complete event receipt')
    if durable_receipt!=receipt:
        raise OOMProbeError('Durable event and complete acknowledgement carry different receipts')
    answer=base._object(complete.get('answer'),'durable answer')
    receipt_branch=base._object(receipt.get('branch'),'receipt branch')
    tokens=answer.get('tokens')
    if (type(tokens) is not list or len(tokens)!=output_count
            or any(type(token) is not int for token in tokens)
            or answer.get('snapshot')!=receipt.get('snapshot')
            or receipt_branch.get('tip')!=receipt.get('snapshot')
            or not isinstance(branch_name,str) or not branch_name
            or receipt_branch.get('name')!=branch_name
            or type(branch_before.get('revision')) is not int
            or receipt_branch.get('revision')!=branch_before.get('revision')+1
            or branch_after.get('revision')!=receipt_branch.get('revision')
            or branch_after.get('tip')!=receipt_branch.get('tip')):
        raise OOMProbeError('Complete receipt, answer snapshot, and durable branch head do not agree')
    return dict(snapshot=receipt['snapshot'],branch=receipt_branch,
                output_tokens=list(tokens),event='complete',durable_event_index=durable_rows[0][0],
                complete_event_index=complete_rows[0][0])


def _service_stop_receipt(service):
    """Add PID absence observed for the exact owned service process to stop receipt."""
    process=getattr(service,'process',None)
    pid=getattr(process,'pid',None)
    if type(pid) is not int or pid<=0:
        raise OOMProbeError('Owned HTTP service has no valid process identity before stop')
    receipt=dict(service.stop())
    receipt['pid']=pid
    receipt['pid_absent_after_stop']=not Path('/proc',str(pid)).exists()
    return receipt


def validate_http_source_binding(http_path, http_sha256, profile_imports, expected_sha256):
    """Validate HTTP code independently, while accepting an identical profile-pinned file."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None:
        raise OOMProbeError('Expected HTTP SHA-256 pin is malformed')
    if http_sha256 != expected_sha256:
        raise OOMProbeError('HTTP service source differs from its explicit SHA-256 pin')
    if http_path in profile_imports:
        if profile_imports[http_path] != http_sha256:
            raise OOMProbeError('Profile-selected HTTP source conflicts with explicit HTTP pin')
        return {'path':http_path,'sha256':http_sha256,
                'binding':'same_file_in_selected_profile_closure'}
    return {'path':http_path,'sha256':http_sha256,
            'binding':'separate_explicit_service_source_pin'}


def validate_recovery_result(result):
    result = base._object(result, 'recovery result')
    _need(result.get('status') == 'PASS', 'recovery result is not PASS')
    oom = base._object(result.get('native_oom'), 'native OOM result')
    response = base._object(oom.get('response'), 'native OOM response')
    stats_after = base._object(oom.get('stats_after'), 'post-OOM native statistics')
    _need(response.get('status') == 'error'
          and type(oom.get('oom_failure_delta')) is int and oom['oom_failure_delta'] > 0
          and isinstance(oom.get('original_error'), str) and oom['original_error']
          and response.get('message') == oom.get('original_error')
          and stats_after.get('vmm_last_failed_operation') == 'cuMemCreate'
          and stats_after.get('vmm_last_driver_error') == 2,
          'original cuMemCreate error and driver code were not retained')
    fault = base._object(result.get('fault_http'), 'fault HTTP event evidence')
    fault_error = base._object(fault.get('evidence'), 'fault HTTP error event').get('error')
    _need(type(fault_error) is dict and fault_error.get('error') == response.get('message'),
          'The client-visible error did not preserve the exact native error message')
    source = base._object(result.get('source_before'), 'source before')
    same = base._object(result.get('same_service'), 'same-service result')
    restart = base._object(result.get('clean_restart'), 'clean restart result')
    validate_same_service_recovery(same, source_snapshot=source['tip'],
        source_payload_sha256=source['payload_sha256'], http_pid=same['http_pid'],
        old_native_pid=same['old_native_pid'])
    validate_clean_restart_branch(restart, source_snapshot=source['tip'],
        source_payload_sha256=source['payload_sha256'])
    transition = base._object(same.get('worker_transition'), 'persisted worker transition')
    replacement = base._object(same.get('replacement'), 'same-service replacement')
    new_worker = base._object(replacement.get('new_native_process'), 'replacement native process')
    native_path=result.get('native_binary')
    if (not isinstance(native_path,str) or not native_path
            or os.path.realpath(new_worker.get('exe',''))!=os.path.realpath(native_path)):
        raise OOMProbeError('Replacement native process executable differs from the pinned binary')
    if (transition.get('old_pid') != same.get('old_native_pid')
            or transition.get('new_pid') != new_worker.get('pid')
            or transition.get('restart_count') != 1
            or type(transition.get('old_returncode')) is not int):
        raise OOMProbeError('Persisted trace does not bind the observed replacement lifecycle')
    if (type(transition.get('old_returncode')) is not int
            or transition.get('old_returncode') != replacement['old_native_exit'].get('exit_code')
            or replacement.get('factory_after_old_exit') is not True):
        raise OOMProbeError('Persisted trace does not prove old-worker exit before replacement')
    initial = base._object(result.get('first_service'), 'first service')
    old_worker=base._object(initial.get('old_native'),'initial native process')
    if os.path.realpath(old_worker.get('exe',''))!=os.path.realpath(native_path):
        raise OOMProbeError('Initial native process executable differs from the pinned binary')
    clean_service = base._object(result.get('clean_restart_service'), 'clean-restart service')
    stop = base._object(clean_service.get('stop'), 'clean-restart stop')
    first_stop = base._object(initial.get('stop'), 'recovered first-service stop')
    if (type(clean_service.get('http_pid')) is not int
            or clean_service['http_pid'] == initial.get('http_pid')
            or os.path.realpath(base._object(clean_service.get('native_process'),'clean native process').get('exe',''))!=os.path.realpath(native_path)
            or type(stop.get('exit_code')) is not int or stop['exit_code'] != 0
            or stop.get('pid_absent_after_stop') is not True
            or type(first_stop.get('exit_code')) is not int or first_stop['exit_code'] != 0
            or first_stop.get('pid_absent_after_stop') is not True):
        raise OOMProbeError('Independent clean restart process identity/terminal exit is unverified')
    validate_clean_worker_exit_trace(clean_service.get('trace_document'),
        http_pid=clean_service['http_pid'], native_pid=clean_service['native_process']['pid'])
    validate_clean_worker_exit_trace(initial.get('trace_document'),
        http_pid=initial['http_pid'], native_pid=new_worker['pid'])
    helper = base._object(result.get('helper'), 'pressure helper evidence')
    _need(helper.get('released_before_gate_abort') is True
          and helper.get('exit_code') == 0 and helper.get('pid_absent') is True,
          'pressure helper was not released and verified before gate cleanup')
    source_after = base._object(result.get('source_after_all'), 'final source audit')
    _need(source_after.get('head') == source.get('head')
          and source_after.get('record_sha256') == source.get('record_sha256')
          and source_after.get('payload_sha256') == source.get('payload_sha256')
          and source_after.get('checkpoint') == source.get('checkpoint'),
          'Pinned source checkpoint changed during recovery')
    return 'PASS'


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--api-src', required=True)
    parser.add_argument('--api-scripts', required=True)
    parser.add_argument('--branch', default='CliA')
    parser.add_argument('--expected-source-snapshot', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--expected-profile-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--expected-http-sha256', required=True)
    parser.add_argument('--expected-helper-sha256', required=True)
    parser.add_argument('--pressure-helper', required=True)
    parser.add_argument('--recovery-branch', required=True)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--preflight-only', action='store_true')
    return parser


def _payload_from_fingerprint(store, snapshot, fingerprint_fn):
    files = fingerprint_fn(store, snapshot)
    payloads = [(name, row) for name, row in files.items() if name.endswith('.state')]
    if len(payloads) != 1:
        raise OOMProbeError('Pinned source checkpoint fingerprint must contain exactly one payload')
    name, row = payloads[0]
    return row['sha256'], {'files': files, 'payload_name': name,
                           'payload_sha256': row['sha256'], 'payload_bytes': row['bytes']}


def _recovery_preflight(args):
    """Validate the selected profile/runtime while accepting an independently pinned HTTP file
    when that same file is already part of the profile closure.
    """
    if not sys.platform.startswith('linux'):
        raise OOMProbeError('Real CUDA OOM recovery requires Linux')
    profile_path=Path(args.profile).expanduser().resolve(strict=True)
    store=Path(args.store).expanduser().resolve(strict=True)
    runtime_root=Path(args.runtime_root).expanduser().resolve(strict=True)
    api_src=Path(args.api_src).expanduser().resolve(strict=True)
    api_scripts=Path(args.api_scripts).expanduser().resolve(strict=True)
    helper_path=Path(args.pressure_helper).expanduser().resolve(strict=True)
    profile_sha=base._sha256(profile_path); helper_sha=base._sha256(helper_path)
    if profile_sha!=args.expected_profile_sha256 or helper_sha!=args.expected_helper_sha256:
        raise OOMProbeError('Profile or pressure helper differs from its pinned SHA-256')
    profile=json.loads(profile_path.read_text(encoding='utf-8'))
    execution=profile.get('execution')
    if (type(execution) is not dict or execution.get('GPU')!=[0]
            or execution.get('capacity')!=98304 or execution.get('prefill_chunk')!=8192
            or execution.get('kv_dtype')!='FP16'):
        raise OOMProbeError('Selected execution profile is not the qualified single-GPU Q4 configuration')
    command=execution.get('sut_command')
    if type(command) is not list or not command:
        raise OOMProbeError('Profile does not select a native SUT binary')
    native=Path(command[0]).expanduser().resolve(strict=True); native_sha=base._sha256(native)
    if native_sha!=args.expected_native_sha256:
        raise OOMProbeError('Native SUT binary differs from its selected hash pin')
    if not (store/'refs.json').is_file() or not (store/'checkpoints').is_dir():
        raise OOMProbeError('Input Store is incomplete')
    script_dir=runtime_root/'experimental'/'branching_context'/'scripts'
    http_path=runtime_root/'serve'/'cow_http.py'
    if not script_dir.is_dir() or not (script_dir/'run_cow_client_recovery.py').is_file():
        raise OOMProbeError('Selected runtime lacks its current client-recovery helper')
    for path in (runtime_root,api_src,api_scripts,script_dir):
        if path.is_symlink(): raise OOMProbeError(f'Runtime import path is a symlink: {path}')
        if str(path) not in sys.path: sys.path.insert(0,str(path))
    from run_q4_publication_fault import _runtime_closure
    closure=_runtime_closure(profile,runtime_root)
    profile_imports={row['path']:row['sha256'] for row in closure['files'].values()}
    modules={
        'context_time_travel.real':api_src/'context_time_travel'/'real.py',
        'context_time_travel.coordinator':api_src/'context_time_travel'/'coordinator.py',
        'context_time_travel.strata_backend':api_src/'context_time_travel'/'strata_backend.py',
        'context_time_travel.execution_receipt':api_src/'context_time_travel'/'execution_receipt.py',
        'context_time_travel.checkpoint_cache':api_src/'context_time_travel'/'checkpoint_cache.py',
    }
    for name,path in modules.items():
        resolved=str(path.resolve(strict=True)); expected=profile_imports.get(resolved)
        if expected is None or base._sha256(Path(resolved))!=expected:
            raise OOMProbeError(f'Execution profile does not pin runtime API import {name}')
    http_path=http_path.resolve(strict=True); http_sha=base._sha256(http_path)
    service_source=validate_http_source_binding(str(http_path),http_sha,profile_imports,
                                                 args.expected_http_sha256)
    if str(http_path) not in profile_imports:
        profile_imports[str(http_path)]=http_sha
    expected_imports=dict(profile_imports)
    from context_time_travel.store import Store
    source_head,source_history=_store_head_history(Store(store),args.branch,
        expected_tip=args.expected_source_snapshot)
    if source_history.get('controller_status')!='active':
        raise OOMProbeError('Pinned source branch controller must be active for recovery continuation')
    return {'profile_path':str(profile_path),'profile_sha256':profile_sha,'profile':profile,
        'store':str(store),'runtime_root':str(runtime_root),'api_src':str(api_src),
        'api_scripts':str(api_scripts),'helper_path':str(helper_path),'helper_sha256':helper_sha,
        'native_binary':str(native),'native_sha256':native_sha,'runtime_source_closure':closure,
        'profile_api_imports':{k:v for k,v in profile_imports.items() if k!=str(http_path)},
        'expected_imports':expected_imports,
        'service_build_source':service_source,'source_head':source_head,
        'source_history':source_history}


def _store_head_history(store, branch, *, expected_tip=None):
    """Read token-only canonical history under a lease; no presentation namespace is required."""
    head=store.head(branch)
    row={'name':head.name,'revision':head.revision,'tip':head.tip,'root':head.root}
    if expected_tip is not None and row['tip']!=expected_tip:
        raise OOMProbeError('Canonical source branch tip differs from its snapshot pin')
    with store.read_lease(row['tip']):
        current=store.head(branch)
        current_row={'name':current.name,'revision':current.revision,
                     'tip':current.tip,'root':current.root}
        if current_row!=row:
            raise OOMProbeError('Canonical branch head changed while acquiring its read lease')
        snapshot=store.get_snapshot(row['tip'])
        root,tokens=store.lineage(row['tip'])
        controller=snapshot.get('controller')
        if (root!=row['root'] or type(controller) is not dict
                or controller.get('status') not in ('active','stopped')):
            raise OOMProbeError('Canonical branch snapshot has invalid root or controller history')
    history={'id':branch,'revision':row['revision'],'tip':row['tip'],'root':row['root'],
             'controller_status':controller['status'],'token_count':len(tokens)}
    return row,history


def _fork_token_branch(client, snapshot, new_branch):
    """Fork a token-only branch without inventing presentation namespace metadata."""
    status,result=client.post_json('/cow/fork',{'name':new_branch,'snapshot':snapshot})
    if status not in (200,201) or type(result) is not dict:
        raise OOMProbeError(f'Token-only snapshot fork failed (HTTP {status})')
    head=result.get('branch') if isinstance(result.get('branch'),dict) else result
    if head.get('name')!=new_branch or head.get('tip')!=snapshot:
        raise OOMProbeError('Token-only fork receipt does not identify the requested snapshot')
    return head


def _head_history(client, store, branch):
    """Cross-check HTTP-selected head with leased canonical token history."""
    from urllib.parse import quote
    status,head=client.get('/cow/head?branch='+quote(branch,safe=''))
    if status!=200 or type(head) is not dict:
        raise OOMProbeError(f'Cannot read branch head (HTTP {status})')
    if (head.get('name')!=branch or type(head.get('revision')) is not int
            or not isinstance(head.get('tip'),str) or not isinstance(head.get('root'),str)):
        raise OOMProbeError('HTTP branch head is malformed')
    canonical,history=_store_head_history(store,branch,expected_tip=head['tip'])
    if canonical!=head:
        raise OOMProbeError('HTTP branch head is stale or differs from canonical Store state')
    return head,history


def run_probe(args):
    """Run the OOM, same-HTTP worker recovery, and separate clean-restart branch."""
    pre = _recovery_preflight(args)
    store_path = Path(pre['store'])
    # The profile closure, helper and native executable are exactly the same pinned
    # inputs used by the baseline fault harness.
    run_dir = Path(args.run_dir).expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    from run_q4_publication_fault import (_snapshot_file_fingerprint,
        _snapshot_record_fingerprint, _generation_body,
        validate_idle_metrics,
        validate_source_checkpoint_unchanged)
    from context_time_travel.store import Store
    canonical_store=Store(store_path)
    from run_cow_client_recovery import HttpClient, ProgressTracker, _ready_url, ServiceProcess, _trace_worker
    nonce = uuid.uuid4().hex
    fault_op = 'op-q4-recovery-fault-' + nonce
    recovery_op = 'op-q4-recovery-followup-' + nonce
    restart_op = 'op-q4-restart-followup-' + nonce
    fault_branch = 'OOMF' + nonce[:12]
    restart_branch = args.recovery_branch
    result = {'schema':'q4-cuda-oom-recovery-v1', 'status':'RUNNING',
        'qualification_scope':'actual Q4 CUDA OOM, same-HTTP native worker replacement, and independent clean-restart branch; numerical comparison is separately audited',
        'profile_path':pre['profile_path'], 'profile_sha256':pre['profile_sha256'],
        'native_binary':pre['native_binary'], 'native_sha256':pre['native_sha256'],
        'runtime_root':pre['runtime_root'], 'store':pre['store'],
        'source_branch':args.branch, 'source_snapshot':args.expected_source_snapshot,
        'fault_branch':fault_branch, 'restart_branch':restart_branch,
        'fault_operation':fault_op, 'same_service_operation':recovery_op,
        'restart_operation':restart_op, 'max_tokens_fault':args.max_tokens,
        'continuation_tokens':8, 'runner_sha256':base._sha256(Path(__file__).resolve()),
        'steps':[]}
    output = run_dir / 'result.json'
    wrapper = base._write_wrapper(run_dir)
    control_path = Path('/tmp') / f'q4-recovery-{nonce[:16]}.sock'
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(control_path)); os.chmod(control_path, 0o600); listener.listen(8)
    progress = ProgressTracker(run_dir, 'q4-cuda-oom-recovery'); progress.start()
    service = helper = None
    helper_id = None
    controls=[]; threads=[]; cleanup=[]; error=None
    try:
        progress.phase('start_fault_http_service')
        service, ready, trace = base._launch_service(runtime_root=Path(pre['runtime_root']),
            api_src=Path(pre['api_src']), api_scripts=Path(pre['api_scripts']),
            profile=Path(pre['profile_path']), store=Path(pre['store']), wrapper=wrapper,
            run_dir=run_dir, control_path=control_path, operation=fault_op,
            branch=fault_branch, expected_imports=pre['expected_imports'], args=args,
            progress=progress, suffix='fault')
        pid_http = service.process.pid
        import_event, conn = base._control_event(listener, args.timeout, progress)
        result['imports'] = base.validate_import_witness(import_event, pre['expected_imports']); conn.close()
        children = sorted({int(pid) for p in (Path('/proc')/str(pid_http)/'task').glob('*/children')
                           for pid in p.read_text().split()})
        if len(children) != 1: raise OOMProbeError(f'Expected one native child, got {children}')
        old = base._proc_identity(children[0])
        if (old.get('parent_pid')!=pid_http
                or os.path.realpath(old.get('exe',''))!=str(Path(pre['native_binary']))):
            raise OOMProbeError('Initial native child identity differs from pinned executable/service')
        result['first_service']={'http_pid':pid_http,'old_native':old,'ready':ready,'trace':str(trace)}
        client=HttpClient(_ready_url(ready),timeout=args.timeout,progress=progress)
        status,health=client.get('/health')
        if status!=200 or health.get('service')!='context-cow': raise OOMProbeError('Service health check failed')
        source_head,source_history=_head_history(client,canonical_store,args.branch)
        if source_head.get('tip')!=args.expected_source_snapshot: raise OOMProbeError('Source branch tip differs from pin')
        if source_history.get('controller_status')!='active':
            raise OOMProbeError('Pinned source branch controller must be active for the recovery probe')
        source_rec=_snapshot_record_fingerprint(store_path,args.expected_source_snapshot)
        source_files=_snapshot_file_fingerprint(store_path,args.expected_source_snapshot)
        payload_sha,payload_receipt=_payload_from_fingerprint(store_path,args.expected_source_snapshot,
                                                               _snapshot_file_fingerprint)
        result['source_before']={'tip':source_head['tip'],'head':source_head,'record_sha256':source_rec,
                                 'payload_sha256':payload_sha,'checkpoint':source_files,'payload':payload_receipt}
        # Both branches are forked from P before any fault or continuation.
        # These token-only fixtures have no presentation namespace to fork from.
        fault_head=_fork_token_branch(client,args.expected_source_snapshot,fault_branch)
        restart_head=_fork_token_branch(client,args.expected_source_snapshot,restart_branch)
        fh,fhist=_head_history(client,canonical_store,fault_branch); rh,rhist=_head_history(client,canonical_store,restart_branch)
        if fh.get('tip')!=args.expected_source_snapshot or rh.get('tip')!=args.expected_source_snapshot:
            raise OOMProbeError('Both test branches must initially point to the immutable source snapshot')
        route,body=_generation_body(fhist,fh,fault_branch,fault_op,args.max_tokens,'Continue the selected Q4 branch.')
        box={}; thread=threading.Thread(target=base._stream_request,args=(client,route,body,box),daemon=True)
        threads.append(thread); thread.start()
        gate_conn=None; error_conn=None
        gate_event,gate_conn=base._control_event(listener,args.timeout,progress); controls.append(gate_conn)
        gate=base.validate_gate_record(gate_event,operation_id=fault_op,branch=fault_branch,
                                       native_pid=old['pid'],expected_forward=0)
        result['gate']=gate
        progress.phase('start_pressure_helper')
        helper,helper_id,readyframe=base._helper_start(Path(pre['helper_path']),run_dir/'helper.stderr.log')
        result['helper']={'ready':readyframe,'identity':helper_id,'state':'ready'}
        progress.phase('fill_device_pressure')
        pressure=base._helper_fill(helper,args.timeout); base.validate_pressure_frame(pressure)
        result['helper'].update({'pressure':pressure,'state':'filled'})
        base._reply(gate_conn,'continue'); gate_conn.close(); controls.remove(gate_conn)
        err_event,error_conn=base._control_event(listener,args.timeout,progress); controls.append(error_conn)
        if (err_event.get('event')!='native_generate_error' or err_event.get('operation_id')!=fault_op
            or err_event.get('native_pid')!=old['pid']): raise OOMProbeError('OOM observation is not bound to the gated old worker')
        oom=base.validate_native_oom(err_event.get('original_response'),err_event.get('stats_before'),
                                    err_event.get('stats_after'),expected_forward=0)
        progress.phase('release_device_pressure')
        release,helper_exit=base._helper_release(helper,helper_id,args.timeout)
        base.validate_helper_pressure(pressure,release)
        base.validate_owned_process_exit(helper_exit,expected_pid=helper_id['pid'],
            expected_start_time=helper_id['start_time'],expected_exe=helper_id['exe'])
        result['helper'].update({'pressure':pressure,'release':release,
            'identity':helper_id,'exit':helper_exit,'exit_code':helper_exit['exit_code'],
            'pid_absent':helper_exit['pid_absent_after_wait'],
            'released_before_gate_abort':True,'state':'released'})
        helper=None; base._reply(error_conn,'continue'); error_conn.close(); controls.remove(error_conn)
        thread.join(args.timeout)
        if thread.is_alive() or 'value' not in box: raise OOMProbeError('Faulted request did not terminate cleanly')
        events,disconnected=box['value']; fault_http=base.validate_fault_http_events(events)
        fault_after,_=_head_history(client,canonical_store,fault_branch)
        source_after,_=_head_history(client,canonical_store,args.branch)
        validate_source_checkpoint_unchanged(source_files,_snapshot_file_fingerprint(store_path,args.expected_source_snapshot),phase='after OOM')
        if source_after!=source_head or fault_after!=fh or _snapshot_record_fingerprint(store_path,args.expected_source_snapshot)!=source_rec:
            raise OOMProbeError('OOM changed source or fault branch state')
        # Observe automatic replacement before issuing another SUT request.
        progress.phase('wait_for_worker_replacement')
        deadline=time.monotonic()+args.timeout; new=None; worker_status=None
        native_exe=os.path.realpath(pre['native_binary'])
        while time.monotonic()<deadline:
            current={int(x) for p in (Path('/proc')/str(pid_http)/'task').glob('*/children') for x in p.read_text().split()}
            if old['pid'] not in current and current:
                candidates=sorted(current)
                if len(candidates)==1:
                    try: new=base._proc_identity(candidates[0])
                    except FileNotFoundError: new=None
                    if (new is not None and new['pid']!=old['pid']
                            and new.get('parent_pid')==pid_http
                            and os.path.realpath(new.get('exe',''))==native_exe):
                        status_code,service_status=client.get('/cow/status')
                        if status_code==200:
                            try:
                                worker_status=validate_worker_status(service_status,new_pid=new['pid'])
                            except OOMProbeError:
                                worker_status=None
                            else:
                                status,health=client.get('/health')
                                if status==200 and health.get('service')=='context-cow':
                                    break
                        new=None
            time.sleep(.1)
        if new is None or worker_status is None:
            raise OOMProbeError('Same HTTP process did not reach a healthy replacement worker before deadline')
        status_code,service_status=client.get('/cow/status')
        if status_code!=200: raise OOMProbeError('Same HTTP service status endpoint failed after replacement')
        validate_idle_metrics(service_status)
        result['native_oom']=oom; result['fault_http']={'events':events,'disconnected':disconnected,'evidence':fault_http}
        progress.phase('same_service_followup')
        same_route,same_body=_generation_body(fhist,fh,fault_branch,recovery_op,8,'Continue after native worker replacement.')
        same_events,same_disconnected=client.post_stream(same_route,same_body)
        same_terminal=base._terminal_event(same_events)
        if same_terminal.get('event')!='complete': raise OOMProbeError('Same-service continuation did not complete')
        same_answer=same_terminal.get('answer')
        if type(same_answer) is not dict: raise OOMProbeError('Same-service completion has no answer')
        same_head,same_hist=_head_history(client,canonical_store,fault_branch)
        source_after_recovery,source_history_after_recovery=_head_history(client,canonical_store,args.branch)
        source_payload_after_recovery,source_receipt_after_recovery=_payload_from_fingerprint(
            store_path,args.expected_source_snapshot,_snapshot_file_fingerprint)
        validate_source_checkpoint_unchanged(source_files,
            _snapshot_file_fingerprint(store_path,args.expected_source_snapshot),
            phase='after same-service recovery')
        if source_after_recovery!=source_head or _snapshot_record_fingerprint(
                store_path,args.expected_source_snapshot)!=source_rec:
            raise OOMProbeError('Same-service recovery changed the canonical source branch or record')
        # Work counters are copied from durable answer metadata, not an out-of-owner native pipe read.
        counter_deltas=same_answer.get('counter_deltas') or same_answer.get('native_counter_deltas')
        native_error=oom['response'].get('message')
        if not isinstance(native_error,str) or fault_http['error'].get('error')!=native_error:
            raise OOMProbeError('Fault HTTP error did not preserve the exact original native error')
        oom['original_error']=native_error
        same_payload_sha,same_payload_receipt=_payload_from_fingerprint(store_path,
            args.expected_source_snapshot,_snapshot_file_fingerprint)
        result['same_service']={'http_pid':pid_http,'old_native_pid':old['pid'],
            'service_status':service_status,
            'replacement':{'old_native_pid':old['pid'],
                'old_native_exit':{'pid':old['pid'],'pid_absent':not Path('/proc',str(old['pid'])).exists()},
                'new_native_process':new,'restart_count':worker_status['restarts'],
                'worker_status':worker_status},
            'followup':{'event':'complete','answer':same_answer,'events':same_events,
                'disconnected':same_disconnected,'counter_deltas':counter_deltas,
                'branch':fault_branch,'branch_before':{'branch':fault_branch,**fh},'branch_after':same_head,
                'controller':same_hist.get('controller_status')},
            'source_after':{'tip':source_after_recovery.get('tip'),'payload_sha256':source_payload_after_recovery,
                            'payload':source_receipt_after_recovery}}
        progress.phase('stop_recovered_first_service')
        first_stop=_service_stop_receipt(service); result['first_service']['stop']=first_stop; service=None
        if first_stop.get('exit_code')!=0: raise OOMProbeError('First HTTP process did not stop cleanly after recovery')
        transition_doc=json.loads(Path(trace).read_text(encoding='utf-8'))
        result['first_service']['trace_document']=transition_doc
        transition=validate_worker_transition_trace(transition_doc,old_pid=old['pid'],
            new_pid=new['pid'],http_pid=pid_http)
        result['same_service']['replacement']['old_native_exit']['exit_code']=transition['old_returncode']
        result['same_service']['replacement']['factory_after_old_exit']=(
            transition['close_sequence'] < transition['event_sequence'])
        result['same_service']['replacement']['restart_count']=transition['restart_count']
        result['same_service']['worker_transition']=transition
        result['same_service']['replacement']['old_native_exit']['pid_absent']=not Path('/proc',str(old['pid'])).exists()
        validate_same_service_recovery(result['same_service'],source_snapshot=args.expected_source_snapshot,
            source_payload_sha256=payload_sha,http_pid=pid_http,old_native_pid=old['pid'])
        # Start a fresh service bound to the pre-fault independent branch.
        progress.phase('start_independent_clean_restart_service')
        service,ready2,trace2=base._launch_service(runtime_root=Path(pre['runtime_root']),
            api_src=Path(pre['api_src']),api_scripts=Path(pre['api_scripts']),profile=Path(pre['profile_path']),
            store=Path(pre['store']),wrapper=wrapper,run_dir=run_dir,control_path=control_path,
            operation=restart_op+'-observer-disabled',branch=restart_branch,expected_imports=pre['expected_imports'],args=args,
            progress=progress,suffix='clean-restart')
        imp,ic=base._control_event(listener,args.timeout,progress)
        restart_http_pid=service.process.pid
        restart_children=sorted({int(x) for p in (Path('/proc')/str(restart_http_pid)/'task').glob('*/children')
                                 for x in p.read_text().split()})
        if len(restart_children)!=1: raise OOMProbeError(f'Expected one clean-restart native child, got {restart_children}')
        restart_native=base._proc_identity(restart_children[0])
        if (restart_native.get('parent_pid')!=restart_http_pid
                or os.path.realpath(restart_native.get('exe',''))!=str(Path(pre['native_binary']))):
            raise OOMProbeError('Clean restart native child identity differs from pinned executable/service')
        result['clean_restart_service']={'http_pid':restart_http_pid,'native_process':restart_native,
            'ready':ready2,'imports':base.validate_import_witness(imp,pre['expected_imports'])}; ic.close()
        client2=HttpClient(_ready_url(ready2),timeout=args.timeout,progress=progress)
        before,hist=_head_history(client2,canonical_store,restart_branch)
        before_payload=_payload_from_fingerprint(store_path,args.expected_source_snapshot,
                                                  _snapshot_file_fingerprint)[0]
        if before.get('tip')!=args.expected_source_snapshot: raise OOMProbeError('Independent restart branch no longer points at P')
        progress.phase('clean_restart_followup')
        rroute,rbody=_generation_body(hist,before,restart_branch,restart_op,8,'Continue from the independent source branch after restart.')
        revents,rdisc=client2.post_stream(rroute,rbody); terminal=base._terminal_event(revents)
        if terminal.get('event')!='complete' or type(terminal.get('answer')) is not dict:
            raise OOMProbeError('Independent clean-restart continuation did not complete')
        answer=terminal['answer']; restart_after,restart_after_hist=_head_history(client2,canonical_store,restart_branch)
        work=answer.get('counter_deltas') or answer.get('native_counter_deltas')
        result['clean_restart']={'branch_before':{'branch':restart_branch,**before},
            'fault_branch':fault_branch,'source_after':{'tip':source_head.get('tip'),
                'payload_sha256':_payload_from_fingerprint(store_path,args.expected_source_snapshot,
                    _snapshot_file_fingerprint)[0]},
            'followup':{'event':'complete','answer':answer,'events':revents,'disconnected':rdisc,
                'counter_deltas':work,'branch':restart_branch,
                'branch_before':{'branch':restart_branch,**before},'branch_after':restart_after,
                'controller':restart_after_hist.get('controller_status')},
            'trace_path':str(trace2)}
        validate_clean_restart_branch(result['clean_restart'],source_snapshot=args.expected_source_snapshot,
            source_payload_sha256=before_payload)
        source_final,source_final_history=_head_history(client2,canonical_store,args.branch)
        source_final_payload,source_final_receipt=_payload_from_fingerprint(store_path,
            args.expected_source_snapshot,_snapshot_file_fingerprint)
        if source_final!=source_head or _snapshot_record_fingerprint(
                store_path,args.expected_source_snapshot)!=source_rec:
            raise OOMProbeError('Clean-restart request changed the canonical source branch or record')
        validate_source_checkpoint_unchanged(source_files,
            _snapshot_file_fingerprint(store_path,args.expected_source_snapshot),
            phase='after clean-restart continuation')
        result['source_after_all']={'head':source_final,'history':source_final_history,
            'record_sha256':_snapshot_record_fingerprint(store_path,args.expected_source_snapshot),
            'checkpoint':_snapshot_file_fingerprint(store_path,args.expected_source_snapshot),
            'payload_sha256':source_final_payload,'payload':source_final_receipt}
        progress.phase('stop_clean_restart_service')
        stop=_service_stop_receipt(service); result['clean_restart_service']['stop']=stop; service=None
        if stop.get('exit_code')!=0: raise OOMProbeError('Clean-restart HTTP service did not stop with exit zero')
        result['clean_restart_service']['trace_document']=json.loads(Path(trace2).read_text(encoding='utf-8'))
        result['status']='PASS'; validate_recovery_result(result)
    except BaseException as exc:
        result['status']='FAIL'; result['error']={'type':type(exc).__name__,'message':str(exc)}
        if isinstance(exc,base.ServiceStartupFailure):
            result.setdefault('service_startup_cleanup',{})['receipt']=exc.cleanup_receipt
            if (not exc.cleanup_receipt.get('pid_absent',False)
                    or exc.cleanup_receipt.get('cleanup_failed',False)):
                cleanup.append({'resource':'service_startup','error':'startup process cleanup was not verified'})
        error=exc
    finally:
        # The allocation helper owns device memory; release it before aborting
        # a gated request so the service cannot replace its worker under pressure.
        if helper is not None:
            try:
                helper_receipt=base._stop_owned_helper(helper,helper_id,args.timeout)
                result['helper_cleanup']=helper_receipt
                terminal=helper_receipt.get('terminal') or helper_receipt.get('forced_terminal') or {}
                if helper_receipt.get('cleanup_errors') or not terminal.get('pid_absent_after_wait'):
                    cleanup.append({'resource':'helper','error':'helper cleanup was not verified'})
            except BaseException as exc:
                cleanup.append({'resource':'helper','error':f'{type(exc).__name__}: {exc}'})
            helper=None
        for conn in controls:
            try: base._reply(conn,'abort')
            except BaseException as exc: cleanup.append({'resource':'control_abort','error':str(exc)})
            try: conn.close()
            except BaseException as exc: cleanup.append({'resource':'control_close','error':str(exc)})
        controls.clear()
        for thread in threads:
            thread.join(timeout=2)
            if thread.is_alive(): cleanup.append({'resource':'request_thread','error':'owned request thread remained live'})
        if service is not None:
            try:
                stop_receipt=_service_stop_receipt(service)
                result.setdefault('service_cleanup',{})['stop']=stop_receipt
                if (stop_receipt.get('exit_code')!=0
                        or stop_receipt.get('pid_absent_after_stop') is not True):
                    cleanup.append({'resource':'service','error':'HTTP service cleanup was not clean'})
            except BaseException as exc: cleanup.append({'resource':'service','error':str(exc)})
        if listener is not None: listener.close()
        try: control_path.unlink()
        except FileNotFoundError: pass
        except BaseException as exc: cleanup.append({'resource':'control_socket','error':str(exc)})
        result['completed_at_unix_ns']=time.time_ns()
        finish_error=base._finish_and_persist(result,output,progress,cleanup,error)
        if finish_error is not None and error is None: error=finish_error
    if error is not None: raise error
    return result


def main(argv=None):
    args=_parser().parse_args(argv)
    try:
        if args.preflight_only:
            pre=_recovery_preflight(args)
            print(json.dumps({'status':'PREPARED_NOT_RUN','profile_sha256':pre['profile_sha256'],
                'native_sha256':pre['native_sha256']},sort_keys=True,separators=(',',':')))
            return 0
        return 0 if run_probe(args)['status']=='PASS' else 1
    except BaseException as exc:
        print(f'q4-cuda-recovery: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
