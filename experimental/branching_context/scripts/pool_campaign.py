"""Real GPU pool switching/restart, independent matching-operator references.

Eight trees/four clients by default. Prepare and resume are separate Python
processes, each closing its native workers. No HTTP server is started.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from native_request_probe import reference, publish, frontier, _record_native_shutdown
from context_time_travel.strata_backend import _unpack_frontier
from durable_resume import sha
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig, PoolTransitionTrace
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend, _digest


def tree_root(fixture,tree,roots):
    mapping=fixture.get('tree_roots')
    if mapping is None: return tree%roots
    if tree >= len(mapping) or type(mapping[tree]) is not int or not 0 <= mapping[tree] < roots:
        raise ValueError('Invalid tree/root fixture mapping')
    return mapping[tree]


def validate_rewind(before,after,root,sibling_before,sibling_after):
    expected=dict(before,tip=root,revision=before['revision']+1)
    if after != expected or sibling_before != sibling_after:
        raise AssertionError('Rewind changed the wrong reference or did not advance its revision')


def rewind_record(root_index,tree,before,after,receipt,sibling,snapshot):
    return dict(root_index=root_index,tree=tree,before=asdict(before),after=asdict(after),
                receipt=asdict(receipt),sibling=asdict(sibling),snapshot=snapshot)


def _finalize_phase(record, destination, action, coordinator, transition_trace,
                    stop, thread, started):
    """Always stop heartbeat and persist phase cleanup, including native exit codes."""
    try:
        if coordinator is not None:
            processes={gpu:w['model'].backend.process for gpu,w in coordinator._workers.items()
                       if w['model'] is not None}
            _record_native_shutdown(record, processes, coordinator.close)
        else:
            record.update(native_pids={},native_exit_codes={},native_exit_observed=False,
                          native_exit_success=False)
            record['status']='FAIL'
            record['cleanup_error']='Coordinator unavailable during native shutdown accounting'
        if transition_trace is not None:
            trace_path=destination.parent/(action+'-pool-transitions.json')
            publish(trace_path,transition_trace.snapshot())
            record['pool_transition_trace']=str(trace_path)
    except BaseException as exc:
        record['status']='FAIL'
        record['cleanup_error']=str(exc)
    finally:
        stop.set()
        try:
            thread.join()
        except BaseException as exc:
            record['status']='FAIL'
            record['heartbeat_cleanup_error']=str(exc)
        record['elapsed_seconds']=time.monotonic()-started
        publish(destination,record)


def _finish_phase(record, destination, action, coordinator, transition_trace,
                  stop, thread, started):
    """Persist cleanup and fail a direct phase invocation on native shutdown failure."""
    exception_active=sys.exc_info()[0] is not None
    _finalize_phase(record,destination,action,coordinator,transition_trace,
                    stop,thread,started)
    if record.get('status')!='PASS' and not exception_active:
        raise SystemExit(1)


def pressure_tree_order(trees,mode='default'):
    if type(trees) is not int or trees <= 0: raise ValueError('Invalid tree count')
    if mode=='default': return list(range(trees))
    if mode=='interleaved-roots':
        if trees!=6: raise ValueError('Interleaved-root pressure order requires the six-tree fixture')
        return list(range(0,trees,2))+list(range(1,trees,2))
    raise ValueError('Unknown pressure tree order')


def validate_spill_reload_trace(document):
    """Require a freed same-key RAM image to be loaded successfully later in one phase/process."""
    if not isinstance(document,dict) or document.get('overflowed'):
        raise ValueError('Pool transition trace overflowed or is missing')
    events=document.get('events')
    if not isinstance(events,list) or type(document.get('event_count')) is not int or document['event_count']!=len(events):
        raise ValueError('Pool transition trace is incomplete')
    phase,process_id,phase_id=(document.get(name) for name in ('phase','process_id','phase_id'))
    if not phase or process_id is None or not phase_id:
        raise ValueError('Pool transition trace lacks phase/process identity')
    for sequence,row in enumerate(events):
        if not isinstance(row,dict) or row.get('sequence')!=sequence:
            raise ValueError('Pool transition trace sequence is incomplete')
        if row.get('process_id')!=process_id:
            raise ValueError('Pool transition trace process identity mismatch')
        if row.get('phase')!=phase or row.get('phase_id')!=phase_id:
            raise ValueError('Pool transition trace phase identity mismatch')
        if not isinstance(row.get('key'),list) or len(row['key'])!=3 or any(
                not isinstance(part,str) or not part for part in row['key']):
            raise ValueError('Pool transition key must bind model contract, profile and snapshot')
        if row.get('event')=='ram_clean_eviction' and (
                type(row.get('freed_bytes')) is not int or row['freed_bytes']<0):
            raise ValueError('RAM eviction trace has invalid freed-byte count')
        if row.get('event')=='disk_preload_succeeded' and (
                row.get('success') is not True or type(row.get('bytes')) is not int or row['bytes']<=0):
            raise ValueError('Disk success trace lacks positive loaded-byte evidence')
        if row.get('event')=='disk_preload_started' and (
                type(row.get('load_id')) is not int or row['load_id']!=sequence):
            raise ValueError('Disk start trace lacks its unique load identity')
        if row.get('event') in ('disk_preload_succeeded','disk_preload_failed') and (
                type(row.get('load_id')) is not int):
            raise ValueError('Disk completion trace lacks its load identity')
    pending_evictions={}
    active_loads={}
    matches=[]
    for row in events:
        key=tuple(row['key'])
        if row['event']=='ram_clean_eviction' and row['freed_bytes']>0:
            pending_evictions.setdefault(key,deque()).append(row)
        elif row['event']=='disk_preload_started':
            evictions=pending_evictions.get(key)
            # Keep the eviction available until a corresponding read succeeds:
            # admission may fail and the coordinator may retry this same key.
            eviction=evictions[0] if evictions else None
            active_loads[row['load_id']]=dict(key=key,eviction=eviction,start=row['sequence'])
        elif row['event'] in ('disk_preload_succeeded','disk_preload_failed'):
            attempt=active_loads.pop(row['load_id'],None)
            if attempt is None or attempt['key']!=key or row['sequence']<=attempt['start']:
                raise ValueError('Disk completion does not match its exact started load')
            if row['event']=='disk_preload_succeeded' and attempt['eviction'] is not None:
                eviction=attempt['eviction']
                evictions=pending_evictions[key]
                if not evictions or evictions[0]['sequence']!=eviction['sequence']:
                    raise ValueError('Disk success lost its pending eviction identity')
                evictions.popleft()
                matches.append(dict(key=list(key),eviction_sequence=eviction['sequence'],
                    load_start_sequence=attempt['start'],load_sequence=row['sequence'],
                    load_id=row['load_id'],freed_bytes=eviction['freed_bytes'],loaded_bytes=row['bytes']))
    if not matches:
        raise ValueError('no same-process same-key freed eviction followed by successful disk preload')
    return matches


def config_and_profile(path,output):
    config = json.loads(path.read_text())
    pool = PoolConfig(gpus=tuple(config['GPU']),ram_bytes=config.get('ram_bytes',8*1024**3),
        disk_bytes=config.get('disk_bytes',64*1024**3),
        min_available_bytes=config.get('min_available_bytes',16*1024**3))
    profile = json.loads(Path(config['profile']).read_text())
    profile['execution']['GPU'] = list(pool.gpus)
    # The independent reference is also explicitly bound to a selected device.
    command = profile['execution']['oracle_command']
    if '--gpu' in command: raise ValueError('Use an unbound pinned base profile for this campaign')
    command += ['--gpu',str(pool.gpus[0])]
    profile['execution_profile_id'] = _digest(dict(build=profile['native_build'],
        execution=profile['execution'],hardware=profile['hardware']))
    publish(output/'profile.json',profile)
    return pool,profile


def get_state(coordinator,snapshot):
    model = next(w['model'] for w in coordinator._workers.values())
    manifest = model.validate_checkpoint(snapshot)
    file = model.checkpoint_directory/(manifest['payload_sha256']+'.state')
    with file.open('rb') as stream:
        stream.seek(manifest['state_bytes']);raw=stream.read(4*manifest['logits_count'])
    audit = coordinator.store.root.parent/'frontiers'
    audit.mkdir(exist_ok=True)
    path = audit/(snapshot+'.f32');path.write_bytes(raw)
    return dict(snapshot=snapshot,manifest=manifest,frontier_sha256=hashlib.sha256(raw).hexdigest(),
        frontier_file=str(path),history=list(model.history(snapshot)),controller=model.controller(snapshot),payload_file=str(file))


def continuation_limit(state,already_generated,limit):
    """EOS is a completed turn, not an invitation to restart its controller."""
    if state['controller']['status']=='stopped': return 0
    return max(0,limit-already_generated)


def phase(args):
    pool,profile = config_and_profile(args.config,args.output)
    config=json.loads(args.config.read_text())
    trace_enabled=bool(config.get('trace_pool_transitions') or config.get('require_spill_reload_same_process'))
    transition_trace=(PoolTransitionTrace(args.action,
        max_events=config.get('pool_trace_max_events',8192)) if trace_enabled else None)
    result = dict(status='RUNNING',phase=args.action,python_pid=os.getpid(),GPU=list(pool.gpus),
        scope='real GPU, greedy identical native operators; not cumulative gate certification',
        execution_profile_id=profile['execution_profile_id'],requests=[])
    destination=args.output/(args.action+'.json')
    publish(destination,result)
    coordinator=None
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):
            print(json.dumps(dict(event='heartbeat',phase=args.action,python_pid=os.getpid(),
                completed=len(result['requests']),seconds=time.monotonic()-start)),flush=True)
    start=time.monotonic()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        def factory(gpu):
            backend=StrataBackend(profile,gpu_id=gpu)
            return BranchingModel(args.output/'store',backend,profile)
        coordinator=ContextCoordinator(args.output/'store',pool,factory,transition_trace=transition_trace)
        coordinator.ready()
        result['startup_seconds']=time.monotonic()-start
        result['native_pids']={gpu:w['model'].backend.process.pid for gpu,w in coordinator._workers.items()}
        # Actual PID/GPU residency, not only the requested routing list.
        result['gpu_processes']=subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
        fixture=json.loads((args.output/'fixture.json').read_text())
        event_path=args.output/(args.action+'-events.jsonl')
        def complete(request):
            while not request.durable.done():
                try: event=request.events.get(timeout=1)
                except queue.Empty: continue
                with event_path.open('a') as stream: stream.write(json.dumps(event)+'\n')
            while not request.events.empty():
                with event_path.open('a') as stream: stream.write(json.dumps(request.events.get())+'\n')
            receipt=request.durable.result()
            answer=request.snapshot.result()
            row=dict(operation=request.operation,answer=answer,receipt=receipt,
                state=get_state(coordinator,receipt['snapshot']),pool=coordinator.metrics())
            result['requests'].append(row);publish(destination,result)
            print(json.dumps(dict(event='completed',phase=args.action,operation=request.operation,
                outputs=len(answer['tokens']),ram=row['pool']['ram'])),flush=True)
            return row
        if args.action=='prepare':
            result['roots']=[]
            for index,ids in enumerate(fixture['prefixes']):
                row=complete(coordinator.create('base-'+str(index),ids,'base-create-'+str(index)))
                result['roots'].append(row['state'])
            for tree in range(args.trees): coordinator.fork('tree-'+str(tree),result['roots'][tree_root(fixture,tree,args.roots)]['snapshot'])
            tree_ids=pressure_tree_order(args.trees,args.pressure_order)
            jobs=[coordinator.submit('tree-'+str(tree),fixture['suffixes'][tree],min(4,args.generated//2),
                'client-'+str(tree//2)+'-tree-'+str(tree)+'-first') for tree in tree_ids]
            for job in jobs: complete(job)
            result['parent_hashes']={s['payload_file']:sha(s['payload_file']) for s in result['roots']}
        elif args.action=='resume':
            prepared=json.loads((args.output/'prepare.json').read_text())
            if prepared['status']!='PASS': raise ValueError('Prepare must pass before restart')
            first={row['operation'].split('-tree-')[1].split('-')[0]:row
                for row in prepared['requests'] if '-tree-' in row['operation']}
            jobs=[]
            result['already_stopped']=[]
            for tree in range(args.trees):
                initial=first[str(tree)]
                count=continuation_limit(initial['state'],len(initial['answer']['tokens']),args.generated)
                if count:
                    jobs.append(coordinator.submit('tree-'+str(tree),(),count,
                        'client-'+str(tree//2)+'-tree-'+str(tree)+'-resume'))
                else:
                    # Validate retained bytes and controller in the new process.
                    # No inference is requested or claimed for an ended turn.
                    sid=coordinator.store.head('tree-'+str(tree)).tip
                    state=get_state(coordinator,sid)
                    if state['controller']!=initial['state']['controller']:
                        raise AssertionError('Stopped controller changed across restart')
                    result['already_stopped'].append(tree)
                    result['requests'].append(dict(operation='client-'+str(tree//2)+'-tree-'+str(tree)+'-resume',
                        mode='already-stopped/no-inference-request',answer=dict(snapshot=sid,tokens=[],counter_deltas=None),
                        state=state))
            if not jobs: raise AssertionError('Fixture exercised no generated restart continuation')
            for job in jobs: complete(job)
            if any(sha(p)!=digest for p,digest in prepared['parent_hashes'].items()):
                raise AssertionError('Saved parent disk payload changed')
            result['parent_bytes_unchanged']=True
        else:
            prepared=json.loads((args.output/'prepare.json').read_text())
            resumed=json.loads((args.output/'resume.json').read_text())
            if prepared['status']!='PASS' or resumed['status']!='PASS':
                raise ValueError('Rewind requires completed prepare/resume phases')
            result['rewinds']=[]
            for root_index in range(args.roots):
                trees=[t for t in range(args.trees) if tree_root(fixture,t,args.roots)==root_index]
                if len(trees)!=2: raise ValueError('Rewind trace requires two questions per root')
                a,b=trees
                branch='tree-'+str(a)
                before=coordinator.store.head(branch)
                sibling=coordinator.fork('saved-A-'+str(root_index),before.tip)
                root=prepared['roots'][root_index]['snapshot']
                receipt=coordinator.rewind(branch,before.revision,root,'rewind-'+str(root_index))
                after=coordinator.store.head(branch)
                validate_rewind(asdict(before),asdict(after),root,asdict(sibling),
                                asdict(coordinator.store.head(sibling.name)))
                row=complete(coordinator.submit(branch,fixture['suffixes'][b],args.generated,
                    'source-'+str(root_index)+'-changed',expected_revision=after.revision))
                if coordinator.store.head(sibling.name)!=sibling:
                    raise AssertionError('Continuation changed retained A sibling')
                result['rewinds'].append(rewind_record(root_index,b,before,after,receipt,sibling,
                                                       row['state']['snapshot']))
            if any(sha(p)!=digest for p,digest in prepared['parent_hashes'].items()):
                raise AssertionError('Rewind campaign mutated saved root payload')
            result['parent_bytes_unchanged']=True
        result['pool']=coordinator.metrics()
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=str(exc));raise
    finally:
        _finish_phase(result,destination,args.action,coordinator,transition_trace,
                      stop,thread,start)


def campaign(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if args.fixture is not None:
        from build_codebase_fixture import normalize_context_fixture
        config=json.loads(args.config.read_text())
        profile=json.loads(Path(config['profile']).read_text())
        fixture=normalize_context_fixture(json.loads(args.fixture.read_text()),
            profile['execution']['capacity'],args.generated)
        args.roots=len(fixture['prefixes']);args.trees=len(fixture['suffixes'])
        args.prefix=max(map(len,fixture['prefixes']));args.suffix=max(map(len,fixture['suffixes']))
    else:
        fixture=dict(prefixes=[[100+r*40+i%17 for i in range(args.prefix)] for r in range(args.roots)],
            suffixes=[[117+t*3+i%11 for i in range(args.suffix)] for t in range(args.trees)])
    publish(args.output/'fixture.json',fixture)
    config=json.loads(args.config.read_text())
    if config.get('require_spill_reload_same_process') and args.pressure_order!='interleaved-roots':
        raise ValueError('Same-process spill/reload evidence requires interleaved-roots ordering')
    if args.exercise_rewind and fixture.get('tree_roots') != [r for r in range(args.roots) for _ in range(2)]:
        raise ValueError('Rewind campaign requires ordered A/B pairs per distinct source root')
    record=dict(status='RUNNING',scope='real model pool campaign; matching native operators, greedy only',
        sizes=dict(prefix=args.prefix,suffix=args.suffix,output_limit=args.generated,trees=args.trees,roots=args.roots),
        phases={},comparisons=[],numerical_max_error=0.,harness_sha256=sha(__file__),
        pressure_order=args.pressure_order)
    record['actual_token_counts']=dict(prefixes=list(map(len,fixture['prefixes'])),
                                     suffixes=list(map(len,fixture['suffixes'])))
    record['rewind_comparisons']=[]
    publish(args.output/'result.json',record)
    try:
        for action in ('prepare','resume')+(('rewind',) if args.exercise_rewind else ()):
            command=[sys.executable,str(Path(__file__).resolve()),action,'--config',str(args.config),
                '--output',str(args.output),'--prefix',str(args.prefix),'--suffix',str(args.suffix),
                '--generated',str(args.generated),'--trees',str(args.trees),'--roots',str(args.roots),
                '--pressure-order',args.pressure_order]
            began=time.monotonic()
            with (args.output/(action+'.stderr')).open('wb') as err:
                child=subprocess.Popen(command,stderr=err)
                code=child.wait(timeout=1800)
            phase_record=json.loads((args.output/(action+'.json')).read_text())
            record['phases'][action]=dict(python_pid=child.pid,observed_exit=child.poll() is not None,
                exit_code=code,seconds=time.monotonic()-began,native_pids=phase_record.get('native_pids'),
                native_exit_codes=phase_record.get('native_exit_codes'),
                native_exit_observed=phase_record.get('native_exit_observed'),
                native_exit_success=phase_record.get('native_exit_success'))
            publish(args.output/'result.json',record)
            native_codes=phase_record.get('native_exit_codes')
            if (code or phase_record['status']!='PASS' or not phase_record.get('native_exit_observed')
                    or phase_record.get('native_exit_success') is not True
                    or not isinstance(native_codes,dict) or not native_codes
                    or any(type(value) is not int or value != 0 for value in native_codes.values())):
                raise AssertionError(action+' did not complete and close all workers')
        prepared=json.loads((args.output/'prepare.json').read_text())
        resumed=json.loads((args.output/'resume.json').read_text())
        profile=json.loads((args.output/'profile.json').read_text())
        for key in list(os.environ):
            if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        first={row['operation'].split('-tree-')[1].split('-')[0]:row
            for row in prepared['requests'] if '-tree-' in row['operation']}
        second={row['operation'].split('-tree-')[1].split('-')[0]:row for row in resumed['requests']}
        for tree in range(args.trees):
            prefix=fixture['prefixes'][tree_root(fixture,tree,args.roots)];suffix=fixture['suffixes'][tree]
            fresh=reference(profile,args.output,'fresh-tree-'+str(tree),prefix+suffix,
                [len(prefix),len(suffix)],args.generated)
            a,b=first[str(tree)],second[str(tree)]
            tokens=a['answer']['tokens']+b['answer']['tokens']
            pairs=[(a['state'],fresh['generated_frontiers'][len(a['answer']['tokens'])-1]),
                (b['state'],fresh['generated_frontiers'][-1])]
            for state,expected in pairs:
                actual=Path(state['frontier_file']).read_bytes()
                if hashlib.sha256(actual).hexdigest()!=state['frontier_sha256']:
                    raise AssertionError('Frontier audit file changed')
                if actual!=expected:
                    av,bv=_unpack_frontier(actual,248320),_unpack_frontier(expected,248320)
                    record['numerical_max_error']=max(record['numerical_max_error'],max(abs(x-y) for x,y in zip(av,bv)))
                    (args.output/'mismatch-expected.f32').write_bytes(expected)
                    (args.output/'mismatch-actual.f32').write_bytes(actual)
                    raise AssertionError('Matching native frontier mismatch for tree '+str(tree))
            if tokens!=fresh['tokens']: raise AssertionError('Matching native tokens differ for tree '+str(tree))
            if args.exercise_rewind and tree%2==1:
                rewound=json.loads((args.output/'rewind.json').read_text())
                row=rewound['requests'][tree//2]
                if row['answer']['tokens']!=fresh['tokens'] or Path(row['state']['frontier_file']).read_bytes()!=fresh['generated_frontiers'][-1]:
                    raise AssertionError('Rewound changed question disagrees with fresh matching reference')
                record['rewind_comparisons'].append(dict(tree=tree,status='PASS',
                    outputs=len(row['answer']['tokens']),max_absolute_error=0.,full_frontier_checks=1))
            if b['answer']['tokens'] and b['answer']['counter_deltas']['forward_tokens']!=len(b['answer']['tokens']):
                raise AssertionError('Restart continuation recomputed a prefix')
            record['comparisons'].append(dict(tree=tree,client=tree//2,status='PASS',outputs=len(tokens),
                resumed_outputs=len(b['answer']['tokens']),already_stopped=not b['answer']['tokens'],
                checked_output_boundaries=sorted({len(a['answer']['tokens']),len(tokens)}),
                full_frontier_checks=2,max_absolute_error=0.,oracle_work=fresh['work']))
            publish(args.output/'result.json',record)
            print(json.dumps(dict(event='reference_pass',tree=tree,outputs=len(tokens))),flush=True)
        phase_pools={action:json.loads((args.output/(action+'.json')).read_text())['pool']
                     for action in record['phases']}
        if any(any(n!=1 for n in pool['max_active_per_gpu'].values()) for pool in phase_pools.values()):
            raise AssertionError('Worker did not execute exactly one active request at a time')
        record['phase_pools']=phase_pools
        record['clean_evictions']=sum(pool['ram']['clean_evictions'] for pool in phase_pools.values())
        if json.loads(args.config.read_text()).get('require_clean_evictions') and not record['clean_evictions']:
            raise AssertionError('Required RAM eviction was not exercised')
        config=json.loads(args.config.read_text())
        if config.get('trace_pool_transitions') or config.get('require_spill_reload_same_process'):
            for action in record['phases']:
                trace_path=args.output/(action+'-pool-transitions.json')
                trace=json.loads(trace_path.read_text())
                if trace.get('overflowed') or trace.get('event_count')!=len(trace.get('events',[])):
                    raise AssertionError(action+' pool-transition trace overflowed or is incomplete')
        if config.get('require_spill_reload_same_process'):
            trace_path=args.output/'prepare-pool-transitions.json'
            trace=json.loads(trace_path.read_text())
            record['same_process_spill_reload_keys']=validate_spill_reload_trace(trace)
            record['executed_test_ids']=['POOL-REAL-SAME-PROCESS-EVICT-RELOAD']
        record.update(status='PASS',profile_id=profile['execution_profile_id'],
            model_contract_id=profile['model_contract_id'],model_weights_sha256=profile['model_weights_sha256'],
            build_sha256=profile['native_build']['sha256'],hardware=profile['hardware'],
            executed_test_ids=record.get('executed_test_ids',[])+['POOL-REAL-ROOTS',('POOL-REAL-EIGHT-TREES' if args.trees==8 else 'POOL-REAL-SOURCE-TREES'),'POOL-REAL-RESTART',
                'POOL-REAL-ZERO-PREFIX-WORK','POOL-REAL-PARENT-BYTES','POOL-REAL-MATCHING-REFERENCES'],
            failures=[],skips=[],xfails=[])
        if args.exercise_rewind: record['executed_test_ids'].append('POOL-REAL-REWIND-CHANGED-QUESTION')
    except BaseException as exc:
        record.update(status='FAIL',error=str(exc));raise
    finally: publish(args.output/'result.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['campaign','prepare','resume','rewind'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',type=int,default=128);parser.add_argument('--suffix',type=int,default=4)
    parser.add_argument('--generated',type=int,default=8);parser.add_argument('--trees',type=int,default=8)
    parser.add_argument('--roots',type=int,default=2)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--exercise-rewind',action='store_true')
    parser.add_argument('--pressure-order',choices=['default','interleaved-roots'],default='default')
    args=parser.parse_args()
    if min(args.prefix,args.suffix,args.trees,args.roots)<=0 or args.generated<2: parser.error('Invalid fixture sizes')
    (campaign if args.action=='campaign' else phase)(args)
