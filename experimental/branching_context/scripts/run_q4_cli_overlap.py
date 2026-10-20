"""Real CLI publication overlap, with controlled proof separate from timings.

The controlled mode pauses one publication before the original writer runs.
Benchmark mode uses the unchanged server and never deliberately holds a save.
Both modes use one native GPU worker and canonical immutable branches.
"""
import argparse,base64,codecs,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from urllib.parse import urlencode

def need(condition,message):
    if not condition:raise AssertionError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while raw:=stream.read(8*1024**2):digest.update(raw)
    return digest.hexdigest()
def publish(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
def events(trace):
    return [(row['arrival_monotonic_ns'],json.loads(base64.b64decode(row['raw_line_b64'],validate=True)))
            for row in trace['requests'][0]['events']]
def validate_controlled_overlap(admission,sources,observations,holds,pending):
    need(len(admission)==4 and len(set(admission))==4,'Need four admitted clients')
    need([row['operation_id'] for row in sources]==admission,'Native source acquisition violated FIFO')
    entered=[row for row in holds if row['event']=='publication_enter']
    released=[row for row in holds if row['event']=='publication_release']
    need(len(entered)==len(released)==1 and entered[0]['operation_id']==released[0]['operation_id']==admission[0],'Ambiguous or incorrect publication hold')
    begin,end=entered[0]['monotonic_ns'],released[0]['monotonic_ns']
    need(begin<end,'Publication hold clock order')
    first_later=[stamp for stamp,event in events(observations[admission[1]]) if event['event'] in ('text','turn_complete')]
    need(first_later and begin<first_later[0]<end,'No later GPU token while publication held')
    for operation in admission:
        row=pending[operation]
        need(row['durable'] is False and row['branch_unchanged'] is True and row['client_exit'] is None,'Request was not pending before publication release')
        stamps=[stamp for stamp,event in events(observations[operation]) if event['event']=='durable']
        need(len(stamps)==1 and stamps[0]>=end,'False or early durable acknowledgment')
    return dict(later_gpu_token_before_first_publication_release=True,first_publication_held_ns=end-begin,
                later_first_token_monotonic_ns=first_later[0],publication_release_monotonic_ns=end,
                FIFO_source_order=admission,no_early_durable_ack=True)


BOOTSTRAP=r'''import hashlib,importlib,json,os,sys,time
from pathlib import Path
profile=json.loads(Path(sys.argv[1]).read_bytes());rows={}
for name in ('context_time_travel.coordinator','context_time_travel.strata_backend','context_time_travel.real','context_time_travel.store','serve.cow_http'):
 module=importlib.import_module(name);path=Path(module.__file__).resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest();assert profile['native_build']['sha256'][str(path)]==digest;rows[name]={'path':str(path),'sha256':digest}
print(json.dumps({'event':'runtime_imports','modules':rows}),flush=True)
configuration=json.loads(Path(sys.argv[2]).read_bytes())
if configuration['mode']=='controlled':
 from context_time_travel.coordinator import ContextCoordinator
 original=ContextCoordinator._publish
 def observed(self,model,job,prepared,key,pin,lease):
  if job.request.operation==configuration['hold_operation']:
   event_path=Path(configuration['hold_events'])
   def record(event):
    with event_path.open('a') as stream:stream.write(json.dumps({'event':event,'operation_id':job.request.operation,'monotonic_ns':time.monotonic_ns()})+'\n');stream.flush();os.fsync(stream.fileno())
   record('publication_enter')
   deadline=time.monotonic()+90
   while not Path(configuration['release_file']).exists():
    if time.monotonic()>=deadline:raise TimeoutError('Controlled publication hold timed out')
    time.sleep(.01)
   record('publication_release')
  return original(self,model,job,prepared,key,pin,lease)
 ContextCoordinator._publish=observed
sys.argv=['cow-http',*sys.argv[3:]]
importlib.import_module('serve.cow_http').main()
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','profile','fixtures','store','observer','run-dir','reference-result'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--observer-sha256',required=True)
    parser.add_argument('--mode',choices=('controlled','benchmark'),required=True)
    parser.add_argument('--outputs',type=int,default=128)
    args=parser.parse_args()
    need(1<=args.outputs<=128,'Output bound')
    runtime=args.runtime_root.resolve(strict=True);profile=read(args.profile);fixtures=read(args.fixtures)
    baseline=read(args.reference_result)
    need(sha(args.profile)==fixtures['profile_sha256']==baseline['profile_sha256'],'Profile binding differs')
    need(sha(args.fixtures)==baseline['fixture_set_sha256'] and baseline['status']=='PASS','Require qualified exact reference fixtures')
    need(profile['execution']['GPU']==[0] and profile['execution']['resume_contract']=='identical-native-operators-v1','Wrong profile')
    need(sha(args.observer)==args.observer_sha256,'Observer changed')
    need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy')
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),str(runtime/'experimental/branching_context/scripts')]
    from run_q4_publication_fault import _runtime_closure
    from run_cow_client_recovery import ServiceProcess,ProgressTracker,HttpClient,_ready_url,_trace_worker,checkpoint_fingerprint
    from run_root_reuse_probe import expected_controller,compare_vectors
    from context_time_travel.store import Store
    from tools.strata_tokenizer import Tokenizer
    from tools.cow_cli import safe_text
    # This helper is explicitly staged beside this runner, outside the pinned runtime.
    import run_q4_cli_trees as cli_tools
    need(sha(Path(cli_tools.__file__))=='3a6b2b2ebb9d494753fbdd68b0b5e0914c117958c1dc0eb012aaac27cd776ea2','Harness helper changed')
    validate_observation=cli_tools.validate_observation
    closure=_runtime_closure(profile,runtime)
    cli=runtime/'tools/cow_cli.py';need(sha(cli)==fixtures['imports']['tools.cow_cli']['sha256'],'CLI differs')
    cases=[]
    for item in fixtures['contexts']:
        need(sha(item['fixture'])==item['fixture_sha256'],'Fixture changed');cases.append(read(item['fixture']))
    output=args.run_dir.absolute();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    (output/'calls').mkdir(mode=0o700)
    run=uuid.uuid4().hex[:10];hold_operation='op-overlap-controlled-0-'+run
    release_file=output/'release-publication';hold_events=output/'hold-events.jsonl'
    configuration=dict(mode=args.mode,hold_operation=hold_operation,release_file=str(release_file),hold_events=str(hold_events))
    publish(output/'configuration.json',configuration)
    bootstrap=output/'server_bootstrap.py';bootstrap.write_text(BOOTSTRAP,encoding='utf-8',newline='\n')
    tracker=ProgressTracker(output,'cli-publication-'+args.mode).start()
    result=dict(schema='cow-cli-publication-overlap-v1',status='RUNNING',mode=args.mode,outputs=args.outputs,
        profile_sha256=sha(args.profile),execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
        cli_sha256=sha(cli),observer_sha256=args.observer_sha256,runner_sha256=sha(Path(__file__)),
        bootstrap_sha256=sha(bootstrap),reference_result_sha256=sha(args.reference_result),fixture_set_sha256=sha(args.fixtures),
        reference_scope='Reuses independently initialized uncached references from qualified identical fixtures/profile; no new oracle computation',
        numerical_max_error=0,comparisons=0,roots={},generations=[],groups=[],failures=[],runtime_source_closure=closure,
        controller_scope='Greedy only',ram_gib=96,disk_gib=128,hold_scope='First publication paused before original writer; controlled mode is not a speed measurement')
    result_path=output/'result.json';publish(result_path,result)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join((str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'))))
    native_trace=output/'native.jsonl';native_trace.touch(mode=0o600)
    server_env=dict(env,CTT_NATIVE_TRACE=str(native_trace))
    command=[sys.executable,'-B',str(bootstrap),str(args.profile.resolve()),str(output/'configuration.json'),
        '--profile',str(args.profile.resolve()),'--store',str(args.store.absolute()),'--host','127.0.0.1','--port','0','--text',
        '--ram-gib','96','--disk-gib','128','--pool-trace-output',str(output/'pool.json')]
    service=ServiceProcess(command,cwd=runtime,env=server_env,stdout_path=output/'server.stdout',stderr_path=output/'server.stderr',timeout=180,progress=tracker)
    store=Store(args.store);tokenizer=Tokenizer.from_gguf(profile['model_path']);owned=[];client=None
    def wait_for(predicate,label,timeout=40):
        tracker.phase(label);deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            value=predicate()
            if value:return value
            need(service.process.poll() is None,'Service exited while waiting')
            time.sleep(.01)
        raise TimeoutError(label)
    def cli_command(index,arguments,trace=None):
        base=[sys.executable,'-B',str(cli)]
        if trace is not None:base=[sys.executable,'-B',str(args.observer),'--runtime-root',str(runtime),'--trace-file',str(trace),'--']
        return base+['--url',url,'--state-dir',str(output/('client-'+str(index)+'-journal')),'--timeout','180',*arguments]
    def fork(index,branch,operation):
        case=cases[index%3]
        argv=cli_command(index,['fork','--source-branch',case['branch'],'--name',branch,'--operation-id',operation])
        process=subprocess.run(argv,cwd=runtime,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
        (output/'calls'/(branch+'-fork.stderr')).write_bytes(process.stderr)
        need(process.returncode==0,'CLI fork failed: '+process.stderr.decode('utf-8','replace')[-500:])
        head=store.head(branch);need(head.tip==result['roots'][str(index%3)]['head']['tip'],'Fork selected another root')
        return head
    def wait_client(item):
        if 'finished_ns' not in item:
            code=item['process'].wait(timeout=180)
            item['finished_ns']=time.monotonic_ns()
            item['stdout'].close();item['stderr'].close()
            need(code==0,'CLI request failed: '+item['stderr_path'].read_text()[-1200:])
    def collect(item):
        wait_client(item)
        trace=read(item['trace']);need(trace['cli_exit_code']==0,'Observer CLI failed')
        decoded=events(trace);complete=[event for _,event in decoded if event['event']=='complete']
        need(len(complete)==1,'Missing completion')
        receipt=complete[0]['receipt'];answer=complete[0]['answer'];tokens=answer['tokens']
        validate_observation(trace,result['cli_sha256'],item['branch'],item['operation'],receipt,tokens)
        case=cases[item['context']];suffix=case['branches'][item['question']]['suffix_token_ids'];prompt=case['root_token_ids']+suffix
        head=store.head(item['branch']);snapshot=store.get_snapshot(head.tip)
        need(head.tip==receipt['snapshot']==answer['snapshot'] and head.revision==item['head'].revision+1,'Wrong durable branch head')
        need(snapshot['parent']==item['head'].tip and snapshot['inputs']==suffix+tokens,'Wrong canonical child')
        need(store.lineage(head.tip)[1]==prompt+tokens,'Wrong canonical history')
        expected=expected_controller(item['operation'],prompt,tokens,tokenizer,profile['execution']['eos_tokens'],248320)
        need(snapshot['controller']==expected,'Controller mismatch')
        counters=answer['counter_deltas'];need(counters['forward_tokens']==len(suffix)+len(tokens) and counters['prefill_tokens']==len(suffix),'Saved prefix recomputed')
        manifest=read(args.store/'checkpoints'/(head.tip+'.json'))['manifest'];payload=args.store/'checkpoints'/(manifest['payload_sha256']+'.state')
        with payload.open('rb') as stream:stream.seek(manifest['state_bytes']);frontier=stream.read()
        name=f'tree{item["context"]}-{item["question"]}-fresh0';saved=next(row for row in baseline['references'] if row['name']==name)
        refdir=args.reference_result.parent/name
        need((refdir/(name+'.tokens')).read_text()==','.join(map(str,prompt)),'Reference input differs')
        need(saved['schedule']['requests']==[len(case['root_token_ids']),len(suffix)],'Reference schedule differs')
        need(tokens==saved['tokens'][:len(tokens)] and 0<len(tokens)<=args.outputs,'Tokens differ from qualified independent reference')
        with (refdir/(name+'.f32.decode.f32')).open('rb') as stream:stream.seek((len(tokens)-1)*993280);reference=stream.read(993280)
        need(hashlib.sha256(reference).hexdigest()==saved['generated_frontier_sha256'][len(tokens)-1],'Reference frontier changed')
        error=compare_vectors(frontier,reference,item['branch']+'-final',output);result['comparisons']+=1;result['numerical_max_error']=max(result['numerical_max_error'],error)
        (output/'calls'/(item['branch']+'.f32')).write_bytes(frontier)
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');pieces=[]
        for token in tokens:
            if token in profile['execution']['eos_tokens']:pieces.append(decoder.decode(b'',final=True));break
            pieces.append(decoder.decode(tokenizer.token_bytes(token),final=False))
        need(item['stdout_path'].read_text()==safe_text(''.join(pieces))+'\n','CLI text differs from consumed tokens')
        row=dict(branch=item['branch'],operation=item['operation'],group=item['group'],context=item['context'],question=item['question'],
            snapshot=head.tip,source_snapshot=item['head'].tip,outputs=len(tokens),tokens=tokens,controller_exact=True,history_exact=True,
            final_frontier_sha256=hashlib.sha256(frontier).hexdigest(),trace=str(item['trace']),counter_deltas=counters,
            arrival_summary=trace['requests'][0]['summary'],engine_generation_seconds=answer['generation_seconds'],
            cli_process_start_ns=item['started_ns'],cli_process_end_ns=item['finished_ns'])
        result['generations'].append(row);publish(result_path,result)
        return trace
    def group(label,queued,held=False):
        specs=[(0,'A'),(1,'A'),(2,'A'),(0,'B')];items=[];admission=[];pending={}
        # Precreate every branch outside the measured request group.
        for index,(context,question) in enumerate(specs):
            branch='Ov'+label+str(index)+run;operation=(hold_operation if held and index==0 else 'op-overlap-'+label+'-'+str(index)+'-'+run)
            # Client 3 uses context 0, independently of the first client.
            head=fork(index,branch,'op-fork-'+branch)
            items.append(dict(context=context,question=question,branch=branch,operation=operation,head=head,client=index,group=label))
        group_start=time.monotonic_ns();observations={}
        for index,item in enumerate(items):
            tracker.phase(label+'-launch-'+str(index));prefix=output/'calls'/item['branch']
            item.update(trace=prefix.with_suffix('.trace.json'),stdout_path=prefix.with_suffix('.stdout'),stderr_path=prefix.with_suffix('.stderr'))
            item['stdout']=item['stdout_path'].open('xb');item['stderr']=item['stderr_path'].open('xb')
            argv=cli_command(item['client'],['ask','--branch',item['branch'],'--text',cases[item['context']]['branches'][item['question']]['question'],
                '--max-tokens',str(args.outputs),'--operation-id',item['operation']],item['trace'])
            item['started_ns']=time.monotonic_ns();item['process']=subprocess.Popen(argv,cwd=runtime,env=env,stdout=item['stdout'],stderr=item['stderr'])
            owned.append(item)
            if queued:
                def admitted():
                    code,value=client.get('/cow/operation?'+urlencode(dict(operation=item['operation'])))
                    return value if code==200 else None
                state=wait_for(admitted,label+'-admit-'+str(index));need(state['operation_id']==item['operation'],'Wrong admitted operation');admission.append(item['operation'])
                if held and index==0:
                    wait_for(lambda:hold_events.exists() and 'publication_enter' in hold_events.read_text(),'first-publication-entered')
            else:
                wait_client(item);admission.append(item['operation'])
        if held:
            def all_consumed():
                states={item['operation']:client.get('/cow/operation?'+urlencode(dict(operation=item['operation']))) for item in items}
                return states if all(code==200 and value['state']=='persisting' for code,value in states.values()) else None
            states=wait_for(all_consumed,'all-four-consumed-before-first-save')
            for item in items:
                state=states[item['operation']][1]
                pending[item['operation']]=dict(durable=state['durable'],branch_unchanged=store.head(item['branch'])==item['head'],client_exit=item['process'].poll())
            result['pending_before_release']=pending;publish(result_path,result)
            release_file.touch(mode=0o600,exist_ok=False)
        if queued:
            for item in items:wait_client(item)
        elapsed=time.monotonic_ns()-group_start
        # Numerical/controller inspection is deliberately outside group timing.
        for item in items:observations[item['operation']]=collect(item)
        row=dict(label=label,queued=queued,deliberate_hold=held,seconds=elapsed/1e9,admission_order=admission,
                 outputs=sum(result['generations'][-len(items)+i]['outputs'] for i in range(len(items))),
                 operation_traces={item['operation']:str(item['trace']) for item in items})
        row['outputs_per_second_to_all_durable']=row['outputs']/row['seconds']
        result['groups'].append(row);publish(result_path,result)
        return row,observations
    failure=None
    try:
        started=time.monotonic();ready=service.start();url=_ready_url(ready);client=HttpClient(url,timeout=30)
        need(ready['execution_profile_id']==profile['execution_profile_id'],'Ready profile differs')
        result.update(http_pid=service.process.pid,service_startup_seconds=time.monotonic()-started)
        for index,case in enumerate(cases):
            head=store.head(case['branch']);need(head.tip==head.root and store.lineage(head.tip)[1]==case['root_token_ids'],'Root history differs')
            result['roots'][str(index)]=dict(branch=case['branch'],head=vars(head),checkpoint_before=checkpoint_fingerprint(args.store,head.tip))
        publish(result_path,result)
        if args.mode=='controlled':
            controlled,observations=group('controlled',True,True)
            result['_controlled_observations']=observations
        else:
            group('warmup',False)
            for repeat in range(3):
                for queued in ((False,True) if repeat%2==0 else (True,False)):
                    group(('queued' if queued else 'serial')+str(repeat),queued)
        code,status=client.get('/cow/status');need(code==200 and status['max_active_per_gpu']=={'0':1} and status['queued']==0,'Single GPU scheduling invariant')
        need(not any(status['active_per_gpu'].values()),'GPU still active after clients completed');result['final_status']=status
        receipt=service.stop();need(receipt['exit_code']==0 and not Path('/proc',str(service.process.pid)).exists(),'HTTP did not exit')
        result['service_stop']=receipt;service=None
        worker=_trace_worker(output/'pool.json',result['http_pid']);need(worker['native_exit_code']==0 and not Path('/proc',str(worker['native_pid'])).exists(),'Native did not exit');result['worker']=worker
        pool=read(output/'pool.json');need(not pool['overflowed'],'Pool trace overflow')
        acquired=[event for event in pool['events'] if event['event']=='request_source_acquired']
        for row in result['groups']:
            selected=[event for event in acquired if event.get('operation_id') in row['admission_order']]
            need([event['operation_id'] for event in selected]==row['admission_order'],'GPU source order violated FIFO')
        if args.mode=='controlled':
            holds=[json.loads(line) for line in hold_events.read_text().splitlines()]
            result['controlled_witness']=validate_controlled_overlap(result['groups'][0]['admission_order'],acquired,
                result.pop('_controlled_observations'),holds,result['pending_before_release'])
        else:
            windows=[]
            for row in result['groups']:
                if not row['queued']:continue
                traces=[read(row['operation_traces'][operation]) for operation in row['admission_order']]
                for before,after in zip(traces,traces[1:]):
                    durable=next(stamp for stamp,event in events(before) if event['event']=='durable')
                    first=next(stamp for stamp,event in events(after) if event['event'] in ('text','turn_complete') and 'token_id' in event)
                    windows.append(dict(earlier_operation=before['requests'][0]['operation_id'],later_operation=after['requests'][0]['operation_id'],
                        later_first_token_before_earlier_durable=first<durable,overlap_seconds=(durable-first)/1e9))
            result['natural_overlap_windows']=windows
        for index,row in result['roots'].items():
            need(vars(store.head(row['branch']))==row['head'],'Original root branch moved')
            need(checkpoint_fingerprint(args.store,row['head']['tip'])['files']==row['checkpoint_before']['files'],'Original root bytes changed')
            row['unchanged']=True
        result['status']='PASS'
    except BaseException as exc:
        failure=exc;result['status']='FAIL';result['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        if hasattr(exc,'maximum'):result['numerical_max_error']=exc.maximum
    finally:
        if args.mode=='controlled' and not release_file.exists():release_file.touch(mode=0o600)
        for item in owned:
            if item['process'].poll() is None:
                item['process'].terminate()
                try:item['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait(timeout=10)
            item['stdout'].close();item['stderr'].close()
        if service is not None:
            try:result['cleanup_stop']=service.stop()
            except BaseException as exc:
                result['failures'].append(dict(type=type(exc).__name__,message='cleanup: '+str(exc)));result['status']='FAIL'
        tracker.finish(result['status'],str(failure) if failure else None);tracker.close();publish(result_path,result)
    print(json.dumps({key:result[key] for key in ('status','mode','comparisons','numerical_max_error','failures')}),flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
