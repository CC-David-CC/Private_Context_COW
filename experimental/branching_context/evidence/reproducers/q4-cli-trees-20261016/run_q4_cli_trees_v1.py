"""Bounded serial terminal-client workload against the pinned COW service.

This is a measurement harness, not a new history or serving implementation.
Fresh numerical qualification runs separately after the service exits.
"""
from __future__ import annotations
import argparse,codecs,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def need(value,message):
    if not value:raise AssertionError(message)
def publish(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','profile','fixtures','run-dir','store'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','campaign'),required=True)
    p.add_argument('--observer',type=Path)
    p.add_argument('--observer-sha256')
    p.add_argument('--outputs',type=int,default=128)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--preflight-only',action='store_true')
    return p

def main(argv=None):
    a=args_parser().parse_args(argv)
    runtime=a.runtime_root.resolve(strict=True);profile=read(a.profile);fixtures=read(a.fixtures)
    need(sha(a.profile)==fixtures['profile_sha256'],'Fixture profile checksum')
    need(profile['execution']['GPU']==[0] and profile['execution']['resume_contract']=='identical-native-operators-v1','Unsupported execution profile')
    need(profile['execution']['generated_operator']=='normal-verifier-decode','Wrong generated operator')
    need(1<=a.outputs<=128 and 1<=a.repetitions<=3,'Bounded output/repetition limits')
    need(len(fixtures['contexts'])==3,'Need three context trees')
    cli=runtime/'tools/cow_cli.py'
    need(sha(cli)==fixtures['imports']['tools.cow_cli']['sha256'],'CLI source changed')
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),str(runtime/'experimental/branching_context/scripts')]
    from run_q4_publication_fault import _runtime_closure
    closure=_runtime_closure(profile,runtime)
    from context_time_travel.store import Store,NotFound
    from run_cow_client_recovery import ServiceProcess,ProgressTracker,HttpClient,_ready_url,_trace_worker,checkpoint_fingerprint
    from run_root_reuse_probe import expected_controller
    from tools.strata_tokenizer import Tokenizer
    from tools.cow_cli import safe_text
    cases=[]
    for entry in fixtures['contexts']:
        need(sha(entry['fixture'])==entry['fixture_sha256'],'Fixture file changed')
        case=read(entry['fixture']);need(sha(case['context_file'])==case['context_sha256'],'Context text changed');cases.append(case)
    if a.phase=='campaign':
        need(a.observer is not None and sha(a.observer)==a.observer_sha256,'Observer checksum mismatch')
    if a.preflight_only:
        need(not a.run_dir.exists(),'Preflight output exists')
        print(json.dumps(dict(status='PREPARED_NOT_RUN',phase=a.phase,root_tokens=[c['root_tokens'] for c in cases],profile_sha256=sha(a.profile),cli_sha256=sha(cli))))
        return 0
    need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU compute process already present')
    output=a.run_dir.absolute();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    (output/'calls').mkdir(mode=0o700)
    tracker=ProgressTracker(output,'serial-cli-'+a.phase).start()
    record=dict(schema='cow-serial-cli-trees-v1',status='RUNNING',phase=a.phase,
        profile_sha256=sha(a.profile),execution_profile_id=profile['execution_profile_id'],
        model_contract_id=profile['model_contract_id'],native_sha256=sha(Path(profile['execution']['sut_command'][0])),
        cli_sha256=sha(cli),runner_sha256=sha(Path(__file__)),fixture_set_sha256=sha(a.fixtures),
        observer_sha256=a.observer_sha256,store=str(a.store.absolute()),calls=[],generations=[],services=[],roots={},
        output_limit=a.outputs,repetitions=a.repetitions,source_scope=fixtures['source_scope'],
        numerical_qualification='PENDING_SEPARATE_FRESH_REFERENCE',filesystem_cache='warm/uncontrolled; no drop_caches',
        ram_gib=96,disk_gib=128,failures=[],source_imports={})
    result_path=output/'result.json';publish(result_path,record)
    run_id=uuid.uuid4().hex[:10];service=None;client=None;ordinal=0;store=None;counter=0
    tokenizer=Tokenizer.from_gguf(profile['model_path'])
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=os.pathsep.join((str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src')))
    bootstrap=output/'server_bootstrap.py'
    bootstrap.write_text('import hashlib,importlib,json,sys\nfrom pathlib import Path\np=json.loads(Path(sys.argv[1]).read_bytes())\nrows={}\nfor name in ("context_time_travel.coordinator","context_time_travel.strata_backend","context_time_travel.real","context_time_travel.store","serve.cow_http"):\n m=importlib.import_module(name);f=Path(m.__file__).resolve();h=hashlib.sha256(f.read_bytes()).hexdigest();assert p["native_build"]["sha256"][str(f)]==h;rows[name]={"path":str(f),"sha256":h}\nprint(json.dumps({"event":"runtime_imports","modules":rows}),flush=True)\nsys.argv=["cow-http",*sys.argv[2:]]\nimportlib.import_module("serve.cow_http").main()\n',encoding='utf-8',newline='\n')
    def stats():
        if ordinal==0:return {}
        path=output/('native-'+str(ordinal)+'.jsonl')
        values=[read_line['response'] for line in path.read_text().splitlines() if line and (read_line:=json.loads(line)).get('command')=='stats']
        return values[-1] if values else {}
    def status():
        code,value=client.get('/cow/status');need(code==200,'Status request failed');return value
    def start():
        nonlocal service,client,ordinal,store
        ordinal+=1;tracker.phase('start_service_'+str(ordinal))
        native_trace=output/('native-'+str(ordinal)+'.jsonl');native_trace.touch(mode=0o600)
        server_env=dict(env,CTT_NATIVE_TRACE=str(native_trace))
        command=[sys.executable,'-B',str(bootstrap),str(a.profile.resolve()),'--profile',str(a.profile.resolve()),'--store',str(a.store.absolute()),'--host','127.0.0.1','--port','0','--text','--ram-gib','96','--disk-gib','128','--pool-trace-output',str(output/('pool-'+str(ordinal)+'.json'))]
        t=time.monotonic();service=ServiceProcess(command,cwd=runtime,env=server_env,stdout_path=output/('server-'+str(ordinal)+'.stdout'),stderr_path=output/('server-'+str(ordinal)+'.stderr'),timeout=180,progress=tracker)
        ready=service.start();need(ready['execution_profile_id']==profile['execution_profile_id'],'Ready execution identity')
        client=HttpClient(_ready_url(ready),timeout=180);store=Store(a.store)
        rows=[json.loads(line) for line in (output/('server-'+str(ordinal)+'.stdout')).read_text().splitlines() if line.startswith('{')]
        imports=[r['modules'] for r in rows if r.get('event')=='runtime_imports'];need(len(imports)==1,'Missing runtime import witness')
        record['source_imports'][str(ordinal)]=imports[0]
        record['services'].append(dict(ordinal=ordinal,http_pid=service.process.pid,ready=ready,startup_seconds=time.monotonic()-t))
        tracker.phase('service_'+str(ordinal)+'_ready');publish(result_path,record)
    def stop():
        nonlocal service,client
        owned=service;pid=owned.process.pid;tracker.phase('stop_service_'+str(ordinal))
        receipt=owned.stop();service=None;client=None
        need(receipt['exit_code']==0 and not Path('/proc',str(pid)).exists(),'HTTP shutdown')
        witness=_trace_worker(output/('pool-'+str(ordinal)+'.json'),pid)
        need(witness['native_exit_code']==0 and not Path('/proc',str(witness['native_pid'])).exists(),'Native shutdown')
        record['services'][-1].update(stop=receipt,worker=witness)
        tracker.native_started(witness['native_pid']);publish(result_path,record)
    def invoke(label,i,*arguments,observed=False,expect=0):
        nonlocal counter
        counter+=1;tracker.phase(label)
        before=stats();before_status=status();base=[sys.executable,'-B',str(cli)]
        trace=output/'calls'/(f'{counter:02d}-{label}.trace.json')
        if observed:base=[sys.executable,'-B',str(a.observer),'--runtime-root',str(runtime),'--trace-file',str(trace),'--']
        cliargs=['--url',_ready_url(record['services'][-1]['ready']),'--state-dir',str(output/('client-'+str(i)+'-journal')),'--timeout','180',*arguments]
        started=time.monotonic();child=subprocess.run(base+cliargs,cwd=runtime,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=210);elapsed=time.monotonic()-started
        prefix=output/'calls'/f'{counter:02d}-{label}';prefix.with_suffix('.stdout').write_bytes(child.stdout);prefix.with_suffix('.stderr').write_bytes(child.stderr)
        after=stats();after_status=status()
        row=dict(label=label,client=i,service=ordinal,command=base+cliargs,exit_code=child.returncode,
            process_seconds=elapsed,observed=observed,trace_file=str(trace) if observed else None,
            stdout_file=str(prefix.with_suffix('.stdout')),stderr_file=str(prefix.with_suffix('.stderr')),
            stdout_sha256=hashlib.sha256(child.stdout).hexdigest(),stderr_sha256=hashlib.sha256(child.stderr).hexdigest(),
            counter_deltas={k:after[k]-v for k,v in before.items() if type(v) in (int,float) and type(after.get(k)) in (int,float)},
            status_before=before_status,status_after=after_status)
        record['calls'].append(row);publish(result_path,record)
        need(child.returncode==expect,label+' CLI failed: '+child.stderr.decode('utf-8','replace')[-1800:])
        text=child.stderr.decode('utf-8');need(text.count('durable receipt:\n')==1,label+' lacks one durable receipt')
        receipt=json.loads(text.split('durable receipt:\n',1)[1]);row['receipt']=receipt
        if observed:row['observation']=read(trace)
        return receipt,row
    def ensure_root(i):
        case=cases[i];branch=case['branch']
        try:head=store.head(branch)
        except NotFound:
            ns,_=invoke('namespace-'+str(i),i,'namespace','--title',case['name'],'--operation-id','op-ns-'+run_id+'-'+str(i))
            receipt,row=invoke('load-'+str(i),i,'load','--namespace-id',ns['namespace_id'],'--branch',branch,'--file',case['context_file'],'--operation-id','op-load-'+run_id+'-'+str(i))
            head=store.head(branch)
            need(row['counter_deltas'].get('prefill_tokens')==case['root_tokens'],'Root prefill work differs')
        need(head.tip==head.root and head.revision==0,'Source root is not untouched idle root')
        need(store.lineage(head.tip)[1]==case['root_token_ids'] and store.get_snapshot(head.tip)['controller']['status']=='idle','CLI root token framing differs')
        record['roots'][str(i)]=dict(branch=branch,head=vars(head),checkpoint_before=checkpoint_fingerprint(a.store,head.tip))
        publish(result_path,record)
    def ask(i,label,tag,observed):
        case=cases[i];branch=f'T{i}{label}{tag}{run_id}';op='op-ask-'+str(i)+'-'+label+'-'+tag+'-'+run_id
        invoke('fork-'+branch,i,'fork','--source-branch',case['branch'],'--name',branch,'--operation-id','op-fork-'+branch)
        source=store.head(branch);source_ids=store.lineage(source.tip)[1]
        receipt,row=invoke('ask-'+branch,i,'ask','--branch',branch,'--text',case['branches'][label]['question'],'--max-tokens',str(a.outputs),'--operation-id',op,observed=observed)
        target=store.head(branch);snapshot=store.get_snapshot(target.tip)
        suffix=case['branches'][label]['suffix_token_ids'];need(snapshot['parent']==source.tip,'Wrong generated parent')
        need(snapshot['inputs'][:len(suffix)]==suffix,'Generated snapshot omitted or changed question inputs')
        tokens=snapshot['inputs'][len(suffix):];need(0<len(tokens)<=a.outputs,'Output count outside limit')
        need(target.revision==source.revision+1 and target.tip==receipt['snapshot'],'Receipt not bound to actual branch head')
        need(store.lineage(target.tip)[1]==source_ids+suffix+tokens,'Wrong canonical tree')
        counters=row['counter_deltas'];need(counters['prefill_tokens']==len(suffix) and counters['forward_tokens']==len(suffix)+len(tokens),'Saved prefix recomputed or work incorrect')
        expected=expected_controller(op,source_ids+suffix,tokens,tokenizer,profile['execution']['eos_tokens'],248320)
        need(snapshot['controller']==expected,'Full controller differs from consumed tokens')
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');pieces=[]
        for token in tokens:
            if token in profile['execution']['eos_tokens']:pieces.append(decoder.decode(b'',final=True));break
            pieces.append(decoder.decode(tokenizer.token_bytes(token),final=False))
        actual=Path(row['stdout_file']).read_bytes().decode('utf-8');need(actual==safe_text(''.join(pieces))+'\n','CLI stdout differs from canonical output IDs')
        manifest=read(a.store/'checkpoints'/(target.tip+'.json'))['manifest'];payload=a.store/'checkpoints'/(manifest['payload_sha256']+'.state')
        with payload.open('rb') as f:f.seek(manifest['state_bytes']);frontier=f.read()
        need(len(frontier)==248320*4,'Incomplete full frontier')
        frontier_path=output/'calls'/(branch+'.f32');frontier_path.write_bytes(frontier)
        gen=dict(call_label=row['label'],operation=op,context_index=i,question=label,tag=tag,source_snapshot=source.tip,snapshot=target.tip,
            branch=branch,output_tokens=tokens,output_count=len(tokens),controller_status=snapshot['controller']['status'],
            source_tokens=len(source_ids),suffix_tokens=len(suffix),final_tokens=len(source_ids)+len(suffix)+len(tokens),
            counter_deltas=counters,process_seconds=row['process_seconds'],observed=observed,trace_file=row['trace_file'],
            frontier_file=str(frontier_path),frontier_sha256=hashlib.sha256(frontier).hexdigest(),
            payload_sha256=manifest['payload_sha256'],payload_bytes=payload.stat().st_size)
        record['generations'].append(gen);publish(result_path,record)
        return gen,row
    error=None
    try:
        start();ensure_root(0)
        if a.phase=='smoke':
            gen,row=ask(0,'A','smoke',False)
            before=store.head(gen['branch'])
            retry,_=invoke('retry-original-operation',0,'retry-request','--operation-id',gen['operation'])
            need(store.head(gen['branch'])==before and retry==row['receipt'],'Idempotent retry changed history or receipt')
        else:
            ensure_root(1);ensure_root(2)
            for i in range(3):ask(i,'A','warmup',True)
            for repeat in range(a.repetitions):
                for question in ('A','B'):
                    for i in range(3):ask(i,question,'round'+str(repeat),True)
            for repeat in range(a.repetitions):
                ask(2,'A','plain'+str(repeat),False);ask(2,'A','observed'+str(repeat),True)
            stop();start()
            for i in range(3):ask(i,'A','disk-restart',True)
        final_status=status();need(final_status['max_active_per_gpu']=={'0':1} and final_status['queued']==0 and not any(final_status['active_per_gpu'].values()),'Single-request worker or final idle invariant')
        record['final_status']=final_status
        stop()
        for i,row in record['roots'].items():
            need(vars(store.head(row['branch']))==row['head'],'Source root reference changed')
            after=checkpoint_fingerprint(a.store,row['head']['tip']);need(after['files']==row['checkpoint_before']['files'],'Source root checkpoint bytes changed')
            row['unchanged']=True
        record['status']='PASS_PIPELINE_NUMERICS_PENDING'
    except BaseException as exc:
        error=exc;record['status']='FAIL';record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        if service is not None:
            try:stop()
            except BaseException as exc:
                record['status']='FAIL';record['failures'].append(dict(type=type(exc).__name__,message='cleanup: '+str(exc)));error=error or exc
        tracker.finish(record['status'],str(error) if error else None);tracker.close();publish(result_path,record)
    print(json.dumps(dict(status=record['status'],calls=len(record['calls']),generations=len(record['generations']),failures=record['failures'])))
    return 1 if error else 0
if __name__=='__main__':raise SystemExit(main())
