"""Two hard-coded long cases: saved 32K/64K + 8K context + up to 128 greedy outputs."""
import argparse
import array
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('api',type=Path)
    parser.add_argument('original_profile',type=Path)
    parser.add_argument('binaries',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.api/'src'))
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend,_digest,_validate_model,_validate_execution
    from context_time_travel.checkpoint_cache import CheckpointCache
    from context_time_travel import controller
    original=json.loads(args.original_profile.read_text())
    for key in list(os.environ):
        if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
    os.environ.update(original['execution']['environment'])
    _validate_execution(original)
    verification=_validate_model(original)
    profile=copy.deepcopy(original)
    capacity,chunk,added=(4096,128,128) if args.smoke else (81920,8192,8192)
    output_bound=8 if args.smoke else 128
    lengths=[128] if args.smoke else [32768,65536]
    assert capacity >= max(lengths)+added+output_bound, 'Capacity leaves no room for output'
    for key,binary in [('sut_command','strata-batched-branch'),('oracle_command','strata-batched-fresh')]:
        command=profile['execution'][key]
        command[0]=str(args.binaries/binary)
        for flag,value in [('--max-context',capacity),('--prefill',chunk),('--max-new',output_bound)]:
            command[command.index(flag)+1]=str(value)
    assert profile['execution']['eos_tokens']==[248044,248046], 'Native stopping ids must match inspected model ids'
    profile['execution'].update(capacity=capacity,operator='batched-prefill-plus-normal-verifier-decode-experimental',
        derived_checkpoint_byte_limit=16*1024**3,native_ram_checkpoint_byte_limit=16*1024**3,
        benchmark_sampling=dict(temperature=0,output_work_bound=output_bound,eos_tokens=[248044,248046],excluded_ids=[]))
    profile['model_contract']['probe_capacity']=capacity
    profile['model_contract_id']=_digest(profile['model_contract'])
    profile['native_build']['revision']='isolated-long-batched-prefill-probe-20261013'
    for name in ('strata-batched-branch','strata-batched-fresh'):
        path=args.binaries/name;profile['native_build']['sha256'][str(path)]=sha(path)
    profile['native_build']['sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=profile['execution'],hardware=profile['hardware']))
    profile['numerical_gate_qualified']=False
    (args.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    report=dict(schema=1,status='RUNNING',gate_status='NOT_QUALIFIED',scope='fixed aligned prefix/chunk prototype; no full acceptance qualification',
        hardware=profile['hardware'],model_verification=verification,profile_sha256=sha(args.output/'profile.json'),
        model_contract_id=profile['model_contract_id'],execution_profile_id=profile['execution_profile_id'],
        sampling=profile['execution']['benchmark_sampling'],cases=[],phase='start')
    def save():
        report['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        pending=args.output/'result.json.tmp'
        pending.write_text(json.dumps(report,indent=2)+'\n')
        pending.replace(args.output/'result.json')
    def stage(phase,**fields):
        report.update(phase=phase,**fields);save()
        print(json.dumps(dict(phase=phase,**fields)),flush=True)
    def values(path):
        out=array.array('f');out.frombytes(Path(path).read_bytes())
        if len(out)!=248320: raise AssertionError('Incomplete vocabulary frontier')
        return out
    def equal(actual,expected,label,row):
        if len(actual)!=len(expected): raise AssertionError('Incomplete reference vector')
        if any(not math.isfinite(x) for vector in (actual,expected) for x in vector):
            raise AssertionError('Nonfinite numerical frontier')
        error=max(abs(a-b) for a,b in zip(actual,expected))
        row.setdefault('comparisons',[]).append(dict(label=label,max_absolute_error=error,logits=len(actual)))
        save()
        if error!=0: raise AssertionError(label+': '+str(error))
    save()
    try:
        for length in lengths:
            case_dir=args.output/('prefix-'+str(length));case_dir.mkdir()
            seed=tuple(profile['fixtures']['prefix'])
            prefix=(seed*((length+len(seed)-1)//len(seed)))[:length]
            tail_seed=tuple(profile['fixtures']['suffix_a']+profile['fixtures']['suffix_b'])
            tail=(tail_seed*((added+len(tail_seed)-1)//len(tail_seed)))[:added]
            case=dict(id='LONG-'+str(length),prefix_tokens=length,added_tokens=added,requested_output_bound=output_bound,status='RUNNING',paths=[])
            report['cases'].append(case)
            tokens=case_dir/'prompt.tokens';tokens.write_text(','.join(map(str,prefix+tail)))
            oracle_path=case_dir/'fresh.f32'
            command=list(profile['execution']['oracle_command'])+['--tokens-file',str(tokens)]
            stage('independent fresh batch and decode',prefix_tokens=length)
            started=time.monotonic()
            with (case_dir/'fresh.stdout').open('wb') as out,(case_dir/'fresh.stderr').open('wb') as err:
                result=subprocess.run(command,env=dict(os.environ,STRATA_DUMP_FIRST_LOGITS=str(oracle_path)),stdout=out,stderr=err,timeout=3600)
            if result.returncode: raise RuntimeError('Fresh native oracle exited '+str(result.returncode))
            oracle=json.loads((case_dir/'fresh.stdout').read_text())
            outputs=len(oracle['generated'])
            if not 0<outputs<=output_bound or oracle['prefill_tokens']!=length+added: raise AssertionError('Incomplete independent work')
            if outputs<output_bound and oracle['generated'][-1] not in profile['execution']['eos_tokens']: raise AssertionError('Unexplained early stop')
            if oracle['consumed']!=length+added+outputs or oracle['decode_tokens']!=outputs: raise AssertionError('Unconsumed reference output')
            if Path(str(oracle_path)+'.decode.f32').stat().st_size!=outputs*248320*4: raise AssertionError('Incomplete output trace')
            case['actual_generated_tokens']=outputs
            case['oracle']=dict(command=command,prompt_tokens=length+added,outputs=oracle['generated'],counters=oracle,
                process_seconds=time.monotonic()-started,frontier_sha256=sha(oracle_path),trace_sha256=sha(str(oracle_path)+'.decode.f32'))
            case['recompute_baseline']=dict(measured_requests=1,extrapolated_requests=8,
                measured_prefill_seconds=oracle['prefill_seconds'],measured_decode_seconds=oracle['decode_seconds'],
                extrapolated_prefill_seconds=8*oracle['prefill_seconds'],extrapolated_decode_seconds=8*oracle['decode_seconds'],
                limitation='Single uncached native run multiplied by eight; excludes loading, canonical publication and checkpoint capture.')
            expected=values(oracle_path)
            backend=StrataBackend(profile)
            backend.new_cache=lambda:CheckpointCache(16*1024**3,256)
            try:
                sut=BranchingModel(case_dir/'store',backend,profile)
                stage('prepare immutable prefix',prefix_tokens=length)
                started=time.monotonic();root=sut.create(prefix)
                case['prepare_seconds']=time.monotonic()-started
                held=sut.cache[root].blob
                root_bytes=sut.snapshot_bytes(root)
                audit=backend.audit(held)
                started=time.monotonic();files=sut.checkpoint(root)
                case['save_seconds']=time.monotonic()-started
                payload_sha=sha(files['payload_path']);manifest_sha=sha(files['manifest_path'])
                case['checkpoint']=dict(bytes=files['payload_path'].stat().st_size,sha256=payload_sha,manifest_sha256=manifest_sha)
                lanes=('recompute','resident','disk') if args.smoke else ('resident',)*4+('disk',)*4
                for request_index,lane in enumerate(lanes):
                    branch_name=lane+'-'+str(request_index)
                    stage(lane+' prefill',prefix_tokens=length,request_index=request_index)
                    branch=sut.store.fork(branch_name,root)
                    if lane=='disk': sut.cache.clear();backend.release_checkpoints()
                    before=backend._request('stats')
                    row=dict(lane=lane,request_index=request_index,status='RUNNING');case['paths'].append(row);save()
                    started=time.monotonic()
                    manager=backend.lease() if lane=='recompute' else sut.materialize(root)
                    with manager as materialized:
                        lease=materialized if lane=='recompute' else materialized[0]
                        if lane=='recompute': backend.consume(lease,prefix)
                        after_restore=time.monotonic()
                        backend.consume(lease,tail)
                        logits=backend.logits(lease)
                        after_tail=time.monotonic()
                        equal(logits,expected,lane+' prompt frontier',row)
                        sid=sut.store.append(root,tail,{'status':'idle'})
                        decoder=controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),'long-prefill-greedy',prefix+tail,backend.vocab_size)
                        sid=sut.store.append(sid,(),decoder)
                        after_prefill=backend._request('stats')
                        generated=[]
                        row.update(restore_or_recompute_seconds=after_restore-started,added_prefill_seconds=after_tail-after_restore,
                            prompt_frontier_seconds=after_tail-started,prefill_counter_delta=after_prefill['prefill_tokens']-before['prefill_tokens'],
                            native_load_bytes=after_prefill['loaded_file_bytes']-before['loaded_file_bytes'])
                        if row['prefill_counter_delta']!=(length+added if lane=='recompute' else added): raise AssertionError('Repeated prefix work or missing suffix work')
                        if (lane=='disk')!=(row['native_load_bytes']>0): raise AssertionError('Wrong disk/resident path')
                        stage(lane+' decode',prefix_tokens=length,accepted_outputs=0)
                        decode_start=time.monotonic()
                        with Path(str(oracle_path)+'.decode.f32').open('rb') as trace:
                            for index in range(output_bound):
                                if decoder['status']=='stopped': break
                                token,next_decoder=controller.choose(logits,decoder)
                                response=backend._request('decode '+str(token))
                                if response['consumed']!=lease.consumed+1: raise AssertionError('Generated token not consumed')
                                lease.consumed+=1;lease.image_identity=None
                                logits=backend.logits(lease)
                                ref=array.array('f');ref.frombytes(trace.read(248320*4))
                                equal(logits,ref,lane+' output '+str(index),row)
                                if index>=outputs or token!=oracle['generated'][index]: raise AssertionError('Output token disagreement')
                                decoder=controller.consume(next_decoder,token,backend.piece(token),backend.is_eog(token))
                                sid=sut.store.append(sid,(token,),decoder) # durable publication after actual consumption
                                generated.append(token)
                                if index%16==15: stage(lane+' decode',prefix_tokens=length,accepted_outputs=index+1)
                        if generated!=oracle['generated']: raise AssertionError('Output count or token disagreement')
                        row['decode_seconds']=time.monotonic()-decode_start
                        row.update(actual_generated_tokens=len(generated),controller_status=decoder['status'],
                            stopping_reason='eos' if decoder['status']=='stopped' else 'output_bound')
                        if row['stopping_reason']!=oracle['stopping_reason']: raise AssertionError('Stop-state disagreement')
                        saved=sut._save(lease,prefix+tail+tuple(generated))
                    sut.cache[sid]=saved
                    sut.store.move(branch_name,branch.revision,sid,'completed-'+branch_name)
                    row['total_seconds']=time.monotonic()-started
                    row['outputs']=generated
                    row['snapshot']=sid
                    after=backend._request('stats')
                    row['counters']={k:after[k]-before[k] for k in ('forward_tokens','prefill_tokens','windows','loaded_file_bytes','cow_pages','copied_bytes')}
                    if row['counters']['windows']!=len(generated) or row['counters']['forward_tokens']!=row['prefill_counter_delta']+len(generated): raise AssertionError('Decode/prefill work counter mismatch')
                    if sut.history(sid)!=prefix+tail+tuple(generated): raise AssertionError('Wrong canonical history')
                    if sut.snapshot_bytes(root)!=root_bytes: raise AssertionError('Mutated immutable root')
                    current_audit=backend.audit(held)
                    if audit['sha256']!=current_audit['sha256']: raise AssertionError('Changed saved native prefix bytes')
                    row['parent_audit']=dict(bytes=current_audit['bytes'],sha256=current_audit['sha256'])
                    row['sampled_memory']=dict(sample='after completed request; not peak')
                    for name,pid in [('python',os.getpid()),('native',backend.process.pid)]:
                        status=Path('/proc/'+str(pid)+'/status').read_text()
                        row['sampled_memory'][name+'_rss_bytes']=int(next(line for line in status.splitlines() if line.startswith('VmRSS:')).split()[1])*1024
                    row['sampled_memory']['gpu_used_mib']=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    if sha(files['payload_path'])!=payload_sha or sha(files['manifest_path'])!=manifest_sha: raise AssertionError('Changed disk checkpoint')
                    row['status']='PASS';save()
                case['measured_reuse_totals']={lane:dict(requests=sum(p['lane']==lane for p in case['paths']),
                    prompt_frontier_seconds=sum(p['prompt_frontier_seconds'] for p in case['paths'] if p['lane']==lane),
                    decode_seconds=sum(p['decode_seconds'] for p in case['paths'] if p['lane']==lane),
                    whole_request_seconds=sum(p['total_seconds'] for p in case['paths'] if p['lane']==lane)) for lane in set(lanes)}
                case['status']='PASS';stage('size complete',prefix_tokens=length)
            finally:
                backend.close()
        report['status']='PASS';stage('complete')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));stage('failed');raise


if __name__=='__main__':main()
