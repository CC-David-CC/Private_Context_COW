"""Bounded native Min-P branch/restart audit with an uncached raw reference.

The reference reads full logits and samples in independent Python arithmetic;
the candidate uses the native full-vocabulary sampler. Matching operator/batch
boundaries are required. This is not original G1-G5 certification or a benchmark.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import statistics
import sys
import time


POLICY=dict(temperature=.70,min_p=.05,top_k=0,top_p=1.0,repetition_penalty=1.0)


def full_mask(vocab):
    if type(vocab) is not int or vocab<=0:raise ValueError('Invalid vocabulary')
    return ((1<<vocab)-1).to_bytes((vocab+7)//8,'little')


def frontier_bytes(values,vocab):
    if len(values)!=vocab or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('Invalid complete finite frontier')
    return struct.pack('<'+'f'*vocab,*values)


def check_delta(before,after,suffix,outputs,*,vocab):
    if type(vocab) is not int or vocab<=0:raise ValueError('Invalid vocabulary')
    for key,expected in dict(forward_tokens=suffix+outputs,prefill_tokens=suffix,
                             decode_timed_tokens=outputs,sampling_calls=outputs,
                             sampling_logit_bytes=vocab*4*outputs,
                             sampling_mask_bytes=((vocab+7)//8)*outputs).items():
        if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key]-before[key]!=expected:
            raise AssertionError('Unexpected native work: '+key)


def compare_record(actual,expected):
    for key in ('tokens','controller','samples'):
        if json.dumps(actual[key],sort_keys=True,allow_nan=False)!=json.dumps(expected[key],sort_keys=True,allow_nan=False):
            raise AssertionError('Exact same-schedule comparison failed: '+key)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def reference_choice(session, values, mask):
    """Select with independent Python arithmetic, never the native sampler."""
    from context_time_travel.min_p import choose
    grammar = session._state.get('constraint') is not None
    if session._state['schema'] == 4:
        from context_time_travel.xtc_pipeline import choose_v5
        from context_time_travel.sampling_proposal import SelectionProposal
        def select(legal_mask, temperature, min_p, *, draw_start,
                   draw_candidates, source_digest, **filters):
            if filters.get('adaptive_target', -1.) >= 0.:
                from context_time_travel.adaptive_p_pipeline import choose_v7
                saved = filters['sampler_state']
                selected = choose_v7(values, legal_mask, saved, temperature=temperature,
                    min_p=min_p, draw53=draw_candidates[0],
                    **{key: value for key, value in filters.items() if key != 'sampler_state'})
                return SelectionProposal(selected['token'], draw_start, 1,
                                         selected['feedback_state'], source_digest)
            if filters.get('mirostat') == 2:
                from context_time_travel import mirostat2
                from context_time_travel.min_p import DRY_DEFAULTS, validate_policy, dry_enabled
                saved = mirostat2.validate_state(filters['sampler_state'])
                policy = validate_policy(dict(temperature=temperature, min_p=min_p,
                    repetition_penalty=1., **{key:value for key,value in filters.items()
                        if key not in ('sampler_state','dry_history','dry_breakers')}))
                frontier_bytes(values,len(values))
                scores = list(values)
                if dry_enabled(policy):
                    from context_time_travel.dry import FLOAT32_MAX, penalties_from_ids
                    penalties = penalties_from_ids(filters['dry_history'], filters['dry_breakers'],
                        **{key:policy[key] for key in DRY_DEFAULTS})
                    for token,penalty in penalties.items():
                        adjusted = max(-FLOAT32_MAX,scores[token]-penalty)
                        scores[token] = struct.unpack('<f',struct.pack('<f',adjusted))[0]
                selected = mirostat2.choose(scores,legal_mask,temperature=temperature,
                    draw53=draw_candidates[0],mu=saved['mu'],
                    tau=policy['mirostat_tau'],eta=policy['mirostat_eta'])
                feedback = dict(saved,mu=selected['next_mu'])
                return SelectionProposal(selected['token'],draw_start,1,feedback,source_digest)
            selected = choose_v5(values, legal_mask, temperature=temperature,
                min_p=min_p, draw53=draw_candidates[0] if draw_candidates else 0,
                draw53_second=draw_candidates[1] if len(draw_candidates)>1 else 1<<53,
                **filters)
            return SelectionProposal(selected['token'], draw_start,
                                     selected['draw_count'], None, source_digest)
        return session._choose_proposal(select) if grammar else session.choose(mask, select)
    if grammar:
        return session.choose(values)
    return session.choose(mask, lambda m,t,p,d,**filters:
        choose(values,m,temperature=t,min_p=p,draw53=d,**filters)['token'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('campaign','reference-A','reference-B','prepare','resume'),required=True)
    for name in ('runtime-root','profile','fixture','run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--grammar',type=Path)
    parser.add_argument('--prefix',type=int,default=128)
    parser.add_argument('--suffix',type=int,default=4)
    parser.add_argument('--outputs',type=int,default=8)
    parser.add_argument('--timed-repetitions',type=int,default=0)
    parser.add_argument('--sampling-json',type=json.loads)
    parser.add_argument('--resume-sampling-json',type=json.loads)
    args=parser.parse_args()
    if args.resume_sampling_json is not None and not args.stage.startswith('reference'):
        raise ValueError('Policy-switch reference is for the serving/sweep harness only')
    if not(0<args.prefix<=8192 and 0<args.suffix<=512 and 4<args.outputs<=128):
        raise ValueError('Unsupported bounded fixture size')
    if not 0<=args.timed_repetitions<=3:raise ValueError('Unsupported repetition count')
    runtime=args.runtime_root.resolve(strict=True)
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),
                  str(runtime/'experimental/branching_context/scripts')]
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from grammar_long_reference import NativePipe,Heartbeat,_native_environment,_apply_profile_environment
    from grammar_long_reference import _sample_frontier,validate_feed_receipt,validate_decode_receipt
    from run_grammar_first_probe import publish,run_stage,finish_backend
    from context_time_travel.strata_backend import StrataBackend,_validate_execution,_validate_model
    from context_time_travel.real import BranchingModel
    from context_time_travel.sampled_controller import PlainSamplingSession,start
    from context_time_travel.min_p import validate_policy
    from context_time_travel.grammar_first import GrammarEngine
    from tools.strata_tokenizer import Tokenizer
    profile=read(args.profile)
    policy=validate_policy(POLICY if args.sampling_json is None else args.sampling_json)
    resumed_policy=None if args.resume_sampling_json is None else validate_policy(args.resume_sampling_json)
    _apply_profile_environment(profile)
    _validate_execution(profile)
    output=args.run_dir.absolute()
    if args.stage=='campaign':
        output.mkdir(mode=0o700,parents=True,exist_ok=False)
        stages=[]
        result=dict(status='RUNNING',stages=stages,scope=__doc__,script_sha256=sha(__file__),
                    profile_sha256=sha(args.profile),prefix=args.prefix,suffix=args.suffix,outputs=args.outputs)
        publish(output/'campaign.json',result)
        for stage in ('reference-A','reference-B','prepare','resume'):
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--stage',stage,
                '--runtime-root',str(runtime),'--profile',str(args.profile.resolve()),
                '--fixture',str(args.fixture.resolve()),'--run-dir',str(output),
                '--prefix',str(args.prefix),'--suffix',str(args.suffix),'--outputs',str(args.outputs),
                '--timed-repetitions',str(args.timed_repetitions)]
            if args.grammar:command+=['--grammar',str(args.grammar.resolve())]
            if args.sampling_json is not None:command+=['--sampling-json',json.dumps(policy)]
            receipt=run_stage(command,output/(stage+'.stdout'),output/(stage+'.stderr'))
            stages.append(dict(stage=stage,**receipt))
            target=output/stage/'result.json'
            child=read(target) if target.exists() else {}
            healthy=(receipt['exit_code']==0 and receipt['pid_absent'] and not receipt['live_group_pids']
                     and not receipt['timed_out'] and not receipt['cleanup_needed'] and child.get('status')=='PASS'
                     and child.get('native_exit_code')==0 and child.get('native_pid_absent') is True)
            result['status']='RUNNING' if healthy else 'FAIL'
            publish(output/'campaign.json',result)
            if not healthy:raise AssertionError('Stage failed: '+stage)
        result['status']='PASS'
        result['numerical_max_error']=0
        publish(output/'campaign.json',result)
        return
    directory=output/args.stage
    directory.mkdir(mode=0o700,exist_ok=False)
    heartbeat=Heartbeat(directory/'heartbeat.json').start()
    result=dict(status='RUNNING',python_pid=os.getpid(),failures=[],rows=[],samples={},
        profile_sha256=sha(args.profile),script_sha256=sha(__file__),
        controller_scope=dict(policy=policy,resumed_policy=resumed_policy,grammar=bool(args.grammar)),
        frontier_sampling='every generated frontier',numerical_max_error=None)
    publish(directory/'result.json',result)
    backend=None
    raw=None
    try:
        result['model_verification']=_validate_model(profile)
        case=read(args.fixture)
        if len(case['root_token_ids'])<args.prefix:raise ValueError('Fixture prefix too short')
        prefix=tuple(case['root_token_ids'][:args.prefix])
        originals={b:case['branches'][b]['suffix_token_ids'] for b in ('A','B')}
        # Keep the original closing user/assistant template. Arbitrarily taking
        # the first few suffix IDs can leave the model inside a user message.
        suffixes={branch:tuple(originals[branch][-args.suffix:]) for branch in ('A','B')}
        if any(len(v)!=args.suffix for v in suffixes.values()) or suffixes['A']==suffixes['B']:
            raise ValueError('Need two distinct complete suffixes')
        selected=dict(source_fixture=str(args.fixture),source_sha256=sha(args.fixture),
            prefix_token_ids=list(prefix),suffix_token_ids={b:list(v) for b,v in suffixes.items()},
            selection='first N prefix IDs; last M suffix IDs, retaining the assistant boundary')
        publish(directory/'selected-fixture.json',selected)
        result['selected_fixture_sha256']=sha(directory/'selected-fixture.json')
        tokenizer=Tokenizer.from_gguf(profile['model_path'])
        vocab=len(tokenizer.tokens)
        eos=set(profile['execution']['eos_tokens'])
        mask=full_mask(vocab)
        engine=None if args.grammar is None else GrammarEngine(tokenizer,
            {'REVIEW':{'format':'gbnf','source':args.grammar.read_text()}},eos_tokens=profile['execution']['eos_tokens'])
        def session_for(branch,history):
            if engine:return engine.restore(engine.start('REVIEW','minp-'+branch,history,vocab,sampling=policy),history)
            state=start(policy,'minp-'+branch,history,vocab,profile['execution']['eos_tokens'])
            return PlainSamplingSession.restore(state,history,vocab,tokenizer.token_bytes,lambda t:t in eos)
        def session_choice(session,values):
            return reference_choice(session,values,mask)
        def sample(raw_bytes,label):
            frontier_bytes(struct.unpack('<'+'f'*vocab,raw_bytes),vocab)
            path=directory/(label+'.f32')
            path.write_bytes(raw_bytes)
            return sha(path)
        def request(model,snapshot,suffix,count,branch,resume=False):
            if engine:return model.run_state_request(snapshot,suffix,count,engine=engine,state='REVIEW',
                stream='minp-'+branch,sampling=policy,resume=resume)
            return model.run_sampled_request(snapshot,suffix,count,sampling=policy,stream='minp-'+branch,resume=resume)
        heartbeat.set('native-start')
        started=time.monotonic()
        if args.stage.startswith('reference'):
            branch=args.stage[-1]
            result['reference_scope']='Fresh raw native feed/decode/logits/stats only; no adapter, cache, checkpoint or native sampler'
            raw=NativePipe(profile['execution']['sut_command'],_native_environment(profile),directory)
            result['native_pid']=raw.process.pid
            result['startup_seconds']=time.monotonic()-started
            if raw.startup_error:raise raw.startup_error
            if raw.ready.get('sampling_protocol')!=profile['execution']['sampling_protocol'] or raw.ready.get('vocab')!=vocab:
                raise AssertionError('Native sampler/vocabulary mismatch')
            position=0
            samples={}
            for label,ids in [('prefix',prefix),('suffix',suffixes[branch])]:
                heartbeat.set(label,raw.process.pid)
                receipt=raw.request('feed '+' '.join(map(str,ids)),'feed')
                validate_feed_receipt(receipt,position=position,count=len(ids),chunk_size=profile['execution']['prefill_chunk'])
                position+=len(ids)
                descriptor=_sample_frontier(raw,directory,label,position,vocab)
                samples[label]=descriptor['sha256']
            session=session_for(branch,prefix+suffixes[branch])
            tokens=[]
            for n in range(1,args.outputs+1):
                if n==5 and branch=='A' and resumed_policy is not None:
                    saved=session.current_controller
                    history=prefix+suffixes[branch]+tuple(tokens)
                    session=(engine.restore(saved,history) if engine else
                        PlainSamplingSession.restore(saved,history,vocab,tokenizer.token_bytes,lambda t:t in eos))
                    session.configure_generation(resumed_policy)
                heartbeat.set('sample-'+str(n),raw.process.pid)
                source=directory/('suffix.f32' if n==1 else 'generated-'+str(n-1)+'.f32')
                values=struct.unpack('<'+'f'*vocab,source.read_bytes())
                token=session_choice(session,values)
                receipt=raw.request('decode '+str(token),'decode')
                validate_decode_receipt(receipt,position=position)
                position+=1
                session._consume_selected(token,tokenizer.token_bytes(token),token in eos)
                tokens.append(token)
                samples['generated-'+str(n)]=_sample_frontier(raw,directory,'generated-'+str(n),position,vocab)['sha256']
                if n==4:result['after_four_controller']=session.current_controller
                result.update(tokens=tokens,samples=samples,controller=session.current_controller)
                publish(directory/'result.json',result)
                if session.is_stopped():break
            if len(tokens)<=4:raise AssertionError('Early stop prevented nonzero-draw restart fixture')
            result['stats']=raw.request('stats','stats')
            assert result['stats']['forward_tokens']==len(prefix)+len(suffixes[branch])+len(tokens)
            assert result['stats']['sampling_calls']==0
            result['commands']=raw.commands
        else:
            backend=StrataBackend(profile)
            result['native_pid']=backend.process.pid
            result['startup_seconds']=time.monotonic()-started
            heartbeat.set('open-store',backend.process.pid)
            model=BranchingModel(output/'store',backend,profile)
            references={b:read(output/('reference-'+b)/'result.json') for b in ('A','B')}
            for ref in references.values():
                assert ref['status']=='PASS' and ref['native_pid_absent'] is True
                assert not Path('/proc',str(ref['native_pid'])).exists()
                assert not Path('/proc',str(ref['python_pid'])).exists()
            if args.stage=='prepare':
                before=backend.work_counters()
                started=time.monotonic();root=model.create(prefix)
                result['prefix_seconds']=time.monotonic()-started
                assert backend.work_counters()['forward_tokens']-before['forward_tokens']==len(prefix)
                head=model.store.fork('Prefix',root)
                model.save_branch(head.name,head.revision,root,'save-prefix')
                result['root']=root
                result['parent_audit_before']=backend.audit(model.cache[root].blob)
                branch='A';suffix=suffixes[branch];count=4;snapshot=root;resume=False
                wanted=dict(tokens=references['A']['tokens'][:4],controller=references['A']['after_four_controller'],
                    samples={k:v for k,v in references['A']['samples'].items() if k in ('prefix','suffix') or k.startswith('generated-') and int(k.split('-')[1])<=4})
                tasks=[(branch,snapshot,suffix,count,resume,wanted)]
            else:
                prepared=read(output/'prepare/result.json')
                assert prepared['native_pid_absent'] is True and prepared['status']=='PASS'
                assert not Path('/proc',str(prepared['native_pid'])).exists()
                assert not Path('/proc',str(prepared['python_pid'])).exists()
                root=prepared['root']
                tasks=[(b,root,suffixes[b],args.outputs,False,references[b]) for b in ('A','B')]
                remaining=len(references['A']['tokens'])-4
                # The reference retains the exact original output scope and RNG cursor.
                expected=dict(tokens=references['A']['tokens'][4:],controller=references['A']['controller'],
                    samples={k:v for k,v in references['A']['samples'].items() if k.startswith('generated-') and int(k.split('-')[1])>4})
                tasks.append(('A',prepared['after_four'],(),remaining,True,expected))
            for task_index,(branch,snapshot,suffix,count,resume,wanted) in enumerate(tasks):
                label=branch+('-continued' if resume else '')
                heartbeat.set('restore-'+label,backend.process.pid)
                receipt=None
                if args.stage=='resume':
                    receipt=model.restore_checkpoint(snapshot)
                    assert receipt['forward_tokens']==0 and receipt['consumed']==len(model.history(snapshot))
                original=model.snapshot_bytes(root)
                image=model.cache[root].blob
                audit_before=backend.audit(image)
                samples={}
                if not resume:
                    samples['prefix']=sample(frontier_bytes(model.cache[snapshot].frontier,vocab),label+'-prefix')
                original_feed,original_decode=backend.consume,backend.consume_generated
                generated=4 if resume else 0
                def feed(lease,ids):
                    assert tuple(ids)==tuple(suffix) and 'suffix' not in samples
                    original_feed(lease,ids)
                    samples['suffix']=sample(frontier_bytes(backend.logits(lease),vocab),label+'-suffix')
                def decode(lease,ids):
                    nonlocal generated
                    original_decode(lease,ids)
                    generated+=len(ids)
                    samples['generated-'+str(generated)]=sample(frontier_bytes(backend.logits(lease),vocab),label+'-generated-'+str(generated))
                before=backend.work_counters()
                heartbeat.set('generate-'+label,backend.process.pid)
                backend.consume,backend.consume_generated=feed,decode
                try:answer=request(model,snapshot,suffix,count,branch,resume)
                finally:backend.consume,backend.consume_generated=original_feed,original_decode
                after=backend.work_counters()
                check_delta(before,after,len(suffix),len(answer['tokens']),vocab=vocab)
                actual=dict(tokens=answer['tokens'],controller=model.controller(answer['snapshot']),samples=samples)
                for name,digest in samples.items():
                    if digest!=wanted['samples'].get(name):
                        expected_file=output/('reference-'+branch)/(name+'.f32')
                        actual_file=directory/(label+'-'+name+'.f32')
                        left=struct.unpack('<'+'f'*vocab,actual_file.read_bytes())
                        right=struct.unpack('<'+'f'*vocab,expected_file.read_bytes())
                        result['numerical_max_error']=max(abs(a-b) for a,b in zip(left,right))
                        result['mismatch_frontier']=label+':'+name
                        raise AssertionError('Matching-schedule logit mismatch')
                compare_record(actual,wanted)
                assert model.history(answer['snapshot'])==model.history(snapshot)+tuple(suffix)+tuple(answer['tokens'])
                assert model.snapshot_bytes(root)==original
                assert backend.audit(image)['sha256']==audit_before['sha256']
                row=dict(label=label,answer=answer,restore_receipt=receipt,counters_before=before,counters_after=after,
                    comparisons=len(samples),controller_next_draw=actual['controller']['next_draw'],samples=samples)
                result['rows'].append(row)
                if args.stage=='prepare':
                    head=model.store.fork('AfterFour',answer['snapshot'])
                    model.save_branch(head.name,head.revision,answer['snapshot'],'save-four')
                    result['after_four']=answer['snapshot']
                publish(directory/'result.json',result)
            if args.stage=='resume' and args.timed_repetitions:
                # These rows exclude per-token full-frontier audit reads and
                # whole-parent byte audits. Integrity checks in the backend
                # remain enabled. Validate every timed answer, then its final
                # full vector, outside the measured request interval.
                result['timed_rows']=[]
                for repetition in range(args.timed_repetitions+1):
                    for branch in ('A','B'):
                        heartbeat.set('resident-timing-'+branch,backend.process.pid)
                        before=backend.work_counters()
                        consumed_times=[]
                        def on_token(token):consumed_times.append(time.monotonic())
                        started=time.monotonic()
                        if engine:
                            answer=model.run_state_request(root,suffixes[branch],args.outputs,engine=engine,state='REVIEW',
                                sampling=policy,stream='minp-'+branch,on_token=on_token)
                        else:
                            answer=model.run_sampled_request(root,suffixes[branch],args.outputs,sampling=policy,
                                stream='minp-'+branch,on_token=on_token)
                        elapsed=time.monotonic()-started
                        after=backend.work_counters()
                        check_delta(before,after,len(suffixes[branch]),len(answer['tokens']),vocab=vocab)
                        assert after['loaded_file_bytes']==before['loaded_file_bytes']
                        last='generated-'+str(len(answer['tokens']))
                        final=frontier_bytes(model.cache[answer['snapshot']].frontier,vocab)
                        actual=dict(tokens=answer['tokens'],controller=model.controller(answer['snapshot']),
                                    samples={last:hashlib.sha256(final).hexdigest()})
                        expected=dict(references[branch],samples={last:references[branch]['samples'][last]})
                        compare_record(actual,expected)
                        intervals=[b-a for a,b in zip(consumed_times,consumed_times[1:])]
                        row=dict(phase='warmup' if repetition==0 else 'measured',repetition=repetition,branch=branch,
                            output_count=len(answer['tokens']),generation_seconds=answer['generation_seconds'],
                            generation_tps=len(answer['tokens'])/answer['generation_seconds'],request_seconds=elapsed,
                            request_tps=len(answer['tokens'])/elapsed,first_consumed_seconds=consumed_times[0]-started,
                            median_inter_consumed_seconds=statistics.median(intervals) if intervals else None,
                            counters_before=before,counters_after=after,final_frontier_sha256=actual['samples'][last])
                        result['timed_rows'].append(row)
                        publish(directory/'result.json',result)
                result['timing_medians']={branch:{key:statistics.median(row[key] for row in result['timed_rows']
                    if row['phase']=='measured' and row['branch']==branch) for key in
                    ('generation_tps','request_seconds','request_tps','first_consumed_seconds','median_inter_consumed_seconds')}
                    for branch in ('A','B')}
            result['stats']=backend.work_counters()
        result['numerical_max_error']=0
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL'
        result['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if backend is not None:finish_backend(backend,result)
        if raw is not None:
            code,absent,errors=raw.close()
            result.update(native_exit_code=code,native_pid_absent=absent,native_cleanup_errors=errors)
            if code!=0 or not absent or errors:result['status']='FAIL'
        heartbeat.close()
        publish(directory/'result.json',result)
    if result['status']!='PASS':raise AssertionError('Native cleanup failed')


if __name__=='__main__':main()
